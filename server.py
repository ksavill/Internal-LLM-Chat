# Standard library imports
import json
import logging
import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Third-party library imports
import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------
# Database Connection Pool
# -----------------------
db_pool = ThreadPoolExecutor(max_workers=10)  # Adjust based on expected database load

async def get_db_connection_async():
    """
    Asynchronous wrapper around the database connection.
    Uses a thread pool to avoid blocking the event loop.
    """
    def _get_conn():
        import sqlite3
        conn = sqlite3.connect('conversations.db')
        conn.row_factory = sqlite3.Row
        return conn
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(db_pool, _get_conn)

# -----------------------
# Graceful Shutdown Handler
# -----------------------
def handle_sigterm(*args):
    """
    Handle SIGTERM signal for graceful shutdown.
    This is especially important in containerized environments.
    """
    raise KeyboardInterrupt()

# Register signal handlers
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

# -----------------------
# Application Lifecycle Management
# -----------------------
@asynccontextmanager
async def lifespan(app):
    """
    Manages application lifecycle - startup and shutdown events.
    Used to initialize and cleanup shared resources.
    """
    # Create a semaphore to limit concurrent model requests
    app.state.model_semaphore = asyncio.Semaphore(10)  # Adjust based on your server capacity
    
    # Create connection pools for APIs that will be shared across requests
    app.state.openai_interfaces = {}  # Cache of OpenAI interfaces by model name
    
    # Tracker for background tasks
    app.state.background_tasks = set()
    
    # A tracker for active streaming connections
    app.state.active_streams = set()
    
    # Initialize database
    from database import init_db
    init_db()
    
    logger.info("Server startup complete - ready to handle requests")
    
    yield  # Server is running and processing requests
    
    # Shutdown cleanup
    logger.info("Server shutting down - cleaning up resources")
    
    # Close any open OpenAI sessions
    close_tasks = []
    for interface in app.state.openai_interfaces.values():
        if hasattr(interface, 'close') and callable(interface.close):
            close_tasks.append(interface.close())
    
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    # Cancel any active background tasks
    if app.state.background_tasks:
        for task in app.state.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete/cancel
        await asyncio.gather(*app.state.background_tasks, return_exceptions=True)
    
    logger.info("Server shutdown complete")

# -----------------------
# Application Setup
# -----------------------
app = FastAPI(
    title="LLM API Server",
    description="A FastAPI server for handling LLM requests with Ollama and OpenAI",
    lifespan=lifespan,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    max_request_body_size=20 * 1024 * 1024,  # 20MB for image uploads
)

# -----------------------
# Import Local Applications
# -----------------------
from async_ollama_interface import AsyncOllamaInterface
from openai_api import OpenAIInterface
from database import (
    create_conversation, create_user, get_conversation,
    get_db_connection, get_user_conversations,
    update_conversation, verify_user
)
from ollama_list_models import list_models

# -----------------------
# Configuration
# -----------------------
SECRET_KEY = "kebin"  # Consider loading from environment variables in production

# Security setup for JWT
security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

# -----------------------
# Background task management
# -----------------------
def register_background_task(app, coro):
    """
    Register and track a background task with proper cleanup.
    """
    task = asyncio.create_task(coro)
    app.state.background_tasks.add(task)
    task.add_done_callback(lambda t: app.state.background_tasks.remove(t))
    return task

# -----------------------
# Authentication Functions
# -----------------------
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload["user_id"]
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security)) -> Optional[int]:
    """
    If a token is provided and valid, return user_id; otherwise return None.
    """
    if credentials is None:
        return None
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("user_id")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

# -----------------------
# Data Models for Chat API
# -----------------------
class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None  # Support images in conversation messages

class ChatRequest(BaseModel):
    model: Optional[str] = "qwen2.5-coder:7b"
    backup_model: Optional[str] = "llama3.2"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    image_b64: Optional[Union[str, List[str]]] = None
    conversation_id: Optional[str] = None

class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

# -----------------------
# Helper: Improved interface management
# -----------------------
async def get_interface(app, model_name: str):
    """
    Get an appropriate interface for the specified model with efficient connection pooling.
    Caches OpenAI interfaces to reuse sessions and reduce connection overhead.
    """
    openai_like = [
        "gpt-4", "gpt-4o", "gpt-3.5",
        "gpt-4o-mini", "o1", "o1-mini",
        "o3", "o3-mini"
    ]
    
    # Determine if we should use OpenAI interface
    is_openai = any(model_name.startswith(m) for m in openai_like)
    
    if is_openai:
        # Try to get a cached interface for this model
        if model_name in app.state.openai_interfaces:
            return app.state.openai_interfaces[model_name]
        
        # Create a new interface and cache it
        interface = OpenAIInterface(model=model_name)
        app.state.openai_interfaces[model_name] = interface
        return interface
    else:
        # For Ollama, create a new instance each time
        # because AsyncClient manages its own session
        return AsyncOllamaInterface(model=model_name)

# -----------------------
# Authentication Endpoints
# -----------------------
@app.post("/signup")
async def signup(signup_req: SignupRequest):
    try:
        create_user(signup_req.username, signup_req.password)
        return {"message": "User created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login(login_req: LoginRequest):
    user_id = verify_user(login_req.username, login_req.password)
    if user_id:
        payload = {
            "user_id": user_id,
            "username": login_req.username,
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        return {"token": token}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

# -----------------------
# Conversation Endpoints
# -----------------------
@app.get("/conversations")
async def get_conversations(current_user: int = Depends(get_current_user)):
    """
    Return a JSON-serializable list of user conversations.
    """
    def _get_convos():
        rows = get_user_conversations(current_user)
        return [dict(r) for r in rows]

    loop = asyncio.get_event_loop()
    conversations = await loop.run_in_executor(db_pool, _get_convos)
    return {"conversations": conversations}

@app.get("/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str, current_user: int = Depends(get_current_user)):
    """
    Return the messages for a single conversation, if the user is authorized.
    """
    def _check_access_and_get_convo():
        conn = get_db_connection()
        cursor = conn.execute(
            'SELECT user_id FROM conversations WHERE conversation_id = ?',
            (conversation_id,)
        )
        row = cursor.fetchone()
        if not row or row['user_id'] != current_user:
            return None
        rows = get_conversation(conversation_id)
        return [dict(r) for r in rows]

    loop = asyncio.get_event_loop()
    messages = await loop.run_in_executor(db_pool, _check_access_and_get_convo)

    if messages is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
    
    return {"messages": messages}

# -----------------------
# Models Endpoints
# -----------------------
@app.get("/ollama-models")
async def get_ollama_models():
    """
    List local models available to Ollama.
    """
    try:
        from fastapi.concurrency import run_in_threadpool
        models = await run_in_threadpool(list_models)
        return JSONResponse(content={"models": models})
    except Exception as e:
        logger.exception(f"Error fetching Ollama models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openai-models")
async def get_openai_models():
    """
    Return a static list of recognized "OpenAI" model names.
    """
    openai_interface = OpenAIInterface(model="gpt-4o-mini")
    if not openai_interface.is_api_key_configured():
        return {"models": []}
        
    models = [
        {"NAME": "o3-mini-high"},
        {"NAME": "o3-mini-medium"},
        {"NAME": "o3-mini-low"},
        {"NAME": "o1-preview"},
        {"NAME": "gpt-4o"},
        {"NAME": "gpt-4o-mini"}
    ]
    return {"models": models}

# -----------------------
# Chat Completion Endpoint
# -----------------------
@app.post("/chat-completion")
async def chat_completion(
    chat_req: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Optional[int] = Depends(get_optional_current_user)
):
    """
    Chat completion endpoint with concurrency and fallback handling.
    """
    # Limit concurrent model calls with a semaphore
    async with request.app.state.model_semaphore:
        try:
            # Determine conversation ID based on user authentication
            if current_user and chat_req.conversation_id:
                # Use a synchronous function for DB check
                def check_conversation_access_sync():
                    conn = get_db_connection()
                    cursor = conn.execute(
                        'SELECT user_id FROM conversations WHERE conversation_id = ?',
                        (chat_req.conversation_id,)
                    )
                    return cursor.fetchone()

                loop = asyncio.get_event_loop()
                row = await loop.run_in_executor(db_pool, check_conversation_access_sync)
                if not row or row['user_id'] != current_user:
                    raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
                conversation_id = chat_req.conversation_id

            elif current_user:
                # Create a new conversation for authenticated users without an ID
                def create_new_conversation_sync():
                    return create_conversation(current_user, chat_req.messages)

                loop = asyncio.get_event_loop()
                conversation_id = await loop.run_in_executor(db_pool, create_new_conversation_sync)
            else:
                # No conversation tracking for unauthenticated users
                conversation_id = None

            primary_interface = await get_interface(request.app, chat_req.model)
            backup_interface = await get_interface(request.app, chat_req.backup_model) if chat_req.backup_model else None

            # Prepare images if provided
            images = chat_req.image_b64 if chat_req.image_b64 else None
            if images and not isinstance(images, list):
                images = [images]

            conversation_history = [m.model_dump() for m in chat_req.messages]

            # Handle vision requests (for Ollama)
            if isinstance(primary_interface, AsyncOllamaInterface) and images:
                cleaned_images = [img.split(",")[1] if "," in img else img for img in images]
                prompt = " ".join([m.content for m in chat_req.messages if m.role == "user"])
                backup_used = False

                try:
                    response = await primary_interface.send_vision(prompt, cleaned_images)
                    if "error" in response:
                        logger.error(f"Primary model vision failed: {response['error']}")
                        if backup_interface and backup_interface._supports("image"):
                            content = await _fallback_vision(backup_interface, prompt, cleaned_images)
                            backup_used = True
                        else:
                            raise ValueError("Primary vision failed, no backup available")
                    else:
                        content = primary_interface.extract_content_from_response(response, is_chat=False)
                except Exception as e:
                    logger.exception(f"Vision request failed: {e}")
                    if backup_interface and backup_interface._supports("image"):
                        try:
                            content = await _fallback_vision(backup_interface, prompt, cleaned_images)
                            backup_used = True
                        except Exception as backup_err:
                            logger.exception(f"Backup vision model failed: {backup_err}")
                            raise HTTPException(status_code=500, detail="Both primary and backup vision models failed")
                    else:
                        raise HTTPException(status_code=500, detail="Vision model failed and no backup available")

                messages = chat_req.messages + [ChatMessage(role="assistant", content=content)]
                if conversation_id:
                    def save_conversation_sync():
                        conn = get_db_connection()
                        update_conversation(conversation_id, messages)
                        if backup_used:
                            conn.execute(
                                'UPDATE conversations SET backup_used = ? WHERE conversation_id = ?',
                                (1, conversation_id)
                            )
                            conn.commit()

                    background_tasks.add_task(
                        lambda: asyncio.run_coroutine_threadsafe(
                            save_conversation_sync(), asyncio.get_event_loop()
                        )
                    )
                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return JSONResponse(content={"message": content}, headers=headers)

            # Handle streaming chat requests
            if chat_req.stream:
                full_response = []
                try_backup = False
                backup_activated = False

                async def streamer() -> AsyncGenerator[bytes, None]:
                    nonlocal try_backup, backup_activated, full_response
                    primary_stream_id = None
                    backup_stream_id = None

                    try:
                        try:
                            stream_generator = primary_interface.send_chat_streaming(conversation_history)
                            primary_stream_id = id(stream_generator)
                            request.app.state.active_streams.add(primary_stream_id)

                            async for chunk in stream_generator:
                                if "error" in chunk:
                                    logger.warning(f"Error in primary model stream: {chunk.get('error')}")
                                    try_backup = True
                                    break
                                content = primary_interface.extract_content_from_chunk(chunk)
                                if content:
                                    full_response.append(content)
                                    yield (json.dumps({"message": content}) + "\n").encode("utf-8")
                        except Exception as e:
                            logger.exception(f"Error in primary model stream: {e}")
                            try_backup = True
                        finally:
                            if primary_stream_id in request.app.state.active_streams:
                                request.app.state.active_streams.remove(primary_stream_id)

                        if try_backup and backup_interface:
                            try:
                                backup_activated = True
                                assistant_partial = "".join(full_response)
                                updated_history = conversation_history
                                if assistant_partial.strip():
                                    updated_history = conversation_history + [
                                        {"role": "assistant", "content": assistant_partial}
                                    ]
                                backup_stream = backup_interface.send_chat_streaming(updated_history)
                                backup_stream_id = id(backup_stream)
                                request.app.state.active_streams.add(backup_stream_id)

                                async for chunk in backup_stream:
                                    if "error" in chunk:
                                        logger.warning(f"Error in backup model stream: {chunk.get('error')}")
                                        raise HTTPException(status_code=500, detail="Both primary and backup models failed")
                                    content = backup_interface.extract_content_from_chunk(chunk)
                                    if content:
                                        full_response.append(content)
                                        yield (json.dumps({"message": content}) + "\n").encode("utf-8")
                            except Exception as e2:
                                logger.exception(f"Error in backup model stream: {e2}")
                                raise HTTPException(status_code=500, detail="Both primary and backup models failed")
                            finally:
                                if backup_stream_id in request.app.state.active_streams:
                                    request.app.state.active_streams.remove(backup_stream_id)
                    except Exception as outer_e:
                        logger.exception(f"Unhandled exception in streamer: {outer_e}")
                        return

                async def save_conversation_async():
                    await asyncio.sleep(0.5)
                    def db_save():
                        combined = "".join(full_response)
                        msgs = chat_req.messages + [ChatMessage(role="assistant", content=combined)]
                        if conversation_id:
                            update_conversation(conversation_id, msgs)
                            if backup_activated:
                                conn = get_db_connection()
                                conn.execute(
                                    'UPDATE conversations SET backup_used = ? WHERE conversation_id = ?',
                                    (1, conversation_id)
                                )
                                conn.commit()
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(db_pool, db_save)

                if conversation_id:
                    background_tasks.add_task(save_conversation_async)

                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return StreamingResponse(streamer(), media_type="application/json", headers=headers)
            else:
                backup_used = False
                try:
                    response = await primary_interface.send_chat_nonstreaming(conversation_history)
                    if "error" in response:
                        logger.warning(f"Primary model failed: {response.get('error')}")
                        if not backup_interface:
                            raise HTTPException(status_code=500, detail="Primary model failed and no backup provided")
                        backup_used = True
                        response = await backup_interface.send_chat_nonstreaming(conversation_history)
                        if "error" in response:
                            raise HTTPException(status_code=500, detail="Both primary and backup models failed")
                    else:
                        content = primary_interface.extract_content_from_response(response, is_chat=True)
                except Exception as e:
                    logger.exception(f"Error with primary model: {e}")
                    if not backup_interface:
                        raise HTTPException(status_code=500, detail=f"Primary model failed: {str(e)}")
                    try:
                        backup_used = True
                        response = await backup_interface.send_chat_nonstreaming(conversation_history)
                        if "error" in response:
                            raise HTTPException(status_code=500, detail="Both primary and backup models failed")
                    except Exception as backup_e:
                        logger.exception(f"Backup model failed: {backup_e}")
                        raise HTTPException(status_code=500, detail="Both primary and backup models failed")

                content = (backup_interface if backup_used else primary_interface).extract_content_from_response(
                    response, is_chat=True
                )

                msgs = chat_req.messages + [ChatMessage(role="assistant", content=content)]
                if conversation_id:
                    async def save_conversation_async():
                        def db_save():
                            update_conversation(conversation_id, msgs)
                            if backup_used:
                                conn = get_db_connection()
                                conn.execute(
                                    'UPDATE conversations SET backup_used = ? WHERE conversation_id = ?',
                                    (1, conversation_id)
                                )
                                conn.commit()
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(db_pool, db_save)

                    background_tasks.add_task(save_conversation_async)

                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return JSONResponse(content={"message": content}, headers=headers)

        except Exception as e:
            logger.exception(f"Unhandled error in chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Helper for Vision Fallback
# -----------------------
async def _fallback_vision(interface, prompt, images):
    if isinstance(interface, AsyncOllamaInterface):
        resp = await interface.send_vision(prompt, images)
        if "error" in resp:
            raise ValueError(resp["error"])
        return interface.extract_content_from_response(resp, is_chat=False)
    elif isinstance(interface, OpenAIInterface):
        vision_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images]
            ]}
        ]
        resp = await interface.send_chat_nonstreaming(vision_messages)
        if "error" in resp:
            raise ValueError(resp["error"])
        return interface.extract_content_from_response(resp, is_chat=True)
    else:
        raise ValueError("Unsupported backup interface type")

# -----------------------
# Static File Endpoints
# -----------------------
@app.get("/", response_class=FileResponse)
async def get_index():
    return FileResponse("static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

# -----------------------
# Server Configuration
# -----------------------
if __name__ == "__main__":
    import uvicorn
    
    config = {
        "app": "server:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "limit_concurrency": 100,
        "timeout_keep_alive": 120,
        "workers": 1
    }
    
    uvicorn.run(**config)
