# Standard library imports
import json
import logging
import asyncio
import signal
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union, AsyncGenerator, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Third-party library imports
import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from uvicorn import Config, Server

# Local imports
from async_ollama_interface import AsyncOllamaInterface
from openai_api import OpenAIInterface
from config import load_config, get_value, set_value
from database import (
    create_conversation, create_user, get_conversation,
    get_db_connection, get_user_conversations,
    update_conversation, verify_user
)
from ollama_list_models import list_models

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

    # Initialize config and lock
    app.state.config_lock = asyncio.Lock()
    app.state.openai_interface_lock = asyncio.Lock()
    await load_config(app)

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
# Customer-facing app (port 23323)
customer_app = FastAPI(
    title="Customer Chat API",
    description="Customer-facing chat completion endpoint",
    lifespan=lifespan,
    docs_url=None,  # No docs for customer app
    redoc_url=None,
)

# Full-access app (port 8000)
app = FastAPI(
    title="LLM API Server",
    description="Full-access FastAPI server for LLM requests",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    max_request_body_size=20 * 1024 * 1024,
)

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
    backup_models: Optional[Union[str, List[str]]] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    image_b64: Optional[Union[str, List[str]]] = None
    conversation_id: Optional[str] = None
    timeout_threshold: Optional[float] = 30.0

class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ConfigUpdate(BaseModel):
    value: Any

# -----------------------
# Helper: Improved interface management
# -----------------------
async def get_interface(app: FastAPI, model_name: str):
    """
    Get an appropriate interface for the specified model with efficient connection pooling.
    Caches OpenAI interfaces to reuse sessions and reduce connection overhead.

    Args:
        app (FastAPI): The FastAPI application instance.
        model_name (str): The name of the model to get an interface for.

    Returns:
        An instance of OpenAIInterface or AsyncOllamaInterface.

    Raises:
        ValueError: If model_name is invalid.
        HTTPException: If a pro model is requested but not allowed.
    """
    # Validate input
    if not model_name or not isinstance(model_name, str):
        raise ValueError("model_name must be a non-empty string")

    # Comprehensive list of OpenAI model prefixes
    openai_prefixes = [
        "gpt-4",          # Covers gpt-4, gpt-4-turbo, etc.
        "gpt-4o",         # Covers gpt-4o, gpt-4o-mini
        "gpt-3.5",        # Covers gpt-3.5-turbo, etc.
        "o1",             # Future-proofing for o1 series
        "o3",             # Hypothetical future models
        "o1-pro-low"      # Pro model
        "o1-pro",         # Pro model
        "o1-pro-high"     # Pro model
        "gpt-4.5-preview" # Pro model
    ]

    # List of pro model prefixes requiring special permission
    pro_model_prefixes = [
        "o1-pro",
        "gpt-4.5-preview"
    ]

    # Determine if the model is an OpenAI model
    if model_name.startswith("ft:"):
        is_openai = True
    else:
        is_openai = any(model_name.startswith(prefix) for prefix in openai_prefixes)

    if is_openai:
        # Check if it's a pro model and if pro models are allowed
        is_pro_model = any(model_name.startswith(prefix) for prefix in pro_model_prefixes)
        if is_pro_model and not await get_value(app, "openai.pro_models_allowed"):
            raise HTTPException(status_code=403, detail="Pro models are not allowed")

        # Safely create or retrieve cached OpenAI interface
        async with app.state.openai_interface_lock:
            if model_name in app.state.openai_interfaces:
                logger.debug(f"Using cached OpenAI interface for model: {model_name}")
                return app.state.openai_interfaces[model_name]
            logger.info(f"Creating new OpenAI interface for model: {model_name}")
            interface = OpenAIInterface(model=model_name)
            app.state.openai_interfaces[model_name] = interface
            return interface
    else:
        # Handle non-OpenAI models with Ollama
        logger.debug(f"Creating new Ollama interface for model: {model_name}")
        return AsyncOllamaInterface(model=model_name)
    
# -----------------------
# Authentication Endpoints
# -----------------------
async def shared_signup(signup_req: SignupRequest):
    try:
        create_user(signup_req.username, signup_req.password)
        return {"message": "User created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

async def shared_login(login_req: LoginRequest):
    user_id = verify_user(login_req.username, login_req.password)
    if user_id:
        payload = {"user_id": user_id, "username": login_req.username, "exp": datetime.now(timezone.utc) + timedelta(hours=24)}
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        return {"token": token}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

app.add_api_route("/signup", shared_signup, methods=["POST"])
app.add_api_route("/login", shared_login, methods=["POST"])

customer_app.add_api_route("/signup", shared_signup, methods=["POST"])
customer_app.add_api_route("/login", shared_login, methods=["POST"])

# -----------------------
# Conversation Endpoints
# -----------------------
async def shared_get_conversations(current_user: int = Depends(get_current_user)):
    def _get_convos():
        rows = get_user_conversations(current_user)
        return [dict(r) for r in rows]
    loop = asyncio.get_event_loop()
    conversations = await loop.run_in_executor(db_pool, _get_convos)
    return {"conversations": conversations}

async def shared_get_conversation_detail(conversation_id: str, current_user: int = Depends(get_current_user)):
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

app.add_api_route("/conversations", shared_get_conversations, methods=["GET"])
app.add_api_route("/conversations/{conversation_id}", shared_get_conversation_detail, methods=["GET"])

customer_app.add_api_route("/conversations", shared_get_conversations, methods=["GET"])
customer_app.add_api_route("/conversations/{conversation_id}", shared_get_conversation_detail, methods=["GET"])

# -----------------------
# Models Endpoints
# -----------------------
async def shared_get_ollama_models():
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

async def shared_get_openai_models(request: Request):
    """
    Return a list of recognized OpenAI model names. 
    - Includes pro models only if 'openai.pro_models_allowed' is true.
    - Includes fine-tuned models only if 'openai.fine_tuned_models_allowed' is true.
    """
    # Check pro models permission
    pro_models_allowed = await get_value(request.app, "openai.pro_models_allowed")
    if pro_models_allowed is None:
        pro_models_allowed = False

    # Check fine-tuned models permission
    fine_tuned_models_allowed = await get_value(request.app, "openai.fine_tuned_models_allowed")
    if fine_tuned_models_allowed is None:
        fine_tuned_models_allowed = False

    # Create a quick OpenAI interface to list models and fine-tunes
    openai_interface = OpenAIInterface(model="gpt-4o-mini")
    if not openai_interface.is_api_key_configured():
        # If API key not available, just return an empty list
        return {"models": []}

    # Base list of non-pro models (always included)
    non_pro_models = [
        {"NAME": "o3-mini-high"},
        {"NAME": "o3-mini-medium"},
        {"NAME": "o3-mini-low"},
        {"NAME": "o1-preview"},
        {"NAME": "gpt-4o"},
        {"NAME": "gpt-4o-mini"},
        {"NAME": "gpt-4o-mini-search-preview"},
        {"NAME": "gpt-4o-search-preview"}
    ]

    # If pro models are allowed
    pro_models = []
    if pro_models_allowed:
        # Example pro models
        pro_models = [
            {"NAME": "gpt-4.5-preview"},
            {"NAME": "o1-pro-low"},
            {"NAME": "o1-pro"},
            {"NAME": "o1-pro-high"}
        ]

    # Combine base + pro
    all_models = non_pro_models + pro_models

    # If fine-tuned models are allowed, fetch them
    if fine_tuned_models_allowed:
        try:
            fine_tuning_jobs = await openai_interface.list_fine_tuning_jobs()
            fine_tuned_ids = openai_interface.get_successful_fine_tuned_models(fine_tuning_jobs)
            fine_tuned_list = [{"NAME": model_id} for model_id in fine_tuned_ids]
            all_models += fine_tuned_list
        except Exception as e:
            logger.warning(f"Could not retrieve fine-tuned models: {e}")

    return {"models": all_models}

app.add_api_route("/ollama-models", shared_get_ollama_models, methods=["GET"])
app.add_api_route("/openai-models", shared_get_openai_models, methods=["GET"])

customer_app.add_api_route("/ollama-models", shared_get_ollama_models, methods=["GET"])
customer_app.add_api_route("/openai-models", shared_get_openai_models, methods=["GET"])

# -----------------------
# Chat Completion Endpoint
# -----------------------
async def chat_completion(
    chat_req: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Optional[int] = Depends(get_optional_current_user)
) -> Union[JSONResponse, StreamingResponse]:
    async with request.app.state.model_semaphore:
        loop = asyncio.get_event_loop()

        # Load or create conversation
        if current_user and chat_req.conversation_id:
            def check_conversation_access():
                conn = get_db_connection()
                c = conn.execute("SELECT user_id FROM conversations WHERE conversation_id = ?", (chat_req.conversation_id,))
                return c.fetchone()
            row = await loop.run_in_executor(db_pool, check_conversation_access)
            if not row or row["user_id"] != current_user:
                raise HTTPException(403, "Not authorized")
            conversation_id = chat_req.conversation_id

            def load_existing_messages():
                rows = get_conversation(chat_req.conversation_id)
                return [ChatMessage(**dict(r)) for r in rows]
            existing_messages = await loop.run_in_executor(db_pool, load_existing_messages)
            for msg in chat_req.messages:
                existing_messages.append(msg)
        elif current_user:
            def create_new_convo():
                return create_conversation(current_user, chat_req.messages)
            conversation_id = await loop.run_in_executor(db_pool, create_new_convo)
            existing_messages = chat_req.messages
        else:
            conversation_id = None
            existing_messages = chat_req.messages

        conversation_history = [m.model_dump() for m in existing_messages]
        backup_models = [chat_req.backup_models] if isinstance(chat_req.backup_models, str) else (chat_req.backup_models or [])
        models_to_try = [chat_req.model] + backup_models
        images = chat_req.image_b64 if isinstance(chat_req.image_b64, list) else ([chat_req.image_b64] if chat_req.image_b64 else [])
        cleaned_images = [img.split(",")[1] if "," in img else img for img in images] if images else None
        prompt = " ".join(m.content for m in chat_req.messages if m.role == "user") if images else None

        try:
            # Vision (non-streaming)
            if cleaned_images:
                content, backup_used = None, False
                for i, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    try:
                        if isinstance(interface, AsyncOllamaInterface):
                            resp = await interface.send_vision(prompt, cleaned_images)
                            if "error" not in resp:
                                content = interface.extract_content_from_response(resp, is_chat=False)
                                backup_used = i > 0
                                break
                        else:
                            vision_msgs = [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in cleaned_images]
                                ]
                            }]
                            resp = await interface.send_chat_nonstreaming(vision_msgs)
                            if "error" not in resp:
                                content = interface.extract_content_from_response(resp, is_chat=True)
                                backup_used = i > 0
                                break
                    except Exception:
                        pass
                if not content:
                    raise HTTPException(500, "All models failed")
                existing_messages.append(ChatMessage(role="assistant", content=content))

                if conversation_id:
                    def save_vision():
                        update_conversation(conversation_id, existing_messages)
                        if backup_used:
                            conn = get_db_connection()
                            conn.execute("UPDATE conversations SET backup_used = ? WHERE conversation_id = ?", (1, conversation_id))
                            conn.commit()

                    background_tasks.add_task(save_vision)

                return JSONResponse({"message": content}, headers={"X-Conversation-ID": conversation_id} if conversation_id else {})

            # Streaming
            elif chat_req.stream:
                full_response = []
                backup_used = False

                # Attempt each model until we get a valid first chunk
                selected_stream_gen = None
                selected_interface = None
                for i, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    try:
                        gen = interface.send_chat_streaming(conversation_history, timeout_threshold=chat_req.timeout_threshold)
                        first_chunk = await gen.__anext__()
                        if "error" in first_chunk:
                            raise RuntimeError("Model error")
                        selected_stream_gen = gen
                        selected_interface = interface
                        backup_used = i > 0
                        chunk_content = interface.extract_content_from_chunk(first_chunk)
                        if chunk_content:
                            full_response.append(chunk_content)
                        break
                    except Exception:
                        pass
                if not selected_stream_gen:
                    raise HTTPException(500, "All models failed")

                async def streamer() -> AsyncGenerator[bytes, None]:
                    models_left = models_to_try
                    idx = models_left.index(selected_interface.model)
                    combined = "".join(full_response)
                    yield (json.dumps({"message": combined}) + "\n").encode("utf-8")

                    # Continue with the same generator
                    try:
                        async for chunk in selected_stream_gen:
                            if "error" in chunk:
                                raise RuntimeError("Model error")
                            c = selected_interface.extract_content_from_chunk(chunk)
                            if c:
                                full_response.append(c)
                                yield (json.dumps({"message": c}) + "\n").encode("utf-8")
                    except Exception:
                        pass

                    # If streaming fails, try backups
                    while True:
                        idx += 1
                        if idx >= len(models_left):
                            break
                        next_model = models_left[idx]
                        try:
                            interface = await get_interface(request.app, next_model)
                            stream_gen = interface.send_chat_streaming(conversation_history, timeout_threshold=chat_req.timeout_threshold)
                            async for chunk in stream_gen:
                                if "error" in chunk:
                                    raise RuntimeError("Model error")
                                c = interface.extract_content_from_chunk(chunk)
                                if c:
                                    full_response.append(c)
                                    yield (json.dumps({"message": c}) + "\n").encode("utf-8")
                            break
                        except Exception:
                            pass

                async def save_streamed():
                    await asyncio.sleep(1)
                    def _save():
                        combined_resp = "".join(full_response)
                        existing_messages.append(ChatMessage(role="assistant", content=combined_resp))
                        if conversation_id:
                            update_conversation(conversation_id, existing_messages)
                            if backup_used:
                                conn = get_db_connection()
                                conn.execute("UPDATE conversations SET backup_used = ? WHERE conversation_id = ?", (1, conversation_id))
                                conn.commit()
                    await loop.run_in_executor(db_pool, _save)

                if conversation_id:
                    background_tasks.add_task(save_streamed)

                return StreamingResponse(streamer(), media_type="application/json",
                                         headers={"X-Conversation-ID": conversation_id} if conversation_id else {})

            # Non-streaming
            else:
                content, backup_used = None, False
                for i, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    try:
                        resp = await interface.send_chat_nonstreaming(conversation_history)
                        if "error" not in resp:
                            content = interface.extract_content_from_response(resp, is_chat=True)
                            backup_used = i > 0
                            break
                    except Exception:
                        pass
                if not content:
                    raise HTTPException(500, "All models failed")

                existing_messages.append(ChatMessage(role="assistant", content=content))

                if conversation_id:
                    async def save_non_stream():
                        def _save():
                            update_conversation(conversation_id, existing_messages)
                            if backup_used:
                                conn = get_db_connection()
                                conn.execute("UPDATE conversations SET backup_used = ? WHERE conversation_id = ?", (1, conversation_id))
                                conn.commit()
                        await loop.run_in_executor(db_pool, _save)
                    background_tasks.add_task(save_non_stream)

                return JSONResponse({"message": content}, headers={"X-Conversation-ID": conversation_id} if conversation_id else {})

        except Exception as e:
            logger.exception(f"Error in chat completion: {e}")
            raise HTTPException(500, str(e))

app.add_api_route("/chat-completion", chat_completion, methods=['POST'], response_model=None)

customer_app.add_api_route("/chat-completion", chat_completion, methods=['POST'], response_model=None)
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
async def shared_get_index():
    return FileResponse("static/index.html")

async def shared_favicon():
    return FileResponse("static/favicon.ico")

async def shared_get_styles():
    return FileResponse("static/styles.css")

app.add_api_route("/", shared_get_index, response_class=FileResponse, methods=["GET"])
app.add_api_route("/favicon.ico", shared_favicon, include_in_schema=False, methods=["GET"])
app.add_api_route("/styles.css", shared_get_styles, response_class=FileResponse, methods=["GET"])

customer_app.add_api_route("/", shared_get_index, response_class=FileResponse, methods=["GET"])
customer_app.add_api_route("/favicon.ico", shared_favicon, include_in_schema=False, methods=["GET"])
customer_app.add_api_route("/styles.css", shared_get_styles, response_class=FileResponse, methods=["GET"])

# ----------------------
# Config Management
# ----------------------
@app.get("/api/config/{key}")
async def get_config_value(key: str, request: Request, current_user: int = Depends(get_current_user)):
    """
    Retrieve a configuration value by key.
    """
    # Optionally enforce role-based logic here if only admins can read config, etc.
    # e.g., if current_user != <admin_user_id>: raise HTTPException(403, "Not authorized")

    value = await get_value(request.app, key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")

    return JSONResponse(content={"key": key, "value": value})

@app.post("/api/config/{key}")
async def set_config_value(key: str, data: ConfigUpdate, request: Request, current_user: int = Depends(get_current_user)):
    """
    Set or update a configuration value by key.
    """
    # Optionally enforce role-based logic here if only admins can write config, etc.
    # e.g., if current_user != <admin_user_id>: raise HTTPException(403, "Not authorized")

    try:
        await set_value(request.app, key, data.value)
        return JSONResponse(content={"message": f"Config key '{key}' updated successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Server Initialization
# -----------------------
async def main():
    customer_config = Config(
        app=customer_app,
        host="0.0.0.0",
        port=23323,
        limit_concurrency=100,
        timeout_keep_alive=120,
    )
    customer_server = Server(customer_config)

    full_config = Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        limit_concurrency=100,
        timeout_keep_alive=300,
    )
    full_server = Server(full_config)

    # Run both servers concurrently
    await asyncio.gather(customer_server.serve(), full_server.serve())

if __name__ == "__main__":
    asyncio.run(main())