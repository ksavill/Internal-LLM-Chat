# Standard library imports
import json
import logging
import asyncio
import signal
from datetime import datetime, timedelta
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
        "o1-pro",         # Pro model
        "gpt-4.5-preview" # Pro model
    ]

    # List of pro model prefixes requiring special permission
    pro_model_prefixes = [
        "o1-pro",
        "gpt-4.5-preview"
    ]

    # Determine if the model is an OpenAI model
    is_openai = any(model_name.startswith(prefix) for prefix in openai_prefixes)

    if is_openai:
        # Check if it's a pro model and if pro models are allowed
        is_pro_model = any(model_name.startswith(prefix) for prefix in pro_model_prefixes)
        if is_pro_model and not get_value(app, "openai.pro_models_allowed"):
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
        payload = {"user_id": user_id, "username": login_req.username, "exp": datetime.utcnow() + timedelta(hours=1)}
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
            {"NAME": "gpt-4.5-preview"}
            # o1-pro is exclusive to the Response API (this service is Chat Completion based)
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
) -> JSONResponse | StreamingResponse:
    """
    Chat completion endpoint with support for multiple backup models.
    Handles vision, streaming, and non-streaming requests with fallback logic.

    Args:
        chat_req: The ChatRequest object containing model, backup_models, messages, etc.
        request: The FastAPI Request object.
        background_tasks: FastAPI BackgroundTasks for scheduling tasks.
        current_user: Optional user ID from authentication.

    Returns:
        JSONResponse for non-streaming/vision requests, StreamingResponse for streaming requests.
    """
    async with request.app.state.model_semaphore:
        try:
            # **Conversation ID Handling**
            if current_user and chat_req.conversation_id:
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
                def create_new_conversation_sync():
                    return create_conversation(current_user, chat_req.messages)

                loop = asyncio.get_event_loop()
                conversation_id = await loop.run_in_executor(db_pool, create_new_conversation_sync)
            else:
                conversation_id = None

            # **Normalize backup_models for Backward Compatibility**
            backup_models = []
            if chat_req.backup_models:
                if isinstance(chat_req.backup_models, str):
                    backup_models = [chat_req.backup_models]  # Convert single string to list
                else:
                    backup_models = chat_req.backup_models

            # List of models to try: primary model followed by backups
            models_to_try = [chat_req.model] + backup_models

            # **Prepare Images if Provided**
            images = chat_req.image_b64 if chat_req.image_b64 else None
            if images and not isinstance(images, list):
                images = [images]
            cleaned_images = [img.split(",")[1] if "," in img else img for img in images] if images else None

            conversation_history = [m.model_dump() for m in chat_req.messages]
            prompt = " ".join([m.content for m in chat_req.messages if m.role == "user"]) if images else None

            # **Handle Vision Requests (Non-Streaming)**
            if images:
                content = None
                backup_used = False
                for idx, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    try:
                        if isinstance(interface, AsyncOllamaInterface):
                            response = await interface.send_vision(prompt, cleaned_images)
                        elif isinstance(interface, OpenAIInterface):
                            vision_messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        *[
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                                            }
                                            for img in cleaned_images
                                        ]
                                    ]
                                }
                            ]
                            response = await interface.send_chat_nonstreaming(vision_messages)
                        if "error" not in response:
                            content = interface.extract_content_from_response(
                                response, is_chat=isinstance(interface, OpenAIInterface)
                            )
                            backup_used = idx > 0
                            break
                    except Exception as e:
                        logger.warning(f"Model {model} failed for vision request: {e}")

                if content is None:
                    raise HTTPException(status_code=500, detail="All models failed for vision request")

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

            # **Handle Streaming Chat Requests**
            elif chat_req.stream:
                full_response = []
                selected_interface = None
                stream_generator = None
                first_content = None
                backup_used = False

                # Select the first model that successfully provides an initial chunk
                for idx, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    try:
                        stream_generator = interface.send_chat_streaming(conversation_history)
                        first_chunk = await stream_generator.__anext__()
                        if "error" not in first_chunk:
                            selected_interface = interface
                            first_content = selected_interface.extract_content_from_chunk(first_chunk)
                            if first_content:
                                full_response.append(first_content)
                            backup_used = idx > 0
                            break
                    except Exception as e:
                        logger.warning(f"Model {model} failed to start streaming: {e}")
                        continue
                else:
                    raise HTTPException(status_code=500, detail="All models failed to start streaming")

                async def streamer() -> AsyncGenerator[bytes, None]:
                    remaining_models = models_to_try.copy()  # List of models to try (e.g., ["openai", "backup1", "backup2"])
                    current_model_index = 0
                    full_response = []  # Track the conversation so far

                    while current_model_index < len(remaining_models):
                        model = remaining_models[current_model_index]
                        interface = await get_interface(request.app, model)
                        try:
                            # Start streaming with the current model
                            stream_generator = interface.send_chat_streaming(conversation_history)
                            async for chunk in stream_generator:
                                if "error" in chunk:
                                    raise ValueError(f"Error in chunk: {chunk['error']}")
                                content = interface.extract_content_from_chunk(chunk)
                                if content:
                                    full_response.append(content)
                                    yield (json.dumps({"message": content}) + "\n").encode("utf-8")
                            break  # If streaming completes successfully, exit the loop
                        except Exception as e:
                            logger.warning(f"Model {model} failed during streaming: {e}")
                            current_model_index += 1  # Move to the next model
                            if current_model_index < len(remaining_models):
                                logger.info(f"Switching to backup model: {remaining_models[current_model_index]}")
                                # Optionally update conversation_history with full_response to continue where it left off
                                conversation_history.append({"role": "assistant", "content": "".join(full_response)})
                            else:
                                raise HTTPException(status_code=500, detail="All models failed during streaming")

                async def save_conversation_async():
                    await asyncio.sleep(1)  # Delay to allow streaming to complete
                    def db_save():
                        combined = "".join(full_response)
                        msgs = chat_req.messages + [ChatMessage(role="assistant", content=combined)]
                        if conversation_id:
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

                if conversation_id:
                    background_tasks.add_task(save_conversation_async)

                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return StreamingResponse(streamer(), media_type="application/json", headers=headers)

            # **Handle Non-Streaming Chat Requests**
            else:
                content = None
                backup_used = False
                for idx, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    try:
                        response = await interface.send_chat_nonstreaming(conversation_history)
                        if "error" not in response:
                            content = interface.extract_content_from_response(response, is_chat=True)
                            backup_used = idx > 0
                            break
                    except Exception as e:
                        logger.warning(f"Model {model} failed for chat request: {e}")

                if content is None:
                    raise HTTPException(status_code=500, detail="All models failed for chat request")

                messages = chat_req.messages + [ChatMessage(role="assistant", content=content)]
                if conversation_id:
                    async def save_conversation_async():
                        def db_save():
                            update_conversation(conversation_id, messages)
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