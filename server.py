# Standard library imports
import json
import logging
import asyncio
import signal
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Third-party library imports
import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from uvicorn import Config, Server

# Local imports
from async_ollama_interface import AsyncOllamaInterface
# from proxy_interface import ProxyInterface # This will be unused code for the forseeable future. We would need to ensure that our response can mimic that of a raw streamed response if in proxy mode
from openai_api import OpenAIInterface
from config import load_config, get_value, set_value
from model_whitelist import is_model_allowed
from model_aliases import resolve_model_alias
from database import (
    create_conversation, create_user, get_conversation,
    get_db_connection, get_user_conversations,
    update_conversation, verify_user, get_conversation_openai_response_id
)
from ollama_list_models import list_models
from profiles_manager import load_profile, load_all_profiles

from models import (
    ChatMessage, ChatRequest, SignupRequest, LoginRequest, ConfigUpdate,
    MessageResponse, TokenResponse, ConversationsListResponse,
    ConversationDetailResponse, OllamaModelsResponse, OpenAIModelsResponse,
    ChatCompletionResponse, ConfigValue, ConfigMessage,
)

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
    logger.info("Acquiring database connection from pool")
    def _get_conn():
        import sqlite3
        conn = sqlite3.connect('conversations.db')
        conn.row_factory = sqlite3.Row
        return conn
    
    loop = asyncio.get_event_loop()
    conn = await loop.run_in_executor(db_pool, _get_conn)
    logger.info("Database connection acquired")
    return conn

# -----------------------
# Graceful Shutdown Handler
# -----------------------
def handle_sigterm(*args):
    """
    Handle SIGTERM signal for graceful shutdown.
    This is especially important in containerized environments.
    """
    logger.info("Received SIGTERM signal, initiating graceful shutdown")
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
    logger.info("Starting application lifecycle")
    # Initialize config and lock
    app.state.config_lock = asyncio.Lock()
    app.state.openai_interface_lock = asyncio.Lock()
    logger.info("Loading configuration")
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
    logger.info("Initializing database")
    init_db()
    logger.info("Database initialized")
    
    logger.info("Server startup complete - ready to handle requests")
    
    yield  # Server is running and processing requests
    
    # Shutdown cleanup
    logger.info("Server shutting down - cleaning up resources")
    
    # Close any open OpenAI sessions
    close_tasks = []
    for interface in app.state.openai_interfaces.values():
        if hasattr(interface, 'close') and callable(interface.close):
            logger.info(f"Closing OpenAI interface for model: {interface.model}")
            close_tasks.append(interface.close())
    
    if close_tasks:
        logger.info("Closing all OpenAI interfaces")
        await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info("Closed all OpenAI interfaces")
    
    # Cancel any active background tasks
    if app.state.background_tasks:
        logger.info("Cancelling active background tasks")
        for task in app.state.background_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*app.state.background_tasks, return_exceptions=True)
        logger.info("All background tasks cancelled")
    
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
    logger.info(f"Registering background task: {coro}")
    task = asyncio.create_task(coro)
    app.state.background_tasks.add(task)
    task.add_done_callback(lambda t: app.state.background_tasks.remove(t))
    logger.info(f"Background task registered: {task}")
    return task

# -----------------------
# Authentication Functions
# -----------------------
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    logger.info("Verifying JWT token for current user")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload["user_id"]
        logger.info(f"Token verified, user_id: {user_id}")
        return user_id
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        logger.error("Invalid token provided")
        raise HTTPException(status_code=401, detail="Invalid token")

def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security)) -> Optional[int]:
    """
    If a token is provided and valid, return user_id; otherwise return None.
    """
    if credentials is None:
        logger.info("No credentials provided, treating as guest user")
        return None
    token = credentials.credentials
    logger.info("Verifying optional JWT token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("user_id")
        logger.info(f"Optional token verified, user_id: {user_id}")
        return user_id
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        logger.warning(f"Invalid or expired optional token: {e}")
        return None

# -----------------------
# Helper: Improved interface management
# -----------------------
async def get_interface(app: FastAPI, model_name: str):
    """
    Get an appropriate interface for the specified model with efficient connection pooling.
    Caches OpenAI interfaces to reuse sessions and reduce connection overhead.
    """
    logger.info(f"Getting interface for model: {model_name}")
    # Validate input
    if not model_name or not isinstance(model_name, str):
        logger.error("Invalid model_name provided")
        raise ValueError("model_name must be a non-empty string")

    # Comprehensive list of OpenAI model prefixes
    openai_prefixes = [
        "gpt-4", "gpt-4o", "gpt-3.5", "o1", "o3", "o1-pro-low", "o1-pro", "o1-pro-high", "gpt-4.5-preview"
    ]

    # List of pro model prefixes requiring special permission
    pro_model_prefixes = ["o1-pro", "gpt-4.5-preview"]

    # Determine if the model is an OpenAI model
    if model_name.startswith("ft:"):
        is_openai = True
    else:
        is_openai = any(model_name.startswith(prefix) for prefix in openai_prefixes)

    try:
        if is_openai:
            # Check if it's a pro model and if pro models are allowed
            is_pro_model = any(model_name.startswith(prefix) for prefix in pro_model_prefixes)
            if is_pro_model and not await get_value(app, "openai.pro_models_allowed"):
                logger.error(f"Pro model {model_name} requested but not allowed")
                raise HTTPException(status_code=403, detail="Pro models are not allowed")

            # Safely create or retrieve cached OpenAI interface
            async with app.state.openai_interface_lock:
                if model_name in app.state.openai_interfaces:
                    logger.info(f"Using cached OpenAI interface for model: {model_name}")
                    return app.state.openai_interfaces[model_name]
                logger.info(f"Creating new OpenAI interface for model: {model_name}")
                interface = OpenAIInterface(model=model_name)
                app.state.openai_interfaces[model_name] = interface
                logger.info(f"OpenAI interface created and cached for model: {model_name}")
                return interface
        else:
            logger.info(f"Creating new Ollama interface for model: {model_name}")
            interface = AsyncOllamaInterface(model=model_name)
            logger.info(f"Ollama interface created for model: {model_name}")
            return interface
    except Exception as e:
        logger.exception(f"Error creating interface for model {model_name}: {e}")
        raise

# -----------------------
# Authentication Endpoints
# -----------------------
@app.post("/signup", response_model=MessageResponse)
async def shared_signup(signup_req: SignupRequest):
    logger.info(f"Signup request for username: {signup_req.username}")
    try:
        create_user(signup_req.username, signup_req.password)
        logger.info(f"User created successfully: {signup_req.username}")
        return {"message": "User created successfully"}
    except ValueError as e:
        logger.error(f"Error creating user {signup_req.username}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login", response_model=TokenResponse)
async def shared_login(login_req: LoginRequest):
    logger.info(f"Login request for username: {login_req.username}")
    user_id = verify_user(login_req.username, login_req.password)
    if user_id:
        payload = {"user_id": user_id, "username": login_req.username, "exp": datetime.now(timezone.utc) + timedelta(hours=24)}
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        logger.info(f"User logged in successfully: {login_req.username}, user_id: {user_id}")
        return {"token": token}
    else:
        logger.warning(f"Failed login attempt for username: {login_req.username}")
        raise HTTPException(status_code=401, detail="Invalid username or password")

# -----------------------
# Conversation Endpoints
# -----------------------
@app.get("/conversations", response_model=ConversationsListResponse)
async def shared_get_conversations(current_user: int = Depends(get_current_user)):
    logger.info(f"Fetching conversations for user: {current_user}")
    def _get_convos():
        rows = get_user_conversations(current_user)
        return [dict(r) for r in rows]
    loop = asyncio.get_event_loop()
    conversations = await loop.run_in_executor(db_pool, _get_convos)
    logger.info(f"Retrieved {len(conversations)} conversations for user: {current_user}")
    return {"conversations": conversations}

@app.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def shared_get_conversation_detail(conversation_id: str, current_user: int = Depends(get_current_user)):
    logger.info(f"Fetching conversation {conversation_id} for user: {current_user}")
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
        logger.warning(f"User {current_user} not authorized to access conversation {conversation_id}")
        raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
    logger.info(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
    return {"messages": messages}

# -----------------------
# Models Endpoints
# -----------------------
@app.get("/ollama-models", response_model=OllamaModelsResponse)
async def shared_get_ollama_models():
    """
    List local models available to Ollama, filtered by the whitelist.
    """
    logger.info("Fetching Ollama models")
    try:
        from fastapi.concurrency import run_in_threadpool
        models = await run_in_threadpool(list_models)
        filtered_models = [m for m in models if is_model_allowed(m["NAME"])]
        logger.info(f"Retrieved {len(filtered_models)} Ollama models after filtering")
        return JSONResponse(content={"models": filtered_models})
    except Exception as e:
        logger.exception(f"Error fetching Ollama models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openai-models", response_model=OpenAIModelsResponse)
async def shared_get_openai_models(request: Request):
    """
    Return a list of recognized OpenAI model names.
    """
    logger.info("Fetching OpenAI models")
    pro_models_allowed = await get_value(request.app, "openai.pro_models_allowed")
    if pro_models_allowed is None:
        pro_models_allowed = False
        logger.info("openai.pro_models_allowed not set, defaulting to False")

    fine_tuned_models_allowed = await get_value(request.app, "openai.fine_tuned_models_allowed")
    if fine_tuned_models_allowed is None:
        fine_tuned_models_allowed = False
        logger.info("openai.fine_tuned_models_allowed not set, defaulting to False")

    openai_interface = OpenAIInterface(model="gpt-4o-mini")
    if not openai_interface.is_api_key_configured():
        logger.warning("OpenAI API key not configured, returning empty model list")
        return {"models": []}

    non_pro_models = [
        {"NAME": "gpt-3.5-turbo"},
        {"NAME": "gpt-4o"},
        {"NAME": "gpt-4o-mini"},
        {"NAME": "gpt-4o-mini-search-preview"},
        {"NAME": "gpt-4o-search-preview"},
        {"NAME": "gpt-4.1"},
        {"NAME": "gpt-4.1-mini"},
        {"NAME": "gpt-4.1-nano"},
        {"NAME": "o1-preview"},
        {"NAME": "o3-mini-high"},
        {"NAME": "o3-mini-medium"},
        {"NAME": "o3-mini-low"},
        {"NAME": "o3"},
        {"NAME": "o3-high"},
    ]

    if pro_models_allowed:
        pro_models = [
            {"NAME": "gpt-4.5-preview"}, {"NAME": "o1-pro-low"},
            {"NAME": "o1-pro"}, {"NAME": "o1-pro-high"}
        ]
    else:
        pro_models = []
        logger.info("Pro models not allowed, excluding from list")

    all_models = non_pro_models + pro_models

    if fine_tuned_models_allowed:
        try:
            fine_tuning_jobs = await openai_interface.list_fine_tuning_jobs()
            fine_tuned_ids = openai_interface.get_successful_fine_tuned_models(fine_tuning_jobs)
            fine_tuned_list = [{"NAME": model_id} for model_id in fine_tuned_ids]
            all_models += fine_tuned_list
            logger.info(f"Added {len(fine_tuned_list)} fine-tuned models")
        except Exception as e:
            logger.warning(f"Could not retrieve fine-tuned models: {e}")

    whitelisted_models = [m for m in all_models if is_model_allowed(m["NAME"])]
    logger.info(f"Retrieved {len(whitelisted_models)} OpenAI models after filtering")

    return {"models": whitelisted_models}

@app.get("/request-profiles", response_model=List[str])
async def list_request_profiles():
    logger.info("Fetching all request profile names")
    all_profiles = await load_all_profiles()
    logger.info(f"Retrieved {len(all_profiles)} request profiles")
    return list(all_profiles.keys())

@app.get("/request-profiles/{profile_name}", response_model=dict)
async def get_request_profile_details(profile_name: str):
    logger.info(f"Fetching details for request profile: {profile_name}")
    all_profiles = await load_all_profiles()
    if profile_name not in all_profiles:
        logger.warning(f"Profile not found: {profile_name}")
        raise HTTPException(status_code=404, detail="Profile not found")
    logger.info(f"Retrieved details for profile: {profile_name}")
    return all_profiles[profile_name]

# Helper function to append only the new user messages.
def append_new_user_messages(db_msgs: List[ChatMessage], incoming_msgs: List[ChatMessage]) -> List[ChatMessage]:
    """
    Assumes the client sends the complete conversation history.
    Returns a merged list that consists of the DB messages plus any new messages 
    (i.e. those beyond the count already stored).
    
    For example, if there are 2 messages in DB and the client sends 3 messages, then
    the third message (i.e. the delta) is returned.
    """
    if len(incoming_msgs) <= len(db_msgs):
        # If the incoming messages are not longer than what's stored,
        # nothing new to append.
        return db_msgs
    tail = incoming_msgs[len(db_msgs):]
    return db_msgs + tail

@app.post("/chat-completion", response_model=ChatCompletionResponse)
@app.post("/api/chat-completion", response_model=ChatCompletionResponse)
async def chat_completion(
    chat_req: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Optional[int] = Depends(get_optional_current_user)
) -> Union[JSONResponse, StreamingResponse]:
    logger.info(f"Received chat completion request from user: {current_user}")
    async with request.app.state.model_semaphore:
        loop = asyncio.get_event_loop()

        # --------------------------------------------------------------------
        # Pre-lookup the previous_response_id if a conversation_id is provided.
        # This avoids duplicate logic later and allows subsequent messages to
        # correctly leverage that stored API response ID.
        previous_response_id: Optional[str] = None
        if current_user and chat_req.conversation_id:
            def load_prev_response_id():
                return get_conversation_openai_response_id(chat_req.conversation_id)
            previous_response_id = await loop.run_in_executor(db_pool, load_prev_response_id)
            logger.info(f"Found previous_response_id: {previous_response_id} for conversation {chat_req.conversation_id}")

        # 1) Apply request profile if specified.
        if chat_req.request_profile:
            logger.info(f"Loading request profile: {chat_req.request_profile}")
            loaded_profile = await load_profile(chat_req.request_profile)
            if not loaded_profile:
                logger.error(f"Request profile '{chat_req.request_profile}' not found")
                raise HTTPException(status_code=400, detail=f"Request profile '{chat_req.request_profile}' not found.")
            profile_model = loaded_profile.get("model")
            profile_backups = loaded_profile.get("backup_models", [])
            if not profile_model:
                logger.error(f"Profile '{chat_req.request_profile}' is missing a 'model' field")
                raise HTTPException(status_code=400, detail=f"Profile '{chat_req.request_profile}' is missing a 'model' field.")
            chat_req.model = profile_model
            chat_req.backup_models = profile_backups
            logger.info(f"Applied profile: model={profile_model}, backups={profile_backups}")

        # 2) Determine conversation: continue one if conversation_id is provided, otherwise create new.
        conversation_id: Optional[str] = None
        existing_db_messages: List[ChatMessage] = []
        # This list is what will be used to build the LLM payload.
        combined_for_model: List[ChatMessage] = []

        if current_user and chat_req.conversation_id:
            # Continue existing conversation.
            logger.info(f"Checking access to conversation {chat_req.conversation_id} for user {current_user}")
            def check_conversation_access():
                conn = get_db_connection()
                c = conn.execute(
                    "SELECT user_id FROM conversations WHERE conversation_id = ?",
                    (chat_req.conversation_id,)
                )
                return c.fetchone()
            row = await loop.run_in_executor(db_pool, check_conversation_access)
            if not row or row["user_id"] != current_user:
                logger.warning(f"User {current_user} not authorized for conversation {chat_req.conversation_id}")
                raise HTTPException(status_code=403, detail="Not authorized")
            conversation_id = chat_req.conversation_id
            logger.info(f"Continuing conversation ID: {conversation_id}")

            def load_existing_db_messages():
                rows = get_conversation(chat_req.conversation_id)
                return [ChatMessage(**dict(r)) for r in rows]
            existing_db_messages = await loop.run_in_executor(db_pool, load_existing_db_messages)
            logger.info(f"Loaded {len(existing_db_messages)} messages from DB for conversation {conversation_id}")

            # We expect the client to send the full conversation history.
            # We'll consider any messages beyond those already stored to be new.
            combined_for_model = existing_db_messages + chat_req.messages

        elif current_user:
            # No conversation_id provided: always create a new conversation.
            logger.info(f"No conversation_id provided; creating new conversation for user: {current_user}")
            def create_new_convo():
                return create_conversation(current_user, chat_req.messages)
            conversation_id = await loop.run_in_executor(db_pool, create_new_convo)
            existing_db_messages = chat_req.messages  # Fresh conversation.
            combined_for_model = chat_req.messages
            logger.info(f"Created new conversation ID: {conversation_id}")
        else:
            logger.info("Processing as guest user; no conversation tracking will be done.")
            conversation_id = None
            existing_db_messages = chat_req.messages
            combined_for_model = chat_req.messages

        # Build conversation history payload for the LLM. (Do not modify this.)
        conversation_history = [m.model_dump() for m in combined_for_model]

        # 3) Determine which models to try.
        if isinstance(chat_req.backup_models, str):
            backup_models = [chat_req.backup_models]
        else:
            backup_models = chat_req.backup_models or []
        logger.info(f"Resolving model aliases: primary={chat_req.model}, backups={backup_models}")
        primary_model = await resolve_model_alias(chat_req.model)
        backup_resolved = await asyncio.gather(*(resolve_model_alias(bm) for bm in backup_models))
        all_requested_models = [primary_model] + backup_resolved
        models_to_try = [m for m in all_requested_models if is_model_allowed(m)]
        if not models_to_try:
            logger.error("No models in request allowed by whitelist")
            raise HTTPException(status_code=403, detail="No models in request allowed by whitelist")
        logger.info(f"Models to try (whitelisted): {models_to_try}")

        # 4) Handle vision requests if images are provided; otherwise process text.
        images = chat_req.image_b64 if isinstance(chat_req.image_b64, list) else ([chat_req.image_b64] if chat_req.image_b64 else [])
        cleaned_images = [img.split(",")[1] if "," in img else img for img in images] if images else None
        prompt = " ".join(m.content for m in chat_req.messages if m.role == "user") if images else None

        try:
            # (A) Vision request branch.
            if cleaned_images:
                logger.info(f"Processing vision request with {len(cleaned_images)} images")
                content = None
                backup_used = False
                for i, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    try:
                        logger.info(f"Attempting vision request with model: {model}")
                        if isinstance(interface, AsyncOllamaInterface):
                            resp = await interface.send_vision(prompt, cleaned_images)
                            logger.debug(f"Vision response from {model}: {resp}")
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
                            logger.debug(f"Vision response from {model}: {resp}")
                            if "error" not in resp:
                                content = interface.extract_content_from_response(resp, is_chat=True)
                                backup_used = i > 0
                                break
                    except Exception as e:
                        logger.warning(f"Model {model} failed vision request: {e}")
                if not content:
                    logger.error("All models failed vision request")
                    raise HTTPException(500, "All models failed vision request")
                # For vision branch, we want to store both the incoming user messages (the delta)
                # and the new assistant message.
                new_assistant = ChatMessage(role="assistant", content=content)
                # Use our simple delta: take any new messages from this request beyond what is in DB.
                new_user_msgs = chat_req.messages[len(existing_db_messages):]
                all_new_msgs = new_user_msgs + [new_assistant]
                final_messages_for_db = existing_db_messages + all_new_msgs

                if conversation_id:
                    def save_vision():
                        openai_response_id = interface.last_response_id if hasattr(interface, "last_response_id") else None
                        # Merge: simply update DB with our final_messages_for_db.
                        update_conversation(conversation_id, final_messages_for_db, openai_response_id=openai_response_id)
                        if backup_used:
                            conn = get_db_connection()
                            conn.execute("UPDATE conversations SET backup_used = ? WHERE conversation_id = ?", (1, conversation_id))
                            conn.commit()
                    background_tasks.add_task(save_vision)
                    logger.info(f"Registered background task to save vision response for conversation {conversation_id}")
                return JSONResponse({"message": content}, headers={"X-Conversation-ID": conversation_id} if conversation_id else {})

            # (B) Streaming chat request branch.
            elif chat_req.stream:
                logger.info("Processing streaming chat request")
                full_response: List[str] = []
                backup_used = False
                selected_stream_gen = None
                selected_interface = None
                for i, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    local_kwargs = {"timeout_threshold": chat_req.timeout_threshold}
                    if isinstance(interface, OpenAIInterface) and interface._uses_responses_api():
                        local_kwargs["previous_response_id"] = previous_response_id
                    try:
                        logger.info(f"Attempting streaming with model: {model}")
                        gen = interface.send_chat_streaming(conversation_history, **local_kwargs)
                        first_chunk = await gen.__anext__()
                        logger.debug(f"First streaming chunk from {model}: {first_chunk}")
                        if "error" in first_chunk:
                            raise RuntimeError("Model error in first chunk")
                        selected_stream_gen = gen
                        selected_interface = interface
                        backup_used = i > 0
                        first_content = interface.extract_content_from_chunk(first_chunk)
                        if first_content:
                            full_response.append(first_content)
                        logger.info(f"Streaming started with model {model}; backup_used={backup_used}")
                        break
                    except Exception as e:
                        logger.warning(f"Model {model} failed streaming initialization: {e}")
                if not selected_stream_gen:
                    logger.error("All models failed streaming request")
                    raise HTTPException(500, "All models failed")
                async def streamer() -> AsyncGenerator[bytes, None]:
                    logger.info("Starting streaming response")
                    if full_response:
                        yield (json.dumps({"message": "".join(full_response)}) + "\n").encode("utf-8")
                    try:
                        async for chunk in selected_stream_gen:
                            if "error" in chunk:
                                raise RuntimeError("Error in streaming chunk")
                            c = selected_interface.extract_content_from_chunk(chunk)
                            if c:
                                full_response.append(c)
                                yield (json.dumps({"message": c}) + "\n").encode("utf-8")
                    except Exception as e:
                        logger.error(f"Error during streaming with model {selected_interface.model}: {e}")
                    logger.info("Streaming response completed")
                async def save_streamed():
                    await asyncio.sleep(1)
                    def _save():
                        combined_resp = "".join(full_response)
                        new_assistant_msg = ChatMessage(role="assistant", content=combined_resp)
                        # Determine new user messages as those beyond what is already stored:
                        new_user_msgs = chat_req.messages[len(existing_db_messages):]
                        final_messages_for_db = existing_db_messages + new_user_msgs + [new_assistant_msg]
                        openai_response_id = selected_interface.last_response_id if hasattr(selected_interface, "last_response_id") else None
                        update_conversation(conversation_id, final_messages_for_db, openai_response_id=openai_response_id)
                        if backup_used:
                            conn = get_db_connection()
                            conn.execute("UPDATE conversations SET backup_used = ? WHERE conversation_id = ?", (1, conversation_id))
                            conn.commit()
                    logger.info(f"Saving streamed response for conversation {conversation_id}")
                    await loop.run_in_executor(db_pool, _save)
                    logger.info("Streamed response saved")
                if conversation_id:
                    background_tasks.add_task(save_streamed)
                    logger.info(f"Registered background task to save streamed response for {conversation_id}")
                return StreamingResponse(
                    streamer(),
                    media_type="application/json",
                    headers={"X-Conversation-ID": conversation_id} if conversation_id else {}
                )

            # (C) Non-streaming chat request branch.
            else:
                logger.info("Processing non-streaming chat request")
                content = None
                backup_used = False
                selected_interface = None
                for i, model in enumerate(models_to_try):
                    interface = await get_interface(request.app, model)
                    local_kwargs = {}
                    if isinstance(interface, OpenAIInterface) and interface._uses_responses_api():
                        local_kwargs["previous_response_id"] = previous_response_id
                    try:
                        logger.info(f"Attempting non-streaming with model: {model}")
                        resp = await interface.send_chat_nonstreaming(conversation_history, **local_kwargs)
                        if "error" not in resp:
                            content = interface.extract_content_from_response(resp, is_chat=True)
                            backup_used = i > 0
                            selected_interface = interface
                            logger.info(f"Non-streaming response from {model}: {content}")
                            break
                    except Exception as e:
                        logger.warning(f"Model {model} failed non-streaming request: {e}")
                if not content:
                    logger.error("All models failed non-streaming request")
                    raise HTTPException(500, "All models failed")
                new_assistant_msg = ChatMessage(role="assistant", content=content)
                new_user_msgs = chat_req.messages[len(existing_db_messages):]
                final_messages_for_db = existing_db_messages + new_user_msgs + [new_assistant_msg]
                if conversation_id:
                    async def save_non_stream():
                        def _save():
                            openai_response_id = selected_interface.last_response_id if (selected_interface and hasattr(selected_interface, "last_response_id")) else None
                            update_conversation(conversation_id, final_messages_for_db, openai_response_id=openai_response_id)
                            if backup_used:
                                conn = get_db_connection()
                                conn.execute("UPDATE conversations SET backup_used = ? WHERE conversation_id = ?", (1, conversation_id))
                                conn.commit()
                        logger.info(f"Saving non-stream response for conversation {conversation_id}")
                        await loop.run_in_executor(db_pool, _save)
                        logger.info("Non-stream response saved")
                    background_tasks.add_task(save_non_stream)
                    logger.info(f"Registered background task to save non-streaming response for {conversation_id}")
                return JSONResponse({"message": content}, headers={"X-Conversation-ID": conversation_id} if conversation_id else {})

        except Exception as e:
            logger.exception(f"Error in chat completion for user {current_user}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Static File Endpoints
# -----------------------
@app.get("/", response_class=FileResponse, response_model=None)
async def shared_get_index():
    logger.info("Serving index.html")
    return FileResponse("static/index.html")

@app.get("/favicon.ico", response_class=FileResponse, response_model=None)
async def shared_favicon():
    logger.info("Serving favicon.ico")
    return FileResponse("static/favicon.ico")

@app.get("/styles.css", response_class=FileResponse, response_model=None)
async def shared_get_styles():
    logger.info("Serving styles.css")
    return FileResponse("static/styles.css")

# ----------------------
# Config Management
# ----------------------
@app.get("/admin/config/{key}", response_model=ConfigValue)
@app.get("/api/config/{key}", response_model=ConfigValue)
async def get_config_value(key: str, request: Request, current_user: int = Depends(get_current_user)):
    logger.info(f"User {current_user} requested config key: {key}")
    value = await get_value(request.app, key)
    if value is None:
        logger.warning(f"Config key '{key}' not found for user {current_user}")
        raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")
    logger.info(f"Retrieved config key '{key}' with value: {value}")
    return JSONResponse(content={"key": key, "value": value})

@app.post("/admin/confog/{key}", response_model=ConfigMessage)
@app.post("/api/config/{key}", response_model=ConfigMessage)
async def set_config_value(key: str, data: ConfigUpdate, request: Request, current_user: int = Depends(get_current_user)):
    logger.info(f"User {current_user} updating config key: {key} to value: {data.value}")
    try:
        await set_value(request.app, key, data.value)
        logger.info(f"Config key '{key}' updated successfully by user {current_user}")
        return JSONResponse(content={"message": f"Config key '{key}' updated successfully"})
    except Exception as e:
        logger.exception(f"Error updating config key '{key}' by user {current_user}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# All Customer App Routes
# -----------------------
customer_app.add_api_route("/signup", shared_signup, methods=["POST"], response_model=MessageResponse)
customer_app.add_api_route("/login", shared_login, methods=["POST"], response_model=TokenResponse)
customer_app.add_api_route("/conversations", shared_get_conversations, methods=["GET"], response_model=ConversationsListResponse)
customer_app.add_api_route("/conversations/{conversation_id}", shared_get_conversation_detail, methods=["GET"], response_model=ConversationDetailResponse)
customer_app.add_api_route("/ollama-models", shared_get_ollama_models, methods=["GET"], response_model=OllamaModelsResponse)
customer_app.add_api_route("/openai-models", shared_get_openai_models, methods=["GET"], response_model=OpenAIModelsResponse)
customer_app.add_api_route("/api/ollama-models", shared_get_ollama_models, methods=["GET"], response_model=OllamaModelsResponse)
customer_app.add_api_route("/api/openai-models", shared_get_openai_models, methods=["GET"], response_model=OpenAIModelsResponse)
customer_app.add_api_route("/request-profiles", list_request_profiles, methods=["GET"], response_model=List[str])
customer_app.add_api_route("/chat-completion", chat_completion, methods=["POST"], response_model=ChatCompletionResponse)
customer_app.add_api_route("/api/chat-completion", chat_completion, methods=["POST"], response_model=ChatCompletionResponse)

customer_app.add_api_route("/", shared_get_index, methods=["GET"], response_class=FileResponse, response_model=None)
customer_app.add_api_route("/favicon.ico", shared_favicon, methods=["GET"], response_class=FileResponse, include_in_schema=False, response_model=None)
customer_app.add_api_route("/styles.css", shared_get_styles, methods=["GET"], response_class=FileResponse, response_model=None)

# -----------------------
# Server Initialization
# -----------------------
async def main():
    logger.info("Initializing servers")
    customer_config = Config(
        app=customer_app,
        host="0.0.0.0",
        port=41025,
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

    logger.info("Starting customer and full-access servers")
    await asyncio.gather(customer_server.serve(), full_server.serve())
    logger.info("Servers stopped")

if __name__ == "__main__":
    logger.info("Starting application")
    asyncio.run(main())
    logger.info("Application terminated")