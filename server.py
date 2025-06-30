# Standard library imports
import json
import logging
import asyncio
import signal
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union, AsyncGenerator, Dict, Any # Added Dict, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Third-party library imports
import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
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
    ForkRequest, ForkResponse
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

# Static files for serving the frontend
app.mount("/static", StaticFiles(directory="static"), name="static_files_main_app")
customer_app.mount("/static", StaticFiles(directory="static"), name="static_files_customer_app")

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

# Helper function for forking logic (can be part of the endpoint or separate)
async def perform_fork_conversation(
    db_pool,
    user_id: int,
    original_conversation_id: str,
    message_index: int
) -> str:
    loop = asyncio.get_event_loop()

    def _get_original_convo_and_check_owner():
        conn = get_db_connection()
        cursor = conn.execute(
            'SELECT user_id, messages_json FROM conversations WHERE conversation_id = ?',
            (original_conversation_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Original conversation not found")
        if row['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to fork this conversation")
        return json.loads(row['messages_json'])

    original_messages_dicts = await loop.run_in_executor(db_pool, _get_original_convo_and_check_owner)
    
    if not isinstance(original_messages_dicts, list):
        logger.error(f"Unexpected message format for conversation {original_conversation_id}")
        raise HTTPException(status_code=500, detail="Error retrieving original conversation messages")

    if message_index < 0 or message_index >= len(original_messages_dicts):
        raise HTTPException(status_code=400, detail="Invalid message index for forking")

    # Convert message dicts to ChatMessage objects for create_conversation
    # Slice to include messages up to and including message_index
    messages_to_fork_dicts = original_messages_dicts[:message_index + 1]
    # Assuming ChatMessage can be instantiated from dicts that include 'role', 'content', and optionally 'images'
    messages_to_fork_pydantic = [ChatMessage(**msg_dict) for msg_dict in messages_to_fork_dicts]
    
    def _create_forked_convo():
        return create_conversation(user_id, messages_to_fork_pydantic)

    new_conversation_id = await loop.run_in_executor(db_pool, _create_forked_convo)
    logger.info(f"User {user_id} forked conversation {original_conversation_id} up to message index {message_index} into new conversation {new_conversation_id}")
    return new_conversation_id

# -----------------------
# Health and Version Endpoints
# -----------------------
@app.get("/healthcheck", response_model=None)
@app.get("/api/healthcheck", response_model=None)
async def shared_healthcheck():
    """
    Simple healthcheck endpoint that returns 200 OK to indicate service is running.
    """
    logger.info("Healthcheck endpoint called")
    return JSONResponse(status_code=200, content={})

@app.get("/version", response_model=None)
@app.get("/api/version", response_model=None)
async def shared_get_version():
    """
    Returns the current version of the application.
    """
    logger.info("Version endpoint called")
    return JSONResponse(status_code=200, content={"version": "25.27.2"})

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
        # Assuming rows are dicts from sqlite3.Row, convert to standard dicts.
        # If ChatMessage needs specific parsing for content/images from DB, it's handled by Pydantic model.
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
        
        # get_conversation should return a list of message dictionaries
        # compatible with ChatMessage Pydantic model (e.g. role, content, images)
        message_dicts = get_conversation(conversation_id) 
        return message_dicts # Already a list of dicts if get_conversation handles json parsing
    
    loop = asyncio.get_event_loop()
    messages_as_dicts = await loop.run_in_executor(db_pool, _check_access_and_get_convo)
    
    if messages_as_dicts is None:
        logger.warning(f"User {current_user} not authorized to access conversation {conversation_id}")
        raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
    
    logger.info(f"Retrieved {len(messages_as_dicts)} messages for conversation {conversation_id}")
    # The response_model ConversationDetailResponse expects {"messages": [ChatMessageCompatibleDicts]}
    # FastAPI will automatically convert these dicts to ChatMessage Pydantic models if they match.
    return {"messages": messages_as_dicts}


@app.post("/conversations/fork", response_model=ForkResponse)
async def fork_conversation_endpoint(
    fork_req: ForkRequest,
    current_user: int = Depends(get_current_user)
):
    logger.info(f"Fork conversation request from user {current_user} for conversation {fork_req.original_conversation_id} at index {fork_req.message_index}")
    try:
        new_id = await perform_fork_conversation(
            db_pool,
            current_user,
            fork_req.original_conversation_id,
            fork_req.message_index
        )
        return ForkResponse(new_conversation_id=new_id)
    except HTTPException as e:
        logger.error(f"HTTPException during fork: {e.detail}")
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error during fork conversation: {e}")
        raise HTTPException(status_code=500, detail="Error forking conversation")

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

# Helper function to transform messages to OpenAI's multimodal format
def _to_openai_multimodal_format(messages_dicts: List[Dict]) -> List[Dict[str, Any]]:
    """
    Transforms a list of message dictionaries (each with 'role', 'content' (str), 
    and optionally 'images' (List[str] of data URLs)) into OpenAI's multimodal format.
    In OpenAI's format, content is a string for text-only messages, or a list of 
    parts (text and image_url) for multimodal messages.
    """

    processed_messages: List[Dict[str, Any]] = []

    for msg_dict in messages_dicts:
        # Shallow-copy so that we do not mutate the caller's data and so that any
        # extra keys not explicitly handled here are preserved (for example
        # `name`, `tool_call_id`, etc.).
        msg_out: Dict[str, Any] = dict(msg_dict)

        image_data_urls = msg_dict.get("images")
        text_content = msg_dict.get("content", "")

        if image_data_urls and isinstance(image_data_urls, list):
            openai_content_parts: List[Dict[str, Any]] = []

            # Always include the textual component first (even if empty) so that
            # the ordering of parts mirrors common usage.
            openai_content_parts.append({"type": "text", "text": text_content})

            for data_url in image_data_urls:
                if isinstance(data_url, str) and data_url.startswith("data:image"):
                    openai_content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
                else:
                    logger.warning(
                        f"Skipping invalid image data URL: {str(data_url)[:50]}"
                    )

            msg_out["content"] = openai_content_parts
            msg_out.pop("images", None)

        processed_messages.append(msg_out)

    return processed_messages


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

        previous_response_id: Optional[str] = None
        if current_user and chat_req.conversation_id:
            def load_prev_response_id():
                return get_conversation_openai_response_id(chat_req.conversation_id)
            previous_response_id = await loop.run_in_executor(db_pool, load_prev_response_id)
            logger.info(f"Found previous_response_id: {previous_response_id} for conversation {chat_req.conversation_id}")

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

        conversation_id: Optional[str] = None
        existing_db_messages_pydantic: List[ChatMessage] = []
        
        if current_user and chat_req.conversation_id:
            logger.info(f"Checking access to conversation {chat_req.conversation_id} for user {current_user}")
            def check_conversation_access():
                conn = get_db_connection()
                c = conn.execute("SELECT user_id FROM conversations WHERE conversation_id = ?", (chat_req.conversation_id,))
                return c.fetchone()
            row = await loop.run_in_executor(db_pool, check_conversation_access)
            if not row or row["user_id"] != current_user:
                logger.warning(f"User {current_user} not authorized for conversation {chat_req.conversation_id}")
                raise HTTPException(status_code=403, detail="Not authorized")
            conversation_id = chat_req.conversation_id
            logger.info(f"Continuing conversation ID: {conversation_id}")

            def load_existing_db_messages():
                # get_conversation returns list of dicts
                message_dicts = get_conversation(chat_req.conversation_id)
                return [ChatMessage(**msg_dict) for msg_dict in message_dicts]
            existing_db_messages_pydantic = await loop.run_in_executor(db_pool, load_existing_db_messages)
            logger.info(f"Loaded {len(existing_db_messages_pydantic)} messages from DB for conversation {conversation_id}")
            
            # The client sends all messages it has; we only care about new ones to append
            # to what's already in the DB. Here, chat_req.messages are only the *new* ones.
            # Or, if client sends full history, we might need to diff.
            # Based on append_new_user_messages, it seems client sends full history.
            # Let's adjust combined_for_model based on that.
            # The current `append_new_user_messages` is not used.
            # Assuming client sends full current state of conversation it knows.
            # We take DB messages + new messages from client if any.
            num_db_messages = len(existing_db_messages_pydantic)
            new_messages_from_request = chat_req.messages[num_db_messages:] if len(chat_req.messages) > num_db_messages else []
            combined_for_model_pydantic = existing_db_messages_pydantic + new_messages_from_request

        elif current_user:
            logger.info(f"No conversation_id provided; creating new conversation for user: {current_user}")
            # chat_req.messages are all messages for the new conversation.
            # This `create_conversation` call happens too early if the LLM call fails.
            # Defer DB creation/update until after successful LLM response.
            # For now, let's keep current logic for simplicity and address in subsequent refactor if needed.
            def create_new_convo():
                return create_conversation(current_user, chat_req.messages) # Save all messages sent in request
            conversation_id = await loop.run_in_executor(db_pool, create_new_convo)
            existing_db_messages_pydantic = [] # No existing messages for new convo from DB perspective
            combined_for_model_pydantic = chat_req.messages
            logger.info(f"Created new conversation ID: {conversation_id}")
        else: # Guest user
            logger.info("Processing as guest user; no conversation tracking will be done.")
            conversation_id = None
            existing_db_messages_pydantic = []
            combined_for_model_pydantic = chat_req.messages

        # Prepare conversation history for LLM
        # This list contains dicts with 'role', 'content' (str), 'images' (Optional[List[str]])
        raw_history_dicts = [m.model_dump(exclude_none=True) for m in combined_for_model_pydantic]
        # Transform into OpenAI's multimodal format
        openai_formatted_history = _to_openai_multimodal_format(raw_history_dicts)

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
        
        try:
            if chat_req.stream:
                logger.info("Processing streaming chat request")
                full_response_chunks: List[str] = []
                backup_used = False
                selected_stream_gen = None
                selected_interface = None # Store the interface that successfully starts streaming
                
                for i, model_name_to_try in enumerate(models_to_try):
                    interface = await get_interface(request.app, model_name_to_try)
                    local_kwargs = {"timeout_threshold": chat_req.timeout_threshold}
                    # Forward tool definitions to the underlying interface when
                    # supplied.  The various interface implementations will
                    # decide how (or whether) to use this information.
                    if chat_req.tools:
                        local_kwargs["tools"] = chat_req.tools
                    if isinstance(interface, OpenAIInterface) and interface._uses_responses_api(): # type: ignore
                        local_kwargs["previous_response_id"] = previous_response_id
                    try:
                        logger.info(f"Attempting streaming with model: {model_name_to_try}")
                        # Pass openai_formatted_history to the interface
                        gen = interface.send_chat_streaming(openai_formatted_history, **local_kwargs)
                        first_chunk = await gen.__anext__() # Check first chunk for errors
                        logger.debug(f"First streaming chunk from {model_name_to_try}: {first_chunk}")
                        
                        # Error checking for the first chunk (adapt to your interface's error reporting)
                        if isinstance(first_chunk, dict) and "error" in first_chunk:
                            raise RuntimeError(f"Model error in first chunk: {first_chunk['error']}")
                        
                        selected_stream_gen = gen
                        selected_interface = interface 
                        backup_used = i > 0
                        
                        logger.info(
                            f"Streaming started with model {model_name_to_try}; backup_used={backup_used}"
                        )
                        break 
                    except Exception as e:
                        logger.warning(f"Model {model_name_to_try} failed streaming initialization: {e}")
                        if selected_stream_gen and hasattr(selected_stream_gen, 'aclose'): # Close generator if opened
                           await selected_stream_gen.aclose()
                        selected_stream_gen = None # Reset for next attempt
                
                if not selected_stream_gen or not selected_interface:
                    logger.error("All models failed streaming request")
                    raise HTTPException(status_code=500, detail="All models failed to start streaming")

                async def streamer() -> AsyncGenerator[bytes, None]:
                    """Generator that streams NDJSON bytes back to the caller.

                    When the caller supplied the optional `tools` parameter we want
                    to preserve the full structure of every chunk emitted by the
                    upstream provider so that the client can correctly interpret
                    tool calls.  In that scenario we simply serialise each chunk
                    verbatim.

                    If no tools are in use we retain the original behaviour of
                    streaming only the assistant message content so that the
                    existing front-end continues to function exactly as before.
                    """

                    logger.info("Starting streaming response")

                    # Helper to serialise a dictionary as NDJSON bytes.
                    def _json_bytes(obj: Any) -> bytes:  # type: ignore[override]
                        """Serialise *obj* (mapping, pydantic model, etc.) to NDJSON."""

                        if not isinstance(obj, (dict, list, str, int, float, bool, type(None))):
                            # Attempt Pydantic model conversion.
                            if hasattr(obj, "model_dump") and callable(obj.model_dump):  # type: ignore[attr-defined]
                                try:
                                    obj = obj.model_dump()
                                except Exception:
                                    pass
                            elif hasattr(obj, "dict") and callable(obj.dict):  # type: ignore[attr-defined]
                                try:
                                    obj = obj.dict()
                                except Exception:
                                    pass
                            else:
                                # Fallback to string
                                obj = str(obj)

                        return (json.dumps(obj, default=str) + "\n").encode("utf-8")

                    # Firstly yield the first chunk that we already pulled above.
                    if chat_req.tools:
                        yield _json_bytes(first_chunk)  # type: ignore[arg-type]
                    else:
                        first_content = selected_interface.extract_content_from_chunk(first_chunk)  # type: ignore
                        if first_content:
                            full_response_chunks.append(first_content)
                            yield _json_bytes({"message": first_content})

                    # Now continue with the remainder of the stream.
                    try:
                        async for chunk in selected_stream_gen:  # type: ignore
                            # Surface errors as-is.
                            if isinstance(chunk, dict) and "error" in chunk:
                                logger.error(f"Error in streaming chunk: {chunk['error']}")
                                yield _json_bytes({"error": chunk["error"]})
                                break

                            if chat_req.tools:
                                # Forward full chunk to client.
                                yield _json_bytes(chunk)
                            else:
                                # Ensure dict for content extraction.
                                if isinstance(chunk, dict):
                                    chunk_dict = chunk
                                elif hasattr(chunk, "model_dump") and callable(chunk.model_dump):  # type: ignore[attr-defined]
                                    try:
                                        chunk_dict = chunk.model_dump()
                                    except Exception:
                                        chunk_dict = {}
                                elif hasattr(chunk, "dict") and callable(chunk.dict):  # type: ignore[attr-defined]
                                    try:
                                        chunk_dict = chunk.dict()
                                    except Exception:
                                        chunk_dict = {}
                                else:
                                    chunk_dict = {}
                                content_from_chunk = selected_interface.extract_content_from_chunk(  # type: ignore
                                    chunk_dict
                                )
                                if content_from_chunk:
                                    full_response_chunks.append(content_from_chunk)
                                    yield _json_bytes({"message": content_from_chunk})
                    except Exception as e:
                        logger.error(
                            f"Error during streaming with model {selected_interface.model}: {e}"
                        )  # type: ignore
                        yield _json_bytes({"error": f"Streaming error: {e}"})
                    finally:
                        if hasattr(selected_stream_gen, "aclose"):
                           await selected_stream_gen.aclose()  # type: ignore
                        logger.info("Streaming response completed")

                async def save_streamed_response_task():
                    await asyncio.sleep(0.1) # Brief pause to allow streamer to finish processing
                    final_content = "".join(full_response_chunks)
                    if not final_content.strip():
                        logger.info("No content generated by LLM, not saving empty assistant message.")
                        return

                    def _save_in_thread():
                        # Messages to save: original DB messages + new user messages from request + new assistant message
                        # Client sends full history in chat_req.messages.
                        # combined_for_model_pydantic already has all user messages.
                        new_assistant_msg_pydantic = ChatMessage(role="assistant", content=final_content, images=None)
                        all_messages_for_db_pydantic = combined_for_model_pydantic + [new_assistant_msg_pydantic]
                        
                        openai_resp_id = getattr(selected_interface, "last_response_id", None)
                        update_conversation(conversation_id, all_messages_for_db_pydantic, openai_response_id=openai_resp_id) # type: ignore
                        
                        if backup_used:
                            conn = get_db_connection()
                            try:
                                conn.execute("UPDATE conversations SET backup_used = ? WHERE conversation_id = ?", (1, conversation_id))
                                conn.commit()
                            finally:
                                conn.close()
                    
                    logger.info(f"Saving streamed response for conversation {conversation_id}")
                    await loop.run_in_executor(db_pool, _save_in_thread)
                    logger.info(f"Streamed response saved for conversation {conversation_id}")

                if conversation_id and not chat_req.tools:
                    background_tasks.add_task(save_streamed_response_task)
                    logger.info(
                        f"Registered background task to save streamed response for {conversation_id}"
                    )
                
                return StreamingResponse(
                    streamer(),
                    media_type="application/x-ndjson", # Standard for NDJSON
                    headers={"X-Conversation-ID": conversation_id} if conversation_id else {}
                )

            else: # Non-streaming
                logger.info("Processing non-streaming chat request")
                final_content_str: Optional[str] = None
                raw_response_obj: Optional[Dict[str, Any]] = None  # For tool calling
                backup_used = False
                selected_interface = None

                for i, model_name_to_try in enumerate(models_to_try):
                    interface = await get_interface(request.app, model_name_to_try)
                    local_kwargs = {}
                    if chat_req.tools:
                        local_kwargs["tools"] = chat_req.tools
                    if isinstance(interface, OpenAIInterface) and interface._uses_responses_api(): # type: ignore
                        local_kwargs["previous_response_id"] = previous_response_id
                    try:
                        logger.info(f"Attempting non-streaming with model: {model_name_to_try}")
                        # Pass openai_formatted_history to the interface
                        response_data = await interface.send_chat_nonstreaming(openai_formatted_history, **local_kwargs)
                        
                        # Error checking (adapt to your interface's error reporting)
                        if isinstance(response_data, dict) and "error" in response_data:
                            raise RuntimeError(f"Model error: {response_data['error']}")
                        
                        if chat_req.tools:
                            # When tools are supplied we forward the entire response
                            # object to the caller so that they have access to
                            # tool call data.
                            raw_response_obj = response_data
                            selected_interface = interface
                            backup_used = i > 0
                            logger.info(
                                f"Non-streaming tool-call response from {model_name_to_try} received."
                            )
                            break
                        else:
                            # Ensure dict for content extraction.
                            if not isinstance(response_data, dict):
                                if hasattr(response_data, "model_dump") and callable(response_data.model_dump):  # type: ignore[attr-defined]
                                    try:
                                        response_data_dict = response_data.model_dump()
                                    except Exception:
                                        response_data_dict = {}
                                elif hasattr(response_data, "dict") and callable(response_data.dict):  # type: ignore[attr-defined]
                                    try:
                                        response_data_dict = response_data.dict()
                                    except Exception:
                                        response_data_dict = {}
                                else:
                                    response_data_dict = {}
                            else:
                                response_data_dict = response_data

                            extracted_content = interface.extract_content_from_response(
                                response_data_dict, is_chat=True
                            )

                            if extracted_content is not None:
                                final_content_str = extracted_content
                                selected_interface = interface
                                backup_used = i > 0
                                logger.info(
                                    f"Non-streaming response from {model_name_to_try} received."
                                )
                                break
                            else:
                                # No content extracted (and no explicit error). Log for diagnostics.
                                logger.warning(
                                    f"Model {model_name_to_try} returned no content and no error."
                                )
                            
                    except Exception as e:
                        logger.warning(f"Model {model_name_to_try} failed non-streaming request: {e}")
                
                # Ensure at least one model succeeded.
                if (not chat_req.tools and final_content_str is None) or (
                    chat_req.tools and raw_response_obj is None
                ):
                    logger.error("All models failed non-streaming request")
                    raise HTTPException(
                        status_code=500,
                        detail="All models failed to generate a response",
                    )

                if conversation_id and not chat_req.tools:
                    async def save_non_streamed_response_task():
                        def _save_in_thread():
                            # combined_for_model_pydantic already has all user messages.
                            new_assistant_msg_pydantic = ChatMessage(role="assistant", content=final_content_str, images=None) # type: ignore
                            all_messages_for_db_pydantic = combined_for_model_pydantic + [new_assistant_msg_pydantic]
                            
                            openai_resp_id = getattr(selected_interface, "last_response_id", None)
                            update_conversation(conversation_id, all_messages_for_db_pydantic, openai_response_id=openai_resp_id) # type: ignore
                            
                            if backup_used:
                                conn = get_db_connection()
                                try:
                                    conn.execute("UPDATE conversations SET backup_used = ? WHERE conversation_id = ?", (1, conversation_id))
                                    conn.commit()
                                finally:
                                    conn.close()
                        
                        logger.info(f"Saving non-streamed response for conversation {conversation_id}")
                        await loop.run_in_executor(db_pool, _save_in_thread)
                        logger.info(f"Non-streamed response saved for conversation {conversation_id}")
                    
                    background_tasks.add_task(save_non_streamed_response_task)
                    logger.info(f"Registered background task to save non-streaming response for {conversation_id}")

                # Ensure the response is JSON serialisable.  The OpenAI
                # interface already returns plain dictionaries but the Ollama
                # python client returns a *ChatResponse* pydantic model which
                # is **not** directly serialisable by FastAPI/JSONResponse.

                def _to_jsonable(obj: Any):  # type: ignore[override]
                    """Best-effort conversion of various response types into plain dicts."""

                    if obj is None:
                        return None

                    # Most common case – obj is already a mapping.
                    if isinstance(obj, dict):
                        return obj

                    # Pydantic v2 models (like ollama.ChatResponse) expose
                    # `.model_dump()`.
                    if hasattr(obj, "model_dump") and callable(obj.model_dump):  # type: ignore[attr-defined]
                        try:
                            return obj.model_dump()
                        except Exception:
                            pass

                    # Pydantic v1 models use `.dict()`.
                    if hasattr(obj, "dict") and callable(obj.dict):  # type: ignore[attr-defined]
                        try:
                            return obj.dict()
                        except Exception:
                            pass

                    # Fallback – attempt to use __dict__ or last-resort string.
                    return getattr(obj, "__dict__", str(obj))

                if chat_req.tools:
                    serialisable_payload = _to_jsonable(raw_response_obj)
                    return JSONResponse(
                        serialisable_payload,  # type: ignore[arg-type]
                        headers={"X-Conversation-ID": conversation_id} if conversation_id else {},
                    )
                else:
                    return JSONResponse(
                        {"message": final_content_str},
                        headers={"X-Conversation-ID": conversation_id} if conversation_id else {},
                    )

        except HTTPException: # Re-raise HTTPExceptions directly
            raise
        except Exception as e:
            logger.exception(f"Error in chat completion for user {current_user}: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


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

@app.post("/admin/confog/{key}", response_model=ConfigMessage) # Typo: confog -> config
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
customer_app.add_api_route("/healthcheck", shared_healthcheck, methods=["GET"], response_model=None)
customer_app.add_api_route("/version", shared_get_version, methods=["GET"], response_model=None)
customer_app.add_api_route("/api/healthcheck", shared_healthcheck, methods=["GET"], response_model=None)
customer_app.add_api_route("/api/version", shared_get_version, methods=["GET"], response_model=None)
customer_app.add_api_route("/signup", shared_signup, methods=["POST"], response_model=MessageResponse)
customer_app.add_api_route("/login", shared_login, methods=["POST"], response_model=TokenResponse)
customer_app.add_api_route("/conversations", shared_get_conversations, methods=["GET"], response_model=ConversationsListResponse)
customer_app.add_api_route("/conversations/{conversation_id}", shared_get_conversation_detail, methods=["GET"], response_model=ConversationDetailResponse)
customer_app.add_api_route("/conversations/fork", fork_conversation_endpoint, methods=["POST"], response_model=ForkResponse)
customer_app.add_api_route("/ollama-models", shared_get_ollama_models, methods=["GET"], response_model=OllamaModelsResponse)
customer_app.add_api_route("/openai-models", shared_get_openai_models, methods=["GET"], response_model=OpenAIModelsResponse)
customer_app.add_api_route("/api/ollama-models", shared_get_ollama_models, methods=["GET"], response_model=OllamaModelsResponse)
customer_app.add_api_route("/api/openai-models", shared_get_openai_models, methods=["GET"], response_model=OpenAIModelsResponse)
customer_app.add_api_route("/request-profiles", list_request_profiles, methods=["GET"], response_model=List[str])
customer_app.add_api_route("/api/request-profiles", list_request_profiles, methods=["GET"], response_model=List[str])
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
        limit_concurrency=1000,
        workers=4,
        timeout_keep_alive=120,
    )
    customer_server = Server(customer_config)

    full_config = Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        limit_concurrency=100,
        workers=1,
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