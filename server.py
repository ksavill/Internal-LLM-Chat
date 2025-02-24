import logging
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Union
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta

from async_ollama_interface import AsyncOllamaInterface
from openai_api import OpenAIInterface
from ollama_list_models import list_models
from database import get_db_connection, init_db, create_user, verify_user, create_conversation, update_conversation, get_user_conversations, get_conversation

logger = logging.getLogger(__name__)

app = FastAPI()
init_db()

SECRET_KEY = "kebin"

# Security setup for JWT
security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

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
    # New: Support images in conversation messages
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    model: Optional[str] = "llama3.2"
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
# Helper: Choose which interface to use
# -----------------------
def get_interface(model_name: str):
    """
    Decide whether to use the Ollama interface or the OpenAI interface
    based on model_name.
    """
    openai_like = [
        "gpt-4", "gpt-4o", "gpt-3.5",
        "gpt-4o-mini", "o1", "o1-mini",
        "o3", "o3-mini"
    ]
    if any(model_name.startswith(m) for m in openai_like):
        return OpenAIInterface(model=model_name)
    else:
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
    conversations = get_user_conversations(current_user)
    return {"conversations": conversations}

@app.get("/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str, current_user: int = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.execute('SELECT user_id FROM conversations WHERE conversation_id = ?', (conversation_id,))
    row = cursor.fetchone()
    if not row or row['user_id'] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
    messages = get_conversation(conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": messages}  # Some users also return {"model": "...", "messages": [...]} if you store a model name

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
    background_tasks: BackgroundTasks,
    req: Request,
    current_user: Optional[int] = Depends(get_optional_current_user)
):
    """
    Main endpoint to request a chat completion from either Ollama or OpenAI.
    If the user is authenticated, we store (and retrieve) conversation info in the DB.
    If the user is not authenticated, we skip DB storage.
    
    The `images` can come in two ways:
      1. As part of chat_req.messages (each ChatMessage can have images).
      2. Via chat_req.image_b64 (a single or list of base64 images).
    """
    try:
        interface = get_interface(chat_req.model)

        # images might come from image_b64 param
        images = chat_req.image_b64
        if images and isinstance(images, str):
            # Convert single string => list of strings
            images = [images]

        # Check conversation ID logic
        if current_user and chat_req.conversation_id:
            conn = get_db_connection()
            cursor = conn.execute(
                'SELECT user_id FROM conversations WHERE conversation_id = ?',
                (chat_req.conversation_id,)
            )
            row = cursor.fetchone()
            if not row or row['user_id'] != current_user:
                raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
            conversation_id = chat_req.conversation_id
        elif current_user:
            # Authenticated user, no conversation ID => create a new conversation
            conversation_id = create_conversation(current_user, chat_req.messages)
        else:
            # Not authenticated => do not track conversation
            conversation_id = None

        # Build conversation history in the format the LLM needs
        # (For Ollama or OpenAI, typically the "content" field is the user/assistant text)
        conversation_history = [m.model_dump() for m in chat_req.messages]

        # =======================
        # If using AsyncOllamaInterface
        # =======================
        if isinstance(interface, AsyncOllamaInterface):
            # If we have images => do a vision request
            if images:
                cleaned_images = []
                for img in images:
                    if img.startswith("data:"):
                        _, _, data = img.partition(",")
                        cleaned_images.append(data)
                    else:
                        cleaned_images.append(img)

                # You might combine the text content of all user messages:
                prompt = " ".join([m.content for m in chat_req.messages])
                response = await interface.send_vision(prompt, cleaned_images)
                content = response.get("response", str(response))

                # Save conversation if we have one
                messages = chat_req.messages + [ChatMessage(role="assistant", content=content)]
                if conversation_id:
                    update_conversation(conversation_id, messages)

                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return PlainTextResponse(content, headers=headers)

            # Normal text chat with streaming or non-streaming
            if chat_req.stream:
                stream_generator = await interface.send_chat_streaming(conversation_history)
                full_response = []

                async def streamer():
                    async for chunk in stream_generator:
                        content = chunk.get("message", {}).get("content", "")
                        full_response.append(content)
                        yield content.encode("utf-8")

                def save_conversation():
                    # Combine user messages with the final assistant message
                    messages = chat_req.messages + [
                        ChatMessage(role="assistant", content="".join(full_response))
                    ]
                    if conversation_id:
                        update_conversation(conversation_id, messages)

                if conversation_id:
                    background_tasks.add_task(save_conversation)

                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return StreamingResponse(streamer(), media_type="text/plain", headers=headers)
            else:
                # Non-streaming
                response = await interface.send_chat_nonstreaming(conversation_history)
                content = response.get("response", str(response))

                messages = chat_req.messages + [ChatMessage(role="assistant", content=content)]
                if conversation_id:
                    update_conversation(conversation_id, messages)

                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return PlainTextResponse(content, headers=headers)

        # =======================
        # If using OpenAIInterface
        # =======================
        elif isinstance(interface, OpenAIInterface):
            # If images => add them to conversation in some custom format
            if images:
                for img in images:
                    conversation_history.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": img}}]
                    })

            if chat_req.stream:
                stream_generator = await interface.send_chat_streaming(conversation_history)
                full_response = []

                async def streamer():
                    async for chunk in stream_generator:
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                        else:
                            content = ""
                        full_response.append(content)
                        yield content.encode("utf-8")

                def save_conversation():
                    messages = chat_req.messages + [
                        ChatMessage(role="assistant", content="".join(full_response))
                    ]
                    if conversation_id:
                        update_conversation(conversation_id, messages)

                if conversation_id:
                    background_tasks.add_task(save_conversation)

                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return StreamingResponse(streamer(), media_type="text/plain", headers=headers)

            else:
                # Non-streaming
                response = await interface.send_chat_nonstreaming(conversation_history)
                # typical openai response => response["choices"][0]["message"]["content"]
                content = response["choices"][0]["message"]["content"] if "choices" in response else str(response)

                messages = chat_req.messages + [ChatMessage(role="assistant", content=content)]
                if conversation_id:
                    update_conversation(conversation_id, messages)

                headers = {"X-Conversation-ID": conversation_id} if conversation_id else {}
                return PlainTextResponse(content, headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Static File Endpoints
# -----------------------
@app.get("/", response_class=FileResponse)
async def get_index():
    return FileResponse("static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
