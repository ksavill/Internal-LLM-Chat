from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional
from pydantic import BaseModel
import base64

from async_ollama_interface import AsyncOllamaInterface
from openai_api import OpenAIInterface

from ollama_list_models import list_models

app = FastAPI()

# -----------------------
# Data Models for Chat API
# -----------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "llama3.2"  # Default model for chat requests.
    messages: List[ChatMessage]

# -----------------------
# Helper: Choose which interface to use
# -----------------------
def get_interface(model_name: str):
    """
    For a given model_name, decide whether to use the Ollama interface 
    or the OpenAI interface.
    This is just a simple example:
      - If the model name *starts with* 'gpt-' or 'gpt4' or something similar,
        we treat it as an OpenAI model.
      - Otherwise, we default to the Ollama interface.
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
# API Endpoints
# -----------------------

@app.get("/ollama-models")
async def get_ollama_models():
    """
    Calls the 'ollama list' command using our synchronous list_models()
    implementation and returns a JSON response containing the structured model list.
    """
    try:
        # Run the blocking CLI function in a thread pool.
        models = await run_in_threadpool(list_models)
        return JSONResponse(content={"models": models})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openai-models")
async def get_openai_models():
    """
    Returns the list of available OpenAI models if an API key is configured.
    If not, it returns an empty list.
    """
    # Instantiate the OpenAIInterface with a default/dummy model.
    openai_interface = OpenAIInterface(model="gpt-4o-mini")
    
    # Check if the API key is configured.
    if not openai_interface.is_api_key_configured():
        # If no API key is configured, return no OpenAI models.
        return {"models": []}
    
    models = [
        {"NAME": "o3-mini"},
        {"NAME": "o1-preview"},
        {"NAME": "gpt-4o"},
        {"NAME": "gpt-4o-mini"}
    ]
    return {"models": models}

@app.post("/chat-completion/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    The client sends a JSON payload with a list of messages.
    The response is streamed back as plain text.
    """
    try:
        interface = get_interface(request.model)

        # If the interface is the async Ollama interface:
        if isinstance(interface, AsyncOllamaInterface):
            # 1. Await the async streaming generator
            stream_generator = await interface.send_chat_streaming(
                [m.dict() for m in request.messages]
            )

            # 2. Create an async generator function that yields chunk data
            async def streamer():
                async for chunk in stream_generator:
                    if "message" in chunk and "content" in chunk["message"]:
                        # Ollama chunk structure
                        text = chunk["message"]["content"]
                        yield text.encode("utf-8")
                    elif "choices" in chunk and "delta" in chunk["choices"][0]:
                        # OpenAI-like chunk structure from Ollama
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            text = delta["content"]
                            yield text.encode("utf-8")

            return StreamingResponse(streamer(), media_type="text/plain")

        # Otherwise, it's the sync OpenAIInterface:
        else:
            # 1. Call the sync streaming generator
            stream_generator = interface.send_chat_streaming(
                [m.dict() for m in request.messages]
            )

            # 2. Wrap the sync generator in an async generator 
            #    so FastAPI can stream it.
            async def streamer():
                for chunk in stream_generator:
                    # chunk is a ChatCompletionChunk object
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        # delta is a ChoiceDelta object
                        if delta.content:
                            yield delta.content.encode("utf-8")

            return StreamingResponse(streamer(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-completion")
async def chat_non_stream(request: ChatRequest):
    """
    Non-streaming chat endpoint.
    The client sends a JSON payload with a list of messages 
    and receives a JSON response.
    """
    try:
        interface = get_interface(request.model)

        if isinstance(interface, AsyncOllamaInterface):
            # Async call:
            response = await interface.send_chat_nonstreaming(
                [m.dict() for m in request.messages]
            )
        else:
            # Sync call:
            response = interface.send_chat_nonstreaming(
                [m.dict() for m in request.messages]
            )

        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vision")
async def vision(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form("llava")
):
    """
    Vision endpoint.
    Accepts a multipart/form-data request with:
      - 'prompt': The text prompt (as a form field).
      - 'image': An image file.
      - 'model': (Optional) The model name, defaulting to "llava".
    The image is read, base64-encoded, and sent to the async interface.
    """
    try:
        # Read and base64-encode the image
        content = await image.read()
        image_base64 = base64.b64encode(content).decode("utf-8")

        # For your custom models, e.g. "llava," we assume AsyncOllamaInterface
        interface = AsyncOllamaInterface(model=model)

        vision_response = await interface.send_vision(
            prompt=prompt,
            images=[image_base64]
        )
        return JSONResponse(content=vision_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=FileResponse)
async def get_index():
    """
    Serves the static HTML file for the chat UI.
    Ensure that the HTML file is placed in the "static" directory.
    """
    return FileResponse("static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Returns the favicon.ico from the static directory.
    This endpoint disables schema inclusion for browsers automatically requesting favicon.
    """
    return FileResponse("static/favicon.ico")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)