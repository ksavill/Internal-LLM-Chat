import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional, Union
from pydantic import BaseModel
import base64

from async_ollama_interface import AsyncOllamaInterface
from openai_api import OpenAIInterface
from ollama_list_models import list_models

logger = logging.getLogger(__name__)

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
    stream: Optional[bool] = False
    image_b64: Optional[Union[str, List[str]]] = None

# -----------------------
# Helper: Choose which interface to use
# -----------------------

def get_interface(model_name: str):
    """
    For a given model_name, decide whether to use the Ollama interface 
    or the OpenAI interface.
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
    try:
        models = await run_in_threadpool(list_models)
        return JSONResponse(content={"models": models})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openai-models")
async def get_openai_models():
    openai_interface = OpenAIInterface(model="gpt-4o-mini")
    if not openai_interface.is_api_key_configured():
        return {"models": []}
    models = [
        # {"NAME": "o3-mini-high"},
        # {"NAME": "o3-mini-medium"},
        # {"NAME": "o3-mini-low"},
        {"NAME": "o3-mini"},
        {"NAME": "o1-preview"},
        {"NAME": "gpt-4o"},
        {"NAME": "gpt-4o-mini"}
    ]
    return {"models": models}

@app.post("/chat-completion")
async def chat_completion(chat_req: ChatRequest, req: Request):
    try:
        interface = get_interface(chat_req.model)

        # Normalize images if provided.
        images = chat_req.image_b64
        if images and isinstance(images, str):
            images = [images]

        # ---- Async Ollama Interface ----
        if isinstance(interface, AsyncOllamaInterface):
            if images:
                # Clean image data: remove any data URI header.
                cleaned_images = []
                for img in images:
                    if img.startswith("data:"):
                        _, _, data = img.partition(",")
                        cleaned_images.append(data)
                    else:
                        cleaned_images.append(img)
                images = cleaned_images

                prompt = " ".join([m.content for m in chat_req.messages])
                response = await interface.send_vision(prompt, images)
                # Extract only the text from the response.
                if hasattr(response, "response"):
                    text_response = response.response
                elif isinstance(response, dict) and "response" in response:
                    text_response = response["response"]
                else:
                    text_response = str(response)
                return PlainTextResponse(text_response)

            # Text-only branch.
            conversation_history = [m.dict() for m in chat_req.messages]
            if chat_req.stream:
                stream_generator = await interface.send_chat_streaming(conversation_history)
                
                async def streamer():
                    async for chunk in stream_generator:
                        try:
                            if "message" in chunk and "content" in chunk["message"]:
                                yield chunk["message"]["content"].encode("utf-8")
                            elif "choices" in chunk and "delta" in chunk["choices"][0]:
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"].encode("utf-8")
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                            logger.info("Client disconnected during streaming (Ollama async): %s", str(e))
                            break
                return StreamingResponse(streamer(), media_type="text/plain")
            else:
                conversation_history = [m.dict() for m in chat_req.messages]
                response = await interface.send_chat_nonstreaming(conversation_history)
                if hasattr(response, "response"):
                    text_response = response.response
                elif isinstance(response, dict) and "response" in response:
                    text_response = response["response"]
                else:
                    text_response = str(response)
                return PlainTextResponse(text_response)

        # ---- OpenAI Interface ----
        elif isinstance(interface, OpenAIInterface):
            conversation_history = [m.dict() for m in chat_req.messages]
            if images:
                for img in images:
                    conversation_history.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img}}
                        ]
                    })
            if chat_req.stream:
                # Get the synchronous generator from OpenAI.
                stream_generator = interface.send_chat_streaming(conversation_history)
                
                async def streamer():
                    try:
                        for chunk in stream_generator:
                            # Check if client has disconnected.
                            if await req.is_disconnected():
                                logger.info("Client disconnected; aborting OpenAI stream.")
                                break
                            if chunk.choices and chunk.choices[0].delta:
                                delta = chunk.choices[0].delta
                                if delta.content:
                                    try:
                                        yield delta.content.encode("utf-8")
                                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                                        logger.info("Client disconnected during yield (OpenAI): %s", str(e))
                                        break
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                        logger.info("Client disconnected during streaming (OpenAI): %s", str(e))
                    except Exception as e:
                        logger.exception("Unexpected error during OpenAI streaming: %s", str(e))
                    finally:
                        # Attempt to close the underlying generator to cancel the OpenAI stream.
                        if hasattr(stream_generator, "close"):
                            try:
                                stream_generator.close()
                            except Exception as e:
                                logger.debug("Error closing OpenAI stream generator: %s", str(e))
                return StreamingResponse(streamer(), media_type="text/plain")
            else:
                response = interface.send_chat_nonstreaming(conversation_history)
                return PlainTextResponse(response)

        # ---- Fallback for unknown interface types ----
        else:
            if images:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{chat_req.model}' does not support image inputs."
                )
            conversation_history = [m.dict() for m in chat_req.messages]
            if chat_req.stream:
                stream_generator = interface.send_chat_streaming(conversation_history)
                async def streamer():
                    for chunk in stream_generator:
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                try:
                                    yield delta.content.encode("utf-8")
                                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                                    logger.info("Client disconnected during yield (Fallback): %s", str(e))
                                    break
                return StreamingResponse(streamer(), media_type="text/plain")
            else:
                response = interface.send_chat_nonstreaming(conversation_history)
                return PlainTextResponse(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vision")
async def vision(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form("llava")
):
    try:
        content = await image.read()
        image_base64 = base64.b64encode(content).decode("utf-8")
        interface = AsyncOllamaInterface(model=model)
        vision_response = await interface.send_vision(prompt=prompt, images=[image_base64])
        return JSONResponse(content=vision_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=FileResponse)
async def get_index():
    return FileResponse("static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)