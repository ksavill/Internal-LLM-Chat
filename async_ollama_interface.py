from ollama import AsyncClient
from typing import List, Dict, Any, AsyncGenerator

class AsyncOllamaInterface:
    def __init__(self, model: str, client: AsyncClient = None):
        """
        model: name of the model, e.g. 'llava:latest' or 'llama3.2'
        client: optional custom AsyncClient instance
        """
        self.model = model
        self.client = client or AsyncClient()

        # Update capabilities based on model name
        vision_models = ['llava', 'bakllava', 'moondream', 'cogvlm']
        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": False,
            "image": any(vm in self.model.lower() for vm in vision_models)
        }

    def _supports(self, feature: str) -> bool:
        """Check if the model supports a specific feature."""
        return self.capabilities.get(feature, False)
    
    def extract_content_from_response(self, response: Dict[str, Any], is_chat: bool = True) -> str:
        """
        Extract content from a response based on whether it's a chat or generate request.
        """
        if is_chat:
            return response.get("message", {}).get("content", "")
        else:
            return response.get("response", "")

    def extract_content_from_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Extract content from a streaming chunk.
        """
        return chunk.get("message", {}).get("content", "")

    async def send_chat_streaming(
            self, 
            messages: List[Dict[str, Any]], 
            timeout_threshold: float = 30.0,
            **kwargs
        ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming chat request without timeout, as Ollama is local and fast.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")
        
        try:
            # Await the chat call to obtain an async iterator before iterating
            stream_generator = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            async for chunk in stream_generator:
                yield chunk
        except Exception as e:
            yield {"error": str(e), "message": {"content": f"\n\n[Error: {str(e)}]"}}

    async def send_chat_nonstreaming(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Non-streaming chat request without timeout.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")
        
        try:
            return await self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                **kwargs
            )
        except Exception as e:
            return {
                "error": str(e),
                "message": {"content": f"[Error: {str(e)}]"}
            }

    async def send_vision(self, prompt: str, images: List[str], **kwargs) -> Dict[str, Any]:
        """
        Vision request without timeout, assuming local Ollama performance.
        """
        if not self._supports("image"):
            raise ValueError(f"Model '{self.model}' does not support image processing.")
        
        try:
            return await self.client.generate(
                model=self.model,
                prompt=prompt,
                images=images,
                **kwargs
            )
        except Exception as e:
            return {
                "error": str(e),
                "response": f"[Error: {str(e)}]"
            }
