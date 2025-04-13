from ollama import AsyncClient
from typing import List, Dict, Any, AsyncGenerator
import logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # ensure debug-level logs


class AsyncOllamaInterface:
    def __init__(self, model: str, client: AsyncClient = None):
        """
        Args:
            model: name of the model, e.g. 'llava:latest' or 'llama2-7b'
            client: optional custom AsyncClient instance
        """
        self.model = model
        self.client = client or AsyncClient()

        # Define supported capabilities. Enables image processing if the model indicates a vision model.
        vision_models = ['llava', 'bakllava', 'moondream', 'cogvlm']
        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": False,
            "image": any(vm in self.model.lower() for vm in vision_models)
        }

        logger.debug(f"Initializing AsyncOllamaInterface with model: {self.model}")
        logger.debug(
            f"Capabilities: chat={self.capabilities['chat']}, generate={self.capabilities['generate']}, "
            f"tool={self.capabilities['tool']}, image={self.capabilities['image']}"
        )

    def _supports(self, feature: str) -> bool:
        """
        Check if the model supports a specific feature.
        """
        support = self.capabilities.get(feature, False)
        logger.debug(f"Feature support check for '{feature}': {support}")
        return support

    def extract_content_from_response(self, response: Dict[str, Any], is_chat: bool = True) -> str:
        """
        Extract content from a response based on whether it's a chat or generate request.
        """
        if is_chat:
            content = response.get("message", {}).get("content", "")
            logger.debug(f"Extracted chat content from response: {content}")
        else:
            content = response.get("response", "")
            logger.debug(f"Extracted generate content from response: {content}")
        return content

    def extract_content_from_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Extract content from a streaming chunk.
        """
        content = chunk.get("message", {}).get("content", "")
        logger.debug(f"Extracted content from chunk: {content}")
        return content

    async def send_chat_streaming(
        self, 
        messages: List[Dict[str, Any]], 
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming chat request with Ollama. Yields each chunk as it is received.
        """
        if not self._supports("chat"):
            error_msg = f"Model '{self.model}' does not support chat requests."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"[Ollama] Initiating streaming chat request for model: {self.model}")
        logger.debug(f"[Ollama] Input messages: {messages}")

        try:
            stream_generator = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            logger.debug(f"[Ollama] Obtained streaming generator for model: {self.model}")
            async for chunk in stream_generator:
                logger.debug(f"[Ollama] Received streaming chunk: {chunk}")
                yield chunk
        except Exception as e:
            logger.error(f"[Ollama] Streaming chat error for model '{self.model}': {e}")
            logger.debug(traceback.format_exc())
            yield {"error": str(e), "message": {"content": f"\n\n[Ollama Error: {str(e)}]"}}

    async def send_chat_nonstreaming(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Non-streaming chat request for Ollama.
        """
        if not self._supports("chat"):
            error_msg = f"Model '{self.model}' does not support chat requests."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"[Ollama] Initiating non-streaming chat request for model: {self.model}")
        logger.debug(f"[Ollama] Input messages: {messages}")

        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                **kwargs
            )
            logger.debug(f"[Ollama] Received non-streaming response: {response}")
            return response
        except Exception as e:
            logger.error(f"[Ollama] Non-streaming chat error for model '{self.model}': {e}")
            logger.debug(traceback.format_exc())
            return {
                "error": str(e),
                "message": {"content": f"[Ollama Error: {str(e)}]"}
            }

    async def send_vision(self, prompt: str, images: List[str], **kwargs) -> Dict[str, Any]:
        """
        Vision request if supported by the local model.
        """
        if not self._supports("image"):
            error_msg = f"Model '{self.model}' does not support image processing."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"[Ollama] Initiating vision request for model: {self.model}")
        logger.debug(f"[Ollama] Vision parameters: prompt='{prompt}', number of images={len(images)}")

        try:
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                images=images,
                **kwargs
            )
            logger.debug(f"[Ollama] Received vision response: {response}")
            return response
        except Exception as e:
            logger.error(f"[Ollama] Vision request error for model '{self.model}': {e}")
            logger.debug(traceback.format_exc())
            return {
                "error": str(e),
                "response": f"[Ollama Error: {str(e)}]"
            }
