import json
import aiohttp
import asyncio
from typing import Dict, List, Any, AsyncGenerator
import logging

"""
Will hold off on finishing this until streaming issues to OpenAI have been resolved, as that could complicate testing this
"""

logger = logging.getLogger(__name__)

class ProxyInterface:
    """
    An interface that routes requests to arbitrary endpoints configured as proxy nodes.
    This allows for distributing LLM workloads across multiple servers.
    """

    def __init__(self, model: str, endpoint_url: str, api_key: str = None):
        """
        Args:
            model: The name of the model to use
            endpoint_url: The URL of the proxy endpoint to route requests to
            api_key: Optional API key for authentication with the proxy endpoint
        """
        self.model = model
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self._session = None
        
        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": True,
            "image": True,
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Create or retrieve a reusable aiohttp session to avoid overhead on repeated calls.
        """
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    def _supports(self, feature: str) -> bool:
        """
        Check if this model supports a given capability.
        """
        return self.capabilities.get(feature, False)

    def extract_content_from_response(self, response: Dict[str, Any], is_chat: bool = True) -> str:
        """
        Extracts the final text from a response.
        """
        content = (
            response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
        )
        
        if content:
            return content
            
        if is_chat:
            return response.get("message", {}).get("content", "")
        else:
            return response.get("response", "")

    def extract_content_from_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Extracts text from a streaming response chunk.
        Handles both OpenAI and Ollama formats.
        """
        choices = chunk.get("choices", [])
        if choices and isinstance(choices, list):
            content = choices[0].get("delta", {}).get("content", "")
            if content:
                return content
        
        evt_type = chunk.get("type", "")
        if evt_type == "response.output_text.delta":
            return chunk.get("delta", "")
        elif evt_type == "response.output_text.done":
            return chunk.get("text", "")
            
        return chunk.get("message", {}).get("content", "")

    async def close(self):
        """Close the aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def send_chat_streaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat completions from the proxy endpoint.
        
        Args:
            messages: A list of conversation messages.
            timeout_threshold: Maximum time (in seconds) to wait for each chunk.
            
        Yields:
            Dict[str, Any]: A parsed chunk of the response.
        """
        if not self._supports("chat"):
            yield {"error": f"Model '{self.model}' does not support chat requests."}
            return

        session = await self._get_session()
        url = f"{self.endpoint_url}/chat-completion"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        try:
            response = await session.post(url, json=payload)
            response.raise_for_status()
            
            while True:
                try:
                    line = await asyncio.wait_for(
                        response.content.readline(),
                        timeout=timeout_threshold
                    )
                except asyncio.TimeoutError:
                    yield {"error": f"No data from proxy in {timeout_threshold} seconds (mid-stream)."}
                    return

                if not line:
                    break

                line_decoded = line.decode('utf-8').strip()
                if not line_decoded:
                    continue

                if line_decoded.startswith("data: "):
                    data = line_decoded[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
                else:
                    try:
                        yield json.loads(line_decoded)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.exception(f"Error in proxy streaming: {e}")
            yield {"error": str(e), "message": {"content": f"[Error: {str(e)}]"}}

    async def send_chat_nonstreaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming chat completions from the proxy endpoint.
        
        Args:
            messages: A list of conversation messages.
            timeout_threshold: Max time (in seconds) to wait for the response.
            
        Returns:
            The parsed JSON response from the proxy endpoint.
        """
        if not self._supports("chat"):
            return {
                "error": f"Model '{self.model}' does not support chat requests.",
                "message": {"content": f"[Error: Model '{self.model}' does not support chat requests.]"}
            }

        session = await self._get_session()
        url = f"{self.endpoint_url}/chat-completion"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = await session.post(url, json=payload)
            response.raise_for_status()
            
            try:
                text_body = await asyncio.wait_for(response.text(), timeout=timeout_threshold)
            except asyncio.TimeoutError:
                return {
                    "error": f"No data from proxy in {timeout_threshold} seconds (non-streaming).",
                    "message": {"content": f"[Error: No data from proxy in {timeout_threshold} seconds.]"}
                }
            
            try:
                return json.loads(text_body)
            except json.JSONDecodeError:
                return {"error": "Malformed JSON from proxy.", "raw": text_body}
                
        except Exception as e:
            logger.exception(f"Error in proxy non-streaming: {e}")
            return {
                "error": str(e),
                "message": {"content": f"[Error: {str(e)}]"}
            }

    async def send_vision(
        self,
        prompt: str,
        images: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Vision request to the proxy endpoint.
        
        Args:
            prompt: The text prompt to send.
            images: List of base64-encoded images.
            
        Returns:
            The parsed JSON response from the proxy endpoint.
        """
        if not self._supports("image"):
            return {
                "error": f"Model '{self.model}' does not support image processing.",
                "response": f"[Error: Model '{self.model}' does not support image processing.]"
            }

        session = await self._get_session()
        url = f"{self.endpoint_url}/chat-completion"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "image_b64": images,
            "stream": False,
            **kwargs
        }
        
        try:
            response = await session.post(url, json=payload)
            response.raise_for_status()
            return await response.json()
        except Exception as e:
            logger.exception(f"Error in proxy vision: {e}")
            return {
                "error": str(e),
                "response": f"[Error: {str(e)}]"
            }