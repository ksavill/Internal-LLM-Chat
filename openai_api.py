import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class OpenAIInterface:
    """
    A unified interface that can talk to either:
      - Chat Completions API (/v1/chat/completions) for normal models
      - Responses API (/v1/responses) for "o1-pro" or similar models
    """

    def __init__(self, model: str, api_key: str = None):
        """
        Args:
            model: e.g. "gpt-4o", "gpt-3.5-turbo", or "o1-pro".
            api_key: explicitly provided or read from environment 
                     (openai_token / OPENAI_API_KEY).
        """
        self.model = model
        if api_key:
            self.api_key = api_key
        else:
            # Priority: 'openai_token', else 'OPENAI_API_KEY'
            if os.environ.get('openai_token'):
                self.api_key = os.environ['openai_token']
            else:
                self.api_key = os.environ.get('OPENAI_API_KEY')

        self._session = None
        # Basic capabilities map
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
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._session

    def is_api_key_configured(self) -> bool:
        """True if the OpenAI API key is set in code or environment."""
        return bool(self.api_key)

    def _supports(self, feature: str) -> bool:
        """
        Check if this model supports a given capability (like "chat", "image", etc.).
        Currently just returns True for everything, but you could refine if needed.
        """
        return self.capabilities.get(feature, False)

    def _uses_responses_api(self) -> bool:
        """
        Return True if we should use /v1/responses for this model (i.e. "o1-pro").
        """
        return self.model.startswith("o1-pro")

    def extract_content_from_response(self, response: Dict[str, Any], is_chat: bool = True) -> str:
        """
        Extracts the final text from a typical ChatCompletion-like response:
          response["choices"][0]["message"]["content"]
        """
        return (
            response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
        )

    def extract_content_from_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Extracts text from a streaming response chunk.
        
        For o1-pro responses, it checks:
          - "response.output_text.delta": returns the text from the "delta" field.
          - "response.output_text.done": returns the final text from the "text" field.
        Otherwise, falls back to the traditional structure.
        """
        evt_type = chunk.get("type", "")
        if evt_type == "response.output_text.delta":
            return chunk.get("delta", "")
        elif evt_type == "response.output_text.done":
            return chunk.get("text", "")
        else:
            # Fallback extraction from a choices list (if available)
            choices = chunk.get("choices", [])
            if choices and isinstance(choices, list):
                return choices[0].get("delta", {}).get("content", "")
            return ""

    async def close(self):
        """Close the aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ----------------------------------------------------------------
    # Example methods for listing fine-tuning jobs (optional)
    # ----------------------------------------------------------------
    async def list_fine_tuning_jobs(self) -> Dict[str, Any]:
        """
        Example helper to list fine-tuning jobs from the official OpenAI API.
        """
        session = await self._get_session()
        url = "https://api.openai.com/v1/fine_tuning/jobs"
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    def get_successful_fine_tuned_models(self, jobs: Dict[str, Any]) -> List[str]:
        """
        Filter out IDs of successfully fine-tuned models from the jobs list.
        """
        return [
            job["fine_tuned_model"]
            for job in jobs.get("data", [])
            if job.get("status") == "succeeded" and job.get("fine_tuned_model") is not None
        ]

    # ----------------------------------------------------------------
    # Utility for older "o3-mini" models (optional)
    # ----------------------------------------------------------------
    def _prepare_payload(self, base_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example logic to handle special naming for certain "o3-mini-*" models.
        Adjust as needed for your environment or remove if not needed.
        """
        if self.model in ["o3-mini-low", "o3-mini-medium", "o3-mini-high"]:
            reasoning_effort = self.model.split("-")[-1]
            base_payload["model"] = "o3-mini"
            base_payload["reasoning_effort"] = reasoning_effort
        elif self.model in ["o1-pro-low", "o1-pro", "o1-pro-high"]:
            reasoning_effort = self.model.split("-")[-1]
            base_payload["model"] = "o1-pro"
            base_payload["reasoning"] = {"effort": reasoning_effort} # validated this is different via the playground
        else:
            base_payload["model"] = self.model
        return base_payload

    # ----------------------------------------------------------------
    # Chat Completions endpoint
    # ----------------------------------------------------------------
    async def send_chat_streaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat completions from /v1/chat/completions (for normal models)
        or from /v1/responses (for o1-pro).
        
        Args:
            messages: A list of conversation messages.
            timeout_threshold (float): Maximum time (in seconds) to wait for *each*
                                       SSE line or chunk of data. Default=30.
        Yields:
            Dict[str, Any]: A parsed chunk of the response.
        """
        if self._uses_responses_api():
            async for chunk in self.send_response_streaming(
                messages, timeout_threshold=timeout_threshold, **kwargs
            ):
                yield chunk
        else:
            if not self._supports("chat"):
                raise ValueError(f"Model '{self.model}' does not support chat requests.")

            session = await self._get_session()
            url = "https://api.openai.com/v1/chat/completions"
            payload = {"messages": messages, "stream": True, **kwargs}
            payload = self._prepare_payload(payload)

            response = await session.post(url, json=payload)
            response.raise_for_status()

            while True:
                try:
                    line = await asyncio.wait_for(
                        response.content.readline(),
                        timeout=timeout_threshold
                    )
                except asyncio.TimeoutError:
                    raise ConnectionError(
                        f"No data from /v1/chat/completions in {timeout_threshold} seconds (mid-stream)."
                    )

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

    async def send_chat_nonstreaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming chat completions.
        Raises a ConnectionError if there's no data within `timeout_threshold`.

        Args:
            messages: A list of conversation messages.
            timeout_threshold: Max time (in seconds) to wait for the entire
                               response from server. Default=30.

        Returns:
            The parsed JSON response from /v1/chat/completions (or from /v1/responses if o1-pro).
        """
        if self._uses_responses_api():
            return await self.send_response_nonstreaming(
                messages, timeout_threshold=timeout_threshold, **kwargs
            )

        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        session = await self._get_session()
        url = "https://api.openai.com/v1/chat/completions"
        payload = {"messages": messages, "stream": False, **kwargs}
        payload = self._prepare_payload(payload)

        response = await session.post(url, json=payload)
        response.raise_for_status()

        try:
            text_body = await asyncio.wait_for(response.text(), timeout=timeout_threshold)
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"No data from /v1/chat/completions in {timeout_threshold} seconds (non-streaming)."
            )

        try:
            return json.loads(text_body)
        except json.JSONDecodeError:
            return {"error": "Malformed JSON from ChatCompletions.", "raw": text_body}

    # ----------------------------------------------------------------
    # "Responses" endpoint for o1-pro
    # ----------------------------------------------------------------
    async def send_response_streaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Sends a request to the /v1/responses endpoint with stream=True
        and yields SSE JSON events as they arrive.

        Args:
            messages: A list of conversation messages.
            timeout_threshold (float): Maximum time (in seconds) to wait for *each* SSE line.
        Yields:
            Dict[str, Any]: SSE chunk data.
        """
        session = await self._get_session()
        url = "https://api.openai.com/v1/responses"
        input_payload = self._messages_to_input_items(messages)
        payload = {
            "model": self.model,
            "input": input_payload,
            "stream": True,
        }
        payload.update(kwargs)
        payload = self._prepare_payload(payload)

        logger.debug(f"Sending payload to OpenAI Responses API: {json.dumps(payload, indent=2)}")

        try:
            response = await session.post(url, json=payload)
            response.raise_for_status()
        except aiohttp.ClientResponseError as e:
            # Attempt to get error detail
            try:
                error_detail = await response.text()
            except: 
                error_detail = "No detail."
            logger.error(f"OpenAI API error: {e.status}, message='{e.message}', detail='{error_detail}'")
            raise

        while True:
            try:
                line = await asyncio.wait_for(response.content.readline(), timeout=timeout_threshold)
            except asyncio.TimeoutError:
                raise ConnectionError(
                    f"No data from /v1/responses in {timeout_threshold} seconds (mid-stream)."
                )

            if not line:
                break

            line_decoded = line.decode("utf-8", errors="replace").strip()
            # Process only lines that start with "data: "
            if not line_decoded or not line_decoded.startswith("data: "):
                continue

            data_str = line_decoded[len("data: "):].strip()
            if not data_str:
                continue

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            yield chunk
            if chunk.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                break

    async def send_response_nonstreaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming request to the /v1/responses endpoint for models like o1-pro.

        Args:
            messages: A list of conversation messages.
            timeout_threshold: Max time (in seconds) to wait for the entire response.

        Returns:
            The parsed JSON response from /v1/responses.
        """
        session = await self._get_session()
        url = "https://api.openai.com/v1/responses"
        input_payload = self._messages_to_input_items(messages)
        payload = {
            "model": self.model,
            "input": input_payload,
            "stream": False
        }
        payload.update(kwargs)

        payload = self._prepare_payload(payload)

        logger.debug(f"Sending payload to OpenAI Responses API: {json.dumps(payload, indent=2)}")
        try:
            response = await session.post(url, json=payload)
            response.raise_for_status()
        except aiohttp.ClientResponseError as e:
            try:
                error_detail = await response.text()
            except:
                error_detail = "No detail."
            logger.error(f"OpenAI API error: {e.status}, message='{e.message}', detail='{error_detail}'")
            raise

        try:
            text_body = await asyncio.wait_for(response.text(), timeout=timeout_threshold)
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"No data from /v1/responses in {timeout_threshold} seconds (non-streaming)."
            )

        try:
            return json.loads(text_body)
        except json.JSONDecodeError:
            return {"error": "Malformed JSON from Responses.", "raw": text_body}

    # ----------------------------------------------------------------
    # Helper to convert user messages into input for "o1-pro"
    # ----------------------------------------------------------------
    def _messages_to_input_items(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Converts a list of chat messages into the proper "input" payload for o1-pro.
        
        If all user messages contain only text and no images, then their contents
        are joined into a single string (which matches the working test using a string).
        Otherwise, returns a list of input items with type "input_text"/"image_url".
        """
        # If messages is already a string, pass it through.
        if isinstance(messages, str):
            return messages

        # Check if any message includes images.
        has_images = any("images" in msg and msg["images"] for msg in messages)
        if not has_images:
            # Join all user message contents into one string.
            texts = [
                msg.get("content", "").strip()
                for msg in messages
                if msg.get("role") == "user" and msg.get("content")
            ]
            return "\n".join(texts)

        # Otherwise, build a list of input items.
        input_items = []
        for msg in messages:
            if msg.get("role") == "user":
                text_content = msg.get("content", "").strip()
                if text_content:
                    input_items.append({"type": "input_text", "text": text_content})
                if msg.get("images"):
                    for image in msg["images"]:
                        input_items.append({
                            "type": "image_url",
                            "image_url": {"url": image}
                        })
        return input_items
