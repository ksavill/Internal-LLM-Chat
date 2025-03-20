import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, AsyncGenerator


class OpenAIInterface:
    def __init__(self, model: str, api_key: str = None):
        """
        model: e.g. "gpt-4o-mini" or another available model name.
        api_key: if not provided, we look for an environment variable.
        """
        self.model = model

        # Set the API key explicitly if provided; else read from env:
        if api_key:
            self.api_key = api_key
        else:
            if os.environ.get('openai_token'):
                self.api_key = os.environ['openai_token']
            else:
                self.api_key = os.environ.get('OPENAI_API_KEY')

        # Session will be created on first use (lazy initialization)
        self._session = None

        # Models capabilities
        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": True,
            "image": True,
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with no built-in timeouts."""
        if self._session is None or self._session.closed:
            # We rely on manual chunk reading timeouts below
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._session

    def _prepare_payload(self, base_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        If the model is one of: o3-mini-low, o3-mini-medium, o3-mini-high,
        change model to "o3-mini" and add 'reasoning_effort' with suffix value.
        """
        if self.model in ["o3-mini-low", "o3-mini-medium", "o3-mini-high"]:
            reasoning_effort = self.model.split("-")[-1]
            base_payload["model"] = "o3-mini"
            base_payload["reasoning_effort"] = reasoning_effort
        else:
            base_payload["model"] = self.model
        return base_payload

    def is_api_key_configured(self) -> bool:
        """True if the OpenAI API key is set."""
        return bool(self.api_key)

    def _supports(self, feature: str) -> bool:
        """Check if model supports a specific feature."""
        return self.capabilities.get(feature, False)

    def extract_content_from_response(self, response: Dict[str, Any], is_chat: bool = True) -> str:
        """
        Extract content from a chat response.
        - Format: response["choices"][0]["message"]["content"]
        """
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")

    def extract_content_from_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Extract content from a streaming chunk.
        - Format: chunk["choices"][0]["delta"]["content"]
        """
        return chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")

    async def send_chat_streaming(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Fully asynchronous streaming chat request with a *manual read* timeout.
        
        We do two things here:
        1. POST the request (no built-in connect/read timeouts).
        2. Read each chunk with `asyncio.wait_for` so that if no chunk arrives
           for 5s, we raise an exception (e.g. user disconnected mid-stream).
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        # Process optional image parameters
        images = kwargs.pop("images", None)
        conversation_history = list(messages)
        if images:
            if isinstance(images, str):
                images = [images]
            for image in images:
                conversation_history.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image}}]
                })

        # Prepare the request payload
        payload = {"messages": conversation_history, "stream": True, **kwargs}
        payload = self._prepare_payload(payload)

        session = await self._get_session()
        url = "https://api.openai.com/v1/chat/completions"

        response = await session.post(url, json=payload)
        response.raise_for_status()

        # Manually read lines from the response stream
        while True:
            # If we don't receive any data within 5 seconds, raise an exception
            try:
                line = await asyncio.wait_for(response.content.readline(), timeout=5.0)
            except asyncio.TimeoutError:
                raise ConnectionError("No data from OpenAI in 5 seconds (mid-stream).")

            if not line:
                # Stream ended
                break

            line = line.decode('utf-8').strip()
            if not line:
                continue

            if line.startswith("data: "):
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    pass

    async def send_chat_nonstreaming(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Non-streaming chat request with a *manual read* approach.
        
        We do a POST to get the response object, then read it in one go with
        `asyncio.wait_for` so that if no data arrives for 5s, we raise.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        # Process optional image parameters
        images = kwargs.pop("images", None)
        conversation_history = list(messages)
        if images:
            if isinstance(images, str):
                images = [images]
            for image in images:
                conversation_history.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image}}]
                })

        # Prepare the request payload
        payload = {"messages": conversation_history, "stream": False, **kwargs}
        payload = self._prepare_payload(payload)

        session = await self._get_session()
        url = "https://api.openai.com/v1/chat/completions"

        response = await session.post(url, json=payload)
        response.raise_for_status()

        try:
            text_body = await asyncio.wait_for(response.text(), timeout=5.0)
        except asyncio.TimeoutError:
            raise ConnectionError("No data from OpenAI in 5 seconds (non-streaming).")

        try:
            return json.loads(text_body)
        except json.JSONDecodeError:
            return {"error": "Malformed JSON from OpenAI.", "raw": text_body}

    async def close(self):
        """Close the aiohttp session when done."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def list_fine_tuning_jobs(self) -> Dict[str, Any]:
        """
        Retrieve a list of fine-tuning jobs from the OpenAI API.
        
        Returns:
            A dictionary containing the list of fine-tuning jobs as returned by the API.
        """
        session = await self._get_session()
        url = "https://api.openai.com/v1/fine_tuning/jobs"
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise

    def get_successful_fine_tuned_models(self, jobs: Dict[str, Any]) -> List[str]:
        """
        Extract IDs of successfully fine-tuned models from the jobs list.
        
        Args:
            jobs: The JSON response from list_fine_tuning_jobs.
        
        Returns:
            A list of fine-tuned model IDs (e.g., "ft:gpt-3.5-turbo:org:abc123").
        """
        return [
            job["fine_tuned_model"]
            for job in jobs.get("data", [])
            if job.get("status") == "succeeded" and job.get("fine_tuned_model") is not None
        ]
