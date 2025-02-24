import os
import json
import requests
import asyncio

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

        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": True,
            "image": True,
        }
    
    def _prepare_payload(self, base_payload: dict) -> dict:
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
        return self.capabilities.get(feature, False)

    async def send_chat_streaming(self, messages: list, **kwargs):
        """
        Asynchronous streaming chat request. Returns an async generator
        that yields chunks (dicts) from OpenAI. Uses 'requests' in a worker thread.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

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

        payload = {
            "messages": conversation_history,
            "stream": True,
            **kwargs
        }
        payload = self._prepare_payload(payload)

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Synchronous generator that reads streaming lines from OpenAI.
        def sync_generator():
            resp = requests.post(url, headers=headers, json=payload, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # OpenAI streams lines like: data: {...}
                if line.strip() == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    json_str = line[len("data: "):].strip()
                    if json_str:
                        try:
                            yield json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # Convert synchronous generator -> async generator
        async def async_generator():
            loop = asyncio.get_event_loop()
            queue = asyncio.Queue()

            def run_in_thread():
                try:
                    for item in sync_generator():
                        queue.put_nowait(item)
                except Exception as e:
                    queue.put_nowait(e)
                finally:
                    queue.put_nowait(None)  # Sentinel

            fut = loop.run_in_executor(None, run_in_thread)

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

            await fut  # ensure thread completed

        return async_generator()

    async def send_chat_nonstreaming(self, messages: list, **kwargs):
        """
        Asynchronous method to get full chat completion at once using requests.
        Runs in a background thread to avoid blocking the event loop.
        Returns the JSON response from OpenAI.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

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

        payload = {
            "messages": conversation_history,
            "stream": False,
            **kwargs
        }
        payload = self._prepare_payload(payload)

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def sync_call():
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_call)