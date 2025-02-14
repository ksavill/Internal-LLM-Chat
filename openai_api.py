import openai
import os

class OpenAIInterface:
    def __init__(self, model: str, api_key: str = None):
        """
        model: e.g. "gpt-4o-mini" or another available model name.
               For models "o3-mini-low", "o3-mini-medium", or "o3-mini-high",
               special payload modifications will be applied.
        api_key: if not provided, we look for an environment variable.
        """
        self.model = model

        # Set the API key explicitly if provided, else read from env:
        if api_key:
            openai.api_key = api_key
        else:
            if os.environ.get('openai_token'):
                openai.api_key = os.environ['openai_token']
            else:
                openai.api_key = os.environ.get('OPENAI_API_KEY')

        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": True,
            "image": True,
        }
    
    def _prepare_payload(self, base_payload: dict) -> dict:
        """
        If the model is one of: o3-mini-low, o3-mini-medium, or o3-mini-high,
        then change the model in the payload to "o3-mini" and add a 'reasoning_effort'
        parameter with the corresponding suffix value.
        """
        if self.model in ["o3-mini-low", "o3-mini-medium", "o3-mini-high"]:
            # Extract the reasoning effort from the model name (low, medium, high)
            reasoning_effort = self.model.split("-")[-1]
            base_payload["model"] = "o3-mini"
            base_payload["reasoning_effort"] = reasoning_effort
        else:
            base_payload["model"] = self.model
        return base_payload

    def is_api_key_configured(self) -> bool:
        """
        Returns True if the OpenAI API Key has been set successfully.
        """
        return bool(openai.api_key)

    def _supports(self, feature: str) -> bool:
        return self.capabilities.get(feature, False)

    def send_chat_streaming(self, messages: list, **kwargs):
        """
        Synchronous generator-based streaming chat request.
        Yields chunks (dicts) from OpenAI one by one.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        # Process images if provided.
        images = kwargs.pop("images", None)
        conversation_history = list(messages)
        if images:
            # Normalize images to a list if needed.
            if isinstance(images, str):
                images = [images]
            for image in images:
                conversation_history.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image}}]
                })

        # Build the payload
        payload = {
            "messages": conversation_history,
            "stream": True,
            **kwargs
        }
        payload = self._prepare_payload(payload)

        response_stream = openai.chat.completions.create(**payload)
        # Yield each chunk from the response stream.
        for chunk in response_stream:
            yield chunk

    def send_chat_nonstreaming(self, messages: list, **kwargs):
        """
        Returns the entire Chat response at once (blocking).
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        # Process images if provided.
        images = kwargs.pop("images", None)
        conversation_history = list(messages)
        if images:
            # Normalize images to a list if needed.
            if isinstance(images, str):
                images = [images]
            for image in images:
                conversation_history.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image}}]
                })

        # Build the payload
        payload = {
            "messages": conversation_history,
            "stream": False,
            **kwargs
        }
        payload = self._prepare_payload(payload)

        completion = openai.chat.completions.create(**payload)
        return completion
