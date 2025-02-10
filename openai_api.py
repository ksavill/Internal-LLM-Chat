import openai
import os

class OpenAIInterface:
    def __init__(self, model: str, api_key: str = None):
        """
        model: e.g. "gpt-4o-mini" or another available model name
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

    def _supports(self, feature: str) -> bool:
        return self.capabilities.get(feature, False)

    def send_chat_streaming(self, messages: list, **kwargs):
        """
        Synchronous generator-based streaming chat request.
        Yields chunks (dicts) from OpenAI one by one.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        # "stream=True" returns a generator that yields partial completions
        response_stream = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        # `response_stream` is an iterable, so yield each chunk
        for chunk in response_stream:
            # print("DEBUG chunk: ", chunk)
            yield chunk

    def send_chat_nonstreaming(self, messages: list, **kwargs):
        """
        Returns the entire Chat response at once (blocking).
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        # "stream=False" (the default) returns the entire response
        completion = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **kwargs
        )
        return completion
