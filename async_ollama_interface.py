# async_ollama_interface.py
from ollama import AsyncClient

class AsyncOllamaInterface:
    def __init__(self, model: str, client: AsyncClient = None):
        """
        model: name of the model, e.g. 'llava' or 'llama3.2'
        client: optional custom AsyncClient instance
        """
        self.model = model
        self.client = client or AsyncClient()

        # Define your model's capabilities.
        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": False,
            "image": True,   # set to True for LLaVA or other multimodal models
        }

    def _supports(self, feature: str) -> bool:
        return self.capabilities.get(feature, False)

    async def send_chat_streaming(self, messages: list, **kwargs):
        """
        Implements a streaming chat request.
        Returns an async generator which yields chunked responses.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        return await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,  # Force streaming on
            **kwargs
        )

    async def send_chat_nonstreaming(self, messages: list, **kwargs):
        """
        Implements a non-streaming chat request.
        Returns a single ChatResponse-like object or dict.
        """
        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        return await self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,  # Disable streaming
            **kwargs
        )

    """
    Not implemented at this time.
    """
    # async def send_tool(self, tool_name: str, **kwargs):
    #     """
    #     (Optional) Simulate or implement a "tool" request.
    #     For example, you could use the 'generate' endpoint with a special prompt.
    #     """
    #     if not self._supports("tool"):
    #         raise ValueError(f"Model '{self.model}' does not support tool requests.")

    #     prompt = f"[tool:{tool_name}] {kwargs.pop('input', '')}"
    #     return await self.client.generate(model=self.model, prompt=prompt, **kwargs)

    async def send_vision(self, prompt: str, images: list, **kwargs):
        """
        (Optional) Multimodal text+image input (e.g., for LLaVA).
        'images' should be a list of base64-encoded image strings.
        """
        if not self._supports("image"):
            raise ValueError(f"Model '{self.model}' does not support image generation.")

        # The Ollama generate endpoint can accept an 'images' parameter for LLaVA models.
        return await self.client.generate(
            model=self.model,
            prompt=prompt,
            images=images,
            **kwargs
        )