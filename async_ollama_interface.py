from ollama import AsyncClient

class AsyncOllamaInterface:
    def __init__(self, model: str, client: AsyncClient = None):
        """
        model: name of the model, e.g. 'llava:latest' or 'llama3.2'
        client: optional custom AsyncClient instance
        """
        self.model = model
        self.client = client or AsyncClient()

        # Only "llava:latest" supports image inputs.
        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": False,
            "image": True if self.model.lower() == "llava:latest" else False,
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

    async def send_vision(self, prompt: str, images: list, **kwargs):
        """
        Multimodal text+image input (e.g., for LLaVA).
        'images' should be a list of base64-encoded image strings.
        """
        if not self._supports("image"):
            raise ValueError(f"Model '{self.model}' does not support image generation.")
        return await self.client.generate(
            model=self.model,
            prompt=prompt,
            images=images,
            **kwargs
        )