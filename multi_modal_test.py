import asyncio
from async_ollama_interface import AsyncOllamaInterface
import base64

async def main():
    ollama_interface = AsyncOllamaInterface(model="llava")

    print("\n---- Vision Example ----")
    try:
        with open("image_test.jpg", "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        vision_response = await ollama_interface.send_vision(
            prompt="What is in this picture?",
            images=[image_base64]
        )
        # The response is a dict/ChatResponse with 'message': 'content'
        print("Vision response:", vision_response['response'])
    except Exception as e:
        print("Error during vision request:", e)

if __name__ == '__main__':
    asyncio.run(main())