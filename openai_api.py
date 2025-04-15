import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, AsyncGenerator
import logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure all debug logs are captured

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
            if os.environ.get('openai_token'):
                self.api_key = os.environ['openai_token']
            else:
                self.api_key = os.environ.get('OPENAI_API_KEY')
        self._session = None
        self.last_response_id = None
        # Basic capabilities map; all features enabled for now.
        self.capabilities = {
            "chat": True,
            "generate": True,
            "tool": True,
            "image": True,
        }
        logger.debug(f"Initialized OpenAIInterface for model: {self.model} with api_key: {self.api_key}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Create or retrieve a reusable aiohttp session to avoid overhead on repeated calls.
        """
        if self._session is None or self._session.closed:
            logger.debug("Creating new aiohttp ClientSession")
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._session

    def is_api_key_configured(self) -> bool:
        """
        True if the OpenAI API key is set in code or environment.
        """
        configured = bool(self.api_key)
        logger.debug(f"API key configured: {configured}")
        return configured

    def _supports(self, feature: str) -> bool:
        """
        Check if this model supports a given capability (like "chat", "image", etc.).
        """
        support = self.capabilities.get(feature, False)
        logger.debug(f"Feature '{feature}' supported: {support}")
        return support

    def _uses_responses_api(self) -> bool:
        """
        Return True if we should use /v1/responses for this model (i.e. "o1-pro").
        """
        uses = self.model.startswith("o1-pro")
        logger.debug(f"Model '{self.model}' uses responses API: {uses}")
        return uses

    def extract_content_from_response(self, response: Dict[str, Any], is_chat: bool = True) -> str:
        """
        Extracts the final text from a typical ChatCompletion-like response.
        Also saves the response ID from the API.
        """
        if "id" in response:
            self.last_response_id = response["id"]
            logger.debug(f"Extracted response ID: {self.last_response_id}")
        else:
            self.last_response_id = None
            logger.debug("No response ID found in API response")
            
        content = response.get("choices", [{}])[0] \
                         .get("message", {}) \
                         .get("content", "")
        logger.debug(f"Extracted content: {content}")
        return content
    
    def extract_content_from_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Extracts text from a streaming response chunk.

        For o1-pro responses, it checks:
        - "response.created": captures the response ID.
        - "response.output_text.delta": returns the text from the "delta" field.
        - "response.output_text.done": returns the final text from the "text" field.
        Otherwise, falls back to the traditional structure.
        """
        evt_type = chunk.get("type", "")
        if evt_type == "response.created":
            # Capture and store the response id for later use.
            response_info = chunk.get("response", {})
            self.last_response_id = response_info.get("id")
            logging.debug(f"Captured response ID from 'response.created': {self.last_response_id}")
            return ""
        elif evt_type == "response.output_text.delta":
            content = chunk.get("delta", "")
            logging.debug(f"Extracted delta content from chunk: {content}")
            return content
        elif evt_type == "response.output_text.done":
            content = chunk.get("text", "")
            logging.debug(f"Extracted done content from chunk: {content}")
            return content
        else:
            # Fallback extraction from a choices list (if available)
            choices = chunk.get("choices", [])
            if choices and isinstance(choices, list):
                content = choices[0].get("delta", {}).get("content", "")
                logging.debug(f"Extracted fallback content from chunk: {content}")
                return content
            logging.debug("No content extracted from chunk")
            return ""

    async def close(self):
        """Close the aiohttp session if open."""
        if self._session and not self._session.closed:
            logger.debug("Closing aiohttp ClientSession")
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
        logger.debug(f"Listing fine-tuning jobs from: {url}")
        async with session.get(url) as response:
            response.raise_for_status()
            jobs = await response.json()
            logger.debug(f"Received fine-tuning jobs: {jobs}")
            return jobs

    def get_successful_fine_tuned_models(self, jobs: Dict[str, Any]) -> List[str]:
        """
        Filter out IDs of successfully fine-tuned models from the jobs list.
        """
        models = [
            job["fine_tuned_model"]
            for job in jobs.get("data", [])
            if job.get("status") == "succeeded" and job.get("fine_tuned_model") is not None
        ]
        logger.debug(f"Successful fine-tuned models: {models}")
        return models

    # ----------------------------------------------------------------
    # Utility for older "o3-mini" or special model preparations.
    # ----------------------------------------------------------------
    def _prepare_payload(self, base_payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.model in ["o3-mini-low", "o3-mini-medium", "o3-mini-high"]:
            reasoning_effort = self.model.split("-")[-1]
            base_payload["model"] = "o3-mini"
            base_payload["reasoning_effort"] = reasoning_effort
            logger.debug(f"Prepared payload for o3-mini with reasoning_effort: {reasoning_effort}")
        elif self.model in ["o1-pro-low", "o1-pro", "o1-pro-high"]:
            reasoning_effort = self.model.split("-")[-1]
            base_payload["model"] = "o1-pro"
            base_payload["reasoning"] = {"effort": reasoning_effort}
            logger.debug(f"Prepared payload for o1-pro with reasoning effort: {reasoning_effort}")
        else:
            base_payload["model"] = self.model
            logger.debug(f"Using model as is in payload: {self.model}")
        return base_payload

    # ----------------------------------------------------------------
    # Chat Completions Endpoint
    # ----------------------------------------------------------------
    async def send_chat_streaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if self._uses_responses_api():
            logger.debug("Using responses API (streaming) for this request")
            async for chunk in self.send_response_streaming(
                messages, timeout_threshold=timeout_threshold, **kwargs
            ):
                yield chunk
            return

        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        session = await self._get_session()
        url = "https://api.openai.com/v1/chat/completions"
        payload = {"messages": messages, "stream": True, **kwargs}
        payload = self._prepare_payload(payload)
        
        logger.debug("Sending streaming chat request with payload:")
        logger.debug(json.dumps(payload, indent=2))
        
        try:
            response = await session.post(url, json=payload)
            if response.status >= 400:
                error_body = await response.text()
                logger.error(f"Chat streaming returned {response.status} => {error_body}")
            response.raise_for_status()

            while True:
                try:
                    line = await asyncio.wait_for(response.content.readline(), timeout=timeout_threshold)
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for a streaming chunk")
                    raise ConnectionError(
                        f"No data from /v1/chat/completions in {timeout_threshold} seconds (mid-stream)."
                    )

                if not line:
                    logger.debug("No more data from stream; ending read loop")
                    break

                line_decoded = line.decode('utf-8').strip()
                if not line_decoded:
                    continue

                if line_decoded.startswith("data: "):
                    data = line_decoded[len("data: "):]
                    if data == "[DONE]":
                        logger.debug("Received [DONE] signal; terminating stream")
                        break
                    try:
                        chunk = json.loads(data)
                        logger.debug(f"Received streaming chunk: {chunk}")
                        yield chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON chunk: {data}")
                        continue

        except aiohttp.ClientResponseError as e:
            logger.exception("Chat streaming request encountered ClientResponseError:")
            try:
                error_detail = await response.text()
            except Exception:
                error_detail = "No detail."
            logger.error(f"Status: {e.status}, message='{e.message}', detail='{error_detail}'\n{traceback.format_exc()}")
            raise
        except Exception as e:
            logger.exception("Unexpected error in send_chat_streaming:")
            raise

    async def send_chat_nonstreaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        if self._uses_responses_api():
            logger.debug("Using responses API (non-streaming) for this request")
            return await self.send_response_nonstreaming(
                messages, timeout_threshold=timeout_threshold, **kwargs
            )

        if not self._supports("chat"):
            raise ValueError(f"Model '{self.model}' does not support chat requests.")

        session = await self._get_session()
        url = "https://api.openai.com/v1/chat/completions"
        payload = {"messages": messages, "stream": False, **kwargs}
        payload = self._prepare_payload(payload)
        
        logger.debug("Sending non-streaming chat request with payload:")
        logger.debug(json.dumps(payload, indent=2))
        
        try:
            response = await session.post(url, json=payload)
            if response.status >= 400:
                err_text = await response.text()
                logger.error(f"Non-streaming chat returned {response.status} => {err_text}")
            response.raise_for_status()

            try:
                text_body = await asyncio.wait_for(response.text(), timeout=timeout_threshold)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for non-streaming chat response")
                raise ConnectionError(
                    f"No data from /v1/chat/completions in {timeout_threshold} seconds (non-streaming)."
                )

            try:
                result = json.loads(text_body)
                logger.debug(f"Non-streaming chat response JSON: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Malformed JSON response: {text_body}")
                return {"error": "Malformed JSON from ChatCompletions.", "raw": text_body}

        except aiohttp.ClientResponseError as e:
            logger.exception("Non-streaming chat request failed with ClientResponseError:")
            try:
                error_detail = await response.text()
            except Exception:
                error_detail = "No detail."
            logger.error(f"Status: {e.status}, message='{e.message}', detail='{error_detail}'\n{traceback.format_exc()}")
            raise
        except Exception as e:
            logger.exception("Unexpected error in send_chat_nonstreaming:")
            raise

    # ----------------------------------------------------------------
    # "Responses" endpoint for o1-pro
    # ----------------------------------------------------------------
    async def send_response_streaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        previous_response_id: str = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        session = await self._get_session()
        url = "https://api.openai.com/v1/responses"
        input_payload = self._messages_to_input_items(messages)
        payload = {
            "model": self.model,
            "input": input_payload,
            "stream": True,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
            logger.debug(f"Including previous_response_id in streaming payload: {previous_response_id}")
        else:
            logger.debug("No previous_response_id provided for streaming request")
        payload.update(kwargs)
        payload = self._prepare_payload(payload)

        logger.debug("Sending streaming 'responses' request with payload:")
        logger.debug(json.dumps(payload, indent=2))
        
        try:
            response = await session.post(url, json=payload)
            if response.status >= 400:
                error_body = await response.text()
                logger.error(f"Responses streaming returned {response.status} => {error_body}")
            response.raise_for_status()

            while True:
                try:
                    line = await asyncio.wait_for(response.content.readline(), timeout=timeout_threshold)
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for SSE chunk in responses streaming")
                    raise ConnectionError(
                        f"No data from /v1/responses in {timeout_threshold} seconds (mid-stream)."
                    )

                if not line:
                    logger.debug("No more SSE lines; ending streaming")
                    break

                line_decoded = line.decode("utf-8", errors="replace").strip()
                if not line_decoded or not line_decoded.startswith("data: "):
                    continue

                data_str = line_decoded[len("data: "):].strip()
                if not data_str:
                    continue

                try:
                    chunk = json.loads(data_str)
                    logger.debug(f"Received SSE chunk: {chunk}")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode SSE chunk: {data_str}")
                    continue

                yield chunk
                if chunk.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                    logger.debug("Received terminal chunk in SSE; ending stream")
                    break

        except aiohttp.ClientResponseError as e:
            logger.exception("Responses streaming request failed with ClientResponseError:")
            try:
                error_detail = await response.text()
            except Exception:
                error_detail = "No detail."
            logger.error(f"Status: {e.status}, message='{e.message}', detail='{error_detail}'\n{traceback.format_exc()}")
            raise
        except Exception as e:
            logger.exception("Unexpected error in send_response_streaming:")
            raise

    async def send_response_nonstreaming(
        self,
        messages: List[Dict[str, Any]],
        timeout_threshold: float = 30.0,
        previous_response_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        session = await self._get_session()
        url = "https://api.openai.com/v1/responses"
        input_payload = self._messages_to_input_items(messages)
        payload = {
            "model": self.model,
            "input": input_payload,
            "stream": False
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
            logger.debug(f"Including previous_response_id in non-streaming payload: {previous_response_id}")
        else:
            logger.debug("No previous_response_id provided for non-streaming request")
        payload.update(kwargs)
        payload = self._prepare_payload(payload)

        logger.debug("Sending non-streaming 'responses' request with payload:")
        logger.debug(json.dumps(payload, indent=2))
        
        try:
            response = await session.post(url, json=payload)
            if response.status >= 400:
                error_body = await response.text()
                logger.error(f"Non-streaming responses returned {response.status} => {error_body}")
            response.raise_for_status()

            try:
                text_body = await asyncio.wait_for(response.text(), timeout=timeout_threshold)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for non-streaming responses")
                raise ConnectionError(
                    f"No data from /v1/responses in {timeout_threshold} seconds (non-streaming)."
                )

            try:
                result = json.loads(text_body)
                logger.debug(f"Non-streaming responses JSON: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Malformed JSON in non-streaming responses: {text_body}")
                return {"error": "Malformed JSON from Responses.", "raw": text_body}

        except aiohttp.ClientResponseError as e:
            logger.exception("Non-streaming responses request failed with ClientResponseError:")
            try:
                error_detail = await response.text()
            except Exception:
                error_detail = "No detail."
            logger.error(f"Status: {e.status}, message='{e.message}', detail='{error_detail}'\n{traceback.format_exc()}")
            raise
        except Exception as e:
            logger.exception("Unexpected error in send_response_nonstreaming:")
            raise

    # ----------------------------------------------------------------
    # Helper: Convert messages to input items for o1-pro
    # ----------------------------------------------------------------
    def _messages_to_input_items(self, messages: List[Dict[str, Any]]) -> Any:
        if isinstance(messages, str):
            logger.debug("Messages provided as a raw string")
            return messages

        latest_user_message = next(
            (msg for msg in reversed(messages) if msg.get("role") == "user" and msg.get("content")),
            None
        )
        
        if latest_user_message is None:
            logger.debug("No user message found in conversation messages")
            return ""
        
        if latest_user_message.get("images"):
            logger.debug("Latest user message contains images; preparing input items")
            input_items = []
            text_content = latest_user_message.get("content", "").strip()
            if text_content:
                input_items.append({"type": "input_text", "text": text_content})
                logger.debug(f"Added text input: {text_content}")
            for image in latest_user_message.get("images"):
                input_items.append({
                    "type": "image_url",
                    "image_url": {"url": image}
                })
                logger.debug(f"Added image input: {image}")
            return input_items

        content = latest_user_message.get("content", "").strip()
        logger.debug(f"Using text from latest user message: {content}")
        return content
