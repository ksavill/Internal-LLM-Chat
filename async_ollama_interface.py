# async_ollama_interface.py
import ollama
from ollama import AsyncClient # Explicit import for clarity
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple

from copy import deepcopy

import logging
import traceback

logger = logging.getLogger(__name__)
# Ensure logger level is set appropriately in your main application if it's not already
# For debugging this module specifically, you can set it here:
# logger.setLevel(logging.DEBUG) 

def _data_url_to_pure_base64(data_url: str) -> Optional[str]:
    """Converts a data URL (e.g., 'data:image/jpeg;base64,XXXX') to a pure base64 string ('XXXX')."""
    try:
        if not data_url.startswith("data:"):
            logger.warning(f"Invalid data URL format (missing 'data:' prefix): {data_url[:60]}")
            return None
        header, encoded = data_url.split(",", 1)
        if ";base64" not in header:
            # This could be a URL-encoded image or other format not directly usable as base64
            logger.warning(f"Data URL does not seem to be base64 encoded: {header}")
            return None # Or handle other encodings if necessary
        return encoded
    except ValueError:
        logger.warning(f"Malformed data URL (could not split header/encoded): {data_url[:60]}")
        return None
    except Exception as e:
        logger.error(f"Error processing data URL '{data_url[:60]}': {e}")
        return None

def _reformat_messages_for_ollama(messages_openai_format: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms messages from OpenAI's multimodal format to Ollama's expected format.
    OpenAI format example: 
        [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}, {'type': 'image_url', 'image_url': {'url': 'data:...'}}]}]
    Ollama format example: 
        [{'role': 'user', 'content': 'Hi', 'images': ['base64_string1']}]
    """
    reformatted_messages: List[Dict[str, Any]] = []

    for msg_openai in messages_openai_format:
        # Start with a *shallow copy* so that any additional keys (e.g.
        # `name`, `tool_call_id`, `tool_calls`, etc.) survive the
        # transformation untouched.
        ollama_msg: Dict[str, Any] = dict(msg_openai)  # shallow copy first

        # ------------------------------------------------------------------
        # 1) Tool-call argument normalisation (OpenAI → Ollama)
        #    Ollama expects ``function.arguments`` **as a dict** whereas the
        #    OpenAI-spec we send back to the client uses a *string* that
        #    contains JSON.  When we forward previous conversation history
        #    (which now contains that string form) we must convert it back to
        #    a real dict so the underlying Pydantic validation inside
        #    `ollama-python` accepts the message.
        # ------------------------------------------------------------------

        try:
            tool_calls = ollama_msg.get("tool_calls")
            if isinstance(tool_calls, list):
                import json as _json
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    func = tc.get("function")
                    if isinstance(func, dict):
                        args_val = func.get("arguments")
                        if isinstance(args_val, str):
                            try:
                                func["arguments"] = _json.loads(args_val)
                                logger.debug(
                                    "Converted tool_call arguments string to dict for function '%s'",
                                    func.get("name"),
                                )
                            except Exception:
                                logger.warning(
                                    "Failed to parse tool_call arguments string for '%s'. Replacing with empty dict to satisfy validation.",
                                    func.get("name"),
                                )
                                func["arguments"] = {}
        except Exception as _exc:
            logger.debug("Failed to normalise tool_call arguments for Ollama: %s", _exc)

        role = msg_openai.get("role")
        ollama_msg["role"] = role  # Ensure role field exists/overrides

        openai_content_parts = msg_openai.get("content")

        final_text_content_parts: List[str] = []
        final_image_base64_parts: List[str] = []

        if isinstance(openai_content_parts, str):
            # Text-only message – no conversion required.
            final_text_content_parts.append(openai_content_parts)
        elif isinstance(openai_content_parts, list):
            # Multimodal: split into text and images.
            for part in openai_content_parts:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    text_val = part.get("text")
                    if isinstance(text_val, str):
                        final_text_content_parts.append(text_val)
                elif part_type == "image_url":
                    image_url_dict = part.get("image_url")
                    if isinstance(image_url_dict, dict):
                        data_url = image_url_dict.get("url")
                        if isinstance(data_url, str):
                            pure_base64 = _data_url_to_pure_base64(data_url)
                            if pure_base64:
                                final_image_base64_parts.append(pure_base64)
        else:
            logger.warning(
                "Unexpected content type (%s) in message for Ollama. Defaulting to empty content.",
                type(openai_content_parts),
            )

        # Compose the transformed content.
        ollama_msg["content"] = "\n".join(final_text_content_parts).strip()

        # Ollama requires non-empty content when images are present.
        if not ollama_msg["content"] and final_image_base64_parts:
            ollama_msg["content"] = " "

        if final_image_base64_parts:
            ollama_msg["images"] = final_image_base64_parts

        # Remove the OpenAI-style content parts list to avoid confusion if it was
        # present – it has been converted above.
        # Remove helper keys not used by Ollama.

        reformatted_messages.append(ollama_msg)

    return reformatted_messages

# ------------------------------------------------------------
# Helper: Extract and remove a system prompt from conversation
# ------------------------------------------------------------

def _pop_system_prompt(
    messages: List[Dict[str, Any]]
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Return *(system_prompt, messages_without_system)*.

    The function searches for the **first** message whose ``role`` equals
    ``"system"`` (case-sensitive to match the ChatML spec).  If found, that
    message's *content* string is returned as the *system_prompt* and the
    message is **removed** from the list that will ultimately be forwarded to
    Ollama.  If there is no such message, ``system_prompt`` is ``None`` and
    the input list is passed through unchanged.

    A *shallow* copy of each message dict is used so that the caller's
    original data is left untouched.
    """

    system_prompt: Optional[str] = None
    remaining: List[Dict[str, Any]] = []

    for msg in messages:
        if system_prompt is None and msg.get("role") == "system":
            # Capture the first system prompt we encounter.
            # Guard against non-string content just in case.
            content_val = msg.get("content")
            if isinstance(content_val, str):
                system_prompt = content_val
            else:
                try:
                    # Attempt best-effort serialisation for e.g. multimodal
                    # system prompts (rare).
                    import json as _json
                    system_prompt = _json.dumps(content_val, default=str)
                except Exception:
                    system_prompt = str(content_val)
            # Skip adding this message to remaining list (i.e. remove it).
            continue

        # Keep all other messages.
        remaining.append(deepcopy(msg))

    return system_prompt, remaining



class AsyncOllamaInterface:
    def __init__(self, model: str, client: Optional[AsyncClient] = None): # Added Optional type hint
        """
        Args:
            model: name of the model, e.g. 'llava:latest' or 'llama2-7b'
            client: optional custom AsyncClient instance
        """
        self.model = model
        self.client = client or AsyncClient()

        vision_models = ['llava', 'bakllava', 'moondream', 'cogvlm'] # Keep this list updated
        # Capability flags are a quick *hint* for upstream logic.  Ollama now
        # supports function/tool calling, therefore mark it as available here.
        self.capabilities = {
            "chat": True,
            "generate": True,  # Still useful for direct generation endpoint.
            "tool": True,
            "image": any(vm in self.model.lower() for vm in vision_models),
        }

        logger.debug(f"Initializing AsyncOllamaInterface with model: {self.model}")
        logger.debug(
            f"Capabilities: chat={self.capabilities['chat']}, generate={self.capabilities['generate']}, "
            f"tool={self.capabilities['tool']}, image={self.capabilities['image']}"
        )

    def _supports(self, feature: str) -> bool:
        """Check if the model supports *feature* based on the capability map."""

        # For *tool* usage we enable support when the caller actually supplies a
        # ``tools`` list.  This avoids advertising tool-calling in capability
        # probes when it is not relevant while still allowing full
        # functionality if requested.

        if feature == "tool":
            return True  # Ollama >=0.1.34 supports tool calling when a tools list is provided.

        support = self.capabilities.get(feature, False)
        logger.debug(f"Feature support check for '{feature}': {support}")
        return support

    def extract_content_from_response(self, response: Dict[str, Any], is_chat: bool = True) -> str:
        """
        Extract content from a non-streaming response.
        """
        if "error" in response and response["error"]:
            logger.warning(f"Attempting to extract content from an error response: {response['error']}")
            return "" # Return empty or specific error string

        if is_chat:
            content = response.get("message", {}).get("content", "")
            logger.debug(f"Extracted chat content from response: '{content[:100]}...'")
        else: # Corresponds to 'generate' endpoint if used directly
            content = response.get("response", "") # 'generate' puts content in 'response'
            logger.debug(f"Extracted generate content from response: '{content[:100]}...'")
        return content

    def extract_content_from_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Extract content from a streaming chunk.
        Handles potential error chunks gracefully.
        """
        if "error" in chunk and chunk["error"]:
            # The server.py stream handler should ideally catch this before calling.
            # If called, we return an empty string or the error message itself.
            logger.warning(f"Extracting content from an error chunk: {chunk['error']}")
            return "" 
            # Or, consider: return f"[Error in chunk: {chunk['error']}]"
        
        content = chunk.get("message", {}).get("content", "")
        # logger.debug(f"Extracted content from chunk: {content}") # Can be very verbose
        return content

    async def send_chat_streaming(
        self, 
        messages_openai_format: List[Dict[str, Any]], 
        timeout_threshold: float = 30.0, # Default from your openai_api.py
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming chat request with Ollama. Transforms messages to Ollama format.
        Yields each chunk as it is received.
        """
        if not self._supports("chat"):
            error_msg = f"Model '{self.model}' does not support chat requests based on capabilities."
            logger.error(error_msg)
            # Yield an error chunk instead of raising, to match server.py's expectation for generators
            yield {"error": error_msg, "message": {"content": f"\n\n[Interface Error: {error_msg}]"}}
            return

        ollama_formatted_messages = _reformat_messages_for_ollama(messages_openai_format)

        # Extract a system prompt (if present) and remove it from the message
        # list.  This provides compatibility with older Ollama releases that
        # ignore system-role messages but honour the explicit *system*
        # parameter.  For newer versions this is also harmless – they will
        # apply the prompt once and we avoid sending a duplicate.

        system_prompt_arg, ollama_formatted_messages = _pop_system_prompt(ollama_formatted_messages)

        logger.debug(
            "[Ollama] Initiating streaming chat for model: %s (system_prompt=%s)",
            self.model,
            (system_prompt_arg[:60] + "…") if system_prompt_arg else None,
        )
        logger.debug(f"[Ollama] Reformatted Ollama messages: {ollama_formatted_messages}")

        # The ollama library's AsyncClient.chat itself doesn't take a direct timeout_threshold.
        # Timeout for the HTTP request is usually part of client construction or options.
        # We can pass 'options' if needed by specific models, but base timeout is on client.
        # The `timeout_threshold` here is more for the readline in OpenAIInterface, not directly applicable
        # in the same way to `ollama-python` which handles its own async iteration.
        
        # Prepare options for the ollama client call if any are relevant from kwargs
        ollama_options = kwargs.get("options", {})
        # If timeout_threshold from server.py logic is meant for request, add it to options:
        # if timeout_threshold: ollama_options['request_timeout'] = timeout_threshold # Check ollama lib for exact option name if needed

        try:
            chat_kwargs = {
                "model": self.model,
                "messages": ollama_formatted_messages,
                "stream": True,
            }

            if ollama_options:
                chat_kwargs["options"] = ollama_options

            if system_prompt_arg:
                chat_kwargs["system"] = system_prompt_arg

            # Forward any additional kwargs except those already handled.
            for k, v in kwargs.items():
                if k not in {"options", "timeout_threshold"}:
                    chat_kwargs[k] = v

            try:
                stream_generator = await self.client.chat(**chat_kwargs)
            except TypeError as exc:
                # Fallback for older ollama-python that doesn't accept the system param.
                if (
                    system_prompt_arg
                    and "unexpected keyword argument 'system'" in str(exc)
                ):
                    logger.debug("'system' param unsupported, falling back to message prepend (streaming)")
                    chat_kwargs.pop("system", None)
                    chat_kwargs["messages"] = [
                        {"role": "system", "content": system_prompt_arg}
                    ] + chat_kwargs["messages"]
                    stream_generator = await self.client.chat(**chat_kwargs)
                else:
                    raise
            logger.debug(f"[Ollama] Obtained streaming generator for model: {self.model}")
            async for chunk in stream_generator:
                # logger.debug(f"[Ollama] Received streaming chunk: {chunk}") # Very verbose
                yield chunk
        except ollama.ResponseError as e: # Catch specific ollama library errors
            logger.error(f"[Ollama] API ResponseError during streaming for '{self.model}': {e.status_code} - {e.error}")
            yield {"error": f"Ollama API Error: {e.status_code} - {e.error}", "message": {"content": f"\n\n[Ollama Error: {e.error}]"}}
        except Exception as e:
            logger.error(f"[Ollama] General streaming chat error for model '{self.model}': {type(e).__name__} - {e}")
            logger.debug(traceback.format_exc())
            yield {"error": str(e), "message": {"content": f"\n\n[Ollama Error: {str(e)}]"}}

    async def send_chat_nonstreaming(
        self, 
        messages_openai_format: List[Dict[str, Any]], 
        timeout_threshold: float = 30.0, # Consistent with streaming, though less directly used by ollama lib here
        **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming chat request for Ollama. Transforms messages to Ollama format.
        """
        if not self._supports("chat"):
            error_msg = f"Model '{self.model}' does not support chat requests based on capabilities."
            logger.error(error_msg)
            return {
                "error": error_msg,
                "message": {"content": f"[Interface Error: {error_msg}]"}
            }

        ollama_formatted_messages = _reformat_messages_for_ollama(messages_openai_format)

        # Extract optional system prompt and strip from message history.
        system_prompt_arg, ollama_formatted_messages = _pop_system_prompt(
            ollama_formatted_messages
        )

        logger.debug(
            "[Ollama] Initiating non-streaming chat for model: %s (system_prompt=%s)",
            self.model,
            (system_prompt_arg[:60] + "…") if system_prompt_arg else None,
        )
        logger.debug(f"[Ollama] Reformatted Ollama messages: {ollama_formatted_messages}")

        ollama_options = kwargs.get("options", {})
        # if timeout_threshold: ollama_options['request_timeout'] = timeout_threshold

        try:
            chat_kwargs = {
                "model": self.model,
                "messages": ollama_formatted_messages,
                "stream": False,
            }

            if ollama_options:
                chat_kwargs["options"] = ollama_options

            if system_prompt_arg:
                chat_kwargs["system"] = system_prompt_arg

            for k, v in kwargs.items():
                if k not in {"options", "timeout_threshold"}:
                    chat_kwargs[k] = v

            try:
                response = await self.client.chat(**chat_kwargs)
            except TypeError as exc:
                if (
                    system_prompt_arg
                    and "unexpected keyword argument 'system'" in str(exc)
                ):
                    logger.debug("'system' param unsupported, falling back to message prepend (non-streaming)")
                    chat_kwargs.pop("system", None)
                    chat_kwargs["messages"] = [
                        {"role": "system", "content": system_prompt_arg}
                    ] + chat_kwargs["messages"]
                    response = await self.client.chat(**chat_kwargs)
                else:
                    raise
            logger.debug(f"[Ollama] Received non-streaming response: {response}")
            return response
        except ollama.ResponseError as e:
            logger.error(f"[Ollama] API ResponseError during non-streaming for '{self.model}': {e.status_code} - {e.error}")
            return {
                "error": f"Ollama API Error: {e.status_code} - {e.error}",
                "message": {"content": f"[Ollama Error: {e.error}]"}
            }
        except Exception as e:
            logger.error(f"[Ollama] Non-streaming chat error for model '{self.model}': {type(e).__name__} - {e}")
            logger.debug(traceback.format_exc())
            return {
                "error": str(e),
                "message": {"content": f"[Ollama Error: {str(e)}]"}
            }

    # Removed send_vision method as its functionality is now integrated into send_chat_streaming/nonstreaming
    # via the multimodal message format and _reformat_messages_for_ollama.

    async def close(self):
        """Close the underlying aiohttp session used by ollama.AsyncClient."""
        if hasattr(self.client, '_client') and self.client._client is not None: # Access internal client if needed
            # The ollama.AsyncClient manages its own session.
            # Its __aenter__ and __aexit__ handle session creation/closing.
            # Explicit close might be needed if not used in an async with block.
            # The `ollama.AsyncClient` itself has an `aclose` method in recent versions (or it uses a session that does)
            if hasattr(self.client, 'aclose') and callable(self.client.aclose):
                 await self.client.aclose()
                 logger.info(f"Ollama async client for model {self.model} gracefully closed via aclose().")
            else:
                 logger.warning(f"Ollama async client for model {self.model} does not have a direct aclose method. Relies on garbage collection or context management for session closure.")
        else:
            logger.debug(f"Ollama async client for model {self.model} doesn't seem to have an active internal client to close, or it's managed differently.")