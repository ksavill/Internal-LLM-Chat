from typing import List, Optional, Union, Any
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    model: Optional[str] = "qwen2.5-coder:7b"
    backup_models: Optional[Union[str, List[str]]] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    image_b64: Optional[Union[str, List[str]]] = None
    conversation_id: Optional[str] = None
    timeout_threshold: Optional[float] = 30.0
    request_profile: Optional[str] = None

class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ConfigUpdate(BaseModel):
    value: Any

class MessageResponse(BaseModel):
    message: str

class TokenResponse(BaseModel):
    token: str

class ConversationsListResponse(BaseModel):
    conversations: List[dict]

class ConversationDetailResponse(BaseModel):
    messages: List[dict]

class OllamaModel(BaseModel):
    NAME: str

class OllamaModelsResponse(BaseModel):
    models: List[OllamaModel]

class OpenAIModelsResponse(BaseModel):
    models: List[OllamaModel]

class ChatCompletionResponse(BaseModel):
    message: str

class ConfigValue(BaseModel):
    key: str
    value: Any

class ConfigMessage(BaseModel):
    message: str

class ForkRequest(BaseModel):
    original_conversation_id: str
    message_index: int # 0-based index of the last message to include in the fork

class ForkResponse(BaseModel):
    new_conversation_id: str