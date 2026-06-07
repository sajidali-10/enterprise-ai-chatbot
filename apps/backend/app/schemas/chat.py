from enum import Enum
from pydantic import BaseModel, Field

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")

class ChatResponse(BaseModel):
    message: str = Field(..., description="Assistant response text")
    role: MessageRole = Field(default=MessageRole.assistant)