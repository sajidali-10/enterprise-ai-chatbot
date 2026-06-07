from typing import Optional, List, Any
from enum import Enum
from pydantic import BaseModel, Field

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")
    mode: str = Field(default="normal", description="Chat mode: 'normal' or 'rag'")

class ChatResponse(BaseModel):
    message: str = Field(..., description="Assistant response text")
    role: MessageRole = Field(default=MessageRole.assistant)
    citations: Optional[List[dict]] = Field(default=None, description="RAG citations when mode is rag")
    debug_info: Optional[dict[str, Any]] = Field(default=None, description="Debug info about retrieval when debug=true")