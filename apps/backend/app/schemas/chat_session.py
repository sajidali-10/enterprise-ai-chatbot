"""
Chat Session Schemas

Pydantic models for chat session API requests/responses.
Phase 20A — Persistent Chat Sessions.
"""

from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field


class ChatSessionMode(str):
    GENERAL = "general"
    RAG = "rag"


class MessageRole(str):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Session schemas
# ---------------------------------------------------------------------------

class ChatSessionCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="Optional session title")
    mode: str = Field(default="general", description="'general' or 'rag'")


class ChatSessionUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="New title for the session")
    archived: bool = Field(False, description="Whether to archive the session")


class ChatSessionResponse(BaseModel):
    id: int
    user_id: int
    title: str
    mode: str
    created_at: datetime
    updated_at: datetime
    archived_at: Optional[datetime] = None
    message_count: int = 0

    class Config:
        from_attributes = True


class ChatSessionListResponse(BaseModel):
    sessions: List[ChatSessionResponse]
    total: int


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------

class ChatMessageContent(BaseModel):
    role: str
    content: str


class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    user_id: int
    role: str
    content: str
    citations_json: Optional[str] = None
    retrieved_documents_json: Optional[str] = None
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    token_count: Optional[int] = None
    latency_ms: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionDetailResponse(BaseModel):
    session: ChatSessionResponse
    messages: List[ChatMessageResponse]


# ---------------------------------------------------------------------------
# Feedback schemas
# ---------------------------------------------------------------------------

class ChatFeedbackCreate(BaseModel):
    rating: str = Field(..., description="'helpful' or 'not_helpful'")
    comment: Optional[str] = Field(None, description="Optional comment")
    reason: Optional[str] = Field(None, description="Reason for 'not_helpful'")


class ChatFeedbackResponse(BaseModel):
    id: int
    message_id: int
    user_id: int
    rating: str
    comment: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True