from typing import Optional, List, Any
from enum import Enum
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMode(str, Enum):
    """Chat mode enum for type safety."""
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_BASE = "knowledge_base"
    DEBUG = "debug"
    # Legacy values for backward compatibility
    NORMAL = "normal"  # Maps to general_chat
    RAG = "rag"  # Maps to knowledge_base


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")
    mode: str = Field(default="general_chat", description="Chat mode: 'general_chat', 'knowledge_base', or 'debug'")
    session_id: Optional[int] = Field(default=None, description="Existing session ID to continue; creates a new session if omitted")


class GroupedSource(BaseModel):
    """Grouped source information for user-friendly display."""
    source_file_name: str = Field(..., description="Document name")
    sections_used: int = Field(..., description="Number of chunks from this document")
    highest_score: float = Field(..., description="Maximum relevance score")
    confidence: str = Field(..., description="Confidence label: High, Medium, or Low")
    excerpts: List[str] = Field(default_factory=list, description="Selected relevant excerpts")
    indices: List[int] = Field(default_factory=list, description="Original citation indices")
    show_debug_details: bool = Field(default=False, description="Whether to show debug details (scores, indices) in UI")


class DebugMetadata(BaseModel):
    """Debug metadata for developer/admin mode."""
    retrieval_method: Optional[str] = Field(default=None, description="Method used for retrieval")
    relevance_threshold: Optional[float] = Field(default=None, description="Minimum relevance threshold applied")
    grounding_decision: Optional[str] = Field(default=None, description="Grounding check result")
    citation_check: Optional[dict] = Field(default=None, description="Citation enforcement check")
    chunks_retrieved: Optional[int] = Field(default=None, description="Number of chunks retrieved")
    top_score: Optional[float] = Field(default=None, description="Top relevance score")


class ChatResponse(BaseModel):
    message: str = Field(..., description="Assistant response text")
    role: MessageRole = Field(default=MessageRole.assistant)
    session_id: Optional[int] = Field(default=None, description="ID of the chat session this message belongs to")
    citations: Optional[List[dict]] = Field(default=None, description="RAG citations when mode is rag - raw citation list for backward compatibility")
    grouped_sources: Optional[List[GroupedSource]] = Field(default=None, description="Grouped sources by document for user-friendly display")
    debug_info: Optional[dict[str, Any]] = Field(default=None, description="Debug info about retrieval when debug=true or mode=debug")
    observation_id: Optional[int] = Field(default=None, description="ID for this observation - used for feedback submission")