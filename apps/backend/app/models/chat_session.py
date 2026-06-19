"""
Chat Session Models

SQLAlchemy models for persistent chat sessions and message storage.
Phase 20A — Persistent Chat Sessions.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, ForeignKey, 
    Enum as SQLEnum, func, Index
)
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum

from app.db.base import Base


class ChatSessionMode(str, PyEnum):
    GENERAL = "general"
    RAG = "rag"


class MessageRole(str, PyEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSession(Base):
    """
    Persistent chat session.
    
    Groups messages between a user and the assistant.
    Supports both general chat and RAG modes.
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New Chat")
    mode = Column(SQLEnum(ChatSessionMode), default=ChatSessionMode.GENERAL, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    archived_at = Column(DateTime, nullable=True)

    # Relationships
    messages = relationship(
        "ChatMessage", 
        back_populates="session", 
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at"
    )

    __table_args__ = (
        Index("ix_chat_sessions_user_created", "user_id", "created_at"),
    )

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, mode='{self.mode}')>"


class ChatMessage(Base):
    """
    Individual message within a chat session.
    
    Stores user messages, assistant responses, and optionally
    citations/metadata from RAG queries.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)  # JSON array of citations
    retrieved_documents_json = Column(Text, nullable=True)  # JSON array of doc metadata
    model_used = Column(String(100), nullable=True)
    provider_used = Column(String(50), nullable=True)
    token_count = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    feedback = relationship(
        "ChatMessageFeedback", 
        back_populates="message", 
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_chat_messages_session_created", "session_id", "created_at"),
    )

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, role='{self.role}')>"


class ChatMessageFeedback(Base):
    """
    Optional feedback on a chat message.
    
    Stores helpfulness ratings and optional comments.
    """
    __tablename__ = "chat_message_feedback"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    rating = Column(String(20), nullable=False)  # "helpful" or "not_helpful"
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    message = relationship("ChatMessage", back_populates="feedback")

    def __repr__(self):
        return f"<ChatMessageFeedback(id={self.id}, message_id={self.message_id}, rating='{self.rating}')>"