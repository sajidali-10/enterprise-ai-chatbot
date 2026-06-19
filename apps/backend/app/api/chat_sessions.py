"""
Chat Sessions API

REST endpoints for managing chat sessions and messages.
Phase 20A — Persistent Chat Sessions.
"""

from datetime import datetime
from typing import Optional
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat_session import (
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionResponse,
    ChatSessionListResponse,
    ChatSessionDetailResponse,
    ChatMessageResponse,
    ChatFeedbackCreate,
    ChatFeedbackResponse,
)
from app.models.chat_session import ChatSession, ChatMessage, ChatMessageFeedback
from app.models.chat_session import ChatSessionMode as DBChatSessionMode, MessageRole as DBMessageRole

try:
    from app.security.auth import AuthContext, authenticate_request
    from app.security.models import UserRole
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    AuthContext = None


router = APIRouter(prefix="/api/chat/sessions", tags=["Chat Sessions"])


def _get_auth(request: Request) -> Optional[AuthContext]:
    if not HAS_SECURITY:
        return None
    return authenticate_request(request)


def _require_auth(auth: Optional[AuthContext]) -> AuthContext:
    if not auth or not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return auth


def _require_admin(auth: AuthContext) -> None:
    if not auth.is_admin():
        raise HTTPException(status_code=403, detail="Admin access required")


def _get_session_or_404(db: Session, session_id: int, user_id: int) -> ChatSession:
    """Get a session belonging to the user or raise 404."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _truncate_title(text: str, max_len: int = 80) -> str:
    """Truncate title to max_len, removing trailing incomplete words."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    # Try to break at a word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_len * 0.6:
        truncated = truncated[:last_space]
    return truncated + "…"


# ---------------------------------------------------------------------------
# Sessions CRUD
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatSessionResponse, status_code=201)
def create_session(
    request: Request,
    data: ChatSessionCreate,
    db: Session = Depends(get_db),
    auth: Optional[AuthContext] = Depends(_get_auth),
):
    """
    Create a new chat session.
    
    The session starts with no messages. Use the chat endpoint to send
    messages after creating the session.
    """
    auth = _require_auth(auth)

    mode = DBChatSessionMode.RAG if data.mode == "rag" else DBChatSessionMode.GENERAL
    title = data.title or "New Chat"

    session = ChatSession(
        user_id=auth.user_id,
        title=title,
        mode=mode,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return ChatSessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        mode=session.mode.value,
        created_at=session.created_at,
        updated_at=session.updated_at,
        archived_at=session.archived_at,
        message_count=0,
    )


@router.get("", response_model=ChatSessionListResponse)
def list_sessions(
    request: Request,
    include_archived: bool = False,
    db: Session = Depends(get_db),
    auth: Optional[AuthContext] = Depends(_get_auth),
):
    """
    List all chat sessions for the authenticated user.
    
    Does NOT include archived sessions by default.
    """
    auth = _require_auth(auth)

    query = db.query(ChatSession).filter(ChatSession.user_id == auth.user_id)
    if not include_archived:
        query = query.filter(ChatSession.archived_at.is_(None))

    sessions = query.order_by(ChatSession.updated_at.desc()).all()
    total = len(sessions)

    return ChatSessionListResponse(
        sessions=[
            ChatSessionResponse(
                id=s.id,
                user_id=s.user_id,
                title=s.title,
                mode=s.mode.value,
                created_at=s.created_at,
                updated_at=s.updated_at,
                archived_at=s.archived_at,
                message_count=len(s.messages),
            )
            for s in sessions
        ],
        total=total,
    )


@router.get("/{session_id}", response_model=ChatSessionDetailResponse)
def get_session(
    request: Request,
    session_id: int,
    db: Session = Depends(get_db),
    auth: Optional[AuthContext] = Depends(_get_auth),
):
    """
    Get a single session with all its messages.
    
    Users can only access their own sessions.
    """
    auth = _require_auth(auth)

    session = _get_session_or_404(db, session_id, auth.user_id)

    messages = [
        ChatMessageResponse(
            id=m.id,
            session_id=m.session_id,
            user_id=m.user_id,
            role=m.role.value,
            content=m.content,
            citations_json=m.citations_json,
            retrieved_documents_json=m.retrieved_documents_json,
            model_used=m.model_used,
            provider_used=m.provider_used,
            token_count=m.token_count,
            latency_ms=m.latency_ms,
            created_at=m.created_at,
        )
        for m in session.messages
    ]

    return ChatSessionDetailResponse(
        session=ChatSessionResponse(
            id=session.id,
            user_id=session.user_id,
            title=session.title,
            mode=session.mode.value,
            created_at=session.created_at,
            updated_at=session.updated_at,
            archived_at=session.archived_at,
            message_count=len(messages),
        ),
        messages=messages,
    )


@router.patch("/{session_id}", response_model=ChatSessionResponse)
def update_session(
    request: Request,
    session_id: int,
    data: ChatSessionUpdate,
    db: Session = Depends(get_db),
    auth: Optional[AuthContext] = Depends(_get_auth),
):
    """
    Update a session (title, archive/unarchive).
    
    Users can only update their own sessions.
    """
    auth = _require_auth(auth)

    session = _get_session_or_404(db, session_id, auth.user_id)

    if data.title is not None:
        session.title = _truncate_title(data.title)

    if data.archived:
        session.archived_at = datetime.utcnow()
    else:
        session.archived_at = None

    db.commit()
    db.refresh(session)

    return ChatSessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        mode=session.mode.value,
        created_at=session.created_at,
        updated_at=session.updated_at,
        archived_at=session.archived_at,
        message_count=len(session.messages),
    )


@router.delete("/{session_id}", status_code=204)
def delete_session(
    request: Request,
    session_id: int,
    db: Session = Depends(get_db),
    auth: Optional[AuthContext] = Depends(_get_auth),
):
    """
    Permanently delete a chat session and all its messages.
    
    Users can only delete their own sessions.
    """
    auth = _require_auth(auth)

    session = _get_session_or_404(db, session_id, auth.user_id)
    db.delete(session)
    db.commit()


# ---------------------------------------------------------------------------
# Message feedback
# ---------------------------------------------------------------------------

@router.post("/messages/{message_id}/feedback", response_model=ChatFeedbackResponse, status_code=201)
def add_message_feedback(
    request: Request,
    message_id: int,
    data: ChatFeedbackCreate,
    db: Session = Depends(get_db),
    auth: Optional[AuthContext] = Depends(_get_auth),
):
    """
    Add feedback to a chat message.
    """
    auth = _require_auth(auth)

    # Find the message and verify ownership
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Verify user owns the session
    session = db.query(ChatSession).filter(
        ChatSession.id == message.session_id,
        ChatSession.user_id == auth.user_id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Message not found")

    # Check for existing feedback
    existing = db.query(ChatMessageFeedback).filter(
        ChatMessageFeedback.message_id == message_id,
        ChatMessageFeedback.user_id == auth.user_id,
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Feedback already submitted")

    feedback = ChatMessageFeedback(
        message_id=message_id,
        user_id=auth.user_id,
        rating=data.rating,
        comment=data.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return ChatFeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        user_id=feedback.user_id,
        rating=feedback.rating,
        comment=feedback.comment,
        created_at=feedback.created_at,
    )