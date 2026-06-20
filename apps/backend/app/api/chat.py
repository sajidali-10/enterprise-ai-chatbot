"""
Chat API Endpoint

Provides /api/chat endpoint for normal and RAG chat.
Supports authentication and audit logging (Phase 6).
Phase 20A: Persistent sessions and message storage.
Phase 20C (refined): Separate conversation context from RAG retrieval query.
"""

import json
import time
from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Query, Request, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm import get_llm_provider
from app.rag.answer_generator import (
    generate_answer_with_rag,
    generate_answer_with_rag_audit,
    generate_answer_without_rag,
    generate_answer_without_rag_audit,
)
from app.rag.citations import format_citations, group_citations_by_source
from app.services.observability import log_chat_observation
from app.services.suggestions import generate_suggestions, is_fallback_response
from app.services.conversation_context import (
    get_recent_conversation_context,
    format_conversation_context_for_prompt,
)
from app.core.rate_limit import rate_limit

# Chat session models (Phase 20A)
try:
    from app.models.chat_session import ChatSession, ChatMessage
    from app.models.chat_session import ChatSessionMode as DBChatSessionMode, MessageRole as DBMessageRole
    HAS_SESSION_MODELS = True
except ImportError:
    HAS_SESSION_MODELS = False
    ChatSession = None
    ChatMessage = None

# Import security modules for Phase 6 / Phase 12
try:
    from app.security.auth import (
        AuthContext,
        authenticate_request,
        DEV_USER_HEADER,
        get_role_permissions,
    )
    from app.security.models import UserRole
    from app.security.dependencies import get_auth_context as _get_auth_context_dep
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    AuthContext = None
    UserRole = None


def _require_permission_for_mode(auth: AuthContext, mode: str) -> None:
    """Raise 403 if auth lacks permission for the requested chat mode."""
    if not HAS_SECURITY or not auth:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    perms = get_role_permissions(auth.role)
    if mode == ChatMode.GENERAL_CHAT and not perms["can_use_general_chat"]:
        raise HTTPException(status_code=403, detail="General Chat is not allowed for this user")
    if mode == ChatMode.KNOWLEDGE_BASE and not perms["can_use_knowledge_base"]:
        raise HTTPException(status_code=403, detail="Knowledge Base is not allowed for this user")
    if mode == ChatMode.DEBUG and not perms["can_use_debug"]:
        raise HTTPException(status_code=403, detail="Debug mode is not allowed for this user")


router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ==============================================================================
# Provider Health Endpoint (Phase 15 — LiteLLM Gateway)
# ==============================================================================
# Separate router so the prefix doesn't conflict with /api/chat
_provider_router = APIRouter(tags=["Health"])


@_provider_router.get("/api/health/provider")
def get_provider_info():
    """
    Return the currently configured LLM provider and its configuration.

    This endpoint NEVER exposes secrets (API keys, master keys).
    Only the base URL host is returned, not the full URL with credentials.

    Response fields:
    - provider: LLM_PROVIDER value (e.g., "openrouter", "litellm", "mock")
    - model: configured model name
    - base_url_host: host portion of base URL only (no credentials or path)
    - gateway_mode: true if using LiteLLM Gateway (LLM_PROVIDER=litellm)
    - litellm_enabled: whether LiteLLM gateway is enabled in config
    """
    import os
    from urllib.parse import urlparse

    provider_name = os.getenv("LLM_PROVIDER", "mock").lower().strip()
    gateway_mode = provider_name == "litellm"
    litellm_enabled = os.getenv("LITELLM_ENABLED", "false").lower() in ("true", "1", "yes")

    # Get model — varies by provider
    if provider_name == "openrouter":
        model = os.getenv("OPENROUTER_MODEL", "unknown")
        base_url_raw = os.getenv("OPENROUTER_BASE_URL", "")
    elif provider_name == "litellm":
        model = os.getenv("LITELLM_MODEL", "unknown")
        base_url_raw = os.getenv("LITELLM_BASE_URL", "")
    elif provider_name == "openai":
        model = os.getenv("OPENAI_MODEL", "unknown")
        base_url_raw = os.getenv("OPENAI_BASE_URL", "")
    elif provider_name == "ollama":
        model = os.getenv("OLLAMA_MODEL", "unknown")
        base_url_raw = os.getenv("OLLAMA_BASE_URL", "")
    else:
        model = "unknown"
        base_url_raw = ""

    # Extract just the host from the base URL — NEVER return full URL or secrets
    base_url_host = ""
    if base_url_raw:
        try:
            parsed = urlparse(base_url_raw if base_url_raw.startswith("http") else f"http://{base_url_raw}")
            base_url_host = parsed.hostname or ""
        except Exception:
            base_url_host = ""

    return {
        "provider": provider_name,
        "model": model,
        "base_url_host": base_url_host,
        "gateway_mode": gateway_mode,
        "litellm_enabled": litellm_enabled,
    }


class ChatMode(str, Enum):
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_BASE = "knowledge_base"
    DEBUG = "debug"


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header first (behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    if request.client:
        return request.client.host
    return "unknown"


def get_auth_context(request: Request) -> Optional[AuthContext]:
    """
    Extract authentication context from request.

    In production, validates JWT tokens.
    In development, supports dev user headers.
    """
    if not HAS_SECURITY:
        return None
    try:
        return _get_auth_context_dep(request)
    except Exception:
        return authenticate_request(request)


def _truncate_title(text: str, max_len: int = 80) -> str:
    """Truncate title to max_len, removing trailing incomplete words."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > max_len * 0.6:
        truncated = truncated[:last_space]
    return truncated + "…"


def _get_or_create_session(
    db: Session,
    auth: AuthContext,
    session_id: Optional[int],
    mode: str,
    first_message: str,
) -> Optional[ChatSession]:
    """
    Get an existing session or create a new one.

    Returns None if session models are not available.
    Raises 404 if session_id refers to a non-existent or non-owned session.
    Raises 400 if session_id refers to an archived session.
    """
    if not HAS_SESSION_MODELS or not auth.user_id:
        return None

    if session_id is not None:
        # Load existing session
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == auth.user_id,
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.archived_at:
            raise HTTPException(status_code=400, detail="Cannot append to archived session")
        return session

    # Create new session
    db_mode = DBChatSessionMode.RAG if mode in (ChatMode.KNOWLEDGE_BASE, ChatMode.DEBUG) else DBChatSessionMode.GENERAL
    title = _truncate_title(first_message)

    session = ChatSession(
        user_id=auth.user_id,
        title=title,
        mode=db_mode,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _store_messages(
    db: Session,
    auth: AuthContext,
    session: ChatSession,
    user_message: str,
    assistant_message: str,
    citations: Optional[list],
    grouped_sources: Optional[list],
    model_used: Optional[str],
    provider_used: Optional[str],
    latency_ms: int,
) -> None:
    """
    Store the user and assistant messages for a session.

    Safely stores citations as display metadata only — does not grant document access.
    """
    if not HAS_SESSION_MODELS or not auth.user_id:
        return

    # Safely serialize citations (display metadata only — no secrets)
    citations_json = None
    retrieved_docs_json = None
    if citations:
        # Store minimal citation info for display — not secrets or raw document content
        safe_citations = []
        for c in citations:
            if isinstance(c, dict):
                safe_citations.append({
                    "source_file_name": c.get("source_file_name"),
                    "content_snippet": c.get("content_snippet", "")[:500] if c.get("content_snippet") else None,
                    "relevance_score": c.get("relevance_score"),
                })
        citations_json = json.dumps(safe_citations)

    if grouped_sources:
        safe_docs = []
        for gs in grouped_sources:
            if isinstance(gs, dict):
                safe_docs.append({
                    "source_file_name": gs.get("source_file_name"),
                    "sections_used": gs.get("sections_used"),
                    "confidence": gs.get("confidence"),
                })
        retrieved_docs_json = json.dumps(safe_docs)

    # Store user message
    user_msg = ChatMessage(
        session_id=session.id,
        user_id=auth.user_id,
        role=DBMessageRole.USER,
        content=user_message,
    )
    db.add(user_msg)

    # Store assistant message
    assistant_msg = ChatMessage(
        session_id=session.id,
        user_id=auth.user_id,
        role=DBMessageRole.ASSISTANT,
        content=assistant_message,
        citations_json=citations_json,
        retrieved_documents_json=retrieved_docs_json,
        model_used=model_used,
        provider_used=provider_used,
        latency_ms=latency_ms,
    )
    db.add(assistant_msg)

    # Update session timestamp
    session.updated_at = datetime.utcnow()
    db.commit()


@router.post("", response_model=ChatResponse)
def post_chat(
    request: Request,
    chat_request: ChatRequest,
    use_hybrid: bool = Query(default=True, description="Use hybrid retrieval (vector + keyword) for RAG"),
    debug: bool = Query(default=False, description="Return debug info about retrieval scores"),
    db: Session = Depends(get_db),
    auth: Optional[AuthContext] = Depends(get_auth_context),
    _=Depends(rate_limit(max_requests=30, window=60)),
) -> ChatResponse:
    """
    Process a chat message and return a response.

    Phase 20A: If session_id is provided, append to that session.
    Otherwise, create a new session automatically.

    Phase 20C (refined): Retrieval query is kept clean (user message only).
    Conversation context is passed separately to the LLM prompt.

    Modes (from request body):
    - general_chat: General AI assistant without document retrieval
    - knowledge_base: RAG-enhanced response using uploaded documents only
    - debug: Admin/developer mode showing retrieval internals (admin only)

    Authentication:
    - In development: Use X-Dev-User header to authenticate (e.g., X-Dev-User: admin_user)
    - Admin users bypass permission checks
    - Dev users (via header) have limited access

    Query Parameters (RAG mode only):
    - use_hybrid: Use hybrid retrieval combining vector and keyword search (default: True)
    - debug: Return detailed retrieval debug info including scores and sources (default: False)
    """
    start_time = time.time()

    # Get client info for audit logging
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "")[:500]

    # Use mode from request body, default to "general_chat"
    mode = chat_request.mode if chat_request.mode else ChatMode.GENERAL_CHAT

    # Handle legacy mode values for backward compatibility
    mode = mode.lower().strip()
    if mode == "normal":
        mode = ChatMode.GENERAL_CHAT
    elif mode == "rag":
        mode = ChatMode.KNOWLEDGE_BASE

    # Enforce authentication and mode permissions (Phase 12)
    _require_permission_for_mode(auth, mode)
    is_debug_mode = mode == ChatMode.DEBUG

    latency_ms = None
    observation_id = None
    session = None
    session_id = None

    # Phase 20A: Get or create session
    if HAS_SESSION_MODELS and auth and auth.is_authenticated and auth.user_id:
        try:
            session = _get_or_create_session(
                db=db,
                auth=auth,
                session_id=chat_request.session_id,
                mode=mode,
                first_message=chat_request.message,
            )
            if session:
                session_id = session.id
        except HTTPException:
            raise
        except Exception:
            # Don't break chat if session storage fails
            session = None

    # Phase 20C: Load conversation context if continuing a session
    conversation_context = ""
    has_conversation_context = False
    if session:
        context_messages, _ = get_recent_conversation_context(
            db=db,
            session_id=session.id,
            user_id=auth.user_id,
        )
        if context_messages:
            conversation_context = format_conversation_context_for_prompt(context_messages)
            has_conversation_context = True

    # Knowledge Base and Debug modes both use RAG
    citations = None
    grouped_sources = None
    metadata = None
    answer = None
    model_used = None
    provider_used = None

    # Phase 20C: Use CLEAN retrieval query (user message only, no context prepended)
    retrieval_query = chat_request.message

    if mode in (ChatMode.KNOWLEDGE_BASE, ChatMode.DEBUG):
        # Pass conversation context SEPARATELY to the prompt, not to retrieval
        if HAS_SECURITY and auth and auth.is_authenticated:
            # Use audit-aware RAG generation with permission filtering
            answer, citations, metadata = generate_answer_with_rag_audit(
                query=retrieval_query,
                auth=auth,
                debug=debug,
                use_hybrid=use_hybrid,
                request_ip=client_ip,
                request_user_agent=user_agent,
                conversation_context=conversation_context,
            )
        else:
            # Fall back to regular RAG without auth/audit
            answer, citations, metadata = generate_answer_with_rag(
                query=retrieval_query,
                use_hybrid=use_hybrid,
                debug=debug,
                conversation_context=conversation_context,
            )

        # Create grouped sources for user-friendly display with answer-aware excerpt selection
        if citations:
            grouped_sources = group_citations_by_source(
                citations,
                question=chat_request.message,
                answer=answer,
                max_excerpts=3,
                debug_mode=is_debug_mode
            )

        response = ChatResponse(
            message=answer,
            role=MessageRole.assistant,
            session_id=session_id,
            citations=citations if citations else None,
            grouped_sources=grouped_sources if grouped_sources else None,
        )

        # Add debug info to response for debug mode or when explicitly requested
        if (is_debug_mode or debug) and metadata:
            response.debug_info = metadata

        # Phase 20B: Add suggested follow-ups based on response characteristics
        # Phase 20C: Contextual suggestions now shown when there's conversation context
        is_fallback = is_fallback_response(answer, citations)
        response.suggested_followups = generate_suggestions(
            mode=mode,
            has_citations=citations is not None and len(citations) > 0,
            is_fallback=is_fallback,
            citations=citations,
            has_conversation_context=has_conversation_context,
        )

    else:
        # General Chat mode - no RAG, no sources
        if HAS_SECURITY and auth and auth.is_authenticated:
            answer = generate_answer_without_rag_audit(
                query=chat_request.message,
                auth=auth,
                request_ip=client_ip,
                request_user_agent=user_agent,
            )
        else:
            answer = generate_answer_without_rag(chat_request.message)

        # General Chat: no citations, no sources, no debug info
        response = ChatResponse(
            message=answer,
            role=MessageRole.assistant,
            session_id=session_id,
        )

        # Phase 20B: Add suggested follow-ups for general chat
        # Phase 20C: Pass conversation context info for contextual suggestions
        response.suggested_followups = generate_suggestions(
            mode=mode,
            has_conversation_context=has_conversation_context,
        )

    # Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)

    # Get provider/model info for storage (no secrets)
    try:
        llm_provider = get_llm_provider()
        if llm_provider:
            provider_used = getattr(llm_provider, 'provider_name', None)
            model_used = getattr(llm_provider, 'model', None)
    except Exception:
        pass

    # Log observability (non-blocking - don't break chat if logging fails)
    try:
        auth_dict = None
        if auth and hasattr(auth, '__dict__'):
            auth_dict = {
                'user_id': getattr(auth, 'user_id', None),
                'username': getattr(auth, 'username', None),
                'role': getattr(auth, 'role', None),
            }

        observation_id = log_chat_observation(
            mode=mode,
            question=chat_request.message,
            answer=answer,
            auth_context=auth_dict,
            citations=citations,
            grouped_sources=grouped_sources,
            metadata=metadata,
            latency_ms=latency_ms,
            blocked=False,
        )
    except Exception:
        pass

    # Add observation_id to response for feedback tracking
    response.observation_id = observation_id

    # Phase 20A: Store messages (non-blocking — don't break chat if storage fails)
    if session:
        try:
            _store_messages(
                db=db,
                auth=auth,
                session=session,
                user_message=chat_request.message,
                assistant_message=answer,
                citations=citations,
                grouped_sources=grouped_sources,
                model_used=model_used,
                provider_used=provider_used,
                latency_ms=latency_ms,
            )
        except Exception:
            # Log but don't break the response
            pass

    return response


@router.get("/auth-info")
def get_auth_info(request: Request):
    """
    Get current authentication info.

    This endpoint is useful for debugging and for the frontend
    to determine what UI to show (admin vs user).
    """
    from app.core.config import settings

    if not HAS_SECURITY:
        return {
            "authenticated": False,
            "username": "anonymous",
            "role": "viewer",
            "user_id": None,
            "is_admin": False,
            "dev_mode": False,
            "permissions": get_role_permissions(UserRole.VIEWER),
        }

    auth = authenticate_request(request)
    perms = get_role_permissions(auth.role)

    return {
        "authenticated": auth.is_authenticated,
        "username": auth.username,
        "role": auth.role.value if hasattr(auth.role, 'value') else str(auth.role),
        "user_id": auth.user_id,
        "is_admin": auth.is_admin(),
        "dev_mode": settings.AUTH_MODE == "dev" and settings.DEV_AUTH_ENABLED,
        "permissions": perms,
    }