"""
Chat API Endpoint

Provides /api/chat endpoint for normal and RAG chat.
Supports authentication and audit logging (Phase 6).
"""

import time
from fastapi import APIRouter, Query, Request, Depends, HTTPException
from enum import Enum
from typing import Optional

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
from app.core.rate_limit import rate_limit

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


@router.post("", response_model=ChatResponse)
def post_chat(
    request: Request,
    chat_request: ChatRequest,
    use_hybrid: bool = Query(default=True, description="Use hybrid retrieval (vector + keyword) for RAG"),
    debug: bool = Query(default=False, description="Return debug info about retrieval scores"),
    auth: Optional[AuthContext] = Depends(get_auth_context),
    _=Depends(rate_limit(max_requests=30, window=60)),
) -> ChatResponse:
    """
    Process a chat message and return a response.
    
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
    
    # Knowledge Base and Debug modes both use RAG
    if mode in (ChatMode.KNOWLEDGE_BASE, ChatMode.DEBUG):
        if HAS_SECURITY and auth and auth.is_authenticated:
            # Use audit-aware RAG generation with permission filtering
            answer, citations, metadata = generate_answer_with_rag_audit(
                query=chat_request.message,
                auth=auth,
                debug=debug,
                use_hybrid=use_hybrid,
                request_ip=client_ip,
                request_user_agent=user_agent,
            )
        else:
            # Fall back to regular RAG without auth/audit
            answer, citations, metadata = generate_answer_with_rag(
                query=chat_request.message,
                use_hybrid=use_hybrid,
                debug=debug,
            )
        
        # Create grouped sources for user-friendly display with answer-aware excerpt selection
        grouped_sources = None
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
            citations=citations if citations else None,
            grouped_sources=grouped_sources if grouped_sources else None,
        )
        
        # Add debug info to response for debug mode or when explicitly requested
        if (is_debug_mode or debug) and metadata:
            response.debug_info = metadata
        
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
        )
        grouped_sources = None
        citations = None
        metadata = None
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
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
            blocked=False,  # Will be determined by the logging service
        )
    except Exception:
        # Observability logging should never break the chat response
        pass
    
    # Add observation_id to response for feedback tracking
    response.observation_id = observation_id
    
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