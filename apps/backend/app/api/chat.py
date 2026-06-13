"""
Chat API Endpoint

Provides /api/chat endpoint for normal and RAG chat.
Supports authentication and audit logging (Phase 6).
"""

import time
from fastapi import APIRouter, Query, Request, Depends
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

# Import security modules for Phase 6
try:
    from app.security.auth import (
        AuthContext,
        authenticate_request,
        DEV_USER_HEADER,
    )
    from app.security.models import UserRole
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    AuthContext = None
    UserRole = None

router = APIRouter(prefix="/api/chat", tags=["Chat"])


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
    
    In production, this would validate JWT tokens, sessions, etc.
    In development, it supports dev user headers.
    """
    if not HAS_SECURITY:
        return None
    return authenticate_request(request)


@router.post("", response_model=ChatResponse)
def post_chat(
    request: Request,
    chat_request: ChatRequest,
    use_hybrid: bool = Query(default=True, description="Use hybrid retrieval (vector + keyword) for RAG"),
    debug: bool = Query(default=False, description="Return debug info about retrieval scores"),
    auth: Optional[AuthContext] = Depends(get_auth_context),
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
    
    # Check if debug mode is allowed (admin only)
    is_debug_mode = mode == ChatMode.DEBUG
    if is_debug_mode:
        if not HAS_SECURITY or not auth or not auth.is_authenticated or not auth.is_admin():
            # Non-admin trying to use debug mode - fall back to knowledge_base
            mode = ChatMode.KNOWLEDGE_BASE
            is_debug_mode = False
    
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
    if not HAS_SECURITY:
        return {
            "authenticated": False,
            "username": "anonymous",
            "role": "viewer",
            "dev_mode_available": False,
        }
    
    auth = authenticate_request(request)
    
    return {
        "authenticated": auth.is_authenticated,
        "username": auth.username,
        "role": auth.role.value if hasattr(auth.role, 'value') else str(auth.role),
        "user_id": auth.user_id,
        "is_admin": auth.is_admin(),
        "dev_mode": True,  # Dev headers are available
    }