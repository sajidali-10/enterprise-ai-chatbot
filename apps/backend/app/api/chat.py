"""
Chat API Endpoint

Provides /api/chat endpoint for normal and RAG chat.
Supports authentication and audit logging (Phase 6).
"""

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
    NORMAL = "normal"
    RAG = "rag"


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
    mode: str = Query(default="normal", description="Chat mode: 'normal' or 'rag'"),
    use_hybrid: bool = Query(default=True, description="Use hybrid retrieval (vector + keyword) for RAG"),
    debug: bool = Query(default=False, description="Return debug info about retrieval scores"),
    auth: Optional[AuthContext] = Depends(get_auth_context),
) -> ChatResponse:
    """
    Process a chat message and return a response.
    
    Modes:
    - normal: Direct LLM response without RAG
    - rag: RAG-enhanced response using document retrieval
    
    Authentication:
    - In development: Use X-Dev-User header to authenticate (e.g., X-Dev-User: admin_user)
    - Admin users bypass permission checks
    - Dev users (via header) have limited access
    
    Query Parameters (RAG mode only):
    - use_hybrid: Use hybrid retrieval combining vector and keyword search (default: True)
    - debug: Return detailed retrieval debug info including scores and sources (default: False)
    """
    # Get client info for audit logging
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "")[:500]
    
    if mode == ChatMode.RAG:
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
        
        response = ChatResponse(
            message=answer,
            role=MessageRole.assistant,
            citations=citations if citations else None,
        )
        
        # Add debug info to response if requested
        if debug and metadata:
            response.debug_info = metadata
        
        return response
    else:
        if HAS_SECURITY and auth and auth.is_authenticated:
            answer = generate_answer_without_rag_audit(
                query=chat_request.message,
                auth=auth,
                request_ip=client_ip,
                request_user_agent=user_agent,
            )
        else:
            answer = generate_answer_without_rag(chat_request.message)
        
        return ChatResponse(
            message=answer,
            role=MessageRole.assistant,
        )


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