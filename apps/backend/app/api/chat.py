"""
Chat API Endpoint

Provides /api/chat endpoint for normal and RAG chat.
"""

from fastapi import APIRouter, Query
from enum import Enum

from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm import get_llm_provider
from app.rag.answer_generator import generate_answer_with_rag, generate_answer_without_rag

router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatMode(str, Enum):
    NORMAL = "normal"
    RAG = "rag"


@router.post("", response_model=ChatResponse)
def post_chat(
    request: ChatRequest,
    mode: str = Query(default="normal", description="Chat mode: 'normal' or 'rag'"),
    use_hybrid: bool = Query(default=True, description="Use hybrid retrieval (vector + keyword) for RAG"),
    debug: bool = Query(default=False, description="Return debug info about retrieval scores"),
) -> ChatResponse:
    """
    Process a chat message and return a response.
    
    Modes:
    - normal: Direct LLM response without RAG
    - rag: RAG-enhanced response using document retrieval
    
    Query Parameters (RAG mode only):
    - use_hybrid: Use hybrid retrieval combining vector and keyword search (default: True)
    - debug: Return detailed retrieval debug info including scores and sources (default: False)
    """
    if mode == ChatMode.RAG:
        answer, citations, metadata = generate_answer_with_rag(
            query=request.message,
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
        answer = generate_answer_without_rag(request.message)
        return ChatResponse(
            message=answer,
            role=MessageRole.assistant,
        )