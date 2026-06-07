"""
Answer Generation for RAG Chat

Provides functions to generate answers with or without RAG.
Supports permission filtering and audit logging (Phase 6).
"""

from typing import Optional
from app.rag.retriever import (
    retrieve_chunks,
    retrieve_chunks_with_settings,
    retrieve_chunks_with_auth,
)
from app.rag.prompt_builder import build_rag_prompt
from app.rag.citations import format_citations
from app.services.llm import get_llm_provider
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole

# Import security modules for Phase 6
try:
    from app.security.auth import AuthContext
    from app.security.audit import get_audit_logger
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    AuthContext = None


def generate_answer_with_rag(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    use_hybrid: bool = True,
    debug: bool = False,
) -> tuple[str, list[dict], dict]:
    """
    Full RAG pipeline: retrieve chunks, build prompt, call LLM, return answer + citations.
    
    Args:
        query: User's question.
        top_k: Number of chunks to retrieve (used when use_hybrid=False).
        score_threshold: Minimum score threshold (used when use_hybrid=False).
        use_hybrid: If True, use hybrid retrieval with all Phase 5 features.
                    If False, use original vector-only retrieval.
        debug: If True, return additional debug metadata about retrieval.
        
    Returns:
        Tuple of (answer, citations, metadata).
        Metadata is empty when debug=False.
    """
    if use_hybrid:
        chunks, retrieval_metadata = retrieve_chunks_with_settings(query, debug=debug)
    else:
        chunks = retrieve_chunks(query=query, limit=top_k, score_threshold=score_threshold)
        retrieval_metadata = {}
    
    if not chunks:
        return (
            "I could not find enough information in the approved knowledge base to answer confidently.",
            [],
            retrieval_metadata,
        )
    
    prompt = build_rag_prompt(query, chunks)
    
    provider = get_llm_provider()
    llm_request = ChatRequest(message=prompt)
    llm_response = provider.chat(llm_request)
    
    answer = llm_response.message
    citations = format_citations(chunks)
    
    return answer, citations, retrieval_metadata


def generate_answer_with_rag_audit(
    query: str,
    auth: AuthContext,
    debug: bool = False,
    use_hybrid: bool = True,
    request_ip: Optional[str] = None,
    request_user_agent: Optional[str] = None,
) -> tuple[str, list[dict], dict]:
    """
    Full RAG pipeline with permission filtering and audit logging (Phase 6).
    
    This function:
    1. Performs permission-aware retrieval (filters unauthorized documents)
    2. Generates answer using LLM
    3. Logs the interaction to audit log
    
    Args:
        query: User's question.
        auth: Authentication context with user info and role.
        debug: If True, return additional debug metadata.
        use_hybrid: Use hybrid retrieval (default True).
        request_ip: Client IP for audit logging.
        request_user_agent: Client user agent for audit logging.
        
    Returns:
        Tuple of (answer, citations, metadata).
    """
    # Get LLM provider info
    provider = get_llm_provider()
    model_provider = getattr(provider, 'provider_name', 'unknown') if provider else 'unknown'
    model_name = getattr(provider, 'model', 'unknown') if provider else 'unknown'
    
    # Perform permission-aware retrieval
    if HAS_SECURITY and auth.is_authenticated:
        chunks, retrieval_metadata = retrieve_chunks_with_auth(
            query=query,
            auth=auth,
            debug=debug,
        )
    else:
        # No auth, fall back to regular retrieval
        chunks, retrieval_metadata = retrieve_chunks_with_settings(query, debug=debug)
    
    # Extract document and chunk IDs for audit
    retrieved_doc_ids = list(set(c.get("document_id") for c in chunks if c.get("document_id")))
    retrieved_chunk_ids = list(set(c.get("chunk_id") for c in chunks if c.get("chunk_id")))
    
    if not chunks:
        # Log failed/empty retrieval
        if HAS_SECURITY and auth.is_authenticated:
            try:
                audit_logger = get_audit_logger()
                audit_logger.log_chat_rag(
                    auth=auth,
                    question=query,
                    retrieved_document_ids=[],
                    retrieved_chunk_ids=[],
                    model_provider=model_provider,
                    model_name=model_name,
                    status="success",  # Empty is not an error
                    request_ip=request_ip,
                    request_user_agent=request_user_agent,
                )
            except Exception:
                pass  # Don't fail the request if audit logging fails
        
        retrieval_metadata["audit_logged"] = False
        return (
            "I could not find enough information in the approved knowledge base to answer confidently.",
            [],
            retrieval_metadata,
        )
    
    # Build prompt and generate answer
    prompt = build_rag_prompt(query, chunks)
    llm_request = ChatRequest(message=prompt)
    llm_response = provider.chat(llm_request)
    answer = llm_response.message
    citations = format_citations(chunks)
    
    # Log successful RAG interaction
    if HAS_SECURITY and auth.is_authenticated:
        try:
            audit_logger = get_audit_logger()
            audit_logger.log_chat_rag(
                auth=auth,
                question=query,
                retrieved_document_ids=retrieved_doc_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                model_provider=model_provider,
                model_name=model_name,
                status="success",
                request_ip=request_ip,
                request_user_agent=request_user_agent,
            )
            retrieval_metadata["audit_logged"] = True
        except Exception:
            retrieval_metadata["audit_logged"] = False
    
    return answer, citations, retrieval_metadata


def generate_answer_without_rag(query: str) -> str:
    """Normal chat without RAG."""
    provider = get_llm_provider()
    llm_request = ChatRequest(message=query)
    llm_response = provider.chat(llm_request)
    return llm_response.message


def generate_answer_without_rag_audit(
    query: str,
    auth: AuthContext,
    request_ip: Optional[str] = None,
    request_user_agent: Optional[str] = None,
) -> str:
    """
    Normal chat without RAG, with audit logging (Phase 6).
    """
    # Log the chat message
    if HAS_SECURITY and auth.is_authenticated:
        try:
            audit_logger = get_audit_logger()
            audit_logger.log_chat_message(
                auth=auth,
                status="success",
                request_ip=request_ip,
                request_user_agent=request_user_agent,
            )
        except Exception:
            pass  # Don't fail the request if audit logging fails
    
    return generate_answer_without_rag(query)