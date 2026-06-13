"""
Answer Generation for RAG Chat

Provides functions to generate answers with or without RAG.
Supports permission filtering and audit logging (Phase 6).

Phase 10.6: Stability fix for citation-aware answering:
- Automatic retry with stricter citation prompt when LLM omits citations
- Backend citation attachment based on content overlap with retrieved chunks
- Deterministic temperature=0 for Knowledge Base mode
"""

import re
from typing import Optional
from app.rag.retriever import (
    retrieve_chunks,
    retrieve_chunks_with_settings,
    retrieve_chunks_with_auth,
)
from app.rag.prompt_builder import build_rag_prompt, build_general_chat_prompt, build_strict_citation_prompt
from app.rag.citations import format_citations, attach_citations_to_answer
from app.rag.grounding import (
    apply_grounding_checks,
    get_debug_info,
    NO_CHUNKS_MESSAGE,
    LOW_RELEVANCE_MESSAGE,
    has_citations,
    check_citations,
)
from app.services.llm import get_llm_provider
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.core.config import settings

# Import security modules for Phase 6
try:
    from app.security.auth import AuthContext
    from app.security.audit import get_audit_logger
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    AuthContext = None


def _call_llm_with_citations(
    query: str,
    chunks: list[dict],
    strict: bool = False,
    temperature: float = 0.0,
) -> str:
    """
    Call LLM to generate answer from retrieved chunks.
    
    Args:
        query: User's question.
        chunks: Retrieved context chunks.
        strict: If True, use stricter citation prompt (for retry).
        temperature: LLM temperature for deterministic output.
        
    Returns:
        LLM generated answer text.
    """
    if strict:
        prompt = build_strict_citation_prompt(query, chunks)
    else:
        prompt = build_rag_prompt(query, chunks)
    
    provider = get_llm_provider()
    llm_request = ChatRequest(message=prompt)
    # Pass temperature to provider if supported
    if hasattr(provider, 'set_temperature'):
        provider.set_temperature(temperature)
    llm_response = provider.chat(llm_request)
    
    return llm_response.message


def _should_retry_for_citations(
    answer: str,
    chunks: list[dict],
    min_relevance_score: float,
) -> bool:
    """
    Determine if we should retry LLM with stricter citation prompt.
    
    Returns True if:
    - Answer lacks citations
    - Chunks exist and are strong (top_score >= threshold)
    - Answer has some content overlap with retrieved chunks (not hallucinated)
    
    Args:
        answer: Generated answer text.
        chunks: Retrieved chunks.
        min_relevance_score: Minimum relevance threshold.
        
    Returns:
        True if retry is warranted.
    """
    if has_citations(answer):
        return False
    
    if not chunks:
        return False
    
    # Check if top chunk score meets threshold
    top_score = chunks[0].get("score", 0) if chunks else 0
    if top_score < min_relevance_score:
        return False
    
    # Check if answer has some content overlap with chunks (not pure hallucination)
    # Use simple keyword overlap as heuristic
    answer_lower = answer.lower()
    chunk_texts = " ".join(c.get("content", "").lower() for c in chunks)
    
    # Extract key terms from answer (words >= 5 chars)
    answer_terms = set(re.findall(r'\b[a-z]{5,}\b', answer_lower))
    chunk_terms = set(re.findall(r'\b[a-z]{5,}\b', chunk_texts))
    
    # If answer has significant overlap with chunk content, it's likely grounded
    if answer_terms and chunk_terms:
        overlap = len(answer_terms & chunk_terms) / len(answer_terms)
        if overlap >= 0.3:  # At least 30% of answer terms appear in chunks
            return True
    
    return False


def generate_answer_with_rag(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    use_hybrid: bool = True,
    debug: bool = False,
    min_relevance_score: Optional[float] = None,
) -> tuple[str, list[dict], dict]:
    """
    Full RAG pipeline: retrieve chunks, build prompt, call LLM, return answer + citations.
    
    Includes grounding checks (Phase 10):
    - Retrieval guardrail: Don't call LLM if no chunks retrieved
    - Minimum relevance threshold: Reject chunks below threshold
    - Citation enforcement: Verify answer includes citations
    - Automatic retry: Retry once with strict citations if LLM omitted them
    - Backend citation attachment: Attach citations based on content overlap
    
    Args:
        query: User's question.
        top_k: Number of chunks to retrieve (used when use_hybrid=False).
        score_threshold: Minimum score threshold (used when use_hybrid=False).
        use_hybrid: If True, use hybrid retrieval with all Phase 5 features.
                    If False, use original vector-only retrieval.
        debug: If True, return additional debug metadata about retrieval.
        min_relevance_score: Minimum relevance score. Defaults to RAG_MIN_RELEVANCE_SCORE.
        
    Returns:
        Tuple of (answer, citations, metadata).
        Metadata is empty when debug=False.
    """
    if min_relevance_score is None:
        min_relevance_score = settings.RAG_MIN_RELEVANCE_SCORE
    
    if use_hybrid:
        chunks, retrieval_metadata = retrieve_chunks_with_settings(query, debug=debug)
    else:
        chunks = retrieve_chunks(query=query, limit=top_k, score_threshold=score_threshold)
        retrieval_metadata = {}
    
    # Phase 10: Apply grounding checks BEFORE calling LLM
    # Phase 11.2: Pass query for topic relevance checking
    should_block, fallback_message, grounding_meta = apply_grounding_checks(
        chunks=chunks,
        answer=None,  # No answer yet, only check retrieval
        threshold=min_relevance_score,
        require_citations=False,  # Can't require citations without an answer
        query=query,  # For topic relevance check
    )
    
    retrieval_metadata["grounding"] = grounding_meta
    
    if should_block:
        retrieval_metadata["blocked"] = True
        retrieval_metadata["block_reason"] = grounding_meta.get("blocked_reason", "unknown")
        if debug:
            retrieval_metadata["debug_info"] = get_debug_info(
                chunks, min_relevance_score, (should_block, fallback_message, grounding_meta)
            )
        return fallback_message, [], retrieval_metadata
    
    retrieval_metadata["blocked"] = False
    
    # Phase 10.6: Track citation repair flow explicitly
    citation_repair_meta = {
        "initial_has_citations": False,
        "retry_attempted": False,
        "retry_has_citations": False,
        "attachment_attempted": False,
        "attachment_has_citations": False,
        "final_has_citations": False,
        "final_citation_count": 0,
        "blocked_reason": None,
    }
    
    # Proceed with LLM call - use temperature=0 for deterministic KB output
    answer = _call_llm_with_citations(query, chunks, strict=False, temperature=0.0)
    citations = format_citations(chunks)
    
    # Phase 10.6: Check initial citations
    citation_repair_meta["initial_has_citations"] = has_citations(answer)
    
    # Phase 10.6: Check answer grounding AFTER LLM generates response
    should_block, fallback_message, answer_grounding_meta = apply_grounding_checks(
        chunks=chunks,
        answer=answer,
        threshold=min_relevance_score,
        require_citations=True,  # Require citations in the answer
    )
    
    retrieval_metadata["grounding"].update(answer_grounding_meta)
    
    # Phase 10.6: Retry logic for missing citations with strong retrieval
    if not citation_repair_meta["initial_has_citations"]:
        top_score = chunks[0].get("score", 0) if chunks else 0
        if top_score >= min_relevance_score:
            # Retry once with strict citation prompt
            citation_repair_meta["retry_attempted"] = True
            answer = _call_llm_with_citations(query, chunks, strict=True, temperature=0.0)
            citations = format_citations(chunks)
            citation_repair_meta["retry_has_citations"] = has_citations(answer)
    
    # Phase 10.6: Backend citation attachment if citations still missing but chunks are strong
    if not has_citations(answer) and chunks:
        top_score = chunks[0].get("score", 0) if chunks else 0
        if top_score >= min_relevance_score:
            # Try to attach citations based on content overlap
            citation_repair_meta["attachment_attempted"] = True
            answer_with_citations, citation_map = attach_citations_to_answer(
                answer, chunks, query
            )
            if citation_map:  # If we found matching content
                answer = answer_with_citations
                citation_repair_meta["attachment_has_citations"] = has_citations(answer)
                citation_repair_meta["citation_map"] = citation_map
                retrieval_metadata["grounding"]["backend_citations_attached"] = True
    
    # Phase 10.6: Final check
    final_has_citations = has_citations(answer)
    citation_repair_meta["final_has_citations"] = final_has_citations
    if final_has_citations:
        _, citation_count_meta = check_citations(answer)
        citation_repair_meta["final_citation_count"] = citation_count_meta.get("citation_count", 0)
    
    # Final block check only if citations still missing after ALL repair attempts
    # Reset should_block to False first - it was set by intermediate checks
    should_block = False
    if not final_has_citations:
        should_block, fallback_message, answer_grounding_meta = apply_grounding_checks(
            chunks=chunks,
            answer=answer,
            threshold=min_relevance_score,
            require_citations=True,
        )
        retrieval_metadata["grounding"].update(answer_grounding_meta)
        citation_repair_meta["blocked_reason"] = "answer_lacks_citations"
    
    # Merge citation repair metadata into grounding metadata
    retrieval_metadata["grounding"]["citation_repair"] = citation_repair_meta
    retrieval_metadata["grounding"]["final_citation_count"] = citation_repair_meta["final_citation_count"]
    retrieval_metadata["grounding"]["final_has_citations"] = final_has_citations
    
    if should_block:
        retrieval_metadata["blocked"] = True
        retrieval_metadata["block_reason"] = citation_repair_meta["blocked_reason"] or "answer_lacks_citations"
        if debug:
            retrieval_metadata["debug_info"] = get_debug_info(
                chunks, min_relevance_score, (should_block, fallback_message, answer_grounding_meta)
            )
        return fallback_message, [], retrieval_metadata
    
    # Add debug info if requested
    if debug:
        retrieval_metadata["debug_info"] = get_debug_info(
            chunks, min_relevance_score, (False, None, retrieval_metadata["grounding"])
        )
    
    return answer, citations, retrieval_metadata


def generate_answer_with_rag_audit(
    query: str,
    auth: AuthContext,
    debug: bool = False,
    use_hybrid: bool = True,
    request_ip: Optional[str] = None,
    request_user_agent: Optional[str] = None,
    min_relevance_score: Optional[float] = None,
) -> tuple[str, list[dict], dict]:
    """
    Full RAG pipeline with permission filtering and audit logging (Phase 6).
    
    Includes grounding checks (Phase 10):
    - Retrieval guardrail: Don't call LLM if no chunks retrieved
    - Minimum relevance threshold: Reject chunks below threshold
    - Citation enforcement: Verify answer includes citations
    
    This function:
    1. Performs permission-aware retrieval (filters unauthorized documents)
    2. Applies grounding checks (Phase 10)
    3. Generates answer using LLM
    4. Logs the interaction to audit log
    
    Args:
        query: User's question.
        auth: Authentication context with user info and role.
        debug: If True, return additional debug metadata.
        use_hybrid: Use hybrid retrieval (default True).
        request_ip: Client IP for audit logging.
        request_user_agent: Client user agent for audit logging.
        min_relevance_score: Minimum relevance score. Defaults to RAG_MIN_RELEVANCE_SCORE.
        
    Returns:
        Tuple of (answer, citations, metadata).
    """
    if min_relevance_score is None:
        min_relevance_score = settings.RAG_MIN_RELEVANCE_SCORE
    
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
    
    # Phase 10: Apply grounding checks BEFORE calling LLM
    # Phase 11.2: Pass query for topic relevance checking
    should_block, fallback_message, grounding_meta = apply_grounding_checks(
        chunks=chunks,
        answer=None,
        threshold=min_relevance_score,
        require_citations=False,
        query=query,  # For topic relevance check
    )
    
    retrieval_metadata["grounding"] = grounding_meta
    
    if should_block:
        retrieval_metadata["blocked"] = True
        retrieval_metadata["block_reason"] = grounding_meta.get("blocked_reason", "unknown")
        
        # Log blocked retrieval
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
                    status="blocked",  # Blocked due to low grounding
                    request_ip=request_ip,
                    request_user_agent=request_user_agent,
                )
            except Exception:
                pass
        
        retrieval_metadata["audit_logged"] = False
        
        if debug:
            retrieval_metadata["debug_info"] = get_debug_info(
                chunks, min_relevance_score, (should_block, fallback_message, grounding_meta)
            )
        
        return fallback_message, [], retrieval_metadata
    
    retrieval_metadata["blocked"] = False
    
    # Extract document and chunk IDs for audit
    retrieved_doc_ids = list(set(c.get("document_id") for c in chunks if c.get("document_id")))
    retrieved_chunk_ids = list(set(c.get("chunk_id") for c in chunks if c.get("chunk_id")))
    
    # Build prompt and generate answer - use temperature=0 for deterministic KB output
    prompt = build_rag_prompt(query, chunks)
    llm_request = ChatRequest(message=prompt)
    if hasattr(provider, 'set_temperature'):
        provider.set_temperature(0.0)
    llm_response = provider.chat(llm_request)
    answer = llm_response.message
    citations = format_citations(chunks)
    
    # Phase 10.6: Track citation repair flow explicitly
    citation_repair_meta = {
        "initial_has_citations": has_citations(answer),
        "retry_attempted": False,
        "retry_has_citations": False,
        "attachment_attempted": False,
        "attachment_has_citations": False,
        "final_has_citations": False,
        "final_citation_count": 0,
        "blocked_reason": None,
    }
    
    # Phase 10.6: Retry logic for missing citations with strong retrieval
    if not citation_repair_meta["initial_has_citations"]:
        top_score = chunks[0].get("score", 0) if chunks else 0
        if top_score >= min_relevance_score:
            # Retry once with strict citation prompt
            citation_repair_meta["retry_attempted"] = True
            strict_prompt = build_strict_citation_prompt(query, chunks)
            llm_request = ChatRequest(message=strict_prompt)
            if hasattr(provider, 'set_temperature'):
                provider.set_temperature(0.0)
            llm_response = provider.chat(llm_request)
            answer = llm_response.message
            citations = format_citations(chunks)
            citation_repair_meta["retry_has_citations"] = has_citations(answer)
    
    # Phase 10.6: Backend citation attachment if citations still missing but chunks are strong
    if not has_citations(answer) and chunks:
        top_score = chunks[0].get("score", 0) if chunks else 0
        if top_score >= min_relevance_score:
            # Try to attach citations based on content overlap
            citation_repair_meta["attachment_attempted"] = True
            answer_with_citations, citation_map = attach_citations_to_answer(
                answer, chunks, query
            )
            if citation_map:  # If we found matching content
                answer = answer_with_citations
                citation_repair_meta["attachment_has_citations"] = has_citations(answer)
                citation_repair_meta["citation_map"] = citation_map
                retrieval_metadata["grounding"]["backend_citations_attached"] = True
    
    # Phase 10.6: Final check
    final_has_citations = has_citations(answer)
    citation_repair_meta["final_has_citations"] = final_has_citations
    if final_has_citations:
        _, citation_count_meta = check_citations(answer)
        citation_repair_meta["final_citation_count"] = citation_count_meta.get("citation_count", 0)
    
    retrieval_metadata["grounding"]["citation_repair"] = citation_repair_meta
    retrieval_metadata["grounding"]["final_citation_count"] = citation_repair_meta["final_citation_count"]
    retrieval_metadata["grounding"]["final_has_citations"] = final_has_citations
    
    # Block only if citations still missing after ALL repair attempts
    if not final_has_citations:
        should_block, fallback_message, answer_grounding_meta = apply_grounding_checks(
            chunks=chunks,
            answer=answer,
            threshold=min_relevance_score,
            require_citations=True,
        )
        retrieval_metadata["grounding"].update(answer_grounding_meta)
        citation_repair_meta["blocked_reason"] = "answer_lacks_citations"
        retrieval_metadata["blocked"] = True
        retrieval_metadata["block_reason"] = "answer_lacks_citations"
        
        # Log blocked due to no citations
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
                    status="blocked_no_citations",
                    request_ip=request_ip,
                    request_user_agent=request_user_agent,
                )
            except Exception:
                pass
        
        retrieval_metadata["audit_logged"] = False
        
        if debug:
            retrieval_metadata["debug_info"] = get_debug_info(
                chunks, min_relevance_score, (should_block, fallback_message, answer_grounding_meta)
            )
        
        return fallback_message, [], retrieval_metadata
    
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
    
    # Add debug info if requested
    if debug:
        retrieval_metadata["debug_info"] = get_debug_info(
            chunks, min_relevance_score, (False, None, retrieval_metadata["grounding"])
        )
    
    return answer, citations, retrieval_metadata


def generate_answer_without_rag(query: str) -> str:
    """General chat without RAG - uses business-friendly prompt."""
    provider = get_llm_provider()
    prompt = build_general_chat_prompt(query)
    llm_request = ChatRequest(message=prompt)
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