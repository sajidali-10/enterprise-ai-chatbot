"""
Answer Grounding Module (Phase 10)

Provides functions to improve grounding, reduce hallucinations, and enforce citations:
- Retrieval guardrail: Don't call LLM if no chunks retrieved
- Minimum relevance threshold: Reject chunks below configurable threshold
- Citation enforcement: Verify answers include proper citations
- Answer grounding check: Validate answer is based on retrieved context

This module is designed to prevent the LLM from answering when there's insufficient
evidence, reducing hallucinations and improving answer quality.
"""

import re
from typing import Optional
from app.core.config import settings


# Default message when no chunks are retrieved
NO_CHUNKS_MESSAGE = "I could not find enough information in the provided sources to answer this question."

# Default message when chunks are below relevance threshold
LOW_RELEVANCE_MESSAGE = "I could not find enough relevant information in the provided sources to answer this question."

# Default message when answer lacks citations
NO_CITATIONS_MESSAGE = "I could not find enough information in the provided sources to answer this question."


def check_retrieval_guardrail(chunks: list[dict]) -> tuple[bool, Optional[str], dict]:
    """
    Check if retrieval returned any chunks.
    
    If no chunks are retrieved, returns a fallback message instead of calling LLM.
    
    Args:
        chunks: List of retrieved chunks.
        
    Returns:
        Tuple of (should_block, fallback_message, metadata).
    """
    if not chunks:
        return True, NO_CHUNKS_MESSAGE, {"blocked_reason": "no_chunks_retrieved", "chunk_count": 0}
    
    return False, None, {"chunk_count": len(chunks)}


def check_minimum_relevance(
    chunks: list[dict],
    threshold: Optional[float] = None,
) -> tuple[bool, Optional[str], dict]:
    """
    Check if top retrieved chunk meets minimum relevance threshold.
    
    If the top chunk's score is below the threshold, returns a fallback message.
    
    Args:
        chunks: List of retrieved chunks (must be sorted by score, descending).
        threshold: Minimum relevance score. Defaults to RAG_MIN_RELEVANCE_SCORE from settings.
        
    Returns:
        Tuple of (should_block, fallback_message, metadata).
    """
    if threshold is None:
        threshold = settings.RAG_MIN_RELEVANCE_SCORE
    
    if not chunks:
        # No chunks - handled by check_retrieval_guardrail
        return False, None, {"threshold_checked": False}
    
    top_score = chunks[0].get("score")
    if top_score is None:
        # No score available - allow through (score might be optional in some retrievers)
        return False, None, {"threshold_checked": False, "top_score": None}
    
    metadata = {
        "threshold_checked": True,
        "threshold_used": threshold,
        "top_score": top_score,
    }
    
    if top_score < threshold:
        return True, LOW_RELEVANCE_MESSAGE, metadata
    
    return False, None, metadata


def check_citations(answer: str) -> tuple[bool, dict]:
    """
    Check if answer includes citations.
    
    Looks for citation patterns like [1], [2], etc. in the answer.
    
    Args:
        answer: The generated answer text.
        
    Returns:
        Tuple of (has_citations, metadata).
    """
    # Pattern to match citations like [1], [2], [1,2], etc.
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    matches = re.findall(citation_pattern, answer)
    
    # Count unique citation numbers
    citation_numbers = set()
    for match in matches:
        for num in match.split(','):
            num = num.strip()
            if num:
                citation_numbers.add(int(num))
    
    citation_found = len(citation_numbers) > 0
    
    return citation_found, {
        "citation_count": len(citation_numbers),
        "citations_found": sorted(citation_numbers),
    }


def has_citations(answer: str) -> bool:
    """
    Check if answer includes any citation markers.
    
    This is a convenience function that wraps check_citations()
    and returns only the boolean result.
    
    Args:
        answer: The generated answer text.
        
    Returns:
        True if answer contains at least one citation marker [N].
    """
    citation_found, _ = check_citations(answer)
    return citation_found


def check_answer_grounding(
    answer: str,
    chunks: list[dict],
    require_citations: bool = True,
) -> tuple[bool, Optional[str], dict]:
    """
    Verify answer is based on retrieved context.
    
    This is a heuristic check that looks for:
    1. Presence of citations when required
    2. No indication of the model refusing to answer based on lack of info
    
    Args:
        answer: The generated answer text.
        chunks: The retrieved chunks used to generate the answer.
        require_citations: If True, citations are required for valid grounding.
        
    Returns:
        Tuple of (is_grounded, fallback_message, metadata).
    """
    metadata = {}
    
    # Check for citations if required
    if require_citations:
        has_citations, citation_meta = check_citations(answer)
        metadata.update(citation_meta)
        
        if not has_citations:
            return True, NO_CITATIONS_MESSAGE, metadata
    
    # Check for common "insufficient information" responses that the model might have
    # incorrectly generated (we already provide this message, but model might ignore it)
    insufficient_patterns = [
        r"i could not find enough information",
        r"i don't have enough information",
        r"the provided sources do not contain",
        r"not enough information to answer",
    ]
    
    answer_lower = answer.lower()
    for pattern in insufficient_patterns:
        if re.search(pattern, answer_lower):
            # Model says it couldn't find info - this could be valid or a hallucination
            # If we have chunks and citations, trust the model
            if chunks and require_citations:
                has_citations, _ = check_citations(answer)
                if has_citations:
                    # Model has citations but also says it couldn't find info - odd but let it through
                    pass
            # If no chunks or no citations, return fallback
            return True, NO_CITATIONS_MESSAGE, metadata
    
    return False, None, metadata


def apply_grounding_checks(
    chunks: list[dict],
    answer: Optional[str] = None,
    threshold: Optional[float] = None,
    require_citations: bool = True,
) -> tuple[bool, Optional[str], dict]:
    """
    Apply all grounding checks in sequence.
    
    Checks:
    1. Retrieval guardrail (no chunks)
    2. Minimum relevance threshold
    3. Answer grounding (if answer provided)
    
    Args:
        chunks: Retrieved chunks.
        answer: Generated answer (optional - if None, only retrieval checks run).
        threshold: Minimum relevance score.
        require_citations: Whether citations are required.
        
    Returns:
        Tuple of (should_block, fallback_message, metadata).
    """
    metadata = {}
    
    # Check 1: Retrieval guardrail
    should_block, message, guardrail_meta = check_retrieval_guardrail(chunks)
    metadata.update(guardrail_meta)
    if should_block:
        return True, message, metadata
    
    # Check 2: Minimum relevance threshold
    should_block, message, relevance_meta = check_minimum_relevance(chunks, threshold)
    metadata.update(relevance_meta)
    if should_block:
        return True, message, metadata
    
    # Check 3: Answer grounding (only if answer provided)
    if answer is not None:
        should_block, message, grounding_meta = check_answer_grounding(
            answer, chunks, require_citations
        )
        metadata.update(grounding_meta)
        if should_block:
            return True, message, metadata
    
    return False, None, metadata


def get_debug_info(
    chunks: list[dict],
    threshold: float,
    grounding_result: tuple[bool, Optional[str], dict],
) -> dict:
    """
    Generate debug information for development mode.
    
    Args:
        chunks: Retrieved chunks.
        threshold: Minimum relevance threshold used.
        grounding_result: Result from apply_grounding_checks.
        
    Returns:
        Dictionary with debug information.
    """
    _, _, metadata = grounding_result
    
    debug_info = {
        "retrieval": {
            "chunk_count": len(chunks),
            "top_score": chunks[0].get("score") if chunks else None,
            "source_files": list(set(c.get("source_file_name", "unknown") for c in chunks)),
        },
        "threshold": {
            "configured": threshold,
            "top_score_met": chunks[0].get("score", 0) >= threshold if chunks else False,
        },
        "grounding": {
            "blocked": grounding_result[0],
            "reason": grounding_result[1],
            "metadata": metadata,
        },
    }
    
    # Add per-chunk details
    if chunks:
        debug_info["retrieval"]["chunks"] = [
            {
                "index": i + 1,
                "score": c.get("score"),
                "source_file_name": c.get("source_file_name"),
                "content_preview": c.get("content", "")[:100] + "..." if c.get("content") else "",
            }
            for i, c in enumerate(chunks[:5])  # Top 5 chunks
        ]
    
    return debug_info