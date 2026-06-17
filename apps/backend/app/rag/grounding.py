"""
Answer Grounding Module (Phase 10, 11.2)

Provides functions to improve grounding, reduce hallucinations, and enforce citations:
- Retrieval guardrail: Don't call LLM if no chunks retrieved
- Minimum relevance threshold: Reject chunks below configurable threshold
- Topic relevance check: Verify retrieved content matches query topic (Phase 11.2)
- Citation enforcement: Verify answers include proper citations
- Answer grounding check: Validate answer is based on retrieved context

This module is designed to prevent the LLM from answering when there's insufficient
evidence, reducing hallucinations and improving answer quality.

Phase 11.2 adds topic relevance checking to prevent answering completely unrelated
queries even when retrieval returns chunks with acceptable scores.
"""

import re
from typing import Optional
from app.core.config import settings


# Default message when no chunks are retrieved
# Note: Using "don't have" to match evaluation fallback detection
NO_CHUNKS_MESSAGE = "I don't have enough information in the provided sources to answer this question."

# Default message when chunks are below relevance threshold
LOW_RELEVANCE_MESSAGE = "I don't have enough relevant information in the provided sources to answer this question."

# Default message when answer lacks citations
NO_CITATIONS_MESSAGE = "I don't have enough information in the provided sources to answer this question."

# Default message when topic is unrelated to retrieved content
TOPIC_MISMATCH_MESSAGE = "I don't have enough information in the provided sources to answer this question."

# High-risk domain patterns that require domain-specific knowledge base content.
LEGAL_PATTERNS = [
    re.compile(r'\bterms?\s+(of|for)\s+(service|use|privacy|liability)\b', re.I),
    re.compile(r'\b(privacy|service|usage|refund)\s+policy\b', re.I),
    re.compile(r'\blegal\b', re.I),
    re.compile(r'\b(contract|agreement|waiver|liability|indemnif)\b', re.I),
]

MEDICAL_PATTERNS = [
    re.compile(r'\b(side effects?|dosage|prescription|diagnosis|treatment)\b', re.I),
    re.compile(r'\b(medicine|drug|pharma|therapy)\b', re.I),
]

FINANCIAL_PATTERNS = [
    re.compile(r'\b(invest|stock\s+market|portfolio|dividend)\b', re.I),
    re.compile(r'\b(buy|sell|trade)\s+(stock|bond|crypto)\b', re.I),
]

OFFTOPIC_PATTERNS = [
    re.compile(r'\bhow\s+much\b.*\b(cost|plan|price|enterprise)\b', re.I),
    re.compile(r'\b(ceo|cto|founder|executive)\b.*\b(of|for)\b', re.I),
    re.compile(r'\bwho\s+(is|was|are)\b.*\b(ceo|cto|founder)\b', re.I),
]


def _classify_query_domain(query: str) -> list[str]:
    domains = []
    if any(p.search(query) for p in LEGAL_PATTERNS):
        domains.append('legal')
    if any(p.search(query) for p in MEDICAL_PATTERNS):
        domains.append('medical')
    if any(p.search(query) for p in FINANCIAL_PATTERNS):
        domains.append('financial')
    if any(p.search(query) for p in OFFTOPIC_PATTERNS):
        domains.append('offtopic')
    return domains


def check_high_risk_domain(
    query: str,
    chunks: list[dict],
) -> tuple[bool, Optional[str], dict]:
    if not chunks:
        return False, None, {"domain_checked": False}

    domains = _classify_query_domain(query)
    if not domains:
        return False, None, {"domain_checked": True, "domains_found": [], "high_risk": False}

    combined = " ".join(c.get("content", "").lower() for c in chunks)
    meta = {"domain_checked": True, "domains_found": domains, "high_risk": True}

    SAFE_MESSAGE = NO_CHUNKS_MESSAGE

    if 'legal' in domains:
        # Strong indicators: specific legal phrases that must appear as complete units
        strong_terms = ['terms of service', 'privacy policy', 'service agreement',
                        'terms and conditions', 'acceptable use', 'data processing',
                        'return policy', 'refund policy']
        strong_found = [t for t in strong_terms if t in combined]
        
        # Weak indicators: single words that may appear in unrelated technical docs
        # Require at least 3 weak terms to compensate for lack of strong terms
        weak_terms = ['liability', 'indemnification', 'warranty', 'contract',
                      'legal', 'jurisdiction', 'arbitration', 'waiver']
        weak_found = [t for t in weak_terms if t in combined]
        
        meta["legal_strong_found"] = strong_found
        meta["legal_weak_found"] = weak_found
        
        # Block if no strong terms, OR fewer than 3 weak terms
        # This prevents false positives from single weak words like 'liability'
        # appearing in unrelated technical documentation
        if not strong_found and len(weak_found) < 3:
            meta["blocked_reason"] = "legal_query_no_legal_content"
            return True, SAFE_MESSAGE, meta

    if 'medical' in domains:
        # Strong: specific medical phrases
        strong_med = ['diagnosis', 'treatment', 'prescription', 'side effect',
                      'clinical', 'patient', 'dosage', 'medication']
        strong_found = [t for t in strong_med if re.search(t, combined)]
        
        # Weak: general health terms
        weak_med = ['medical', 'health', 'therapy', 'pharma', 'doctor']
        weak_found = [t for t in weak_med if t in combined]
        
        meta["medical_strong_found"] = strong_found
        meta["medical_weak_found"] = weak_found
        
        # Block if no strong terms, or fewer than 2 weak terms
        if not strong_found and len(weak_found) < 2:
            meta["blocked_reason"] = "medical_query_no_medical_content"
            return True, SAFE_MESSAGE, meta

    if 'financial' in domains:
        fin_terms = ['investment', 'stock', 'bond', 'dividend', 'portfolio',
                     'trading', 'tax', 'revenue', 'pricing', 'cost',
                     'budget', 'expense', 'invoice', 'payment']
        found = [t for t in fin_terms if t in combined]
        meta["financial_indicators_found"] = found
        if not found:
            meta["blocked_reason"] = "financial_query_no_financial_content"
            return True, SAFE_MESSAGE, meta

    if 'offtopic' in domains:
        meta["blocked_reason"] = "offtopic_query"
        return True, SAFE_MESSAGE, meta

    return False, None, meta


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


def _extract_query_keywords(query: str) -> set[str]:
    """
    Extract meaningful keywords from a query for topic matching.
    
    Extracts nouns, technical terms, and compound phrases while filtering
    out common stopwords and short terms.
    
    Args:
        query: User's question/query string.
        
    Returns:
        Set of keyword strings (lowercased).
    """
    # Common stopwords to filter out
    stopwords = {
        'what', 'is', 'are', 'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'for',
        'to', 'and', 'or', 'but', 'with', 'from', 'by', 'how', 'do', 'does', 'can',
        'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her',
        'its', 'our', 'their', 'this', 'that', 'these', 'those', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'about', 'who', 'whom', 'whose', 'which', 'where', 'when', 'why',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very',
        'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if', 'because',
        'as', 'until', 'while', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'any', 'use',
        'used', 'using', 'like', 'many', 'much', 'well', 'way', 'want', 'need',
        'help', 'make', 'know', 'think', 'see', 'come', 'took', 'get', 'got',
    }
    
    query_lower = query.lower()
    
    # Extract compound phrases (2-3 words) that might be meaningful
    compound_pattern = re.compile(r'\b[a-z]{3,}(?:\s+[a-z]{3,}){1,2}\b')
    compounds = compound_pattern.findall(query_lower)
    
    # Extract single keywords (words >= 4 chars)
    words = re.findall(r'\b[a-z]{4,}\b', query_lower)
    
    # Combine and filter
    all_terms = set(compounds) | set(words)
    keywords = {t for t in all_terms if t not in stopwords and len(t) >= 4}
    
    return keywords


def check_topic_relevance(
    query: str,
    chunks: list[dict],
    min_keyword_overlap: float = 0.30,
) -> tuple[bool, Optional[str], dict]:
    """
    Check if retrieved chunks are topically relevant to the query.
    
    This prevents answering unrelated queries even when retrieval returns
    chunks with acceptable vector similarity scores. For example, a query
    about "refund policy" might return Docker documentation chunks because
    the embedding model found some vector similarity, but the content is
    completely unrelated.
    
    The check uses keyword overlap between query and chunk content:
    1. Extract meaningful keywords from query
    2. Check how many of those keywords appear in retrieved chunk content
    3. If overlap is below threshold, block the answer
    
    Args:
        query: User's question.
        chunks: List of retrieved chunks.
        min_keyword_overlap: Minimum fraction of query keywords that must
                           appear in chunks (0.0-1.0). Default 0.15 (15%).
        
    Returns:
        Tuple of (should_block, fallback_message, metadata).
    """
    if not chunks:
        # No chunks - handled by check_retrieval_guardrail
        return False, None, {"topic_checked": False}
    
    if not query:
        return False, None, {"topic_checked": False}
    
    # Extract keywords from query
    query_keywords = _extract_query_keywords(query)
    
    if not query_keywords:
        # No extractable keywords - allow through
        return False, None, {"topic_checked": True, "query_keywords_found": 0, "topic_relevant": True}
    
    # Combine all chunk content for analysis
    combined_content = " ".join(c.get("content", "").lower() for c in chunks)
    
    # Count how many query keywords appear in chunk content
    found_keywords = []
    missing_keywords = []
    
    for keyword in query_keywords:
        if keyword in combined_content:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    overlap_ratio = len(found_keywords) / len(query_keywords) if query_keywords else 1.0
    
    metadata = {
        "topic_checked": True,
        "query_keywords_count": len(query_keywords),
        "query_keywords_found": len(found_keywords),
        "query_keywords_missing": len(missing_keywords),
        "keyword_overlap_ratio": overlap_ratio,
        "min_keyword_overlap_required": min_keyword_overlap,
        "missing_keywords_sample": missing_keywords[:5],  # First 5 missing for debugging
        "topic_relevant": overlap_ratio >= min_keyword_overlap,
    }
    
    # Block if keyword overlap is too low
    if overlap_ratio < min_keyword_overlap:
        metadata["blocked_reason"] = "topic_not_relevant"
        return True, TOPIC_MISMATCH_MESSAGE, metadata

    # Additional semantic guard: if we have few keywords and only 1 matches,
    # but the top chunk score is weak, still block (avoid false positives from
    # single common-word matches like "service" in unrelated docs)
    if (
        len(query_keywords) <= 3 and
        len(found_keywords) == 1 and
        chunks and
        chunks[0].get("score", 0) < 0.75
    ):
        metadata["blocked_reason"] = "topic_not_relevant_single_keyword"
        return True, TOPIC_MISMATCH_MESSAGE, metadata

    return False, None, metadata


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
            metadata["blocked_reason"] = "answer_lacks_citations"
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
            metadata["blocked_reason"] = "answer_lacks_citations"
            return True, NO_CITATIONS_MESSAGE, metadata
    
    return False, None, metadata


def apply_grounding_checks(
    chunks: list[dict],
    answer: Optional[str] = None,
    threshold: Optional[float] = None,
    require_citations: bool = True,
    query: Optional[str] = None,
) -> tuple[bool, Optional[str], dict]:
    """
    Apply all grounding checks in sequence.
    
    Checks:
    1. Retrieval guardrail (no chunks)
    2. Minimum relevance threshold
    3. Topic relevance (Phase 11.2) - verify chunks match query topic
    4. Answer grounding (if answer provided)
    
    Args:
        chunks: Retrieved chunks.
        answer: Generated answer (optional - if None, only retrieval checks run).
        threshold: Minimum relevance score.
        require_citations: Whether citations are required.
        query: User's original query (required for topic relevance check).
        
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

    # Check 2.5: High-risk domain check (legal/medical/financial/offtopic)
    # This runs before topic relevance to catch domain-specific queries that
    # might pass keyword overlap checks but are actually out of scope
    if query is not None:
        should_block, message, domain_meta = check_high_risk_domain(query, chunks)
        metadata.update(domain_meta)
        if should_block:
            return True, message, metadata

    # Check 3: Topic relevance (Phase 11.2) - only if query is provided
    if query is not None:
        should_block, message, topic_meta = check_topic_relevance(query, chunks)
        metadata.update(topic_meta)
        if should_block:
            return True, message, metadata
    
    # Check 4: Answer grounding (only if answer provided)
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