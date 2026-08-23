import re
from typing import List, Dict, Any, Optional, Set

# Phase 31A — LangSmith tracing for citation processing.
try:
    from app.services.langsmith_tracing import (
        trace_span,
        redact_filenames,
        redact_path,
    )
except Exception:  # pragma: no cover - tracing never required
    from contextlib import contextmanager

    @contextmanager
    def trace_span(*args, **kwargs):
        yield None

    def redact_filenames(value):
        return list(value or [])

    def redact_path(value):
        return value or ""


def get_confidence_label(score: Optional[float]) -> str:
    """
    Convert a relevance score to a confidence label.
    
    Args:
        score: The relevance score (0-1 range)
        
    Returns:
        Confidence label: 'High', 'Medium', or 'Low'
    """
    if score is None:
        return "Low"
    if score >= 0.80:
        return "High"
    elif score >= 0.60:
        return "Medium"
    else:
        return "Low"


def clean_excerpt(content: str, max_length: int = 200) -> str:
    """
    Clean and truncate content to a readable excerpt.
    
    Uses sentence boundary to avoid mid-word truncation when possible.
    Removes noisy prefixes and broken fragments.
    
    Args:
        content: The content string to clean
        max_length: Maximum length before truncation (default: 200)
        
    Returns:
        Cleaned excerpt at or near max_length, ending at sentence boundary if possible
    """
    if not content:
        return ""
    
    # Clean up whitespace
    content = content.strip()
    content = re.sub(r'\s+', ' ', content)
    
    # Remove common noisy prefixes
    noisy_prefixes = [
        r'^\d+\s*[.\)]\s*',  # Numbered list markers like "1. " or "1) "
        r'^[-*•]\s*',  # Bullet markers
        r'^(?:Source|File|Document):\s*\S+\s*',  # Source/file headers
        r'^Chapter \d+[.:]?\s*',  # Chapter headers
        r'^Section \d+[.:]?\s*',  # Section headers
    ]
    
    for prefix in noisy_prefixes:
        content = re.sub(prefix, '', content, flags=re.IGNORECASE)
    
    # Strip again after removing prefixes
    content = content.strip()
    
    if len(content) <= max_length:
        return content
    
    # Try to truncate at a sentence boundary
    truncated = content[:max_length]
    
    # Look for sentence-ending punctuation followed by space or end
    sentence_end_match = re.search(r'[.!?。](?:\s|$)', truncated[::-1])
    if sentence_end_match:
        end_pos = max_length - sentence_end_match.start() - 1
        if end_pos > max_length * 0.6:  # Only use if we keep at least 60% of max
            return content[:end_pos + 1].strip()
    
    # Try to truncate at a word boundary if no good sentence boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.7:  # Only use if we keep at least 70% of max
        return content[:last_space].rstrip() + "..."
    
    # Fallback: hard truncate with ellipsis
    return content[:max_length].rstrip() + "..."


def extract_keywords(text: str) -> Set[str]:
    """
    Extract significant keywords from text for relevance scoring.
    
    Args:
        text: Input text to extract keywords from
        
    Returns:
        Set of lowercase keywords (3+ characters, not common stopwords)
    """
    if not text:
        return set()
    
    # Common stopwords to exclude (keep meaningful adjectives like 'good')
    stopwords = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
        'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'with', 'have',
        'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been',
        'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like',
        'long', 'many', 'over', 'such', 'take', 'than', 'them', 'well',
        'were', 'what', 'would', 'there', 'their', 'said', 'each', 'which',
        'however', 'therefore', 'because', 'although', 'though', 'since',
        'while', 'where', 'after', 'before', 'during', 'about', 'into',
        'through', 'above', 'below', 'between', 'among', 'within', 'without',
        'could', 'should', 'might', 'must', 'shall', 'may',
    }
    
    # Extract words (3+ characters, letters only)
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    
    # Filter out stopwords and return as set
    return {w for w in words if w not in stopwords}


def score_excerpt_relevance(
    excerpt: str,
    relevance_score: float,
    question: str,
    answer: str
) -> float:
    """
    Score excerpt relevance based on multiple factors.
    
    Penalizes generic introductory content and boosts content that
    directly answers the question with specific details.
    
    Args:
        excerpt: The excerpt text
        relevance_score: Original relevance score from retrieval
        question: User's question
        answer: Generated answer
        
    Returns:
        Combined relevance score (higher is better)
    """
    if not excerpt:
        return 0.0

    # Phase 34A.1.2 — by-id chunks (resolved image_content sources)
    # carry ``score=None`` because they were fetched by payload filter
    # rather than semantic similarity. Treat None as a high
    # relevance baseline so the citation-grouping logic does not
    # crash and so the resolved image source ranks above any
    # incidental low-score KB noise.
    if relevance_score is None:
        relevance_score = 1.0

    # Base score from retrieval
    score = relevance_score * 0.4
    
    # Keyword overlap with question
    excerpt_keywords = extract_keywords(excerpt)
    question_keywords = extract_keywords(question)
    answer_keywords = extract_keywords(answer)
    
    if question_keywords:
        question_overlap = len(excerpt_keywords & question_keywords) / len(question_keywords)
        score += question_overlap * 0.30
    
    # Keyword overlap with answer (answer-aware selection)
    if answer_keywords:
        answer_overlap = len(excerpt_keywords & answer_keywords) / len(answer_keywords)
        score += answer_overlap * 0.30
    
    excerpt_lower = excerpt.lower()
    
    # Strong penalty for generic introductory content
    generic_patterns = [
        (r'^\s*what is\b.*?\?', 0.5),  # "What is X?" at start
        (r'^\s*why (do|does|did|would|should)\b.*?\?', 0.5),  # "Why do we need..."
        (r'\b(problem|solution|introduction|overview|background)\s*[:\-]?\s*$', 0.6),
        (r'\bthis (document|guide|article|paper|tutorial)\s+(is|explains|describes|covers)\b', 0.6),
        (r'\bin this\s+(chapter|section|document|guide|article)\b', 0.6),
        (r'\bconsistency across environments\b', 0.6),  # Common generic Docker phrase
        (r'\bplatform designed to help\b', 0.6),  # Generic description language
        # CI/CD and deployment content - heavily penalize when asking about components
        (r'\bcontinuous integration\b', 0.4),
        (r'\bcontinuous deployment\b', 0.4),
        (r'\bci/cd\b', 0.4),
        (r'\bdevops\b', 0.4),
        (r'\bdeveloped,?\s*updated,?\s*deployed\b', 0.35),
        (r'\bdeployed independently\b', 0.35),
        (r'\bservice can be\b', 0.45),
        (r'\beach service can\b', 0.45),
        (r'\bupdated and deployed\b', 0.4),
        (r'\bdeveloped and deployed\b', 0.4),
        (r'\bindependent(ly)?\s+(deploy|update|develop)\b', 0.45),
    ]
    
    for pattern, penalty in generic_patterns:
        if re.search(pattern, excerpt_lower):
            score *= penalty
            break  # Apply strongest matching penalty
    
    return score


def select_best_excerpts_for_source(
    citations: List[dict],
    question: str,
    answer: str,
    max_excerpts: int = 3
) -> List[dict]:
    """
    Select the most relevant excerpts for a source document.
    
    Uses answer-aware selection to prefer excerpts that directly support
    the answer claims and match the user's question.
    
    Args:
        citations: Citations from the same source document
        question: User's original question
        answer: Generated answer text
        max_excerpts: Maximum number of excerpts to return (default: 3)
        
    Returns:
        List of top citations sorted by relevance
    """
    if not citations:
        return []
    
    # Score each citation
    scored_citations = []
    for citation in citations:
        content = citation.get("content_snippet", "")
        relevance = citation.get("relevance_score", 0.5)
        
        combined_score = score_excerpt_relevance(
            content, relevance, question, answer
        )
        
        scored_citations.append((combined_score, citation))
    
    # Sort by combined score descending
    scored_citations.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate similar excerpts (simple similarity check)
    selected = []
    seen_content = set()
    seen_normalized = set()
    
    for score, citation in scored_citations:
        if len(selected) >= max_excerpts:
            break
        
        content = citation.get("content_snippet", "")
        if not content:
            continue
        
        # Simple dedup: check if we've seen similar content
        content_lower = content.lower().strip()
        
        # Create normalized version for better comparison (remove punctuation)
        content_normalized = re.sub(r'[^\w\s]', '', content_lower)
        
        is_duplicate = False
        
        # Check against all previously seen content
        for seen_norm in seen_normalized:
            # If normalized content is very similar
            if content_normalized in seen_norm or seen_norm in content_normalized:
                is_duplicate = True
                break
            
            # Check word overlap on normalized content
            content_words = set(content_normalized.split())
            seen_words = set(seen_norm.split())
            if content_words and seen_words:
                overlap = len(content_words & seen_words) / max(len(content_words), len(seen_words))
                if overlap > 0.85:  # Higher threshold for stricter dedup
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            selected.append(citation)
            seen_content.add(content_lower)
            seen_normalized.add(content_normalized)
    
    return selected


def format_citations(chunks: list[dict]) -> List[dict]:
    """
    Format retrieved chunks as citations for the response.
    Returns list of citation dicts with index, source, and content snippet.
    """
    # Phase 31A — wrap citation formatting in a span. We track the number
    # of citations and the unique source filenames but never the content
    # text itself unless LANGSMITH_LOG_RETRIEVED_CONTEXT is on.
    with trace_span(
        "citation_processing",
        metadata={
            "phase": "citation_processing",
            "step": "format",
            "input_chunk_count": len(chunks or []),
        },
    ) as cite_span:
        citations = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            citation = {
                "index": i,
                "source_file_name": chunk.get("source_file_name", "Unknown"),
                "content_snippet": clean_excerpt(content),
                "relevance_score": chunk.get("score"),
                "chunk_id": chunk.get("id"),
            }
            # Phase 34C -- Persistent Multimodal Knowledge. Image
            # knowledge citations carry extra fields so the citation
            # layer can distinguish them from KB / OCR chunks and so
            # the response renderer can show the original filename
            # and image metadata. We do NOT change the [N] marker
            # format; the distinction lives in the structured fields
            # of each citation dict.
            if (chunk.get("source_type") or chunk.get("content_type")) == "image_knowledge":
                citation.update({
                    "citation_kind": "image_knowledge",
                    "image_id": chunk.get("image_id"),
                    "image_type": chunk.get("image_type") or chunk.get("mime_type"),
                    "vision_provider": chunk.get("vision_provider"),
                    "vision_model": chunk.get("vision_model"),
                    "has_vision": bool(chunk.get("has_vision")),
                    "knowledge_schema_version": chunk.get("knowledge_schema_version"),
                    # Prefer the (shorter, human-readable) vision
                    # description as the snippet when present. Fall
                    # back to a clipped knowledge_text excerpt so the
                    # snippet stays bounded either way.
                    "content_snippet": _format_image_knowledge_snippet(chunk),
                    "document_id": chunk.get("document_id"),
                })
            citations.append(citation)
        if cite_span is not None:
            try:
                cite_span.set_meta("citation_count", len(citations))
                cite_span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list({c.get("source_file_name") for c in citations if c.get("source_file_name")})
                    ),
                )
            except Exception:
                pass
        return citations


def _format_image_knowledge_snippet(chunk: dict, max_length: int = 240) -> str:
    """Pick the best human-readable excerpt for an image knowledge citation.

    Prefers the persisted vision description when present (the
    Phase 34B output). Falls back to the knowledge_text excerpt.
    Never logs OCR or vision raw bodies elsewhere in the pipeline.
    """
    # The retriever embeds the knowledge text on ``content``; the
    # vision description is not separately carried on the chunk dict
    # at this point in the pipeline. We use the knowledge_text as the
    # primary source and clip it for readability.
    raw = chunk.get("content") or chunk.get("knowledge_text") or ""
    if not raw:
        return ""
    return clean_excerpt(raw, max_length=max_length)


def attach_citations_to_answer(
    answer: str,
    chunks: List[dict],
    query: str,
) -> tuple[str, dict]:
    """
    Attempt to attach citations to an answer that lacks them.

    Uses content overlap analysis between the answer and retrieved chunks
    to identify which source chunks support which parts of the answer.

    Args:
        answer: The generated answer text (may lack citations).
        chunks: Retrieved context chunks.
        query: User's original question.

    Returns:
        Tuple of (modified_answer, citation_map) where:
        - modified_answer: Answer with [N] citation markers added
        - citation_map: Dict mapping citation indices to source info
    """
    if not answer or not chunks:
        return answer, {}

    # Phase 31A — separate span for the citation-attachment fallback path
    # so the LangSmith trace shows when the system had to add citations
    # because the model omitted them.
    with trace_span(
        "citation_processing",
        metadata={
            "phase": "citation_processing",
            "step": "attach",
            "input_chunk_count": len(chunks or []),
        },
    ) as attach_span:
        modified_answer, citation_map = _attach_citations_to_answer_impl(
            answer=answer, chunks=chunks, query=query
        )
        if attach_span is not None:
            try:
                attach_span.set_meta("citation_count", len(citation_map))
                attach_span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list(
                            {
                                redact_path(c.get("source_file_name"))
                                for c in chunks
                                if c.get("source_file_name")
                            }
                        )
                    ),
                )
            except Exception:
                pass
        return modified_answer, citation_map


def _attach_citations_to_answer_impl(
    answer: str,
    chunks: List[dict],
    query: str,
) -> tuple[str, dict]:
    
    answer_lower = answer.lower()
    citation_map = {}
    modified_answer = answer
    
    # Sort chunks by score descending
    sorted_chunks = sorted(enumerate(chunks, 1), key=lambda x: x[1].get("score", 0), reverse=True)
    
    for idx, chunk in sorted_chunks:
        content = chunk.get("content", "").lower()
        if not content:
            continue
        
        # Extract significant keywords/phrases from chunk (5+ char words)
        chunk_terms = set(re.findall(r'\b[a-z]{5,}\b', content))
        
        if not chunk_terms:
            continue
        
        # Find terms that appear in both chunk and answer
        answer_terms = set(re.findall(r'\b[a-z]{5,}\b', answer_lower))
        matching_terms = chunk_terms & answer_terms
        
        # If we have substantial overlap (at least 3 matching terms)
        if len(matching_terms) >= 3:
            citation_map[idx] = {
                "source_file_name": chunk.get("source_file_name", "Unknown"),
                "chunk_id": chunk.get("id"),
                "matching_terms": list(matching_terms)[:10],  # Limit for readability
            }
            
            # Try to add citation after sentences that contain matching terms
            # Find sentences in the answer that contain these terms
            sentences = re.split(r'(?<=[.!?])\s+', modified_answer)
            new_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                sentence_has_match = any(term in sentence_lower for term in matching_terms)
                # Check if this specific citation index is already present
                sentence_has_this_citation = re.search(rf'\[{idx}(?:,|\s|\])', sentence)
                
                if sentence_has_match and not sentence_has_this_citation:
                    # Append citation at the end of sentence
                    sentence = sentence.rstrip()
                    # Check if sentence already has other citations - append to them
                    existing_citations = re.search(r'\[(\d+(?:,\s*\d+)*)\]', sentence)
                    if existing_citations:
                        # Append this citation to existing bracketed list
                        # Period is preserved because we check existing_citations before stripping punctuation
                        old_list = existing_citations.group(1)
                        new_list = f"{old_list}, {idx}"
                        sentence = sentence[:existing_citations.start()] + '[' + new_list + ']' + sentence[existing_citations.end():]
                    else:
                        # No existing citations - strip punctuation then add new citation with period
                        if sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?'):
                            sentence = sentence[:-1]  # Remove trailing punctuation
                        sentence = f"{sentence} [{idx}]."
                
                new_sentences.append(sentence)
            
            modified_answer = " ".join(new_sentences)
    
    return modified_answer, citation_map


def group_citations_by_source(
    citations: List[dict],
    question: str = "",
    answer: str = "",
    max_excerpts: int = 3,
    debug_mode: bool = False
) -> List[dict]:
    """
    Group citations by source_file_name, combining multiple chunks from the same document.
    
    Uses answer-aware excerpt selection to show only the most relevant excerpts
    for the user's question and answer.
    
    Args:
        citations: List of citation dicts from format_citations()
        question: User's original question (for relevance scoring)
        answer: Generated answer text (for relevance scoring)
        max_excerpts: Maximum number of excerpts to include per document (default: 3)
        debug_mode: If True, include debug details in output (default: False)
        
    Returns:
        List of grouped source dicts with:
        - source_file_name: Document name
        - sections_used: Number of chunks from this document
        - highest_score: Maximum relevance score
        - confidence: High/Medium/Low label
        - excerpts: List of selected relevant excerpts
        - indices: Original citation indices (debug mode only)
        - show_debug_details: Flag indicating if debug details should be shown
    """
    # Group citations by source
    source_citations: Dict[str, List[dict]] = {}
    
    for citation in citations:
        source_name = citation.get("source_file_name", "Unknown")
        if source_name not in source_citations:
            source_citations[source_name] = []
        source_citations[source_name].append(citation)
    
    # Build grouped sources with answer-aware excerpt selection
    result = []
    
    for source_name, source_cits in source_citations.items():
        # Calculate aggregate stats
        scores = [c.get("relevance_score", 0) for c in source_cits if c.get("relevance_score") is not None]
        highest_score = max(scores) if scores else 0.0
        
        # Get all indices for potential display
        all_indices = [c["index"] for c in source_cits if "index" in c]
        
        # Select best excerpts using answer-aware selection
        best_citations = select_best_excerpts_for_source(
            source_cits, question, answer, max_excerpts
        )
        
        # Extract and clean excerpts
        excerpts = [c.get("content_snippet", "") for c in best_citations if c.get("content_snippet")]
        
        # Build the source dict
        # Always include all fields (for API compatibility), but indicate debug mode
        source_dict = {
            "source_file_name": source_name,
            "sections_used": len(source_cits),  # Total sections from this doc
            "highest_score": highest_score,  # Always include for sorting
            "confidence": get_confidence_label(highest_score),
            "excerpts": excerpts,
            "indices": all_indices if debug_mode else [],  # Empty list in non-debug
            "show_debug_details": debug_mode,  # Flag for frontend
        }

        result.append(source_dict)

    # Sort by highest score descending
    result.sort(key=lambda x: x["highest_score"], reverse=True)

    # Phase 31A — emit one citation_processing span for the grouping step.
    # This keeps the trace hierarchy clean: the parent `citation_processing`
    # for format/attach stays as a child span, and grouping is its own.
    with trace_span(
        "citation_processing",
        metadata={
            "phase": "citation_processing",
            "step": "group",
            "input_citation_count": len(citations or []),
            "grouped_source_count": len(result),
        },
    ) as group_span:
        if group_span is not None:
            try:
                group_span.set_meta("citation_count", len(citations or []))
                group_span.set_meta("grouped_source_count", len(result))
                group_span.set_meta(
                    "source_file_names",
                    redact_filenames([g.get("source_file_name") for g in result]),
                )
            except Exception:
                pass
        return result
