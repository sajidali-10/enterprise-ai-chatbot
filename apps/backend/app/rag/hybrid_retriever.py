"""
Hybrid Retrieval Service

Combines vector search (Qdrant) and keyword search (PostgreSQL)
with configurable fusion and reranking.

Phase 30E additions:
- Generic keyword/phrase/number/acronym boost scoring
- MMR-style diversity filtering
- Retrieval strategy dispatch (similarity | hybrid | mmr | hybrid_mmr)
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from app.services.vector.qdrant_service import search as qdrant_search
from app.services.search.keyword_search import search_chunks_keyword
from app.services.embeddings import get_embedding_provider
from app.rag.reranker import get_reranker, RerankerBase, RerankResult
from app.rag.exact_match import apply_exact_match_boost as _exact_match_boost_helper
from app.services.langsmith_tracing import trace_span


# ============================================================================
# Generic query analysis (document-agnostic; no hardcoded domain terms)
# ============================================================================

# Common stopwords excluded from generic keyword matching.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "these", "those",
    "are", "was", "were", "been", "have", "has", "had", "does", "did",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "you", "your", "they", "their", "them", "our", "ours",
    "about", "into", "onto", "upon", "than", "then", "also", "just",
    "some", "any", "all", "each", "every", "both", "few", "more", "most",
    "such", "not", "only", "same", "very", "much", "many",
})


def _extract_query_signals(query: str) -> dict:
    """
    Extract generic, document-agnostic signals from a query.

    Returns a dict with:
      - phrases: list of quoted phrases (2+ word substrings)
      - numbers: list of numeric tokens
      - acronyms: list of uppercase acronyms (>= 2 chars)
      - words: list of meaningful lowercase words (>= 4 chars, not stopwords)
    """
    if not query:
        return {"phrases": [], "numbers": [], "acronyms": [], "words": []}

    phrases = re.findall(r'"([^"]+)"', query) or re.findall(r"'([^']+)'", query)
    # Also capture 2-3 word lowercase phrases as generic phrase candidates
    bigrams = re.findall(r"\b[a-zA-Z]{3,}(?:\s+[a-zA-Z]{3,})\b", query)
    phrases = list({p.strip().lower() for p in phrases if p.strip()}) + [
        b.lower() for b in bigrams if 5 <= len(b) <= 60
    ]

    numbers = re.findall(r"\b\d+(?:[.,]\d+)*\b", query)

    acronyms = re.findall(r"\b[A-Z]{2,}\b", query)

    raw_words = re.findall(r"\b[a-zA-Z]{4,}\b", query.lower())
    words = [w for w in raw_words if w not in _STOPWORDS]

    # Deduplicate while preserving order
    seen = set()
    dedup_words = []
    for w in words:
        if w not in seen:
            seen.add(w)
            dedup_words.append(w)

    return {
        "phrases": list(dict.fromkeys(phrases)),
        "numbers": list(dict.fromkeys(numbers)),
        "acronyms": list(dict.fromkeys(acronyms)),
        "words": dedup_words,
    }


def _generic_keyword_score(content: str, title: str, signals: dict) -> float:
    """
    Compute a generic keyword/phrase/number/acronym match score for a chunk.

    Returns a float in [0.0, 1.0+] roughly proportional to how many distinct
    query signals appear in the chunk content or title. This is document-agnostic.

    IMPORTANT: This score is used purely for *ranking*. It is NOT used to
    decide whether a chunk actually answers a question. The grounding layer
    (Phase 30E Hotfix v2) independently checks lexical anchors via
    `decide_evidence_level()` and `RAG_MIN_KEYWORD_SCORE`.
    """
    if not content:
        return 0.0

    content_lower = content.lower()
    title_lower = (title or "").lower()
    haystack = content_lower + "\n" + title_lower

    score = 0.0
    matched = 0
    total_signals = 0

    # Exact word matches (highest weight among text matches)
    if signals["words"]:
        total_signals += len(signals["words"])
        for w in signals["words"]:
            if w in haystack:
                matched += 1
                score += 1.0
                # Boost if word appears in title (metadata match)
                if w in title_lower:
                    score += 0.5

    # Phrase matches (strongest signal)
    if signals["phrases"]:
        total_signals += len(signals["phrases"])
        for p in signals["phrases"]:
            if p in haystack:
                matched += 1
                score += 2.0

    # Number matches
    if signals["numbers"]:
        total_signals += len(signals["numbers"])
        for n in signals["numbers"]:
            if n in content:
                matched += 1
                score += 1.5

    # Acronym matches (case-sensitive in original content)
    if signals["acronyms"]:
        total_signals += len(signals["acronyms"])
        for a in signals["acronyms"]:
            # Acronyms are uppercase; match against original-case content
            if a in content or a.lower() in haystack:
                matched += 1
                score += 1.5

    if total_signals == 0:
        return 0.0

    # Normalize by total signals so score is in [0, ~2.0] then clamp to [0, 1]
    coverage = matched / total_signals
    raw = (score / max(total_signals, 1)) * coverage
    return max(0.0, min(1.0, raw))


def compute_keyword_ranking_score(content: str, title: str, query: str) -> float:
    """
    Public helper that computes the keyword ranking score for a chunk against
    a query string. This is a thin wrapper around `_extract_query_signals` and
    `_generic_keyword_score`. It is for *ranking only* — answerability is
    decided by the grounding layer.
    """
    signals = _extract_query_signals(query)
    return _generic_keyword_score(content, title, signals)


def _boost_chunks_with_keyword_signals(
    chunks: list[dict],
    query: str,
    keyword_weight: float = 0.3,
) -> list[dict]:
    """
    Apply generic keyword/phrase/number/acronym boost to chunk scores.

    The boost is additive on top of the existing fused score, scaled by
    `keyword_weight`. Returns a new list with updated `score` and adds
    `_keyword_score` and `_signal_matches` fields for debugging.
    """
    signals = _extract_query_signals(query)
    has_signals = any(signals[k] for k in ("phrases", "numbers", "acronyms", "words"))
    if not has_signals:
        # Nothing to boost; return chunks unchanged
        return chunks

    boosted = []
    for chunk in chunks:
        content = chunk.get("content", "")
        title = chunk.get("title", "")
        kw_score = _generic_keyword_score(content, title, signals)
        base = float(chunk.get("score") or 0.0)
        new_score = base * (1.0 - keyword_weight) + kw_score * keyword_weight
        new_chunk = dict(chunk)
        new_chunk["score"] = new_score
        new_chunk["_keyword_score"] = round(kw_score, 4)
        new_chunk["_signal_matches"] = {
            "phrases": [p for p in signals["phrases"] if p in (content.lower() + " " + (title or "").lower())],
            "numbers": [n for n in signals["numbers"] if n in content],
            "acronyms": [a for a in signals["acronyms"] if a in content or a.lower() in (content.lower())],
        }
        boosted.append(new_chunk)

    boosted.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return boosted


# ============================================================================
# MMR-style diversity filtering (generic, document-agnostic)
# ============================================================================

def _chunk_text_signature(chunk: dict, max_chars: int = 200) -> str:
    """Return a normalized text signature for similarity comparison."""
    content = (chunk.get("content") or "").lower()
    content = re.sub(r"\s+", " ", content).strip()
    return content[:max_chars]


def _jaccard(a: str, b: str) -> float:
    """Compute Jaccard word-set similarity between two strings."""
    if not a or not b:
        return 0.0
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    inter = wa & wb
    union = wa | wb
    return len(inter) / len(union)


def mmr_diversify(
    chunks: list[dict],
    lambda_param: float = 0.7,
    top_n: Optional[int] = None,
    same_doc_penalty: float = 0.1,
) -> list[dict]:
    """
    Apply MMR-style diversity filtering on a list of scored chunks.

    MMR formula (generic, no domain assumptions):
        score_mmr = lambda * relevance - (1 - lambda) * max_similarity_to_selected

    Where:
        - relevance is the chunk's existing score (assumed higher = better)
        - similarity is Jaccard word overlap between chunk text signatures
        - chunks from the same document as already-selected chunks get a small
          extra penalty (configurable; default 0.1) to encourage cross-document diversity

    The function never removes the highest-scoring chunk (so the single best source
    is always retained). It preserves source metadata on returned chunks.

    Args:
        chunks: List of chunk dicts (assumed sorted by score desc).
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
        top_n: If set, return at most this many chunks.
        same_doc_penalty: Extra penalty applied to chunks from already-selected docs.

    Returns:
        New list of chunks, diversified, sorted by MMR score.
    """
    if not chunks:
        return []

    # Defensive copy and normalize lambda
    lambda_param = max(0.0, min(1.0, float(lambda_param)))

    selected: list[dict] = []
    selected_signatures: list[str] = []
    selected_doc_ids: set = set()
    candidates = list(chunks)  # do not mutate input

    # Always keep the top-scoring chunk to preserve the strongest source
    if candidates:
        first = candidates.pop(0)
        selected.append(first)
        selected_signatures.append(_chunk_text_signature(first))
        if first.get("document_id"):
            selected_doc_ids.add(str(first.get("document_id")))

    while candidates:
        best_idx = -1
        best_mmr = float("-inf")
        for i, chunk in enumerate(candidates):
            relevance = float(chunk.get("score") or 0.0)
            sig = _chunk_text_signature(chunk)
            if selected_signatures:
                max_sim = max(_jaccard(sig, s) for s in selected_signatures)
            else:
                max_sim = 0.0
            penalty = 0.0
            if chunk.get("document_id") and str(chunk.get("document_id")) in selected_doc_ids:
                penalty = float(same_doc_penalty)
            mmr = lambda_param * relevance - (1.0 - lambda_param) * max_sim - penalty
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx < 0:
            break
        chosen = candidates.pop(best_idx)
        selected.append(chosen)
        selected_signatures.append(_chunk_text_signature(chosen))
        if chosen.get("document_id"):
            selected_doc_ids.add(str(chosen.get("document_id")))

        if top_n is not None and len(selected) >= top_n:
            break

    return selected


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    vector_top_k: int = 10
    keyword_top_k: int = 10
    final_top_k: int = 5
    min_score: float = 0.0
    reranker_type: str = "noop"
    rerank_final_k: Optional[int] = None
    keyword_weight: float = 0.3
    vector_weight: float = 0.7


@dataclass
class ScoredChunk:
    """A chunk with its retrieval score from hybrid fusion."""
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    source_file_name: str
    title: str
    vector_score: Optional[float]
    keyword_score: Optional[float]
    fused_score: float
    is_from_vector: bool
    is_from_keyword: bool


def _normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max normalize scores to [0, 1] range.
    Handles empty lists and constant scores.
    """
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _fuse_scores(
    vector_results: List[dict],
    keyword_results: List[dict],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[ScoredChunk]:
    """
    Fuse vector and keyword search results using weighted score fusion.
    
    Uses reciprocal rank fusion (RRF) as a fallback/complement to score fusion.
    """
    # Build lookup maps
    chunk_map: dict[str, ScoredChunk] = {}
    
    # Process vector results
    vector_scores = [r.get("score", 0.0) for r in vector_results]
    normalized_vector = _normalize_scores(vector_scores)
    
    for i, result in enumerate(vector_results):
        chunk_id = str(result.get("chunk_id", f"v_{i}"))
        norm_score = normalized_vector[i] if i < len(normalized_vector) else 0.0
        chunk_map[chunk_id] = ScoredChunk(
            chunk_id=chunk_id,
            document_id=str(result.get("document_id", "")),
            chunk_index=result.get("chunk_index", 0),
            content=result.get("content", ""),
            source_file_name=result.get("source_file_name", ""),
            title=result.get("title", ""),
            vector_score=norm_score,
            keyword_score=None,
            fused_score=norm_score * vector_weight,
            is_from_vector=True,
            is_from_keyword=False,
        )
    
    # Process keyword results
    keyword_scores = [r.get("score", 0.0) for r in keyword_results]
    normalized_keyword = _normalize_scores(keyword_scores)
    
    for i, result in enumerate(keyword_results):
        chunk_id = str(result.get("chunk_id", f"k_{i}"))
        norm_score = normalized_keyword[i] if i < len(normalized_keyword) else 0.0
        
        if chunk_id in chunk_map:
            # Chunk already exists from vector search
            existing = chunk_map[chunk_id]
            existing.keyword_score = norm_score
            existing.fused_score = (
                (existing.vector_score or 0.0) * vector_weight +
                norm_score * keyword_weight
            )
            existing.is_from_keyword = True
        else:
            chunk_map[chunk_id] = ScoredChunk(
                chunk_id=chunk_id,
                document_id=str(result.get("document_id", "")),
                chunk_index=result.get("chunk_index", 0),
                content=result.get("content", ""),
                source_file_name=result.get("source_file_name", ""),
                title=result.get("title", ""),
                vector_score=None,
                keyword_score=norm_score,
                fused_score=norm_score * keyword_weight,
                is_from_vector=False,
                is_from_keyword=True,
            )
    
    # Sort by fused score
    sorted_chunks = sorted(chunk_map.values(), key=lambda x: x.fused_score, reverse=True)
    return sorted_chunks


def retrieve_chunks_hybrid(
    query: str,
    config: Optional[RetrievalConfig] = None,
    reranker: Optional[RerankerBase] = None,
    query_analysis: Optional[object] = None,
    exact_match_boost: float = 0.30,
    exact_term_boost: float = 0.15,
) -> Tuple[List[dict], dict]:
    """
    Perform hybrid retrieval combining vector and keyword search.
    
    Args:
        query: User's search query.
        config: Retrieval configuration with top_k, weights, and reranker settings.
        reranker: Optional reranker to apply after fusion. If None, uses config.reranker_type.
        
    Returns:
        Tuple of (list of chunk dicts, retrieval metadata dict with scores info).
    """
    if config is None:
        config = RetrievalConfig()

    if reranker is None:
        reranker = get_reranker(config.reranker_type)

    # Phase 31A — wrap the three retrieval sub-steps in their own spans
    # so the parent `rag_retrieval` span shows a clean hierarchy:
    #     rag_retrieval
    #         qdrant_vector_search
    #         hybrid_keyword_scoring
    #         context_selection
    vector_chunks: list[dict] = []
    keyword_chunks: list[dict] = []

    with trace_span(
        "qdrant_vector_search",
        metadata={"top_k": config.vector_top_k, "phase": "qdrant_vector_search"},
    ) as vector_span:
        # 1. Vector search (Qdrant)
        provider = get_embedding_provider()
        query_embedding = provider.embed([query])[0]
        vector_results = qdrant_search(query_embedding=query_embedding, limit=config.vector_top_k)

        for result in vector_results:
            payload = result.payload if hasattr(result, 'payload') else result
            # Phase 34A.1.1 — extract the structured source / OCR
            # metadata so downstream image-aware routing can classify
            # chunks without round-tripping to Qdrant. Fields written
            # by qdrant_service.upsert_chunks (newer chunks only —
            # legacy chunks fall back to content-based detection in
            # image_routing).
            source_type = payload.get("source_type") or payload.get("content_type")
            image_id = payload.get("image_id")
            vector_chunks.append({
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "document_version_id": payload.get("document_version_id"),
                "chunk_index": payload.get("chunk_index"),
                "content": payload.get("content", ""),
                "source_file_name": payload.get("source_file_name", ""),
                "title": payload.get("title", ""),
                "section_heading": payload.get("section_heading"),
                "score": getattr(result, 'score', 0.0) if hasattr(result, 'score') else 0.0,
                # Phase 34A.1.1 — image / OCR provenance
                "source_type": source_type,
                "content_type": payload.get("content_type"),
                "image_id": image_id,
                "ocr_provider": payload.get("ocr_provider"),
                "ocr_confidence": payload.get("ocr_confidence"),
                "page_number": payload.get("page_number"),
                "mime_type": payload.get("mime_type"),
                "is_ocr": payload.get("is_ocr"),
            })
        if vector_span is not None:
            try:
                vector_span.set_meta("vector_results_count", len(vector_chunks))
                if vector_chunks:
                    vector_span.set_meta(
                        "top_score",
                        round(max(float(c.get("score") or 0.0) for c in vector_chunks), 4),
                    )
            except Exception:
                pass

    with trace_span(
        "hybrid_keyword_scoring",
        metadata={"top_k": config.keyword_top_k, "phase": "hybrid_keyword_scoring"},
    ) as keyword_span:
        # 2. Keyword search (PostgreSQL)
        keyword_chunks = search_chunks_keyword(
            query=query,
            limit=config.keyword_top_k,
            min_score=config.min_score,
        )
        if keyword_span is not None:
            try:
                keyword_span.set_meta("keyword_results_count", len(keyword_chunks))
                if keyword_chunks:
                    keyword_span.set_meta(
                        "top_keyword_score",
                        round(max(float(c.get("score") or 0.0) for c in keyword_chunks), 4),
                    )
            except Exception:
                pass

    # 3. Fuse results
    with trace_span(
        "context_selection",
        metadata={"phase": "context_selection", "vector_weight": config.vector_weight, "keyword_weight": config.keyword_weight},
    ):
        fused_chunks = _fuse_scores(
            vector_results=vector_chunks,
            keyword_results=keyword_chunks,
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
        )

    # 3a. Phase 34A.1 — exact-match reranking.
    # Multiplicative boost per matched error code / technical term so
    # that an exact identifier hit ranks above a higher-vector-score
    # chunk that does not contain the identifier. Skipped when the
    # query analysis reports no identifiers.
    exact_match_count = 0
    if query_analysis is not None and getattr(query_analysis, "has_identifiers", lambda: False)():
        for chunk in fused_chunks:
            if chunk.content and any(
                code.lower() in chunk.content.lower()
                for code in (getattr(query_analysis, "error_codes", []) or [])
            ):
                exact_match_count += 1
        fused_chunks = _apply_exact_match_to_scored_chunks(
            fused_chunks,
            query_analysis,
            exact_match_boost,
            exact_term_boost,
        )
    fused_results_count = len(fused_chunks)
    
    # 4. Apply min_score filter
    if config.min_score > 0:
        fused_chunks = [c for c in fused_chunks if c.fused_score >= config.min_score]
    
    # 5. Convert to dict format for reranker
    chunk_dicts = []
    for chunk in fused_chunks:
        chunk_dicts.append({
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "source_file_name": chunk.source_file_name,
            "title": chunk.title,
            "score": chunk.fused_score,
        })
    
    # 6. Apply reranking
    if config.rerank_final_k is not None and config.rerank_final_k > 0:
        top_chunks_for_rerank = chunk_dicts[:config.rerank_final_k]
    else:
        top_chunks_for_rerank = chunk_dicts
    
    rerank_results = reranker.rerank(query, top_chunks_for_rerank, top_n=config.final_top_k)
    
    # 7. Build final result with all required fields
    final_chunks = []
    for rr in rerank_results:
        final_chunks.append({
            "chunk_id": rr.chunk_id,
            "document_id": rr.document_id,
            "chunk_index": rr.chunk_index,
            "content": rr.content,
            "source_file_name": rr.source_file_name,
            "title": rr.title,
            "score": rr.rerank_score,
            "original_score": rr.original_score,
            "vector_score": chunk_dicts[[c["chunk_id"] for c in chunk_dicts].index(rr.chunk_id)]["score"] if rr.chunk_id in [c["chunk_id"] for c in chunk_dicts] else None,
        })
    
    # Build metadata about retrieval
    metadata = {
        "vector_results_count": len(vector_chunks),
        "keyword_results_count": len(keyword_chunks),
        "fused_results_count": fused_results_count,
        "final_results_count": len(final_chunks),
        "reranker_type": config.reranker_type,
        "vector_weight": config.vector_weight,
        "keyword_weight": config.keyword_weight,
        "exact_match_count": exact_match_count,
    }

    return final_chunks, metadata


def _apply_exact_match_to_scored_chunks(
    fused_chunks: List[ScoredChunk],
    query_analysis: object,
    code_boost: float,
    term_boost: float,
) -> List[ScoredChunk]:
    """Apply multiplicative exact-match boost to scored chunks.

    Mirrors `app.rag.exact_match.apply_exact_match_boost` but operates
    on `ScoredChunk` dataclasses so we do not need to convert to dict
    and back. The boost is computed by re-using the same normalisation
    and code-matching helpers from `exact_match.py`.
    """
    from app.rag.exact_match import compute_exact_match_score

    if not fused_chunks:
        return fused_chunks
    if query_analysis is None or not getattr(
        query_analysis, "has_identifiers", lambda: False
    )():
        return fused_chunks

    boosted: list[ScoredChunk] = []
    for chunk in fused_chunks:
        synthetic = {
            "content": chunk.content,
            "title": chunk.title,
            "section_heading": getattr(chunk, "section_heading", ""),
        }
        match = compute_exact_match_score(synthetic, query_analysis, code_boost, term_boost)
        new_fused = chunk.fused_score * (1.0 + match["score"])
        # Preserve ScoredChunk fields; expose the boost as attributes
        # via __dict__ so downstream code can read them.
        chunk_dict = chunk.__dict__.copy()
        chunk_dict["fused_score"] = new_fused
        chunk_dict["_exact_match_count"] = len(match["matched_codes"]) + len(
            match["matched_terms"]
        )
        chunk_dict["_exact_match_boost"] = match["score"]
        chunk_dict["_matched_codes"] = match["matched_codes"]
        chunk_dict["_matched_terms"] = match["matched_terms"]
        boosted.append(ScoredChunk(**{k: v for k, v in chunk_dict.items() if k in (
            "chunk_id", "document_id", "chunk_index", "content",
            "source_file_name", "title", "vector_score", "keyword_score",
            "fused_score", "is_from_vector", "is_from_keyword",
        )}))

    boosted.sort(key=lambda c: c.fused_score, reverse=True)
    return boosted


# ============================================================================
# Phase 30E — Retrieval Strategy Abstraction
# ============================================================================

VALID_STRATEGIES = ("similarity", "hybrid", "mmr", "hybrid_mmr")


def _apply_strategy_to_chunks(
    chunks: list[dict],
    query: str,
    strategy: str,
    config: RetrievalConfig,
    mmr_lambda: float,
    top_n: Optional[int] = None,
) -> tuple[list[dict], dict]:
    """
    Apply a retrieval strategy to a pre-computed list of chunks.

    This is the testable core of `retrieve_with_strategy`. It does NOT make
    any service calls. Callers who already have chunks (e.g., from tests or
    from a custom retrieval backend) can use this directly.

    Returns:
        Tuple of (processed chunks, metadata dict).
    """
    requested = (strategy or "hybrid_mmr").lower()
    if requested not in VALID_STRATEGIES:
        requested = "hybrid"

    metadata: dict = {
        "strategy": requested,
        "hybrid_applied": False,
        "mmr_applied": False,
        "top_k": top_n if top_n is not None else config.final_top_k,
    }

    target_n = top_n if top_n is not None else config.final_top_k

    if requested == "similarity":
        metadata["selected_chunk_count"] = len(chunks[:target_n])
        return chunks[:target_n], metadata

    if requested == "hybrid":
        boosted = _boost_chunks_with_keyword_signals(
            chunks, query, keyword_weight=config.keyword_weight
        )
        out = boosted[:target_n]
        metadata["hybrid_applied"] = True
        metadata["selected_chunk_count"] = len(out)
        return out, metadata

    if requested == "mmr":
        with trace_span(
            "mmr_filtering",
            metadata={
                "phase": "mmr_filtering",
                "mmr_lambda": mmr_lambda,
                "input_chunk_count": len(chunks),
                "top_n": target_n,
            },
        ):
            out = mmr_diversify(chunks, lambda_param=mmr_lambda, top_n=target_n)
        metadata["mmr_applied"] = True
        metadata["selected_chunk_count"] = len(out)
        return out, metadata

    # hybrid_mmr (default)
    boosted = _boost_chunks_with_keyword_signals(
        chunks, query, keyword_weight=config.keyword_weight
    )
    with trace_span(
        "mmr_filtering",
        metadata={
            "phase": "mmr_filtering",
            "mmr_lambda": mmr_lambda,
            "input_chunk_count": len(boosted),
            "top_n": target_n,
        },
    ):
        out = mmr_diversify(boosted, lambda_param=mmr_lambda, top_n=target_n)
    metadata["hybrid_applied"] = True
    metadata["mmr_applied"] = True
    metadata["selected_chunk_count"] = len(out)
    return out, metadata


def retrieve_with_strategy(
    query: str,
    strategy: Optional[str] = None,
    config: Optional[RetrievalConfig] = None,
    mmr_lambda: Optional[float] = None,
    top_n: Optional[int] = None,
    query_analysis: Optional[object] = None,
    exact_match_boost: float = 0.30,
    exact_term_boost: float = 0.15,
) -> Tuple[List[dict], dict]:
    """
    Retrieve chunks using a named strategy.

    Strategies (document-agnostic):
      - similarity: vector-only baseline
      - hybrid: vector + generic keyword/phrase/number/acronym scoring
      - mmr: vector + MMR-style diversity filtering
      - hybrid_mmr: hybrid scoring + MMR diversity (default)

    Falls back to 'hybrid' if an unknown strategy is supplied (safe default).

    Returns:
        Tuple of (chunks list, metadata dict). Metadata includes:
          - strategy: name of the strategy actually used
          - hybrid_applied: bool
          - mmr_applied: bool
          - top_k: input top_k
          - selected_chunk_count: int
    """
    if config is None:
        config = RetrievalConfig()

    if mmr_lambda is None:
        mmr_lambda = 0.7

    requested = (strategy or "hybrid_mmr").lower()
    if requested not in VALID_STRATEGIES:
        requested = "hybrid"

    # Run the standard hybrid retrieval pipeline (vector + keyword fusion).
    chunks, inner_meta = retrieve_chunks_hybrid(
        query=query,
        config=config,
        query_analysis=query_analysis,
        exact_match_boost=exact_match_boost,
        exact_term_boost=exact_term_boost,
    )

    processed, metadata = _apply_strategy_to_chunks(
        chunks=chunks,
        query=query,
        strategy=requested,
        config=config,
        mmr_lambda=mmr_lambda,
        top_n=top_n,
    )

    # Merge pipeline-level metadata (counts, weights) with strategy metadata
    metadata.update(inner_meta)
    return processed, metadata