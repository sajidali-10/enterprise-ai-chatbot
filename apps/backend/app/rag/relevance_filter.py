"""
Phase 34A.1 — Minimum Relevance Filtering

Applies a hard minimum-score threshold and a maximum-results cap to the
candidate chunk list. Chunks that do not meet the threshold are dropped;
chunks that do are sorted by score and returned in descending order.

Design notes:

- The threshold is intentionally a *floor on the fused retrieval score*
  produced by `retrieve_chunks_hybrid` (vector + keyword fusion). If
  no chunk clears the floor, we return an empty list rather than
  fabricating a source. The answer generator downstream already
  handles the empty-chunks case via its existing grounding checks.

- The cap (`max_results`) is independent of `final_top_k`. It bounds
  the maximum number of chunks presented to the LLM; combined with the
  threshold, it guarantees that `0 <= len(final) <= max_results`.

- Diagnostic counters are exposed so observability code can surface
  `candidate_count`, `filtered_count`, and `final_source_count` without
  recomputing them.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


DEFAULT_MIN_RELEVANCE_SCORE = 0.0
DEFAULT_MAX_RESULTS = 6


def apply_relevance_threshold(
    chunks: List[dict],
    min_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    max_results: Optional[int] = DEFAULT_MAX_RESULTS,
) -> Tuple[List[dict], dict]:
    """Apply minimum-score + max-results filtering to a chunk list.

    Args:
        chunks: Candidate chunks. Each chunk must have a `score`
            attribute (float). The list is NOT mutated.
        min_score: Chunks with score < min_score are dropped. Set to
            0.0 to disable the threshold while keeping the cap.
        max_results: Maximum number of chunks to return. None disables
            the cap.

    Returns:
        Tuple of (filtered chunks, metadata dict). Metadata includes:
          - candidate_count: input chunk count
          - threshold_dropped: chunks dropped because score < min_score
          - cap_dropped: chunks dropped because max_results was hit
          - filtered_count: total chunks dropped (sum of above)
          - final_source_count: chunks in returned list
          - min_score: the threshold applied
          - max_results: the cap applied (None if disabled)
    """
    if not chunks:
        return [], {
            "candidate_count": 0,
            "threshold_dropped": 0,
            "cap_dropped": 0,
            "filtered_count": 0,
            "final_source_count": 0,
            "min_score": float(min_score or 0.0),
            "max_results": max_results,
        }

    candidate_count = len(chunks)

    # 1) Apply the score floor.
    threshold_dropped = 0
    above_threshold: list[dict] = []
    for c in chunks:
        score = c.get("score")
        try:
            score_val = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score_val = 0.0
        if score_val < float(min_score or 0.0):
            threshold_dropped += 1
            continue
        above_threshold.append(c)

    # 2) Sort by score descending. Defensive: chunks without a score are
    # treated as 0.0 and end up at the bottom.
    above_threshold.sort(
        key=lambda c: float(c.get("score") or 0.0),
        reverse=True,
    )

    # 3) Apply the cap.
    cap_dropped = 0
    final = above_threshold
    if max_results is not None and max_results >= 0:
        if len(above_threshold) > max_results:
            cap_dropped = len(above_threshold) - max_results
            final = above_threshold[:max_results]

    filtered_count = threshold_dropped + cap_dropped
    metadata = {
        "candidate_count": candidate_count,
        "threshold_dropped": threshold_dropped,
        "cap_dropped": cap_dropped,
        "filtered_count": filtered_count,
        "final_source_count": len(final),
        "min_score": float(min_score or 0.0),
        "max_results": max_results,
    }
    return final, metadata
