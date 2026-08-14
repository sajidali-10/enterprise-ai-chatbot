"""
Phase 34A.1 — Exact-Match Boosting

Reranking helper that boosts chunks whose content contains the
exact technical identifiers extracted from the query. Conservative:

- An **exact error-code hit** (e.g. the chunk text contains the
  literal string "902", "ORA-12541", "HTTP 404") is a strong signal
  that the chunk answers the user's question, even if the vector
  similarity score is moderate. We multiply the existing score by
  (1 + boost) per matched code.
- An **exact technical-term hit** (e.g. "rejected-forbidden-country")
  is a medium-strength signal. We apply a smaller multiplicative
  boost.
- If the query analysis reports no identifiers, this component is a
  no-op (returns the chunks untouched).

The function is intentionally additive on top of the existing fused
score from `retrieve_chunks_hybrid`. It does NOT touch MMR diversity
filtering — that runs after we mutate the score, on the same list,
so we still get cross-document coverage.

Diagnostics: each chunk gains `_exact_match_count`, `_matched_codes`,
`_matched_terms` for downstream observability. These keys are
prefixed with `_` to mirror the convention already used by
`_keyword_score` / `_signal_matches` in `hybrid_retriever.py`.
"""

from __future__ import annotations

import re
from typing import List, Optional

from app.rag.query_analysis import QueryAnalysis


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-code boost: chunk gets multiplied by (1 + boost) for each match.
# 0.30 = up to a 30% score increase per exact hit. Capped at 1 code
# in practice; multiple matches compound up to a ceiling below.
DEFAULT_EXACT_CODE_BOOST = 0.30
DEFAULT_EXACT_TERM_BOOST = 0.15
# Cap on total exact-match boost applied per chunk so a chunk with
# 5 accidental number coincidences does not run away.
MAX_BOOST_PER_CHUNK = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_for_match(text: str) -> str:
    """Normalise text for exact-match comparison.

    - Lowercase.
    - Collapse runs of whitespace.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _chunk_text(chunk: dict) -> str:
    """Return the searchable text blob for a chunk.

    Combines content + title + section_heading so a code in a chunk's
    title or heading still counts. Whitespace is collapsed.
    """
    parts = [
        str(chunk.get("content") or ""),
        str(chunk.get("title") or ""),
        str(chunk.get("section_heading") or ""),
    ]
    return _normalize_for_match(" ".join(parts))


def _code_present_in_chunk(code: str, haystack: str) -> bool:
    """Return True if an extracted code appears verbatim in the chunk.

    Numeric-only codes (e.g. "902") use a word-boundary regex to avoid
    matching inside larger numbers ("9021", "19021"). Codes with
    non-digit characters (ORA-, HTTP, 0x, HL-, ERR_) do not need
    word boundaries because their prefixes already prevent collisions.
    """
    if not code or not haystack:
        return False

    code_l = code.lower().strip()
    if not code_l:
        return False

    # "HTTP 404" style codes. Match against the haystack with one or
    # more spaces / punctuation between HTTP and the status number.
    # The haystack is already whitespace-normalised, so a single-space
    # literal search usually works; fall back to a regex for safety.
    if code_l.startswith("http "):
        num = code_l.split(" ", 1)[1]
        pattern = re.compile(
            r"\bhttp[\s/_-]*(?:status[\s_-]*)?"
            + re.escape(num)
            + r"\b"
        )
        return bool(pattern.search(haystack))

    # SQLSTATE 08001 — match SQLSTATE + whitespace + code. Use regex
    # so any number of spaces / punctuation between SQLSTATE and the
    # code still matches.
    if code_l.startswith("sqlstate"):
        code_after = code_l.replace("sqlstate", "").strip()
        if not code_after:
            return False
        pattern = re.compile(r"\bsqlstate[\s_-]*" + re.escape(code_after) + r"\b")
        return bool(pattern.search(haystack))

    # Pure-numeric codes: word-boundary match.
    if code_l.isdigit():
        pattern = re.compile(r"\b" + re.escape(code_l) + r"\b")
        return bool(pattern.search(haystack))

    # Hex codes (0x80070005) — match with optional leading/trailing space.
    if code_l.startswith("0x"):
        pattern = re.compile(r"(?<!\w)" + re.escape(code_l) + r"(?!\w)")
        return bool(pattern.search(haystack))

    # Everything else (ORA-12345, HL-1053, ERR_CONNECTION_REFUSED):
    # case-insensitive substring is sufficient because the leading
    # non-digit prefix (ORA-, HL-, ERR_) is unique.
    return code_l in haystack


def _term_present_in_chunk(term: str, haystack: str) -> bool:
    """Return True if a hyphenated technical term appears in the chunk.

    The term and the chunk content may use either '-' or '_' as the
    separator; we accept both interchangeably. Matching is
    word-bounded so "delivery" does not match inside "redelivery".
    """
    if not term or not haystack:
        return False

    t = term.lower().strip()
    # Build a regex where every '-' / '_' in the term matches either
    # '-' or '_' in the haystack.
    parts = re.split(r"[-_]+", t)
    if len(parts) < 2:
        # Single-segment terms fall through to plain substring match.
        pattern = re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)")
        return bool(pattern.search(haystack))

    flexible = r"[-_]?".join(re.escape(p) for p in parts)
    pattern = re.compile(r"(?<!\w)" + flexible + r"(?!\w)")
    return bool(pattern.search(haystack))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_exact_match_score(
    chunk: dict,
    analysis: QueryAnalysis,
    code_boost: float = DEFAULT_EXACT_CODE_BOOST,
    term_boost: float = DEFAULT_EXACT_TERM_BOOST,
) -> dict:
    """Compute exact-match metadata for a single chunk.

    Returns a dict with:
      - score: float in [0.0, MAX_BOOST_PER_CHUNK] indicating total boost
      - matched_codes: list of codes that hit
      - matched_terms: list of terms that hit
    """
    if analysis is None or not analysis.has_identifiers():
        return {"score": 0.0, "matched_codes": [], "matched_terms": []}

    haystack = _chunk_text(chunk)
    if not haystack:
        return {"score": 0.0, "matched_codes": [], "matched_terms": []}

    matched_codes: list[str] = []
    matched_terms: list[str] = []
    boost = 0.0

    for code in analysis.error_codes:
        if _code_present_in_chunk(code, haystack):
            matched_codes.append(code)
            boost += code_boost

    for term in analysis.technical_terms:
        if _term_present_in_chunk(term, haystack):
            matched_terms.append(term)
            boost += term_boost

    boost = min(boost, MAX_BOOST_PER_CHUNK)
    return {"score": boost, "matched_codes": matched_codes, "matched_terms": matched_terms}


def apply_exact_match_boost(
    chunks: List[dict],
    analysis: Optional[QueryAnalysis],
    code_boost: float = DEFAULT_EXACT_CODE_BOOST,
    term_boost: float = DEFAULT_EXACT_TERM_BOOST,
    return_only_scored: bool = False,
) -> List[dict]:
    """Apply exact-match boost to a list of chunks in-place style.

    For each chunk:
      new_score = old_score * (1 + boost)

    The chunk dicts are copied (not mutated in place) so caller's input
    list is untouched. Each returned chunk gains:
      - _exact_match_count: int
      - _exact_match_boost: float
      - _matched_codes: list[str]
      - _matched_terms: list[str]

    If `return_only_scored` is True, chunks with no matches are dropped.
    This is useful when the caller wants ONLY exact-match results
    (e.g. an image-content routing fallback). Default False preserves
    the existing hybrid semantics.

    If `analysis` is None or has no identifiers, this is a no-op and
    the original chunk list is returned with no extra metadata.
    """
    if not chunks:
        return []
    if analysis is None or not analysis.has_identifiers():
        # Nothing to boost. Return shallow copies so the caller can
        # safely assume new chunk dicts (consistent contract).
        return [dict(c) for c in chunks]

    boosted: list[dict] = []
    for chunk in chunks:
        match = compute_exact_match_score(chunk, analysis, code_boost, term_boost)
        new_chunk = dict(chunk)
        old_score = float(new_chunk.get("score") or 0.0)
        new_chunk["score"] = old_score * (1.0 + match["score"])
        new_chunk["_exact_match_count"] = len(match["matched_codes"]) + len(match["matched_terms"])
        new_chunk["_exact_match_boost"] = round(match["score"], 4)
        new_chunk["_matched_codes"] = match["matched_codes"]
        new_chunk["_matched_terms"] = match["matched_terms"]
        if return_only_scored and new_chunk["_exact_match_count"] == 0:
            continue
        boosted.append(new_chunk)

    # Re-sort by score descending to reflect the new ranking.
    boosted.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
    return boosted
