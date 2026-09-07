"""Phase 34C.1 -- Public helpers for historical-image observation queries.

This module owns every regex and every decision function that decides
whether a chat query is asking about a previously uploaded image
(historical-image intent), whether it is asking about product meaning
or troubleshooting (in which case image-knowledge evidence is NOT
authoritative), and how to extract identifier tokens that can be
matched verbatim against OCR text.

It is the **public** surface for:

* ``app.rag.grounding`` -- re-uses the classifier as a thin alias so
  no module reaches into another's private namespace.
* ``app.rag.prompt_builder`` -- same.
* ``app.rag.image_observation_synthesis`` -- primary consumer; uses
  the classifier to decide whether to ship a deterministic synthesized
  answer.

Two safe synthesis paths consume this module:

* **Path A** -- exact identifier match ("Which screenshot showed
  error 902?"). Identifier tokens extracted from the query appear
  verbatim in the OCR body.
* **Path B** -- strong semantic-dominant historical-image match
  ("Which dashboard showed the spike?"). No identifier required but
  exactly one unrelated image must dominate the candidate set.

This module is **pure**: no LLM calls, no Qdrant, no DB, no I/O.
It can be unit-tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Public regexes -- single source of truth.
# ---------------------------------------------------------------------------

# Historical-image observation intent. The user is asking about a
# previously uploaded image rather than the image attached to the
# current chat (which is the in-scope ``image_content`` routing).
HISTORICAL_IMAGE_INTENT_RE = re.compile(
    r"\b("
    r"which\s+(?:screenshot|image|diagram|picture|photo|attachment|figure|dashboard)|"
    r"what\s+(?:screenshot|image|diagram|picture)\s+(?:showed|shows|show|contained|contains|displayed|depicting|depicts)|"
    r"(?:find|show|search|locate)\s+(?:the\s+|a\s+|an\s+|any\s+)?(?:screenshot|image|diagram|picture|photo|dashboard)|"
    r"do\s+we\s+have\s+(?:a|an|any)\s+(?:screenshot|image|diagram|picture|dashboard)|"
    r"previous(?:ly)?\s+(?:uploaded\s+)?(?:screenshot|image|diagram|picture|photo)|"
    r"(?:a|the|any)\s+(?:screenshot|image|dashboard|diagram)\s+(?:showing|with|of|where|that|which)|"
    r"(?:screenshot|image|screen|photo)\s+where|"
    r"any\s+(?:image|screenshot)\s+(?:showing|with|of)|"
    r"dashboard\s+(?:showing|with|of|where)|"
    r"diagram\s+(?:showing|with|of|where)"
    r")\b",
    re.IGNORECASE,
)

# Product-meaning / troubleshooting intent. When this matches the
# image-observation paths MUST refuse to synthesize -- KB authority is
# preserved.
PRODUCT_MEANING_INTENT_RE = re.compile(
    r"\b("
    r"how\s+do\s+i\s+(?:fix|resolve|configure|setup|set\s+up|install|use|troubleshoot|clear|reset|recover|restore|debug|address|handle|work\s+around)|"
    r"how\s+to\s+(?:fix|resolve|configure|setup|set\s+up|install|use|troubleshoot|clear|reset|recover|restore|debug|address|handle|work\s+around)|"
    r"what\s+does\s+.+\s+mean|"
    r"what\s+is\s+(?:the\s+)?(?:meaning|definition|cause)\s+of|"
    r"root\s+cause|"
    r"why\s+(?:is|does|did|am|are|do|don't|has|have|should|would|will|won't|can't|cannot)|"
    r"troubleshoot|troubleshooting|"
    r"configuration\s+guide|"
    r"how\s+can\s+i|"
    r"how\s+should\s+i|"
    r"what\s+should\s+i\s+do|"
    r"please\s+(?:fix|resolve|help)"
    r")\b",
    re.IGNORECASE,
)

# Identifier token shapes. These are tokens that, if found verbatim
# in OCR / knowledge text, prove the chunk matches the query. We are
# intentionally conservative -- the patterns below are pinned by
# ``test_phase34c1_helpers.py`` and tuned to avoid false positives
# such as matching common English words.
IDENTIFIER_TOKEN_RE = re.compile(
    r"(?<![\w])("
    r"\d{3,}[A-Za-z\-]*|"                          # 902, 1234A, 200-OK, 9021
    r"\d{2,}[A-Z][A-Za-z\-]*|"                     # 99A
    r"[A-Z][A-Za-z]+-\d+|"                          # Vodafone-1, Server-42
    r"[A-Z][A-Za-z]+-[A-Z][A-Za-z]+|"               # Vodafone-UK, OOHipText-Two
    r"\+?\d[\d\s\-\(\)]{4,}\d|"                    # +44 7700 900123, 01234-567-890
    r"[A-Za-z]+_[A-Za-z0-9_\-]{2,}|"               # test_a_902, dashboard_main_v2
    r"Screenshot[\s_\-\.]\d{4}[\-_]?\d{2}[\-_]?\d{2}|"  # Screenshot 2026-08-05
    r"[A-Za-z]+\.png|[A-Za-z]+\.jpg|[A-Za-z]+\.jpeg"     # dashboard.png, foo.jpg
    r")(?![\w])",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# LLM refusal detection -- deterministic, conservative.
# ---------------------------------------------------------------------------

# Substring patterns that indicate the LLM refused to answer. Matches
# are case-insensitive. We use a *substring* match (not anchored)
# because the LLM sometimes prefixes or wraps the phrase with extra
# hedging language.
_LLM_REFUSAL_PATTERNS: Sequence[str] = (
    "i could not find enough information",
    "i don't have enough information",
    "i do not have enough information",
    "i cannot find enough information",
    "the provided sources do not contain",
    "not enough information to answer",
    "could not find enough information",
    "do not have enough information in the provided sources",
    "no relevant documents were found",
    "the sources do not contain enough information",
    "i'm unable to answer",
    "i am unable to answer",
    "i cannot answer this question",
)


def is_llm_refusal_answer(answer: Optional[str]) -> bool:
    """Return True when the LLM answer is a known refusal.

    Conservative: returns False (treat as substantive answer) on any
    empty / whitespace / non-string input. The phrase list is the
    same one used elsewhere in the codebase (``grounding.py`` and
    ``agentic_graph.py``); we duplicate it here only to keep this
    module independent of those modules at import time.
    """
    if not answer or not isinstance(answer, str):
        return False
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _LLM_REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# Source-type helpers.
# ---------------------------------------------------------------------------

# Set of source_type / content_type values that count as image-derived
# evidence. Used by the chunk filters.
_IMAGE_SOURCE_TYPES = frozenset({"image_knowledge", "image_ocr"})

# Filename suffix / substring patterns that strongly suggest the chunk
# is image-derived even when ``source_type`` is missing.
_IMAGE_FILENAME_HINTS = (
    "screenshot",
    ".png",
    ".jpg",
    ".jpeg",
    "dashboard",
)


def is_image_knowledge_chunk(chunk: dict) -> bool:
    """Return True when the chunk is image-derived evidence.

    Accepts a chunk dict as returned by the RAG candidate layer. A
    chunk qualifies if:

    * ``source_type`` or ``content_type`` is ``image_knowledge`` /
      ``image_ocr``; or
    * the chunk carries a non-null ``image_id`` (every Phase 34C
      point has one); or
    * the ``source_file_name`` / ``filename`` ends in a recognised
      image suffix or contains "dashboard".
    """
    if not isinstance(chunk, dict):
        return False
    st = chunk.get("source_type") or chunk.get("content_type")
    if st in _IMAGE_SOURCE_TYPES:
        return True
    if chunk.get("image_id") is not None:
        return True
    fn = (chunk.get("source_file_name") or chunk.get("filename") or "").lower()
    if not fn:
        return False
    return any(s in fn for s in _IMAGE_FILENAME_HINTS)


def all_chunks_are_image_knowledge(chunks: Iterable[dict]) -> bool:
    """True when every chunk in the iterable is image-derived."""
    if not chunks:
        return False
    return all(is_image_knowledge_chunk(c) for c in chunks)


# ---------------------------------------------------------------------------
# Identifier extraction / matching.
# ---------------------------------------------------------------------------

def extract_query_identifier_tokens(query: str) -> List[str]:
    """Return the lowercase identifier tokens found in the query.

    Drops tokens shorter than 2 characters and de-duplicates while
    preserving the original order of first occurrence.
    """
    if not query:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for match in IDENTIFIER_TOKEN_RE.finditer(query):
        token = (match.group(1) or "").strip().lower()
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def identifier_matches_chunk_content(token: str, chunk: dict) -> bool:
    """Return True when the lowercase ``token`` appears verbatim in the chunk.

    Looks at ``content``, ``knowledge_text``, and ``source_file_name``
    / ``filename``. Match is case-insensitive and substring-based.
    """
    if not token:
        return False
    needle = token.lower()
    for key in ("content", "knowledge_text", "source_file_name", "filename"):
        raw = chunk.get(key)
        if isinstance(raw, str) and needle in raw.lower():
            return True
    return False


def query_identifier_matches_any_chunk(query: str, chunks: Iterable[dict]) -> bool:
    """True if at least one identifier token from the query appears verbatim in any chunk."""
    tokens = extract_query_identifier_tokens(query)
    if not tokens:
        return False
    for chunk in chunks:
        for token in tokens:
            if identifier_matches_chunk_content(token, chunk):
                return True
    return False


# ---------------------------------------------------------------------------
# Dominance / ambiguity detection for Path B.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ImageKey:
    """Identity tuple for "unrelated image" detection.

    Two images are considered related (not ambiguous) when they share
    a ``document_id`` AND an ``image_type``. Same-document dashboards
    and their zoomed-in insets count as one image, not two.
    """

    document_id: Optional[int]
    image_type: Optional[str]

    @classmethod
    def from_chunk(cls, chunk: dict) -> "_ImageKey":
        doc_id = chunk.get("document_id")
        img_type = chunk.get("image_type") or chunk.get("mime_type")
        try:
            doc_id_int: Optional[int] = int(doc_id) if doc_id is not None else None
        except (TypeError, ValueError):
            doc_id_int = None
        return cls(document_id=doc_id_int, image_type=str(img_type) if img_type else None)


def find_qualifying_images(
    chunks: Sequence[dict],
    *,
    score_threshold: float = 0.65,
    dominance_ratio: float = 1.40,
) -> tuple[List[dict], bool]:
    """Return ``(qualifying_images, is_ambiguous)``.

    * Filters chunks to image-derived ones with ``score >= score_threshold``.
    * Groups by ``_ImageKey`` (document_id + image_type) and picks the
      best chunk per group.
    * If the top group's score is >= ``dominance_ratio`` times the
      second group's score AND there is at most one group, the result
      is unambiguous and ``qualifying_images`` contains just that group.
    * Otherwise the result is ambiguous.

    The ratio compares the **score** of the highest-scoring chunk in
    each group. Empty input -> ``([], False)``.
    """
    if not chunks:
        return [], False

    # Filter to image-derived chunks above the score threshold.
    qualified: List[dict] = []
    for c in chunks:
        if not is_image_knowledge_chunk(c):
            continue
        score = c.get("score")
        if score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if score_f < float(score_threshold):
            continue
        qualified.append(c)

    if not qualified:
        return [], False

    # Group by image identity; keep the best chunk per group.
    groups: dict[_ImageKey, dict] = {}
    for c in qualified:
        key = _ImageKey.from_chunk(c)
        existing = groups.get(key)
        if existing is None or float(c.get("score") or 0.0) > float(existing.get("score") or 0.0):
            groups[key] = c

    if len(groups) == 1:
        return [next(iter(groups.values()))], False

    # Multiple groups -> ambiguous unless one clearly dominates.
    ranked = sorted(
        groups.values(),
        key=lambda c: float(c.get("score") or 0.0),
        reverse=True,
    )
    top = ranked[0]
    runner_up = ranked[1]
    top_score = float(top.get("score") or 0.0)
    runner_score = float(runner_up.get("score") or 0.0)
    if top_score >= dominance_ratio * runner_score:
        return [top], False
    return ranked, True  # ambiguous


# ---------------------------------------------------------------------------
# Decision object.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageObservationDecision:
    """The full decision produced by ``classify_image_observation_query``.

    Attributes are read-only. Callers must inspect ``path_a_eligible``
    or ``path_b_eligible`` (mutually exclusive in practice) to decide
    whether to ship a synthesized answer.
    """

    is_image_observation: bool = False       # historical-image intent matched
    refusing_intent: bool = False            # product-meaning / troubleshooting matched
    all_evidence_is_image: bool = False      # every chunk is image-derived
    authorized: bool = False                 # caller passed authorization
    path_a_eligible: bool = False            # exact identifier matched
    path_b_eligible: bool = False            # strong semantic-dominant match
    path_b_ambiguous: bool = False           # multiple unrelated images at top
    matched_identifier: Optional[str] = None # which token satisfied Path A
    dominant_chunk: Optional[dict] = None    # which chunk satisfied Path B

    def any_path_eligible(self) -> bool:
        """True if either Path A or Path B is eligible."""
        return bool(self.path_a_eligible or self.path_b_eligible)


# ---------------------------------------------------------------------------
# The classifier.
# ---------------------------------------------------------------------------

def classify_image_observation_query(
    query: str,
    chunks: Sequence[dict],
    *,
    authorized: bool = True,
    image_score_threshold: float = 0.65,
    dominance_ratio: float = 1.40,
) -> ImageObservationDecision:
    """Classify the query against the chunk set.

    Pure function. All four preconditions for ANY image-observation
    synthesis path are checked:

    1. ``HISTORICAL_IMAGE_INTENT_RE`` matches the query.
    2. ``PRODUCT_MEANING_INTENT_RE`` does NOT match.
    3. Every supplied chunk is image-derived.
    4. The caller is authorized.

    If those hold, Path A is checked first (exact identifier match),
    then Path B (strong semantic dominance). If neither holds, the
    decision's ``any_path_eligible()`` returns False.

    Parameters
    ----------
    query:
        The user question text.
    chunks:
        Candidate chunks in the order the retriever returned them.
        May be empty.
    authorized:
        Defense-in-depth flag. The retriever already enforced RBAC;
        pass the same conclusion here so a buggy caller cannot
        accidentally let unauthorized chunks through.
    image_score_threshold:
        Minimum top-image score for Path B.
    dominance_ratio:
        Minimum ratio between the top and the runner-up unrelated
        image scores for Path B.
    """
    if not query or not chunks:
        return ImageObservationDecision()

    is_obs = bool(HISTORICAL_IMAGE_INTENT_RE.search(query))
    refusing = bool(PRODUCT_MEANING_INTENT_RE.search(query))
    all_img = all_chunks_are_image_knowledge(chunks)
    is_authorized = bool(authorized)

    if not (is_obs and not refusing and all_img and is_authorized):
        return ImageObservationDecision(
            is_image_observation=is_obs,
            refusing_intent=refusing,
            all_evidence_is_image=all_img,
            authorized=is_authorized,
        )

    # Path A: at least one identifier token from the query appears
    # verbatim in some chunk.
    path_a = False
    matched_token: Optional[str] = None
    for token in extract_query_identifier_tokens(query):
        for chunk in chunks:
            if identifier_matches_chunk_content(token, chunk):
                path_a = True
                matched_token = token
                break
        if path_a:
            break

    if path_a:
        return ImageObservationDecision(
            is_image_observation=True,
            refusing_intent=False,
            all_evidence_is_image=True,
            authorized=True,
            path_a_eligible=True,
            matched_identifier=matched_token,
        )

    # Path B: semantic-dominant historical image. No identifier
    # required but the candidate set must collapse to exactly one
    # unrelated image with a clear score margin.
    qualifying, ambiguous = find_qualifying_images(
        chunks,
        score_threshold=image_score_threshold,
        dominance_ratio=dominance_ratio,
    )
    if ambiguous or not qualifying:
        return ImageObservationDecision(
            is_image_observation=True,
            refusing_intent=False,
            all_evidence_is_image=True,
            authorized=True,
            path_b_ambiguous=bool(ambiguous),
        )

    return ImageObservationDecision(
        is_image_observation=True,
        refusing_intent=False,
        all_evidence_is_image=True,
        authorized=True,
        path_b_eligible=True,
        dominant_chunk=qualifying[0],
    )


# ---------------------------------------------------------------------------
# Public exports.
# ---------------------------------------------------------------------------

__all__ = [
    "HISTORICAL_IMAGE_INTENT_RE",
    "PRODUCT_MEANING_INTENT_RE",
    "IDENTIFIER_TOKEN_RE",
    "is_llm_refusal_answer",
    "is_image_knowledge_chunk",
    "all_chunks_are_image_knowledge",
    "extract_query_identifier_tokens",
    "identifier_matches_chunk_content",
    "query_identifier_matches_any_chunk",
    "find_qualifying_images",
    "ImageObservationDecision",
    "classify_image_observation_query",
]
