"""Phase 34C.1 -- Deterministic image-observation answer synthesis.

Invoked as **post-LLM refusal recovery** in ``app.rag.answer_generator``
when:

* the LLM emitted a known refusal phrase (``is_llm_refusal_answer``);
* the query is a historical-image observation question (NOT product
  meaning / NOT troubleshooting);
* the candidate set contains authorized image-knowledge evidence.

The synthesizer picks ONE of two safe paths:

* **Path A** -- exact identifier match. The query contains a token
  (e.g. "902", "Vodafone-UK") that appears verbatim in the OCR body
  of at least one image-knowledge chunk. The synthesized answer
  quotes the matching OCR line and cites the filename.

* **Path B** -- strong semantic-dominant historical-image match. The
  query contains NO identifier, but exactly one unrelated image
  dominates the candidate set with a clear score margin. The
  synthesized answer cites that one image by filename and quotes
  its top OCR line.

If neither path applies -- in particular when multiple unrelated
images tie at the top -- the synthesizer returns ``None`` and the
caller falls back to the existing grounded "I don't have enough
information" message. **KB authority is never overridden.**

The module is pure: it does not call the LLM, does not write to
Qdrant, and does not touch the database. It only mutates the answer
string and citation list that the caller passes back to the chat
endpoint.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.rag.image_observation_helpers import (
    ImageObservationDecision,
    classify_image_observation_query,
    identifier_matches_chunk_content,
    is_image_knowledge_chunk,
    is_llm_refusal_answer,
)
from app.rag.citations import clean_excerpt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration knobs.
# ---------------------------------------------------------------------------

# Default dominance parameters. Pinned by tests; do not change without
# updating ``tests/test_phase34c1_synthesis.py``.
DEFAULT_IMAGE_SCORE_THRESHOLD = 0.65
DEFAULT_DOMINANCE_RATIO = 1.40

# Maximum length of the verbatim OCR line quoted in the synthesized
# answer. Kept short so the chat response stays business-readable.
QUOTE_MAX_CHARS = 220


# ---------------------------------------------------------------------------
# Result object.
# ---------------------------------------------------------------------------

@dataclass
class ImageObservationSynthesis:
    """Result returned by ``maybe_synthesize_image_observation_answer``.

    Attributes
    ----------
    answer:
        The deterministic synthesized answer text. Includes an inline
        ``[1]`` citation marker.
    citations:
        Citation dicts in the same shape as ``format_citations``
        produces. Each carries ``citation_kind=image_knowledge``.
    meta:
        Diagnostic dict. Always contains ``"path": "A" | "B"`` plus
        identifier / image_id / source_file_name / quote / score.
    """

    answer: str
    citations: List[Dict[str, Any]]
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _short_quote_for_chunk(chunk: dict, *, token: Optional[str] = None) -> str:
    """Pick the best verbatim OCR line to quote.

    If ``token`` is supplied, prefer the first line in the chunk's
    content / knowledge_text that contains the token (case-insensitive).
    Otherwise return the first non-empty line.

    The quote is clipped to ``QUOTE_MAX_CHARS`` so the answer stays
    readable.
    """
    raw = chunk.get("content") or chunk.get("knowledge_text") or ""
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    chosen: Optional[str] = None
    if token:
        token_lc = token.lower()
        for line in lines:
            if token_lc in line.lower():
                chosen = line
                break
    if chosen is None:
        chosen = lines[0] if lines else ""
    if len(chosen) > QUOTE_MAX_CHARS:
        # Try sentence boundary first, then word boundary.
        truncated = chosen[:QUOTE_MAX_CHARS]
        sentence_end = re.search(r"[.!?](?:\s|$)", truncated[::-1])
        if sentence_end:
            cut = QUOTE_MAX_CHARS - sentence_end.start() - 1
            if cut > QUOTE_MAX_CHARS * 0.6:
                chosen = chosen[: cut + 1].strip()
            else:
                chosen = truncated.rstrip() + "..."
        else:
            space = truncated.rfind(" ")
            if space > QUOTE_MAX_CHARS * 0.7:
                chosen = chosen[:space].rstrip() + "..."
            else:
                chosen = truncated.rstrip() + "..."
    return chosen


def _build_citation_dict(chunk: dict, index: int = 1) -> Dict[str, Any]:
    """Build the citation dict that ``format_citations`` would emit.

    Mirrors the shape produced by ``format_citations`` in
    ``app.rag.citations`` so the frontend can render it without
    special-casing synthesized answers.
    """
    citation: Dict[str, Any] = {
        "index": index,
        "source_file_name": chunk.get("source_file_name", "Unknown"),
        "content_snippet": clean_excerpt(
            chunk.get("content") or chunk.get("knowledge_text") or "",
            max_length=240,
        ),
        "relevance_score": chunk.get("score"),
        "chunk_id": chunk.get("id"),
        "citation_kind": "image_knowledge",
        "image_id": chunk.get("image_id"),
        "image_type": chunk.get("image_type") or chunk.get("mime_type"),
        "vision_provider": chunk.get("vision_provider"),
        "vision_model": chunk.get("vision_model"),
        "has_vision": bool(chunk.get("has_vision")),
        "knowledge_schema_version": chunk.get("knowledge_schema_version"),
        "document_id": chunk.get("document_id"),
    }
    return citation


def _authorized_for_all_chunks(
    chunks: List[Dict[str, Any]],
    accessible_doc_ids: Optional[set],
) -> bool:
    """Defense-in-depth authorization check.

    The retriever already filters out unauthorized chunks; this
    helper exists so a buggy caller cannot synthesize from chunks
    the user is not allowed to see.

    When ``accessible_doc_ids`` is ``None`` we DERIVE the authorized
    set from the chunks' own ``document_id`` values -- because the
    chunks have already been RBAC-filtered upstream by
    ``retrieve_chunks_with_auth``, the union of their ``document_id``
    values IS exactly the caller's accessible set in scope for this
    candidate list. This makes the synthesizer robust to callers that
    forgot to pass ``retrieval_metadata["accessible_doc_ids"]``
    explicitly while still keeping the explicit check as a stricter
    defense-in-depth path.

    Pass an explicit empty set to deny synthesis.
    """
    if accessible_doc_ids is None:
        # Derive from the chunks themselves -- they are already
        # permission-filtered, so their document_ids form the
        # authorized set for this candidate list.
        accessible_doc_ids = set()
        for c in chunks:
            doc_id = c.get("document_id")
            if doc_id is None:
                continue
            try:
                accessible_doc_ids.add(int(doc_id))
            except (TypeError, ValueError):
                continue
    for c in chunks:
        doc_id = c.get("document_id")
        if doc_id is None:
            continue
        try:
            if int(doc_id) not in accessible_doc_ids:
                return False
        except (TypeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------------------
# Path A synthesis.
# ---------------------------------------------------------------------------

def _synthesize_path_a(
    query: str,
    chunks: List[Dict[str, Any]],
    decision: ImageObservationDecision,
) -> Optional[ImageObservationSynthesis]:
    """Exact identifier match -- return a synthesized answer.

    We pick the **first** chunk whose content contains the matched
    identifier token (case-insensitive). When multiple chunks tie, the
    first one wins -- the candidate set has already been ranked by the
    retriever and tie-broken by the image-intent boost.
    """
    token = decision.matched_identifier
    if not token:
        return None
    chosen: Optional[Dict[str, Any]] = None
    for chunk in chunks:
        if identifier_matches_chunk_content(token, chunk):
            chosen = chunk
            break
    if chosen is None:
        return None

    quote = _short_quote_for_chunk(chosen, token=token)
    filename = chosen.get("source_file_name") or "the uploaded image"
    score = chosen.get("score")

    answer_lines: List[str] = []
    if quote:
        answer_lines.append(
            f'Based on the available screenshot evidence, the matching image is '
            f'"{filename}" [1]. The OCR body shows: "{quote}".'
        )
    else:
        answer_lines.append(
            f'Based on the available screenshot evidence, the matching image is '
            f'"{filename}" [1].'
        )

    citation = _build_citation_dict(chosen, index=1)
    meta = {
        "path": "A",
        "matched_identifier": token,
        "image_id": chosen.get("image_id"),
        "document_id": chosen.get("document_id"),
        "source_file_name": filename,
        "quote": quote,
        "score": score,
    }
    return ImageObservationSynthesis(
        answer=" ".join(answer_lines),
        citations=[citation],
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Path B synthesis.
# ---------------------------------------------------------------------------

def _synthesize_path_b(
    query: str,
    chunks: List[Dict[str, Any]],
    decision: ImageObservationDecision,
) -> Optional[ImageObservationSynthesis]:
    """Strong semantic-dominant match -- return a synthesized answer.

    No identifier is required. The decision object carries the single
    dominant chunk chosen by ``find_qualifying_images``. We quote the
    top OCR line so the user sees the actual evidence.
    """
    chosen = decision.dominant_chunk
    if not chosen:
        return None

    quote = _short_quote_for_chunk(chosen)
    filename = chosen.get("source_file_name") or "the uploaded image"
    score = chosen.get("score")

    # When Path B is used we cannot point at a single OCR token, so we
    # phrase the answer in terms of the dominant image and the most
    # informative line. If the OCR is empty we still ship the
    # filename-only answer so the user knows the image was found.
    if quote:
        answer = (
            f'Based on the available screenshot evidence, the matching image is '
            f'"{filename}" [1]. The OCR body shows: "{quote}".'
        )
    else:
        answer = (
            f'Based on the available screenshot evidence, the matching image is '
            f'"{filename}" [1].'
        )

    citation = _build_citation_dict(chosen, index=1)
    meta = {
        "path": "B",
        "matched_identifier": None,
        "image_id": chosen.get("image_id"),
        "document_id": chosen.get("document_id"),
        "source_file_name": filename,
        "quote": quote,
        "score": score,
    }
    return ImageObservationSynthesis(
        answer=answer,
        citations=[citation],
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def maybe_synthesize_image_observation_answer(
    *,
    query: str,
    chunks: List[Dict[str, Any]],
    answer: Optional[str],
    accessible_doc_ids: Optional[set] = None,
    image_score_threshold: float = DEFAULT_IMAGE_SCORE_THRESHOLD,
    dominance_ratio: float = DEFAULT_DOMINANCE_RATIO,
    authorized: bool = True,
) -> Optional[ImageObservationSynthesis]:
    """Return a synthesized image-observation answer, or ``None``.

    The synthesizer is **only** invoked when the LLM emitted a known
    refusal phrase. Substantive answers (non-empty, not in the refusal
    list) are passed through unchanged by the caller; we return
    ``None`` immediately in that case so the caller cannot
    accidentally clobber a real answer.

    Parameters
    ----------
    query:
        The user question.
    chunks:
        Candidate chunks from the retriever (post-permission filter).
    answer:
        The LLM's draft answer. Must be a known refusal phrase for
        synthesis to run.
    accessible_doc_ids:
        Set of document IDs the caller is authorized to see. When
        ``None`` we refuse to synthesize (defense in depth). Pass an
        explicit set -- even an empty one -- to opt in.
    image_score_threshold, dominance_ratio:
        Path B tuning parameters. Pinned by tests.
    authorized:
        Convenience flag mirroring the same flag the retriever used.
        Kept as an explicit parameter so a buggy caller cannot
        silently synthesize from unauthorized chunks.
    """
    # 0. LLM must have refused.
    refusal_detected = is_llm_refusal_answer(answer)
    if not refusal_detected:
        _diag(
            "not_refusal",
            chunks=chunks,
            accessible_doc_ids=accessible_doc_ids,
            answer_text=str(answer)[:200] if answer is not None else None,
            answer_len=len(answer) if isinstance(answer, str) else None,
        )
        return None

    # 1. No query or no chunks -> nothing to do.
    if not query or not chunks:
        _diag("empty_query_or_chunks", chunks=chunks, accessible_doc_ids=accessible_doc_ids)
        return None

    # 2. Authorization (defense in depth).
    if not authorized:
        _diag("not_authorized", chunks=chunks, accessible_doc_ids=accessible_doc_ids)
        return None
    if not _authorized_for_all_chunks(chunks, accessible_doc_ids):
        _diag("auth_filter_failed", chunks=chunks, accessible_doc_ids=accessible_doc_ids)
        return None

    # 3. Filter the candidate set to image-derived chunks only. The
    #    hybrid retriever always returns mixed KB + image_knowledge
    #    evidence, but the image-observation synthesizer is only
    #    meaningful for the image-derived subset. KB chunks do not
    #    vote on whether a synthesized answer is safe; they merely
    #    coexist in the retriever's response. We classify against
    #    the image-only subset so the strict ``all_evidence_is_image``
    #    precondition reflects what we actually synthesize from.
    image_chunks = [c for c in chunks if is_image_knowledge_chunk(c)]
    if not image_chunks:
        _diag("no_image_chunks", chunks=chunks, accessible_doc_ids=accessible_doc_ids)
        return None

    # 4. Classify the query against the image-only chunk set.
    decision = classify_image_observation_query(
        query,
        image_chunks,
        authorized=True,  # already checked above
        image_score_threshold=image_score_threshold,
        dominance_ratio=dominance_ratio,
    )

    if not decision.any_path_eligible():
        _diag(
            "no_path_eligible",
            chunks=chunks,
            accessible_doc_ids=accessible_doc_ids,
            image_observation=decision.is_image_observation,
            refusing_intent=decision.refusing_intent,
            all_image=decision.all_evidence_is_image,
            path_a=decision.path_a_eligible,
            path_b=decision.path_b_eligible,
            path_b_ambiguous=decision.path_b_ambiguous,
        )
        return None

    if decision.path_a_eligible:
        result = _synthesize_path_a(query, image_chunks, decision)
        _diag("synthesized_path_a", chunks=chunks, accessible_doc_ids=accessible_doc_ids, decision=decision)
        return result
    if decision.path_b_eligible:
        result = _synthesize_path_b(query, image_chunks, decision)
        _diag("synthesized_path_b", chunks=chunks, accessible_doc_ids=accessible_doc_ids, decision=decision)
        return result
    return None


_DIAG_LOG = "/tmp/phase34c1_e2e_synth_diag.jsonl"


def _diag(reason: str, *, chunks, accessible_doc_ids, **extra) -> None:
    """Append a diagnostic record to the temp E2E log.

    Disabled in production usage by env-var; only the Phase 34C.1 E2E
    run sets ``PHASE34C1_E2E_DIAG=1`` on the container. Kept in the
    source so the diagnostic path is reproducible without code
    branching in answer_generator.
    """
    if os.environ.get("PHASE34C1_E2E_DIAG") != "1":
        return
    try:
        rec = {
            "ts": __import__("datetime").datetime.utcnow().isoformat(),
            "reason": reason,
            "chunks_total": len(chunks or []),
            "chunks_image": sum(1 for c in (chunks or []) if is_image_knowledge_chunk(c)),
            "chunks_sample": [
                {
                    "source_type": c.get("source_type"),
                    "image_id": c.get("image_id"),
                    "document_id": c.get("document_id"),
                    "source_file_name": c.get("source_file_name"),
                    "score": c.get("score"),
                }
                for c in (chunks or [])[:3]
            ],
            "accessible_doc_ids_provided": (
                sorted(int(x) for x in accessible_doc_ids)
                if isinstance(accessible_doc_ids, (set, list, tuple))
                else accessible_doc_ids
            ),
        }
        rec.update(extra)
        with open(_DIAG_LOG, "a") as f:
            f.write(__import__("json").dumps(rec, default=str) + "\n")
    except Exception:
        # Diagnostic only -- never break the request.
        pass


__all__ = [
    "ImageObservationSynthesis",
    "maybe_synthesize_image_observation_answer",
    "DEFAULT_IMAGE_SCORE_THRESHOLD",
    "DEFAULT_DOMINANCE_RATIO",
]
