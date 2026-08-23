"""Phase 34B — RAG integration glue.

Hooks the Vision orchestrator into the existing Phase 34A.1.x RAG
pipeline. Strategy:

    * After OCR chunks are selected for an image-content question,
      run the Vision orchestrator with the same OCR signals.
    * If the orchestrator decides Vision is justified and a
      successful VisionResult is available, prepend a synthetic
      "vision evidence" chunk to the chunk list. The chunk
      carries the same ``source_file_name`` as the underlying
      document so the existing citation pipeline attaches a [N]
      marker pointing at the image-derived document.
    * The LLM prompt is unchanged in shape — the synthetic chunk
      flows through the same prompt builder, citation formatter,
      and grounding checks.
    * Observability metadata (``processing_mode``, ``vision_called``,
      ``vision_cache_hit``, ``vision_latency_ms``, etc.) is merged
      into the retrieval metadata so it surfaces in chat response
      debug payloads and LangSmith traces.

OCR remains the baseline. Vision is purely additive.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import settings
from app.vision.evidence import ImageEvidence
from app.vision.orchestrator import OrchestratorOutcome, process_image_for_question

logger = logging.getLogger(__name__)


_VISION_CHUNK_MARKER = "[Phase 34B Vision Evidence]"
_MAX_VISION_CHUNK_CHARS = 1800  # bounding cap inside the chunk content


def maybe_run_vision(
    db: Optional[Session],
    *,
    question: str,
    chunks: List[Dict[str, Any]],
    image_context: Optional[Dict[str, Any]],
    ocr_text: str = "",
    ocr_confidence: Optional[int] = None,
    ocr_status: str = "success",
    ocr_error: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[OrchestratorOutcome], Dict[str, Any]]:
    """Run the Vision orchestrator and inject evidence into ``chunks``.

    Returns:
        Tuple of (modified_chunks, orchestrator_outcome, vision_meta).

        * ``modified_chunks`` is the input chunk list with a
          synthetic vision-evidence chunk prepended when Vision was
          successfully invoked.
        * ``orchestrator_outcome`` is the full outcome (None when
          the orchestrator was skipped — e.g. no image_context).
        * ``vision_meta`` is a small dict suitable for merging into
          the chat response / LangSmith trace.
    """
    if not image_context or not isinstance(image_context, dict):
        return chunks, None, _empty_meta()

    document_id = image_context.get("document_id")
    image_id = image_context.get("image_id")
    if document_id is None and image_id is None:
        return chunks, None, _empty_meta()

    # Source filename for citations: prefer the first chunk's filename
    # (the OCR-derived chunk carries the document's filename in its
    # payload). Fall back to the explicit image_context filename.
    source_filename = _resolve_source_filename(chunks, image_context)

    try:
        outcome = process_image_for_question(
            db,
            question=question,
            document_id=int(document_id) if document_id is not None else None,
            image_id=int(image_id) if image_id is not None else None,
            image_filename=_image_filename(image_context, source_filename),
            source_type=image_context.get("source_type"),
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            ocr_status=ocr_status,
            ocr_error=ocr_error,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            context_hint=_context_hint(image_context),
        )
    except Exception as exc:
        logger.warning("maybe_run_vision: orchestrator crashed: %s", exc)
        return chunks, None, _empty_meta(error=str(exc)[:200])

    vision_meta = outcome.to_dict()

    if not outcome.evidence.vision_used:
        # OCR-only or Vision failure — return chunks unchanged.
        return chunks, outcome, vision_meta

    synthetic = _build_synthetic_vision_chunk(
        evidence=outcome.evidence,
        source_filename=source_filename,
        document_id=outcome.evidence.document_id,
        image_id=outcome.evidence.image_id,
    )
    if synthetic is not None:
        # Prepend so the LLM sees the vision observation first; the
        # OCR chunks still follow. The exact ordering does not change
        # grounding semantics — the post-LLM citation attache picks
        # the best matches.
        chunks = [synthetic] + list(chunks or [])

    return chunks, outcome, vision_meta


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _empty_meta(error: Optional[str] = None) -> Dict[str, Any]:
    meta = {
        "image_processing_mode": "skipped",
        "vision_required": False,
        "vision_called": False,
        "vision_cache_hit": False,
        "vision_provider": "",
        "vision_model": "",
        "vision_latency_ms": 0,
    }
    if error:
        meta["vision_error"] = error
    return meta


def _resolve_source_filename(
    chunks: List[Dict[str, Any]],
    image_context: Dict[str, Any],
) -> str:
    for chunk in chunks or []:
        name = chunk.get("source_file_name")
        if name:
            return str(name)
    if image_context.get("filename"):
        return str(image_context["filename"])
    if image_context.get("image_id"):
        return f"image_{image_context['image_id']}.png"
    return "image.png"


def _image_filename(
    image_context: Dict[str, Any], fallback: str
) -> str:
    explicit = image_context.get("image_filename") or image_context.get(
        "original_filename"
    )
    return str(explicit) if explicit else fallback


def _context_hint(image_context: Dict[str, Any]) -> str:
    bits: List[str] = []
    if image_context.get("source_type"):
        bits.append(f"source_type: {image_context['source_type']}")
    if image_context.get("document_id"):
        bits.append(f"document_id: {image_context['document_id']}")
    return "; ".join(bits)


def _build_synthetic_vision_chunk(
    *,
    evidence: ImageEvidence,
    source_filename: str,
    document_id: Optional[int],
    image_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Render an ImageEvidence as a chunk the LLM pipeline can cite."""
    section = evidence.to_prompt_section()
    if not section:
        return None
    content = f"{_VISION_CHUNK_MARKER}\n{section}"
    if len(content) > _MAX_VISION_CHUNK_CHARS:
        content = content[:_MAX_VISION_CHUNK_CHARS].rsplit(" ", 1)[0] + "…"

    # Stable chunk_id so the citation pipeline picks the same
    # synthetic chunk every time. The seed intentionally does NOT
    # include any timestamp — cache hits and misses for the same
    # (document_id, image_id, provider, model) MUST produce the
    # same chunk_id so the LLM cites the same [N] marker across
    # repeated calls. ImageEvidence has no `vision_processed_at`
    # attribute (that lives on the DocumentImage row); including
    # `vision_processing_time_ms` here would break cache-hit
    # stability because the cached path rebuilds the VisionResult
    # with processing_time_ms=0.
    seed = f"vision|{document_id}|{image_id}|{evidence.vision_provider}|{evidence.vision_model}"
    chunk_id_seed = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    return {
        "chunk_id": f"vision-{chunk_id_seed}",
        "document_id": document_id,
        "image_id": image_id,
        "source_file_name": source_filename,
        "title": f"{_VISION_CHUNK_MARKER} {source_filename}",
        "section_heading": "Vision Evidence",
        "content": content,
        "score": None,  # synthetic — no similarity score
        # Tag so we can recognise this chunk later in observability.
        "source_type": "vision_synthetic",
        "content_type": "vision_synthetic",
        "metadata": {
            "phase": "phase34b_vision",
            "vision_used": True,
            "vision_provider": evidence.vision_provider,
            "vision_model": evidence.vision_model,
            "vision_image_type": evidence.vision_image_type,
            "vision_confidence": evidence.vision_confidence,
            "vision_cache_hit": evidence.vision_cache_hit,
            "processing_mode": evidence.processing_mode.value,
            "routing_reasons": list(evidence.routing_reasons),
        },
    }
