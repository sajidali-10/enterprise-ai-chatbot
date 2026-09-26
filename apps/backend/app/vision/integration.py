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
    auth: Optional[Any] = None,
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

    ``auth`` (Phase 34D) is the authenticated ``AuthContext``. It
    is forwarded to the advanced Vision orchestrator so its RBAC
    gate sees the same principal as the chat endpoint. When
    ``auth`` is ``None`` the advanced orchestrator refuses to
    invoke the provider (it treats the request as unauthorized),
    but Phase 34B still runs normally — Phase 34D's kill switch
    is independent.
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

    # ------------------------------------------------------------------
    # Phase 34D — Advanced Visual Understanding (additive, optional).
    #
    # Runs INDEPENDENTLY of whether Phase 34B produced a generic
    # VisionResult. In the chat flow Phase 34B often reports
    # ``vision_error='no_image_bytes_available'`` because it does not
    # self-fetch MinIO bytes; the advanced orchestrator, by contrast,
    # performs its own RBAC + MinIO fetch (``_image_fetch``) so it can
    # run whenever the request carries an authorized image_id. Gating the
    # advanced call behind Phase 34B's ``vision_used`` would silently drop
    # the entire advanced layer for every chat request.
    #
    # Single-image ``image_context`` is the trigger here; comparison has
    # its own entry point in the answer generator.
    # ------------------------------------------------------------------
    advanced_meta = _maybe_run_advanced_vision(
        db,
        auth=auth,
        question=question,
        image_id=outcome.evidence.image_id,
        ocr_text=ocr_text,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        document_id=outcome.evidence.document_id,
        context_hint=_context_hint(image_context),
    )
    if advanced_meta:
        vision_meta["advanced_vision"] = dict(advanced_meta)
        # Convenience top-level aliases for downstream consumers.
        vision_meta["advanced_vision_ran"] = bool(advanced_meta.get("advanced_vision_ran"))
        vision_meta["advanced_vision_task_type"] = advanced_meta.get("advanced_vision_task_type")
        vision_meta["advanced_vision_called"] = bool(advanced_meta.get("advanced_vision_called"))
        vision_meta["advanced_vision_cache_hit"] = bool(advanced_meta.get("advanced_vision_cache_hit"))
        vision_meta["advanced_vision_provider"] = advanced_meta.get("advanced_vision_provider", "")
        vision_meta["advanced_vision_model"] = advanced_meta.get("advanced_vision_model", "")
        vision_meta["advanced_vision_latency_ms"] = int(advanced_meta.get("advanced_vision_latency_ms", 0) or 0)
        vision_meta["advanced_vision_skipped_reason"] = advanced_meta.get("advanced_vision_skipped_reason", "")
        advanced_result = advanced_meta.get("advanced_vision_result_dict")
    else:
        advanced_result = None

    if not outcome.evidence.vision_used:
        # OCR-only or Vision failure — no Phase 34B generic synthetic
        # chunk, but IF the advanced orchestrator ran and produced a
        # usable result we still prepend an advanced-vision chunk so the
        # LLM sees and can cite the structured reasoning (the chat flow
        # commonly has Phase 34B vision_used=False due to
        # no_image_bytes_available, while advanced vision self-fetches).
        if isinstance(advanced_result, dict) and advanced_result:
            adv_chunk = _build_advanced_vision_single_chunk(
                advanced_meta=advanced_meta,
                source_filename=source_filename,
                document_id=outcome.evidence.document_id,
                image_id=outcome.evidence.image_id,
            )
            if adv_chunk is not None:
                chunks = [adv_chunk] + list(chunks or [])
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
        # Attach the structured reasoning to the synthetic chunk so
        # downstream rendering can include the JSON without re-running
        # the orchestrator.
        if isinstance(advanced_result, dict):
            synthetic.setdefault("metadata", {})["advanced_vision"] = advanced_result

    return chunks, outcome, vision_meta


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_advanced_vision_single_chunk(
    *,
    advanced_meta: Optional[Dict[str, Any]],
    source_filename: str,
    document_id: Optional[int],
    image_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Phase 34D — build a synthetic chunk from a single-image advanced
    Vision result so the LLM sees and can cite the structured reasoning.

    Used here whenever the advanced orchestrator ran and produced a
    usable result even if Phase 34B's generic VisionResult was not
    available (e.g. ``no_image_bytes_available`` on the chat path).
    Returns ``None`` when the advanced orchestrator did not actually
    run with a usable result.
    """
    if not advanced_meta or not advanced_meta.get("advanced_vision_ran"):
        return None
    result_dict = advanced_meta.get("advanced_vision_result_dict") or {}
    if not isinstance(result_dict, dict) or not result_dict:
        return None
    task_type = str(result_dict.get("task_type") or "general_visual")
    summary = str(result_dict.get("summary") or "")
    provider = str(advanced_meta.get("advanced_vision_provider") or "")
    model = str(advanced_meta.get("advanced_vision_model") or "")
    cache_hit = bool(advanced_meta.get("advanced_vision_cache_hit"))
    image_ids = list(advanced_meta.get("advanced_vision_image_ids") or [])
    img_id = int(image_id) if image_id is not None else (int(image_ids[0]) if image_ids else None)

    lines = [
        "[Phase 34D Advanced Vision Evidence]",
        f"task_type: {task_type}",
        f"summary: {summary}",
    ]
    for obs in list(result_dict.get("observations") or [])[:8]:
        lines.append(f"- {str(obs)[:200]}")
    for rel in list(result_dict.get("relationships") or [])[:8]:
        frm = str(rel.get("from") or "")
        to = str(rel.get("to") or "")
        if frm and to:
            lines.append(f"- relationship: {frm} -> {to}")
    section = "\n".join(lines)
    if len(section) > _MAX_VISION_CHUNK_CHARS:
        section = section[:_MAX_VISION_CHUNK_CHARS].rsplit(" ", 1)[0] + "…"

    seed = f"advanced_vision|{document_id}|{img_id}|{task_type}|{provider}|{model}|{cache_hit}"
    chunk_id_seed = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    image_ids_final = [img_id] if img_id else list(image_ids)

    return {
        "chunk_id": f"advanced-vision-{chunk_id_seed}",
        "document_id": document_id,
        "image_id": img_id,
        "source_file_name": source_filename,
        "title": f"[Phase 34D Advanced Vision Evidence] {source_filename}",
        "section_heading": "Advanced Vision Evidence",
        "content": section,
        "score": None,
        "source_type": "advanced_vision",
        "content_type": "advanced_vision",
        "metadata": {
            "phase": "phase34d_advanced_vision",
            "advanced_vision_task_type": task_type,
            "advanced_vision_ran": True,
            "advanced_vision_provider": provider,
            "advanced_vision_model": model,
            "advanced_vision_cache_hit": cache_hit,
            "advanced_vision_image_count": len(image_ids_final),
            "advanced_vision_image_ids": image_ids_final,
            "advanced_vision_result_dict": result_dict,
        },
    }


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


def _maybe_run_comparison_vision(
    db: Optional[Any],
    *,
    auth: Optional[Any],
    question: str,
    comparison_targets: List[Tuple[int, Optional[int]]],
    context_hint: str,
    ocr_text_a: str = "",
    ocr_text_b: str = "",
) -> Optional[Dict[str, Any]]:
    """Phase 34D — run the IMAGE_COMPARISON orchestrator on two images.

    Resolves each comparison target to a concrete ``DocumentImage.id``
    (because the comparison orchestrator requires image IDs, not just
    document IDs), enforces RBAC on BOTH images independently, then
    delegates to ``run_advanced_visual_comparison``. Returns
    ``None`` on every failure path so the chat pipeline can fall
    back to Phase 34B single-image evidence.

    The helper NEVER raises. The comparison orchestrator already
    enforces the two-gate RBAC contract internally; this helper
    only performs the document_id → image_id resolution that the
    answer generator cannot do without a DB session.
    """
    if not comparison_targets or len(comparison_targets) < 2:
        return None
    try:
        from app.services.advanced_vision.comparison import (
            run_advanced_visual_comparison,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("advanced_vision: comparison import failed: %s", exc)
        return None

    if db is None:
        return None
    try:
        from app.core.config import settings as _settings
        if not bool(getattr(_settings, "ADVANCED_VISION_ENABLED", False)):
            return None
    except Exception:
        return None

    (doc_a, img_a) = comparison_targets[0]
    (doc_b, img_b) = comparison_targets[1]

    # Resolve document_id -> image_id. If the caller already supplied
    # an image_id we use it; otherwise we pick the most recent
    # DocumentImage on the document (matches the recent-image
    # resolver's behaviour).
    image_id_a = img_a
    image_id_b = img_b
    try:
        from app.models.document import DocumentImage
        if not image_id_a:
            row = (
                db.query(DocumentImage.id)
                .filter(DocumentImage.document_id == int(doc_a))
                .order_by(DocumentImage.id.desc())
                .first()
            )
            if row is not None:
                image_id_a = int(row[0])
        if not image_id_b:
            row = (
                db.query(DocumentImage.id)
                .filter(DocumentImage.document_id == int(doc_b))
                .order_by(DocumentImage.id.desc())
                .first()
            )
            if row is not None:
                image_id_b = int(row[0])
    except Exception as exc:
        logger.debug("advanced_vision: comparison image-id resolution failed: %s", exc)
        return None

    if not image_id_a or not image_id_b:
        return None

    try:
        outcome = run_advanced_visual_comparison(
            db,
            auth=auth,
            question=question or "Compare the two images and describe what changed.",
            image_id_a=int(image_id_a),
            image_id_b=int(image_id_b),
            ocr_text_a=ocr_text_a or "",
            ocr_text_b=ocr_text_b or "",
            context_hint=context_hint or "",
        )
    except Exception as exc:
        logger.debug("advanced_vision: comparison orchestrator crashed: %s", exc)
        return None

    payload = outcome.to_advanced_outcome().to_dict()
    if outcome.result is not None:
        try:
            payload["advanced_vision_result_dict"] = outcome.result.to_dict()
        except Exception:
            payload["advanced_vision_result_dict"] = None
    else:
        payload["advanced_vision_result_dict"] = None
    # Carry image_ids for the citation renderer (both images cited).
    payload["advanced_vision_image_ids"] = list(outcome.image_ids or [])
    payload["advanced_vision_image_a_filename"] = outcome.image_a_filename
    payload["advanced_vision_image_b_filename"] = outcome.image_b_filename
    return payload


def _build_comparison_synthetic_chunks(
    *,
    advanced_meta: Dict[str, Any],
    source_filename_a: str,
    source_filename_b: str,
) -> List[Dict[str, Any]]:
    """Phase 34D — build two synthetic chunks (image A, image B) so the
    comparison produces TWO ``[N]`` citation markers — one per image.

    Both chunks share the same ``advanced_vision`` metadata
    (the comparison result) so the LLM prompt shows the change
    once; the citation renderer attaches ``[N]`` to each image
    individually. This is the only Phase 34D path that emits more
    than one synthetic chunk per request.

    Returns an empty list when the advanced meta does not carry a
    successful comparison result — callers fall back to the
    Phase 34B single-image synthetic chunk.
    """
    if not advanced_meta or not advanced_meta.get("advanced_vision_ran"):
        return []
    image_ids = list(advanced_meta.get("advanced_vision_image_ids") or [])
    if len(image_ids) < 2:
        return []
    result_dict = advanced_meta.get("advanced_vision_result_dict") or {}
    summary = str(result_dict.get("summary") or "")
    task_type = str(result_dict.get("task_type") or "image_comparison")
    changes = list(result_dict.get("comparison_changes") or [])
    lines = [
        "[Phase 34D Advanced Vision Comparison]",
        f"task_type: {task_type}",
        f"summary: {summary}",
    ]
    if changes:
        lines.append("comparison_changes:")
        for change in changes[:12]:
            subject = str(change.get("subject") or "")
            kind = str(change.get("kind") or "")
            before = str(change.get("before") or "")
            after = str(change.get("after") or "")
            evidence = str(change.get("evidence") or "")
            lines.append(
                f"- {subject}: {kind} from '{before}' to '{after}' (evidence: {evidence[:80]})"
            )
    section = "\n".join(lines)
    if len(section) > _MAX_VISION_CHUNK_CHARS:
        section = section[:_MAX_VISION_CHUNK_CHARS].rsplit(" ", 1)[0] + "…"

    image_id_a, image_id_b = int(image_ids[0]), int(image_ids[1])

    def _seed(label: str, image_id: int, filename: str) -> Dict[str, Any]:
        seed = f"advanced_vision_comparison|{image_id_a}|{image_id_b}|{label}"
        chunk_id_seed = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return {
            "chunk_id": f"advanced-vision-compare-{chunk_id_seed}",
            "document_id": None,
            "image_id": image_id,
            "source_file_name": filename,
            "title": f"[Phase 34D Advanced Vision Comparison] {filename}",
            "section_heading": "Advanced Vision Comparison",
            "content": section,
            "score": None,
            "source_type": "advanced_vision_comparison",
            "content_type": "advanced_vision_comparison",
            "metadata": {
                "phase": "phase34d_advanced_vision",
                "advanced_vision_task_type": task_type,
                "advanced_vision_ran": True,
                "advanced_vision_image_count": 2,
                "advanced_vision_image_ids": [image_id_a, image_id_b],
                "advanced_vision_provider": advanced_meta.get("advanced_vision_provider", ""),
                "advanced_vision_model": advanced_meta.get("advanced_vision_model", ""),
                "advanced_vision_cache_hit": bool(advanced_meta.get("advanced_vision_cache_hit")),
                "advanced_vision_result_dict": result_dict,
            },
        }

    return [
        _seed("A", image_id_a, source_filename_a or f"image_{image_id_a}.png"),
        _seed("B", image_id_b, source_filename_b or f"image_{image_id_b}.png"),
    ]


def _maybe_run_advanced_vision(
    db: Optional[Any],
    *,
    auth: Optional[Any] = None,
    question: str,
    image_id: Optional[int],
    ocr_text: str,
    image_bytes: Optional[bytes],
    image_mime_type: Optional[str],
    document_id: Optional[int],
    context_hint: str,
) -> Optional[Dict[str, Any]]:
    """Phase 34D — run the advanced Vision orchestrator on one image.

    Returns ``None`` when Phase 34D is disabled, when the request
    lacks a usable image_id, or when the orchestrator is not
    importable for any reason. Always returns a small dict when it
    runs so the caller can surface the advanced_vision_* fields.

    The helper NEVER raises. Failures degrade to ``{"advanced_vision_ran": False, ...}``
    so the chat pipeline can continue with the Phase 34B evidence.

    Authentication: ``auth`` (Phase 34D) is forwarded to the
    orchestrator so its RBAC gate sees the same principal as the
    chat endpoint. When ``auth`` is ``None`` the orchestrator treats
    the request as unauthorized and refuses to send bytes to the
    provider. The non-audit answer-generator path passes ``None``
    on purpose: Phase 34D is only available through the
    authenticated answer-generator paths.
    """
    if image_id is None:
        return None
    try:
        # Lazy import so a missing or partially-installed Phase 34D
        # package never breaks the chat pipeline.
        from app.services.advanced_vision.orchestrator import (
            run_advanced_visual_reasoning,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("advanced_vision: orchestrator import failed: %s", exc)
        return None

    try:
        from app.core.config import settings as _settings
        if not bool(getattr(_settings, "ADVANCED_VISION_ENABLED", False)):
            return None
    except Exception:
        return None

    # Best-effort: the integration layer is invoked from both the
    # audit and non-audit answer-generator paths. We do not have
    # the AuthContext at this seam; we delegate to the orchestrator
    # which performs its own RBAC gate. The orchestrator's first
    # gate (``_gate_authorize``) requires an auth — without one it
    # returns None and the orchestrator records ``unauthorized``.
    # That is the correct, fail-safe behaviour for the non-audit
    # path: Phase 34D is available ONLY through the authenticated
    # answer-generator paths.
    outcome = run_advanced_visual_reasoning(
        db,
        auth=auth,
        question=question or "",
        image_id=int(image_id),
        ocr_text=ocr_text or "",
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        context_hint=context_hint or "",
    )
    payload = outcome.to_dict()
    # Carry the structured reasoning dict (or None) so the synthetic
    # chunk can render it without re-running the orchestrator.
    if outcome.result is not None:
        try:
            payload["advanced_vision_result_dict"] = outcome.result.to_dict()
        except Exception:
            payload["advanced_vision_result_dict"] = None
    else:
        payload["advanced_vision_result_dict"] = None
    return payload


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
