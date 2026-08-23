"""Phase 34C -- Image Knowledge Record builder.

A pure, deterministic builder that turns the persisted Phase 34A OCR
data + Phase 34B Vision fields on a ``DocumentImage`` row into an
``ImageKnowledgeRecord`` and a canonical ``knowledge_text`` string
suitable for embedding.

Design rules (per the Phase 34C plan):

* NEVER call the Vision / LLM / embedding provider here. We only READ
  persisted columns on ``DocumentImage`` and the OCR chunk content
  already in Qdrant. This keeps the indexing path idempotent and
  free of provider cost.
* ``build_knowledge_text`` is a pure function of the record. Same
  input -> same output, every time.
* ``compute_point_id`` is a stable integer hash over
  ``(document_id, image_id, schema_version)``; it does NOT depend on
  timestamps or call counts. Re-running indexing for the same image
  hits the same Qdrant point id and overwrites in place.
* Length cap on knowledge_text is enforced so a hostile Vision
  response cannot inflate the embedding budget. The cap is taken
  from ``settings.MULTIMODAL_KNOWLEDGE_TEXT_MAX_CHARS`` with a hard
  fallback constant.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from app.services.multimodal.config import (
    DEFAULT_SCHEMA_VERSION,
    IMAGE_KNOWLEDGE_POINT_NAMESPACE,
    KNOWLEDGE_TEXT_HARD_CAP_CHARS,
    MAX_ENTITIES,
    MAX_FINDINGS,
    MAX_STATES,
    MAX_TAGS,
    MULTIMODAL_SOURCE_TYPE,
)
from app.services.vector.qdrant_service import fetch_chunks_by_image_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ImageKnowledgeRecord:
    """Structured representation of one image, ready to embed.

    Mirrors the canonical fields documented in the Phase 34C plan.
    ``knowledge_text`` is the deterministic string we embed;
    ``embedding_text`` is the (possibly trimmed) text actually sent to
    the embedding provider. They are equal for non-empty knowledge.
    """

    document_id: int
    image_id: int
    owner_user_id: Optional[int]
    visibility: str
    document_version_id: Optional[int] = None
    source_type: str = MULTIMODAL_SOURCE_TYPE
    original_filename: Optional[str] = None
    mime_type: Optional[str] = None
    image_type: Optional[str] = None
    ocr_text: str = ""
    ocr_confidence: Optional[int] = None
    ocr_status: Optional[str] = None
    vision_description: str = ""
    visual_findings: List[str] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)
    visual_states: List[str] = field(default_factory=list)
    vision_tags: List[str] = field(default_factory=list)
    vision_provider: str = ""
    vision_model: str = ""
    vision_processed_at: Optional[str] = None
    vision_status: Optional[str] = None
    is_ocr_only: bool = True
    knowledge_schema_version: int = DEFAULT_SCHEMA_VERSION
    knowledge_text: str = ""
    embedding_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return default


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except Exception:
        return None
    return max(0, min(100, n))


def _clip(text: str, max_chars: int) -> str:
    """Word-boundary clip a string to ``max_chars`` with an ellipsis.

    Avoids mid-word truncation. Falls back to a hard cut when no good
    boundary is found. Used to keep knowledge_text bounded so a
    pathological Vision response cannot inflate the embedding budget.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[: max(1, max_chars - 1)]
    last_space = truncated.rfind(" ")
    if last_space >= int(max_chars * 0.6):
        return truncated[:last_space].rstrip() + "..."
    return truncated.rstrip() + "..."


def _join_list(items: List[str], max_items: int) -> str:
    """Deduplicate, truncate, and comma-join a list of strings."""
    if not items:
        return ""
    seen: set = set()
    out: List[str] = []
    for raw in items:
        if raw is None:
            continue
        s = _safe_str(raw).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_items:
            break
    return ", ".join(out)


def build_knowledge_text(record: ImageKnowledgeRecord) -> str:
    """Assemble the deterministic knowledge_text for a record.

    Pure function -- no DB or network access. Same record -> same text.
    Sections with no content are omitted entirely (NOT left blank) so
    the resulting text is compact and stable across Vision enrichment
    transitions.
    """
    sections: List[str] = []

    # Header line. Always present so retrieval-side heuristics can
    # detect "this is image knowledge" by the leading token.
    if record.original_filename:
        sections.append(f"Source image: {record.original_filename}")
    if record.image_type:
        sections.append(f"Image type: {record.image_type}")

    if record.ocr_status == "success" and record.ocr_text:
        sections.append("OCR:")
        sections.append(_clip(record.ocr_text, 900))

    if record.vision_status == "success" and record.vision_description:
        sections.append("Visual description:")
        sections.append(_clip(record.vision_description, 700))

    findings = _join_list(record.visual_findings, MAX_FINDINGS)
    if findings:
        sections.append("Visual findings:")
        for finding in findings.split(", "):
            sections.append(f"- {_clip(finding, 200)}")

    entities = _join_list(record.detected_entities, MAX_ENTITIES)
    if entities:
        sections.append(f"Entities: {entities}")

    states = _join_list(record.visual_states, MAX_STATES)
    if states:
        sections.append(f"Visual states: {states}")

    tags = _join_list(record.vision_tags, MAX_TAGS)
    if tags:
        sections.append(f"Tags: {tags}")

    if not record.ocr_text and not record.vision_description:
        # Neither OCR nor Vision contributed anything -- the image
        # had no extractable content. Surface a single-line marker so
        # the indexed chunk is still semantically grounded (an empty
        # embedding would degrade neighbour retrieval).
        sections.append("(no extractable image content)")

    return "\n".join(sections)


def cap_knowledge_text(text: str, max_chars: Optional[int] = None) -> str:
    """Apply the runtime cap from settings, falling back to the hard cap."""
    if not text:
        return ""
    try:
        from app.core.config import settings

        cap = int(max_chars if max_chars is not None else settings.MULTIMODAL_KNOWLEDGE_TEXT_MAX_CHARS)
        if cap <= 0:
            cap = KNOWLEDGE_TEXT_HARD_CAP_CHARS
    except Exception:
        cap = max_chars or KNOWLEDGE_TEXT_HARD_CAP_CHARS
    return _clip(text, cap)


def compute_point_id(
    document_id: int,
    image_id: int,
    schema_version: int = DEFAULT_SCHEMA_VERSION,
) -> int:
    """Deterministic positive integer id for the Qdrant point.

    SHA-256 prefix over a stable namespace string; coerced to a
    positive Python int. Independent of timestamps, so repeated
    indexing of the same image hits the same point and overwrites
    in place (idempotency requirement from the Phase 34C plan).
    """
    key = f"{IMAGE_KNOWLEDGE_POINT_NAMESPACE}:{int(document_id)}:{int(image_id)}:{int(schema_version)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Take the first 8 bytes as an unsigned 64-bit int, then keep it
    # positive. Qdrant accepts both, but staying positive keeps the
    # point id namespace disjoint from any negative-id collisions.
    value = int.from_bytes(digest[:8], "big", signed=False)
    return value & 0x7FFFFFFFFFFFFFFF


def compute_content_hash(text: str) -> str:
    """Stable short hash for observability / diagnostics only."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------


def _load_ocr_text_for_image(image_id: int) -> str:
    """Read the OCR text for an image from the already-indexed Qdrant
    chunk (the Phase 34A path stores OCR text in chunk payloads, not
    on the ``DocumentImage`` row itself -- only ``ocr_text_hash`` is
    persisted on the row).
    """
    if not image_id:
        return ""
    try:
        chunks = fetch_chunks_by_image_id(int(image_id), limit=8)
    except Exception as exc:
        logger.debug("multimodal: failed to fetch OCR chunks for image_id=%s: %s", image_id, exc)
        return ""
    if not chunks:
        return ""
    # Concatenate, de-duplicating by chunk_index. Keep order stable.
    chunks.sort(key=lambda c: (
        int(c.get("chunk_index") or 0) if c.get("chunk_index") is not None else 0,
        str(c.get("chunk_id") or ""),
    ))
    seen: set = set()
    pieces: List[str] = []
    for chunk in chunks:
        cid = str(chunk.get("chunk_id") or "")
        if cid in seen:
            continue
        seen.add(cid)
        content = (chunk.get("content") or "").strip()
        if content:
            pieces.append(content)
    return "\n".join(pieces)


def load_image_record(
    db,
    document_image_id: int,
    schema_version: Optional[int] = None,
) -> Optional[ImageKnowledgeRecord]:
    """Build an ``ImageKnowledgeRecord`` from persisted DB state.

    Returns ``None`` if the image row is missing or has no usable
    content (no OCR success and no Vision). The caller is responsible
    for handling ``None`` -- this function NEVER raises for normal
    missing-data cases.

    The function does NOT call the Vision provider. Vision is read
    from the persisted Phase 34B columns on ``DocumentImage``.
    """
    if not document_image_id:
        return None
    try:
        from app.models.document import Document, DocumentImage
    except Exception as exc:
        logger.error("multimodal: cannot import DocumentImage model: %s", exc)
        return None

    try:
        img = (
            db.query(DocumentImage)
            .filter(DocumentImage.id == int(document_image_id))
            .first()
        )
    except Exception as exc:
        logger.error("multimodal: DB error loading DocumentImage %s: %s", document_image_id, exc)
        return None

    if img is None:
        return None

    doc = None
    try:
        if getattr(img, "document_id", None) is not None:
            doc = db.query(Document).filter(Document.id == int(img.document_id)).first()
    except Exception as exc:
        logger.debug("multimodal: cannot load parent Document for image_id=%s: %s", document_image_id, exc)

    owner_user_id = getattr(doc, "owner_user_id", None) if doc is not None else None
    visibility = getattr(doc, "visibility", "global") if doc is not None else "global"

    ocr_text = _load_ocr_text_for_image(int(img.id)) if getattr(img, "ocr_status", None) == "success" else ""

    vision_status = getattr(img, "vision_status", None) or None
    has_vision = vision_status == "success"

    # Build a raw record first so the deterministic text builder can
    # operate on the canonical structure.
    record = ImageKnowledgeRecord(
        document_id=int(getattr(img, "document_id") or 0),
        image_id=int(img.id),
        owner_user_id=owner_user_id,
        visibility=_safe_str(visibility, "global"),
        document_version_id=getattr(img, "document_version_id", None),
        original_filename=getattr(img, "original_filename", None),
        mime_type=getattr(img, "mime_type", None),
        image_type=getattr(img, "vision_image_type", None) if has_vision else None,
        ocr_text=ocr_text,
        ocr_confidence=_safe_int(getattr(img, "ocr_confidence", None)),
        ocr_status=getattr(img, "ocr_status", None),
        vision_description=_safe_str(getattr(img, "vision_description", None)) if has_vision else "",
        visual_findings=list(getattr(img, "vision_findings", None) or []) if has_vision else [],
        detected_entities=list(getattr(img, "vision_entities", None) or []) if has_vision else [],
        visual_states=list(getattr(img, "visual_states", None) or []) if has_vision else [],
        vision_tags=list(getattr(img, "vision_tags", None) or []) if has_vision else [],
        vision_provider=_safe_str(getattr(img, "vision_provider", None)) if has_vision else "",
        vision_model=_safe_str(getattr(img, "vision_model", None)) if has_vision else "",
        vision_processed_at=(
            getattr(img, "vision_processed_at").isoformat()
            if has_vision and getattr(img, "vision_processed_at", None) is not None
            else None
        ),
        vision_status=vision_status,
        is_ocr_only=not has_vision,
        knowledge_schema_version=int(schema_version if schema_version is not None else DEFAULT_SCHEMA_VERSION),
    )

    record.knowledge_text = build_knowledge_text(record)
    record.embedding_text = cap_knowledge_text(record.knowledge_text)
    return record


# ---------------------------------------------------------------------------
# Re-export the source type so callers can import from one place
# ---------------------------------------------------------------------------


__all__ = [
    "ImageKnowledgeRecord",
    "build_knowledge_text",
    "cap_knowledge_text",
    "compute_point_id",
    "compute_content_hash",
    "load_image_record",
    "MULTIMODAL_SOURCE_TYPE",
]
