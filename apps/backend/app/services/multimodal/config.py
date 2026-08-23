"""Phase 34C -- Multimodal constants.

Centralised constants so the indexing / retrieval / lifecycle modules
all agree on the canonical ``source_type`` and the deterministic
point-id namespace.

The source type ``image_knowledge`` is intentionally distinct from the
existing OCR-derived ``image_ocr`` / ``pdf_ocr`` / ``docx_image_ocr``
values. ``image_ocr`` represents *raw text extracted from pixels*;
``image_knowledge`` represents the *searchable multimodal
representation* built from OCR + Vision evidence in Phase 34C.
"""

from __future__ import annotations


# Canonical source_type for Phase 34C image knowledge points. Mirrors
# the way Phase 34A.1.1 distinguishes OCR chunks via their source_type
# payload field (see app/services/vector/qdrant_service.py:upsert_chunks).
MULTIMODAL_SOURCE_TYPE: str = "image_knowledge"


# Bumped in lockstep with the knowledge_text format. Embedded in the
# deterministic Qdrant point id so a schema change writes fresh points
# and leaves stale points in place (reindex / backfill cleans them).
DEFAULT_SCHEMA_VERSION: int = 1


# Stable namespace string used when hashing (document_id, image_id,
# schema_version) into a deterministic integer point id. Changing this
# string invalidates every existing point; only do it alongside a
# schema version bump.
IMAGE_KNOWLEDGE_POINT_NAMESPACE: str = "hiplink:image_knowledge:v1"


# Hard upper bound on knowledge_text characters BEFORE embedding. The
# indexer also applies the runtime cap from
# ``settings.MULTIMODAL_KNOWLEDGE_TEXT_MAX_CHARS``; this module-level
# constant is the *defensive* ceiling used when settings cannot be
# loaded (e.g. during early import in tests).
KNOWLEDGE_TEXT_HARD_CAP_CHARS: int = 4000


# Maximum number of structured list items (findings / entities /
# states / tags) included in knowledge_text. Mirrors the
# MAX_FINDINGS / MAX_ENTITIES / MAX_STATES / MAX_TAGS caps used by
# app/vision/evidence.py. Keeping these aligned prevents the knowledge
# text from diverging from the evidence the LLM sees.
MAX_FINDINGS: int = 6
MAX_ENTITIES: int = 12
MAX_STATES: int = 6
MAX_TAGS: int = 8
