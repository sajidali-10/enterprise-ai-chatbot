"""Phase 34C -- Persistent Multimodal Knowledge.

This package turns persisted Phase 34A OCR + Phase 34B Vision analysis
into searchable, durable image knowledge points in the existing Qdrant
collection (new ``source_type=image_knowledge``). It does NOT call any
Vision / LLM provider during indexing and does NOT change Phase 34B
behaviour.

Module layout:

    config.py           -- constants (source type, defaults)
    knowledge_record.py -- deterministic ImageKnowledgeRecord builder
    indexer.py          -- idempotent Qdrant upsert / delete
    retrieval.py        -- retrieval overlay (boost + authority demote)
    lifecycle.py        -- delete / reindex / vision-enrichment hooks
    scripts/backfill.py -- explicit, idempotent admin backfill CLI
"""

from app.services.multimodal.config import (
    DEFAULT_SCHEMA_VERSION,
    MULTIMODAL_SOURCE_TYPE,
    IMAGE_KNOWLEDGE_POINT_NAMESPACE,
)

__all__ = [
    "DEFAULT_SCHEMA_VERSION",
    "MULTIMODAL_SOURCE_TYPE",
    "IMAGE_KNOWLEDGE_POINT_NAMESPACE",
]
