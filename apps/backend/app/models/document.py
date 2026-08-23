from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, func, JSON
from sqlalchemy.orm import relationship
from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    status = Column(String, default="pending")
    # Phase 13: visibility (private | shared | global) and ownership.
    visibility = Column(String(20), nullable=False, server_default="global")
    owner_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    images = relationship("DocumentImage", back_populates="document", cascade="all, delete-orphan")
    # Owner relationship is resolved lazily via string reference to avoid circular imports
    # (app.security.models.User is imported elsewhere; SQLAlchemy resolves at mapper-config time).
    owner = relationship("User", foreign_keys=[owner_user_id], lazy="joined")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    version_number = Column(Integer, default=1)
    storage_key = Column(String, nullable=False)
    extractor_type = Column(String, nullable=True)
    extracted_text = Column(Text, nullable=True)
    # Phase 34A: JSON-ish summary of extraction quality (native vs OCR chars,
    # completeness flag, warnings). Nullable; existing rows remain untouched.
    extraction_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    document = relationship("Document", back_populates="versions")
    images = relationship(
        "DocumentImage",
        back_populates="document_version",
        cascade="all, delete-orphan",
    )


class DocumentImage(Base):
    """Phase 34A: image assets associated with a document.

    One row per image — whether the image was uploaded directly,
    rendered from a scanned PDF page, or extracted from a DOCX.
    Captures OCR provider, status, confidence, storage key, dimensions,
    and provenance.

    Phase 34B (Automatic Vision Intelligence) — adds nullable Vision
    columns populated by the Vision orchestrator's cache/persistence
    layer. Vision results are STRICTLY ADDITIONAL to OCR; OCR remains
    the default evidence source. The columns are nullable so all
    Phase 34A / 34A.1 / 34A.1.1 / 34A.1.2 / 34A.2 / 34A.2.1 rows
    remain valid after migration 013.

    See:
      * alembic/versions/013_phase34b_vision_columns.py (schema)
      * app/vision/persistence.py (read/write logic)
      * app/vision/orchestrator.py (cache lookup + write-through)
    """

    __tablename__ = "document_images"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    document_version_id = Column(
        Integer, ForeignKey("document_versions.id"), nullable=True
    )
    storage_key = Column(String(length=512), nullable=False)
    mime_type = Column(String(length=100), nullable=False)
    # direct_image | pdf_page_ocr | docx_image_ocr
    source_type = Column(String(length=40), nullable=False)
    page_number = Column(Integer, nullable=True)
    sequence_number = Column(Integer, nullable=False, default=0)
    original_filename = Column(String(length=512), nullable=True)
    # tesseract | null (when ocr disabled / not run)
    ocr_provider = Column(String(length=40), nullable=True)
    # pending | success | failed | disabled | empty | low_confidence
    ocr_status = Column(String(length=32), nullable=False, default="pending")
    ocr_confidence = Column(Integer, nullable=True)  # 0-100; nullable
    ocr_text_hash = Column(String(length=64), nullable=True)
    ocr_error = Column(Text, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    byte_size = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # ------------------------------------------------------------------
    # Phase 34B — Vision Intelligence columns (migration 013)
    # All nullable. Existing Phase 34A rows keep NULL — non-destructive.
    # ------------------------------------------------------------------
    # pending | success | failed | disabled | skipped
    vision_status = Column(String(length=32), nullable=True)
    # e.g. "mock", "openai-compatible"
    vision_provider = Column(String(length=40), nullable=True)
    # Provider-specific model identifier (e.g. "mock-v1", "gpt-4o-mini")
    vision_model = Column(String(length=120), nullable=True)
    # Structured description used in the Evidence Builder prompt section.
    vision_description = Column(Text, nullable=True)
    # Cheap image-type inference from the Vision provider.
    vision_image_type = Column(String(length=40), nullable=True)
    # JSONB arrays of structured findings / entities / states / tags.
    # We deliberately use the wide-open JSON column type so the
    # provider can return arbitrary structured data without breaking
    # the schema.
    vision_findings = Column(JSON, nullable=True)
    vision_entities = Column(JSON, nullable=True)
    visual_states = Column("vision_states", JSON, nullable=True)
    vision_tags = Column(JSON, nullable=True)
    # 0..100 — coerced by the Evidence Builder to the valid range.
    vision_confidence = Column(Integer, nullable=True)
    # Timestamp of the most recent successful Vision analysis.
    vision_processed_at = Column(DateTime, nullable=True)
    # Safe error string from the most recent failed attempt.
    vision_error = Column(Text, nullable=True)
    # sha256 prefix used as the cache key for reuse decisions.
    # Bumping VISION_CACHE_SCHEMA_VERSION invalidates ALL persisted
    # rows at once.
    vision_cache_key = Column(String(length=200), nullable=True)

    document = relationship("Document", back_populates="images")
    document_version = relationship("DocumentVersion", back_populates="images")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    document_version_id = Column(Integer, ForeignKey("document_versions.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    title = Column(String, nullable=True)
    section_heading = Column(String, nullable=True)
    page_number = Column(Integer, nullable=True)
    source_file_name = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="chunks")
