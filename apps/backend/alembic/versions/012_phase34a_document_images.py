"""phase34a_document_images

Revision ID: 012
Revises: 011
Create Date: 2026-08-09

Phase 34A: OCR & image ingestion foundation.

This migration is additive only. It does NOT touch any existing tables
or columns.

Changes:
    1. Create ``document_images`` table to record every image asset
       associated with a document, whether it was uploaded directly,
       rendered from a scanned PDF page, or extracted from a DOCX.
       Each row captures OCR provider, status, confidence, storage
       key, dimensions, and provenance. This is the foundation for
       future Phase 34B Vision columns (vision_provider,
       vision_description, vision_tags, image_type, detected_entities,
       objects, vision_confidence) which will be added as nullable
       columns in a separate migration.
    2. Add nullable ``extraction_summary`` JSON-ish text column to
       ``document_versions`` to record per-version extraction
       statistics (native chars vs OCR chars, completeness flag,
       warnings). Existing rows keep ``NULL`` — no destructive change.
"""

from alembic import op
import sqlalchemy as sa


revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_images",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("document_version_id", sa.Integer(), sa.ForeignKey("document_versions.id"), nullable=True),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column(
            "source_type",
            sa.String(length=40),
            nullable=False,
        ),  # direct_image | pdf_page_ocr | docx_image_ocr
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("sequence_number", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("original_filename", sa.String(length=512), nullable=True),
        sa.Column(
            "ocr_provider",
            sa.String(length=40),
            nullable=True,
        ),  # tesseract | null (when ocr disabled / not run)
        sa.Column(
            "ocr_status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),  # pending | success | failed | disabled | empty | low_confidence
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("ocr_text_hash", sa.String(length=64), nullable=True),
        sa.Column("ocr_error", sa.Text(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_document_images_document_id", "document_images", ["document_id"])
    op.create_index(
        "ix_document_images_document_id_source_type",
        "document_images",
        ["document_id", "source_type"],
    )
    op.create_index(
        "ix_document_images_document_id_sequence",
        "document_images",
        ["document_id", "sequence_number"],
    )

    # Add extraction_summary to document_versions (nullable).
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "document_versions" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("document_versions")}
        if "extraction_summary" not in cols:
            op.add_column(
                "document_versions",
                sa.Column("extraction_summary", sa.Text(), nullable=True),
            )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "document_versions" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("document_versions")}
        if "extraction_summary" in cols:
            op.drop_column("document_versions", "extraction_summary")

    if "document_images" in insp.get_table_names():
        op.drop_index("ix_document_images_document_id_sequence", "document_images")
        op.drop_index("ix_document_images_document_id_source_type", "document_images")
        op.drop_index("ix_document_images_document_id", "document_images")
        op.drop_table("document_images")
