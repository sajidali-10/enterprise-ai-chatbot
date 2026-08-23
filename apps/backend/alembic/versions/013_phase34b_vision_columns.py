"""phase34b_vision_columns

Revision ID: 013
Revises: 012
Create Date: 2026-08-18

Phase 34B: Automatic Vision Intelligence.

Adds additive, nullable Vision columns to ``document_images`` so the
OCR-first pipeline can persist Vision analysis results alongside OCR
metadata without changing any existing column semantics. All columns
are NULLABLE so existing rows from Phase 34A / 34A.1 / 34A.1.1 /
34A.1.2 / 34A.2 / 34A.2.1 remain valid after upgrade.

New columns on ``document_images``:

    vision_status           VARCHAR(32)   NULL
        pending | success | failed | disabled | skipped
    vision_provider         VARCHAR(40)   NULL  (e.g. mock | openai-compatible)
    vision_model            VARCHAR(120)  NULL
    vision_description      TEXT          NULL
    vision_image_type       VARCHAR(40)   NULL  (dashboard | chart | ...)
    vision_findings         JSONB         NULL  (structured array)
    vision_entities         JSONB         NULL  (structured array)
    vision_states           JSONB         NULL  (structured array)
    vision_tags             JSONB         NULL  (structured array)
    vision_confidence       INTEGER       NULL  (0-100)
    vision_processed_at     TIMESTAMP     NULL
    vision_error            TEXT          NULL
    vision_cache_key        VARCHAR(200)  NULL
        deterministic cache key (sha256 prefix) for reuse across
        (document_image_id, provider, model, schema_version).

JSONB is used for structured arrays to match PostgreSQL/project
conventions. All existing Phase 34A rows keep NULL — no destructive
change. Downgrade drops every new column safely.
"""

from alembic import op
import sqlalchemy as sa


revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


VISION_COLUMNS = [
    ("vision_status", sa.String(length=32), "pending | success | failed | disabled | skipped"),
    ("vision_provider", sa.String(length=40), "Vision provider name (e.g. mock)"),
    ("vision_model", sa.String(length=120), "Vision model identifier"),
    ("vision_description", sa.Text(), "Free-text structured description of image"),
    ("vision_image_type", sa.String(length=40), "dashboard | chart | diagram | ..."),
    ("vision_findings", sa.dialects.postgresql.JSONB(), "Structured visual_findings[]"),
    ("vision_entities", sa.dialects.postgresql.JSONB(), "Structured detected_entities[]"),
    ("vision_states", sa.dialects.postgresql.JSONB(), "Structured visual_states[]"),
    ("vision_tags", sa.dialects.postgresql.JSONB(), "Structured tags[]"),
    ("vision_confidence", sa.Integer(), "0..100 Vision provider confidence"),
    ("vision_processed_at", sa.DateTime(), "Timestamp of last successful Vision analysis"),
    ("vision_error", sa.Text(), "Last Vision error (if any)"),
    ("vision_cache_key", sa.String(length=200), "Stable cache key for reuse decisions"),
]


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "document_images" not in insp.get_table_names():
        # Defensive: the table must already exist (Phase 34A migration 012).
        # If somehow it does not, this migration cannot proceed.
        raise RuntimeError(
            "phase34b_vision_columns: document_images table missing. "
            "Apply Phase 34A migration 012 first."
        )

    existing = {c["name"] for c in insp.get_columns("document_images")}

    for col_name, col_type, _comment in VISION_COLUMNS:
        if col_name in existing:
            continue
        op.add_column(
            "document_images",
            sa.Column(col_name, col_type, nullable=True),
        )

    # Index the cache key so cache lookup stays cheap as the table grows.
    if not insp.get_indexes("document_images"):
        pass
    index_names = {i["name"] for i in insp.get_indexes("document_images")}
    if "ix_document_images_vision_cache_key" not in index_names:
        op.create_index(
            "ix_document_images_vision_cache_key",
            "document_images",
            ["vision_cache_key"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "document_images" not in insp.get_table_names():
        return

    # Drop the cache-key index first.
    index_names = {i["name"] for i in insp.get_indexes("document_images")}
    if "ix_document_images_vision_cache_key" in index_names:
        op.drop_index("ix_document_images_vision_cache_key", "document_images")

    existing = {c["name"] for c in insp.get_columns("document_images")}
    for col_name, _col_type, _comment in VISION_COLUMNS:
        if col_name in existing:
            op.drop_column("document_images", col_name)
