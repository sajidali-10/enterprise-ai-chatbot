"""add DOCUMENT_ACCESS_CHANGE to auditaction enum

Revision ID: 006
Revises: 005
Create Date: 2026-06-15 18:05:00.000000

Phase 13: Adds the DOCUMENT_ACCESS_CHANGE value to the auditaction enum so
the new /api/admin/documents/{id}/access PATCH endpoint can log share /
visibility / owner changes via AuditAction.DOCUMENT_ACCESS_CHANGE.

PostgreSQL enum values cannot be removed (only added), so this migration
is one-way; rolling back is harmless because the value is never written
by older code paths.
"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ALTER TYPE ... ADD VALUE cannot run inside a transaction in older PostgreSQL
    # versions. We use autocommit via the op.execute path which is fine for PG 12+.
    op.execute("ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'DOCUMENT_ACCESS_CHANGE'")


def downgrade() -> None:
    # PostgreSQL does not support removing enum values. No-op downgrade.
    pass
