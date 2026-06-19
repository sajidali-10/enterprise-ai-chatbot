"""repair_audit_action_enum

Revision ID: 010
Revises: 009
Create Date: 2026-06-19

Repair: Migration 008 extended the auditaction enum with Phase 14 values,
but the database was stamped from 006 → 009, skipping 008. This migration
re-applies the missing ADD VALUE statements idempotently.

Missing values (from migration 008):
- RAG_ACCESS_DENIED
- DOCUMENT_REINDEX
- LOGIN_SUCCESS
- LOGIN_FAILURE
- PASSWORD_RESET
- USER_CREATED
- USER_UPDATED
- USER_DEACTIVATED
- ROLE_CHANGED
- ADMIN_ACCESS_DENIED
"""

from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None

_MISSING_VALUES = [
    "RAG_ACCESS_DENIED",
    "DOCUMENT_REINDEX",
    "LOGIN_SUCCESS",
    "LOGIN_FAILURE",
    "PASSWORD_RESET",
    "USER_CREATED",
    "USER_UPDATED",
    "USER_DEACTIVATED",
    "ROLE_CHANGED",
    "ADMIN_ACCESS_DENIED",
]


def upgrade() -> None:
    for value in _MISSING_VALUES:
        op.execute(f"ALTER TYPE auditaction ADD VALUE IF NOT EXISTS '{value}'")


def downgrade() -> None:
    # PostgreSQL does not support removing enum values. No-op downgrade.
    pass