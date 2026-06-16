"""extend AuditAction enum with Phase 14 security events

Revision ID: 008
Revises: 007
Create Date: 2026-06-16

Phase 14: Adds new security-related audit actions to the auditaction enum:
- login_success, login_failure, logout, password_reset
- user_created, user_updated, user_deactivated, role_changed
- document_reindex
- rag_access_denied, admin_access_denied
"""
from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None

_NEW_VALUES = [
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
    for value in _NEW_VALUES:
        op.execute(f"ALTER TYPE auditaction ADD VALUE IF NOT EXISTS '{value}'")


def downgrade() -> None:
    # PostgreSQL does not support removing enum values. No-op downgrade.
    pass
