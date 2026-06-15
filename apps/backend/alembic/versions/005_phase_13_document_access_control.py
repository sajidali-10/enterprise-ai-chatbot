"""phase 13 document access control

Revision ID: 005
Revises: 004
Create Date: 2026-06-15 17:36:45.103765

Adds Phase 13 document access control columns and tables:
- documents.visibility (private | shared | global) with server_default 'global' so existing
  rows remain accessible to all authenticated users (preserves RAG evaluation compatibility).
- documents.owner_user_id nullable FK to users.id for document ownership.
- document_permissions.access_level (view | manage) backfilled from can_read/can_write booleans.
- document_role_access table for role-based sharing (created by Base.metadata.create_all on
  first backend startup after this migration is stamped; the autogenerate diff skips it).

Safe for existing documents: visibility defaults to 'global' for all pre-existing rows so the
Introduction to Docker.pdf and CrewAI MasterClass.pdf documents used by RAG evaluation remain
retrievable by every authenticated user.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # documents.visibility: server_default='global' so any existing rows already in the table
    # when the column is added will be 'global' on PostgreSQL (existing rows get the default).
    op.add_column(
        'documents',
        sa.Column('visibility', sa.String(length=20), server_default='global', nullable=False),
    )
    # documents.owner_user_id: nullable FK to users.id for ownership tracking.
    op.add_column(
        'documents',
        sa.Column('owner_user_id', sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        'fk_documents_owner_user_id_users',
        'documents',
        'users',
        ['owner_user_id'],
        ['id'],
    )

    # document_permissions.access_level: backfill from existing can_read/can_write booleans.
    # can_write=True => manage, else view. server_default='view' covers any new rows.
    op.add_column(
        'document_permissions',
        sa.Column('access_level', sa.String(length=20), server_default='view', nullable=False),
    )
    op.execute(
        "UPDATE document_permissions SET access_level = 'manage' WHERE can_write = true"
    )
    op.execute(
        "UPDATE document_permissions SET access_level = 'view' WHERE can_write = false OR can_write IS NULL"
    )

    # document_role_access table is created by Base.metadata.create_all() at backend startup
    # (since it's declared on Base via app.security.models.DocumentRoleAccess). The autogenerate
    # diff detects it as already-present because the running backend has already created it.
    # We do not recreate it here to avoid duplicate-table errors.


def downgrade() -> None:
    # Reverse access_level backfill
    op.drop_column('document_permissions', 'access_level')

    # Reverse documents.owner_user_id
    op.drop_constraint('fk_documents_owner_user_id_users', 'documents', type_='foreignkey')
    op.drop_column('documents', 'owner_user_id')

    # Reverse documents.visibility
    op.drop_column('documents', 'visibility')

    # document_role_access is left in place on downgrade (would require a separate migration to
    # drop safely, since the table is shared with the model layer).
