"""add token_version column to users table

Revision ID: 007
Revises: 006
Create Date: 2026-06-16

Phase 14: Adds a token_version column to the users table for JWT session
invalidation. When a user's password is reset or account is deactivated,
token_version is incremented, causing any existing JWT tokens to be rejected.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column(
            'token_version',
            sa.Integer(),
            nullable=False,
            server_default='1',
        )
    )
    op.create_index('ix_users_token_version', 'users', ['token_version'])


def downgrade() -> None:
    op.drop_index('ix_users_token_version', 'users')
    op.drop_column('users', 'token_version')
