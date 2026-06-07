"""add pg full-text search index on document_chunks content
Revision ID: 003
Revises: 002
Create Date: 2025-01-01
"""
from alembic import op

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    # Enable pg_trgm extension for trigram similarity
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    
    # Create GIN index for full-text search on content
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_document_chunks_content_fts
        ON document_chunks
        USING GIN (to_tsvector('english', content))
    """)
    
    # Create trigram similarity index for flexible matching
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_document_chunks_content_trgm
        ON document_chunks
        USING GIN (content gin_trgm_ops)
    """)


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_content_trgm")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_content_fts")
    # Note: We don't drop the extension as other tables might use it