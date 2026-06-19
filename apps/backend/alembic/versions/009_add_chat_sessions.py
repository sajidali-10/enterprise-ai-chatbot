"""add_chat_sessions tables

Revision ID: 009
Revises: 008
Create Date: 2026-06-19

Phase 20A: Add persistent chat sessions and message storage.
- chat_sessions: stores session metadata (user, title, mode, timestamps)
- chat_messages: stores individual messages with citations/metadata
- chat_message_feedback: optional feedback on messages
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- chat_sessions -------------------------------------------------------
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False, server_default="New Chat"),
        sa.Column(
            "mode",
            sa.Enum("GENERAL", "RAG", name="chatsessionmode"),
            nullable=False,
            server_default="GENERAL",
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_chat_sessions_id", "chat_sessions", ["id"])
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_index(
        "ix_chat_sessions_user_created", "chat_sessions", ["user_id", "created_at"]
    )

    # ---- chat_messages -------------------------------------------------------
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "session_id", sa.Integer(), sa.ForeignKey("chat_sessions.id"), nullable=False
        ),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "role", sa.Enum("USER", "ASSISTANT", "SYSTEM", name="messagerole"), nullable=False
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=True),
        sa.Column("retrieved_documents_json", sa.Text(), nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("provider_used", sa.String(50), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_chat_messages_id", "chat_messages", ["id"])
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.create_index(
        "ix_chat_messages_session_created", "chat_messages", ["session_id", "created_at"]
    )

    # ---- chat_message_feedback -----------------------------------------------
    op.create_table(
        "chat_message_feedback",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "message_id",
            sa.Integer(),
            sa.ForeignKey("chat_messages.id"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("rating", sa.String(20), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False
        ),
    )
    op.create_index("ix_chat_message_feedback_id", "chat_message_feedback", ["id"])
    op.create_index("ix_chat_message_feedback_message_id", "chat_message_feedback", ["message_id"])


def downgrade() -> None:
    op.drop_table("chat_message_feedback")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.execute("DROP TYPE IF EXISTS chatsessionmode")
    op.execute("DROP TYPE IF EXISTS messagerole")