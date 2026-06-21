"""
Admin RAG Configuration API

Provides a safe, admin-only endpoint for RAG configuration status.
No secrets, API keys, tokens, or raw .env values are exposed.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.security.dependencies import require_admin
from app.security.auth import AuthContext
from app.services.rag_config import get_rag_config

router = APIRouter(prefix="/api/admin/rag", tags=["Admin RAG Config"])


@router.get("/config")
def get_rag_config_status(
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
):
    """
    Get current RAG configuration summary (admin only).

    Returns safe, non-sensitive RAG settings including:
    - Active pipeline, loader, splitter, retriever, reranker providers
    - Embedding model and dimension
    - Vector store provider
    - Retrieval behavior (top_k, score threshold)
    - Chunking parameters
    - Conversation context settings
    - Feature flags (LangChain, reranker)

    No secrets, API keys, tokens, passwords, or raw .env values are exposed.
    No database migration is required for this endpoint.
    """
    _ = db  # explicit DB dependency to ensure session lifecycle is correct
    return get_rag_config()