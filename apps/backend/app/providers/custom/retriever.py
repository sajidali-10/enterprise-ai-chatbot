"""
Custom Retriever Adapter

Wraps the existing retriever functions from app/rag/retriever.py.
Supports both permission-aware (retrieve_chunks_with_auth) and
permission-free (retrieve_chunks_with_settings) retrieval.
"""

from typing import Any

from app.rag.retriever import (
    retrieve_chunks,
    retrieve_chunks_with_settings,
    retrieve_chunks_with_auth,
)
from app.core.config import settings
from app.providers.base import RetrieverProvider, ChunkDict


class CustomRetrieverProvider(RetrieverProvider):
    """
    Phase 22 active provider: delegates to retriever.py functions.

    - With auth: calls retrieve_chunks_with_auth (permission filtering active).
    - Without auth: calls retrieve_chunks_with_settings (admin-level access).
    """

    def __init__(self):
        self._vector_top_k = settings.RETRIEVAL_VECTOR_TOP_K
        self._keyword_top_k = settings.RETRIEVAL_KEYWORD_TOP_K
        self._final_top_k = settings.RETRIEVAL_FINAL_TOP_K
        self._min_score = settings.RETRIEVAL_MIN_SCORE

    def retrieve(
        self,
        query: str,
        auth: Any | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
        debug: bool = False,
    ) -> tuple[list[ChunkDict], dict]:
        if auth is not None:
            chunks, meta = retrieve_chunks_with_auth(query=query, auth=auth, debug=debug)
        else:
            chunks, meta = retrieve_chunks_with_settings(query=query, debug=debug)

        # Apply per-call overrides
        if limit is not None:
            chunks = chunks[:limit]
        if score_threshold is not None:
            chunks = [c for c in chunks if (c.get("score") or 0) >= score_threshold]

        return chunks, meta