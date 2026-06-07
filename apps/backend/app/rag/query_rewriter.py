"""
Query Rewriting Service

Rewrites user queries to improve retrieval quality.
Supports multiple strategies: expand, paraphrase, decompose.
"""

from abc import ABC, abstractmethod
from typing import Optional


class QueryRewriterBase(ABC):
    """Abstract base class for query rewriting strategies."""

    @abstractmethod
    def rewrite(self, query: str) -> str:
        """Rewrite a query and return the rewritten version."""
        ...


class PassthroughRewriter(QueryRewriterBase):
    """
    No-op rewriter that returns the query unchanged.
    Used when query rewriting is disabled or unavailable.
    """

    def rewrite(self, query: str) -> str:
        return query


class MockQueryRewriter(QueryRewriterBase):
    """
    Mock query rewriter that adds context expansion.
    In production, this would call an LLM to expand/paraphrase queries.
    """

    def rewrite(self, query: str) -> str:
        """
        Basic mock rewrite that adds common expansions.
        Real implementation would use an LLM for better rewrites.
        """
        query = query.strip()
        if not query.endswith("?"):
            query = query + "?"
        return query


def get_query_rewriter(rewriter_type: Optional[str] = None) -> QueryRewriterBase:
    """
    Factory to get a query rewriter by name.
    
    Args:
        rewriter_type: One of "passthrough", "mock", "cohere", "bge", or None.
        
    Returns:
        QueryRewriterBase instance.
    """
    if rewriter_type is None or rewriter_type == "passthrough":
        return PassthroughRewriter()
    elif rewriter_type == "mock":
        return MockQueryRewriter()
    elif rewriter_type in ("cohere", "bge"):
        # Placeholder for future implementation
        raise NotImplementedError(
            f"Query rewriter '{rewriter_type}' is not yet implemented. "
            f"Use 'mock' or 'passthrough' for now."
        )
    else:
        raise ValueError(f"Unknown query rewriter type: {rewriter_type}")