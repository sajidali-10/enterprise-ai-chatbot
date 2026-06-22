"""
Provider Interface Definitions

Lightweight ABC/Protocol classes defining the contract for each RAG component.
These interfaces are aligned with the existing implementation shapes so that
custom adapters can wrap existing code without logic duplication.

Phase 22: Only custom, local, qdrant, none providers are active.
Future phases will add LangChain and other provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


class ChunkDict(TypedDict, total=False):
    """Normalised chunk dictionary as used across the RAG pipeline."""
    chunk_id: str
    document_id: int
    chunk_index: int
    content: str
    source_file_name: str
    title: str
    score: float | None


class SearchResult(TypedDict, total=False):
    """Normalised search result returned by VectorStoreProvider."""
    chunk_id: str
    document_id: int
    chunk_index: int
    content: str
    source_file_name: str
    title: str
    score: float | None


# ---------------------------------------------------------------------------
# Document Loader Provider
# ---------------------------------------------------------------------------


class DocumentLoaderProvider(ABC):
    """
    Extract normalised text from uploaded file bytes.

    Implementations must provide:
    - load(file_bytes, mime_type) -> str
    """

    @abstractmethod
    def load(self, file_bytes: bytes, mime_type: str) -> str:
        """
        Parse file bytes and return plain-text content.

        Args:
            file_bytes: Raw file content.
            mime_type: MIME type of the file (e.g. "application/pdf").

        Returns:
            Normalised UTF-8 text string.

        Raises:
            ValueError: If the MIME type is not supported.
        """
        ...


# ---------------------------------------------------------------------------
# Text Splitter Provider
# ---------------------------------------------------------------------------


class TextSplitterProvider(ABC):
    """
    Split plain text into smaller chunks with document metadata preserved.

    Implementations must provide:
    - split(text, metadata) -> list[ChunkDict]
    """

    @abstractmethod
    def split(self, text: str, metadata: dict) -> list[ChunkDict]:
        """
        Split text into chunks, preserving document metadata.

        Args:
            text: Plain text to split.
            metadata: Dict that MUST contain at minimum:
                - source_file_name: str
                Optionally: title, section_heading, document_id, etc.

        Returns:
            List of chunk dicts with keys: chunk_id, document_id, chunk_index,
            content, source_file_name, title, section_heading (optional).
        """
        ...


# ---------------------------------------------------------------------------
# Embedding Provider
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """
    Convert text into vector embeddings.

    Phase 22 implementations: MockEmbeddingProvider, SentenceTransformersEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats), one per input text,
            in the same order.
        """
        ...


# ---------------------------------------------------------------------------
# Vector Store Provider
# ---------------------------------------------------------------------------


class VectorStoreProvider(ABC):
    """
    Index and search vector chunks in a vector database.

    Phase 22 implementations: QdrantVectorStore (wraps qdrant_service.py).
    """

    @abstractmethod
    def index(self, chunks_with_embeddings: list[tuple], metadata: list[dict]) -> None:
        """
        Index chunks and their corresponding vectors.

        Args:
            chunks_with_embeddings: List of (text_content, embedding_vector) tuples,
                one per chunk, in the same order as metadata.
            metadata: List of chunk metadata dicts. Required keys per entry:
                chunk_id, document_id, document_version_id, chunk_index,
                content_hash, source_file_name, title, section_heading.

        Raises:
            RuntimeError: If indexing fails.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        limit: int,
        score_threshold: float,
    ) -> list[SearchResult]:
        """
        Search for the most similar chunks to a query embedding.

        Args:
            query_embedding: The embedded query vector.
            limit: Maximum number of results to return.
            score_threshold: Minimum relevance score (0.0–1.0).

        Returns:
            List of search result dicts, sorted by score descending.
        """
        ...

    @abstractmethod
    def delete_by_document_id(self, document_id: int) -> int:
        """
        Delete all indexed vectors for a specific document.

        Args:
            document_id: The document ID whose vectors should be removed.

        Returns:
            Number of vectors deleted.
        """
        ...


# ---------------------------------------------------------------------------
# Retriever Provider
# ---------------------------------------------------------------------------


class RetrieverProvider(ABC):
    """
    Retrieve relevant chunks for a user query with optional permission filtering.

    Phase 22 implementation: CustomRetrieverProvider (wraps retriever.py).
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        auth: Any | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
        debug: bool = False,
    ) -> tuple[list[ChunkDict], dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User's search/query string.
            auth: AuthContext for permission filtering (None = no auth, full access).
            limit: Override default limit. None = use provider default.
            score_threshold: Override default score threshold. None = use default.
            debug: If True, include debug metadata in the second tuple element.

        Returns:
            Tuple of (chunks list, metadata dict). Metadata keys include at minimum:
            - permission_filtered: bool
            - accessible_count: int
            - total_count: int
            When debug=True, also includes: original_query, rewritten_query, etc.
        """
        ...


# ---------------------------------------------------------------------------
# Reranker Provider
# ---------------------------------------------------------------------------


class RerankerProvider(ABC):
    """
    Re-score and re-rank retrieved chunks for improved relevance.

    Phase 22 implementation: NoOpRerankerProvider (identity pass-through).
    Future phases will add Cohere, BGE, and LangChain rerankers.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[ChunkDict],
        top_n: int | None = None,
    ) -> list[ChunkDict]:
        """
        Re-rank a list of chunks for a query.

        Args:
            query: User's search/query string.
            chunks: List of chunk dicts to re-rank.
            top_n: Return only the top N results. None = return all.

        Returns:
            List of chunk dicts, re-ordered by relevance (best first).
            May include an additional "rerank_score" key per chunk.
        """
        ...


# ---------------------------------------------------------------------------
# RAG Pipeline Provider
# ---------------------------------------------------------------------------


class RagPipelineProvider(ABC):
    """
    Orchestrate the full RAG answer-generation pipeline.

    Phase 22 implementation: CustomRagPipelineProvider (wraps answer_generator.py).
    Future phases will add LangChain and LangGraph pipeline adapters.
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        auth: Any | None = None,
        conversation_context: str = "",
        debug: bool = False,
    ) -> tuple[str, list[dict], dict]:
        """
        Run the full RAG pipeline: retrieve chunks → optionally rerank →
        generate answer → attach citations.

        Args:
            query: User's question.
            auth: AuthContext for permission filtering (None = no auth).
            conversation_context: Formatted prior-turn context string.
            debug: If True, include retrieval + generation debug metadata.

        Returns:
            Tuple of (answer_text, citations_list, metadata_dict).
            metadata dict includes retrieval stats and (when debug=True) full
            retrieval debug info.
        """
        ...