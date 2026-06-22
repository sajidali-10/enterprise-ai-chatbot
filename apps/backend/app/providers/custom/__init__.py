"""
Custom Provider Adapters

Each adapter wraps existing services/functions without duplicating logic.
Phase 22: all custom adapters are the only active providers.
"""

from app.providers.custom.document_loader import CustomDocumentLoaderProvider
from app.providers.custom.text_splitter import CustomTextSplitterProvider
from app.providers.custom.embeddings import CustomEmbeddingProvider
from app.providers.custom.vector_store import CustomVectorStoreProvider
from app.providers.custom.retriever import CustomRetrieverProvider
from app.providers.custom.reranker import NoOpRerankerProvider
from app.providers.custom.rag_pipeline import CustomRagPipelineProvider

__all__ = [
    "CustomDocumentLoaderProvider",
    "CustomTextSplitterProvider",
    "CustomEmbeddingProvider",
    "CustomVectorStoreProvider",
    "CustomRetrieverProvider",
    "NoOpRerankerProvider",
    "CustomRagPipelineProvider",
]