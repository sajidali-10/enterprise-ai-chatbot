"""
LangChain Provider Adapters

Optional LangChain-based document loader and text splitter adapters.
These are activated only when DOCUMENT_LOADER_PROVIDER=langchain or
TEXT_SPLITTER_PROVIDER=langchain is set in the environment.

Phase 24: behavior-preserving — custom providers remain the defaults.
"""

from app.providers.langchain.document_loader import LangChainDocumentLoaderProvider
from app.providers.langchain.text_splitter import LangChainTextSplitterProvider

__all__ = [
    "LangChainDocumentLoaderProvider",
    "LangChainTextSplitterProvider",
]