"""
LangChain Document Loader Adapter

Wraps LangChain community loaders to implement the DocumentLoaderProvider interface.
Supported file types: PDF, DOCX, TXT, MD.

Uses tempfile.NamedTemporaryFile to give LangChain loaders a file path,
since they require pathlib.Path rather than raw bytes.
"""

import tempfile
import os
from typing import Any

from app.providers.base import DocumentLoaderProvider

# MIME type → LangChain loader class mapping
# These are lazily resolved so the factory can detect missing deps.
_LOADER_CLASSES: dict[str, tuple[str, str]] = {
    "application/pdf": (
        "langchain_community.document_loaders",
        "PyPDFLoader",
    ),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        "langchain_community.document_loaders",
        "UnstructuredWordDocumentLoader",
    ),
    "text/plain": (
        "langchain_community.document_loaders",
        "TextLoader",
    ),
    "text/markdown": (
        "langchain_community.document_loaders",
        "TextLoader",
    ),
    "text/x-markdown": (
        "langchain_community.document_loaders",
        "TextLoader",
    ),
}


def _load_loader_class(module: str, cls: str) -> Any:
    import importlib
    mod = importlib.import_module(module)
    return getattr(mod, cls)


class LangChainDocumentLoaderProvider(DocumentLoaderProvider):
    """
    Phase 24 active provider when DOCUMENT_LOADER_PROVIDER=langchain.

    Wraps LangChain community loaders:
      - PyPDFLoader for PDF
      - UnstructuredWordDocumentLoader for DOCX
      - TextLoader for TXT / MD / Markdown

    Temporary files are created and immediately deleted after loading
    so no artifacts remain on disk.
    """

    # Identical mime types to custom document_loader.py (parsers)
    SUPPORTED_MIME_TYPES = list(_LOADER_CLASSES.keys())

    def load(self, file_bytes: bytes, mime_type: str) -> str:
        if mime_type not in _LOADER_CLASSES:
            raise ValueError(
                f"Unsupported MIME type for LangChain loader: '{mime_type}'. "
                f"Supported types: {self.SUPPORTED_MIME_TYPES}"
            )

        module_name, class_name = _LOADER_CLASSES[mime_type]
        loader_cls = _load_loader_class(module_name, class_name)

        # Write bytes to a temporary file so LangChain loaders can read it
        suffix = self._mime_to_suffix(mime_type)
        fd = tempfile.NamedTemporaryFile(suffix=suffix, delete=True)
        try:
            fd.write(file_bytes)
            fd.flush()
            fd.seek(0)
            loader = loader_cls(fd.name)  # PyPDFLoader etc. take a file path
            documents = loader.load()
            # Return concatenated page/paragraphtext separated by newlines
            return "\n".join(doc.page_content for doc in documents)
        finally:
            fd.close()  # NamedTemporaryFile with delete=True cleans up on close

    @staticmethod
    def _mime_to_suffix(mime_type: str) -> str:
        suffix_map = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "text/x-markdown": ".md",
        }
        return suffix_map.get(mime_type, ".bin")