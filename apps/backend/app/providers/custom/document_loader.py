"""
Custom Document Loader Adapter

Wraps the existing app/ingestion/pipeline.py process_document function.
No logic duplication — delegates directly to existing parsers.
"""

from app.ingestion.pipeline import process_document
from app.providers.base import DocumentLoaderProvider


class CustomDocumentLoaderProvider(DocumentLoaderProvider):
    """
    Phase 22 active provider: delegates to the existing parser pipeline.

    Supported MIME types:
    - text/plain, text/markdown, text/x-markdown  (TextParser)
    - application/pdf                               (PDFParser)
    - application/vnd.openxmlformats-officedocument.wordprocessingml.document (DOCXParser)
    """

    def load(self, file_bytes: bytes, mime_type: str) -> str:
        return process_document(file_bytes, mime_type)