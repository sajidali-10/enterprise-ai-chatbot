"""
Custom Text Splitter Adapter

Wraps the existing RecursiveChunker from app/ingestion/chunkers/recursive_chunker.py.
No logic duplication — delegates directly.
"""

from app.ingestion.chunkers.recursive_chunker import RecursiveChunker
from app.providers.base import DocumentLoaderProvider, TextSplitterProvider, ChunkDict


class CustomTextSplitterProvider(TextSplitterProvider):
    """
    Phase 22 active provider: delegates to RecursiveChunker.

    The underlying RecursiveChunker uses paragraph-aware splitting with
    configurable chunk_size and overlap from settings.
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        self._chunker = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)

    def split(self, text: str, metadata: dict) -> list[ChunkDict]:
        chunks = self._chunker.chunk(text, metadata)
        return [
            {
                "chunk_id": f"{metadata.get('source_file_name', '')}_{c.chunk_index}",
                "document_id": metadata.get("document_id", 0),
                "chunk_index": c.chunk_index,
                "content": c.content,
                "source_file_name": c.source_file_name,
                "title": c.title or metadata.get("title"),
                "section_heading": c.section_heading,
            }
            for c in chunks
        ]