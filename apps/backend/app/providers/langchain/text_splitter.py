"""
LangChain Text Splitter Adapter

Wraps LangChain's RecursiveCharacterTextSplitter to implement the
TextSplitterProvider interface. Uses settings for chunk_size (RAG_CHUNK_SIZE)
and chunk_overlap (RAG_CHUNK_OVERLAP).
"""

from app.providers.base import TextSplitterProvider, ChunkDict

# Lazy-initialized splitter instance — created on first use, not at import time.
_splitter = None


def _get_splitter():
    global _splitter
    if _splitter is None:
        from app.core.config import settings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    return _splitter


class LangChainTextSplitterProvider(TextSplitterProvider):
    """
    Phase 24 active provider when TEXT_SPLITTER_PROVIDER=langchain.

    Wraps langchain_text_splitters.RecursiveCharacterTextSplitter with
    RAG_CHUNK_SIZE and RAG_CHUNK_OVERLAP settings.

    Chunk dicts use the same shape as CustomTextSplitterProvider output so
    the rest of the indexing pipeline is unaffected.
    """

    def split(self, text: str, metadata: dict) -> list[ChunkDict]:
        """
        Split text using LangChain RecursiveCharacterTextSplitter and
        return chunks in the standard ChunkDict format.
        """
        splitter = _get_splitter()
        langchain_docs = splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        chunks: list[ChunkDict] = []
        for chunk_index, doc in enumerate(langchain_docs):
            source_file_name = metadata.get("source_file_name", "")
            chunks.append({
                "chunk_id": f"{source_file_name}_{chunk_index}",
                "document_id": metadata.get("document_id", 0),
                "chunk_index": chunk_index,
                "content": doc.page_content,
                "source_file_name": source_file_name,
                "title": metadata.get("title"),
                "section_heading": metadata.get("section_heading"),
            })
        return chunks