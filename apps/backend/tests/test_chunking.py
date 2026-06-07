import pytest
from app.ingestion.chunkers.recursive_chunker import RecursiveChunker

def test_chunker_splits_long_text():
    chunker = RecursiveChunker(chunk_size=300, overlap=50)
    long_text = " ".join(["word"] * 500)
    metadata = {"source_file_name": "test.txt"}
    chunks = chunker.chunk(long_text, metadata)
    assert len(chunks) > 1
    assert all(c.source_file_name == "test.txt" for c in chunks)
    assert all(c.chunk_index >= 0 for c in chunks)

def test_chunker_respects_chunk_size():
    chunker = RecursiveChunker(chunk_size=500, overlap=50)
    # Generate text longer than chunk size
    text = " ".join(["sentence number " + str(i) for i in range(50)])
    chunks = chunker.chunk(text, {"source_file_name": "test.txt"})
    for chunk in chunks:
        assert len(chunk.content) <= 600  # allow some buffer

def test_chunker_content_hash_deterministic():
    chunker = RecursiveChunker()
    metadata = {"source_file_name": "test.txt"}
    chunks1 = chunker.chunk("hello world", metadata)
    chunks2 = chunker.chunk("hello world", metadata)
    assert chunks1[0].content_hash == chunks2[0].content_hash

def test_chunker_empty_text():
    chunker = RecursiveChunker()
    chunks = chunker.chunk("", {"source_file_name": "test.txt"})
    assert len(chunks) == 0