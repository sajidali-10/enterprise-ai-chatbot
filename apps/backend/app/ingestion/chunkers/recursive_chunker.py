import hashlib
from app.ingestion.chunkers.base import BaseChunker, Chunk
from app.core.config import settings

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        source_file_name = metadata.get("source_file_name", "")
        title = metadata.get("title")
        section_heading = metadata.get("section_heading")
        
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    content_hash = hashlib.sha256(current_chunk.encode()).hexdigest()[:16]
                    chunks.append(Chunk(
                        chunk_index=chunk_index,
                        content=current_chunk.strip(),
                        content_hash=content_hash,
                        title=title,
                        section_heading=section_heading,
                        source_file_name=source_file_name,
                    ))
                    chunk_index += 1
                    
                    # Keep overlap — take last overlap chars
                    overlap_start = max(0, len(current_chunk) - self.overlap)
                    current_chunk = current_chunk[overlap_start:] + para + "\n\n"
                else:
                    # Paragraph itself is too big, split by sentences
                    sentences = para.split(". ")
                    for sent in sentences:
                        sent = sent.strip() + ". "
                        if len(current_chunk) + len(sent) <= self.chunk_size:
                            current_chunk += sent
                        else:
                            if current_chunk.strip():
                                content_hash = hashlib.sha256(current_chunk.encode()).hexdigest()[:16]
                                chunks.append(Chunk(
                                    chunk_index=chunk_index,
                                    content=current_chunk.strip(),
                                    content_hash=content_hash,
                                    title=title,
                                    section_heading=section_heading,
                                    source_file_name=source_file_name,
                                ))
                                chunk_index += 1
                                overlap_start = max(0, len(current_chunk) - self.overlap)
                                current_chunk = current_chunk[overlap_start:]
                            current_chunk = sent
        
        # Handle remaining content
        if current_chunk.strip():
            content_hash = hashlib.sha256(current_chunk.encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_index=chunk_index,
                content=current_chunk.strip(),
                content_hash=content_hash,
                title=title,
                section_heading=section_heading,
                source_file_name=source_file_name,
            ))
        
        return chunks