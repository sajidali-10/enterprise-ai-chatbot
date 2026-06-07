from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class Chunk:
    chunk_index: int
    content: str
    content_hash: str
    title: Optional[str] = None
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    source_file_name: str = ""

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """
        Split text into chunks with metadata.
        metadata should contain: source_file_name, title (optional), etc.
        """
        pass