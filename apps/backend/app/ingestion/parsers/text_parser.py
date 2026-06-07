from app.ingestion.parsers.base import BaseParser

class TextParser(BaseParser):
    @property
    def supported_types(self) -> list[str]:
        return ["text/plain", "text/markdown", "text/x-markdown"]
    
    def parse(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="replace")