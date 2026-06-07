from io import BytesIO
from pypdf import PdfReader
from app.ingestion.parsers.base import BaseParser

class PDFParser(BaseParser):
    @property
    def supported_types(self) -> list[str]:
        return ["application/pdf"]
    
    def parse(self, file_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)