from io import BytesIO
from docx import Document
from app.ingestion.parsers.base import BaseParser

class DOCXParser(BaseParser):
    @property
    def supported_types(self) -> list[str]:
        return ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    
    def parse(self, file_bytes: bytes) -> str:
        doc = Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)