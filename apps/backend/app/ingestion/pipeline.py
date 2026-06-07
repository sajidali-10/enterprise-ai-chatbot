from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.base import BaseParser

_PARSERS = [TextParser(), PDFParser(), DOCXParser()]

def get_parser(mime_type: str) -> BaseParser | None:
    for parser in _PARSERS:
        if mime_type in parser.supported_types:
            return parser
    return None

def process_document(file_bytes: bytes, mime_type: str) -> str:
    parser = get_parser(mime_type)
    if not parser:
        raise ValueError(f"Unsupported MIME type: {mime_type}")
    return parser.parse(file_bytes)