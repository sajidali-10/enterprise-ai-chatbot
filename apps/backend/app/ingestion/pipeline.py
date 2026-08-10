"""
Document ingestion pipeline entrypoint (Phase 34A).

Parsers are registered here. New parsers (e.g. PPTX in a future phase)
must be added to ``_PARSERS``.

Two public entrypoints:

    * :func:`process_document` — returns ``str`` (legacy interface).
      Used by ``app.providers.custom.document_loader`` and elsewhere.
    * :func:`process_document_to_result` — returns the full
      :class:`ExtractionResult` with images, OCR metadata, and quality
      flags. Used by the upload endpoint in Phase 34A.
"""

from app.ingestion.parsers.base import BaseParser, ExtractionResult
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.image_parser import ImageParser

_PARSERS: list[BaseParser] = [
    TextParser(),
    PDFParser(),
    DOCXParser(),
    ImageParser(),
]


def _all_supported_types() -> set[str]:
    out: set[str] = set()
    for parser in _PARSERS:
        for t in parser.supported_types:
            out.add(t.lower())
    return out


def get_parser(mime_type: str) -> BaseParser | None:
    mime_type = (mime_type or "").lower().strip()
    for parser in _PARSERS:
        if mime_type in {t.lower() for t in parser.supported_types}:
            return parser
    return None


def process_document(file_bytes: bytes, mime_type: str) -> str:
    """Legacy interface — returns just the joined text."""
    parser = get_parser(mime_type)
    if not parser:
        raise ValueError(f"Unsupported MIME type: {mime_type}")
    return parser.parse_to_text(file_bytes)


def process_document_to_result(
    file_bytes: bytes,
    mime_type: str,
    *,
    original_filename: str | None = None,
    original_storage_key: str | None = None,
    **kwargs,
) -> ExtractionResult:
    """Phase 34A interface — returns the full structured result."""
    parser = get_parser(mime_type)
    if not parser:
        raise ValueError(f"Unsupported MIME type: {mime_type}")
    return parser.parse(
        file_bytes,
        original_filename=original_filename,
        original_storage_key=original_storage_key,
        **kwargs,
    )


def supported_mime_types() -> list[str]:
    """Return the sorted list of supported MIME types (Phase 34A)."""
    return sorted(_all_supported_types())


def supported_image_mime_types() -> list[str]:
    return ImageParser().supported_types
