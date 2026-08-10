"""
Plain-text / markdown parser.

Phase 34A: returns an :class:`ExtractionResult` with no images and no
OCR. The legacy ``str`` interface is preserved via
``BaseParser.parse_to_text``.
"""

from app.ingestion.parsers.base import BaseParser, ExtractionResult


class TextParser(BaseParser):
    @property
    def supported_types(self) -> list[str]:
        return ["text/plain", "text/markdown", "text/x-markdown"]

    def parse(self, file_bytes: bytes, **_) -> ExtractionResult:
        text = file_bytes.decode("utf-8", errors="replace")
        text = (text or "").strip()
        return ExtractionResult(
            text=text,
            extraction_method="text_plain",
            native_text_chars=len(text),
            ocr_text_chars=0,
            images_detected=0,
            images_ocr_processed=0,
            pages_ocr_processed=0,
            ocr_average_confidence=None,
            ocr_provider=None,
            ocr_disabled=False,
            needs_human_review=False,
            warnings=[],
            images=[],
            metadata={},
        )
