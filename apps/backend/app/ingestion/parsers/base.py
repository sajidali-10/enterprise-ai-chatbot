"""
Parser base classes and the shared :class:`ExtractionResult`.

Phase 34A adds image/OCR ingestion to the parser chain. To keep the
upload pipeline consistent across text, PDF, DOCX, and direct images
we extend the parser contract: instead of returning ``str``,
``BaseParser.parse`` now returns an :class:`ExtractionResult` that
carries the final text *and* per-image metadata, OCR statistics, and
quality flags.

Backward compatibility:
    Callers that only need the joined text can call
    ``ExtractionResult.text`` directly. Helpers
    ``process_document_to_text`` and ``BaseParser.parse_to_text``
    preserve the pre-Phase-34A ``str`` interface.

Security:
    Parsers MUST route any OCR-derived text through
    ``app.ingestion.ocr_cleaning.clean_ocr_text`` before exposing it
    in ``ExtractionResult.text``. Sanitization of secrets, keys,
    passwords, etc. happens at the ingestion layer (after parsing)
    via ``app.services.langsmith_tracing.redact_text``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedImage:
    """Per-image extraction metadata.

    The parser produces this; the upload endpoint consumes it to
    upload the original image bytes to MinIO and persist a
    :class:`app.models.document.DocumentImage` row.

    Attributes:
        raw_bytes: Original (or rendered) image bytes. For PDFs/DOCX
            the parser renders/extracts these; for direct image
            uploads the caller supplies the same bytes it received.
        mime_type: MIME type of ``raw_bytes``.
        source_type: One of ``direct_image``, ``pdf_page_ocr``,
            ``docx_image_ocr``.
        page_number: 1-based page index, where applicable.
        sequence_number: Per-document monotonic sequence (0-based).
        original_filename: Original filename hint when available.
        storage_key: MinIO storage key. The upload endpoint
            overwrites this with the key it actually uses.
        ocr_text: Cleaned OCR text. May be empty.
        ocr_provider: Provider name (e.g. ``"tesseract"``) or None.
        ocr_status: One of ``pending``/``success``/``failed``/
            ``disabled``/``empty``/``low_confidence``.
        ocr_confidence: 0-100, may be None.
        ocr_error: Error message on failure, else None.
        ocr_text_hash: SHA-256 prefix of cleaned OCR text.
        width: Pixel width after preprocessing.
        height: Pixel height after preprocessing.
        byte_size: Original byte size of the image (before any
            preprocessing).
    """

    raw_bytes: bytes
    mime_type: str
    source_type: str
    page_number: Optional[int] = None
    sequence_number: int = 0
    original_filename: Optional[str] = None
    storage_key: Optional[str] = None
    ocr_text: str = ""
    ocr_provider: Optional[str] = None
    ocr_status: str = "pending"
    ocr_confidence: Optional[float] = None
    ocr_error: Optional[str] = None
    ocr_text_hash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    byte_size: Optional[int] = None


@dataclass
class ExtractionResult:
    """Output of a parser.

    ``text`` is the joined content the chunker will consume. ``images``
    carries per-image metadata that the upload endpoint persists.

    Quality flags drive document-status decisions in the upload
    endpoint:
        - ``needs_human_review``: True when extraction quality looks
          suspicious (e.g. very little native text but many embedded
          images and OCR is disabled).
        - ``ocr_disabled``: True when OCR was configured off and the
          document was image-only.
    """

    text: str
    extraction_method: str = ""
    native_text_chars: int = 0
    ocr_text_chars: int = 0
    images_detected: int = 0
    images_ocr_processed: int = 0
    pages_ocr_processed: int = 0
    ocr_average_confidence: Optional[float] = None
    ocr_provider: Optional[str] = None
    ocr_disabled: bool = False
    needs_human_review: bool = False
    warnings: List[str] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe summary dict for ``DocumentVersion.extraction_summary``."""
        return {
            "extraction_method": self.extraction_method,
            "native_text_chars": self.native_text_chars,
            "ocr_text_chars": self.ocr_text_chars,
            "images_detected": self.images_detected,
            "images_ocr_processed": self.images_ocr_processed,
            "pages_ocr_processed": self.pages_ocr_processed,
            "ocr_average_confidence": self.ocr_average_confidence,
            "ocr_provider": self.ocr_provider,
            "ocr_disabled": self.ocr_disabled,
            "needs_human_review": self.needs_human_review,
            "warnings": list(self.warnings),
        }

    @property
    def is_empty(self) -> bool:
        """True when the result has no useful text and no images."""
        return not (self.text and self.text.strip())


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes, **kwargs) -> ExtractionResult:
        """Parse bytes into an :class:`ExtractionResult`.

        ``kwargs`` may include parser-specific hints. The current
        contract understands:
            - ``original_storage_key`` (str): MinIO key for the
              original uploaded file (for direct image parser).
            - ``original_filename`` (str): original uploaded filename.
            - ``ocr_provider``: OCR provider instance. If not
              provided, parsers obtain one via
              :func:`app.providers.ocr.get_ocr_provider`.
        """
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[str]: ...

    # Backward-compatible helpers --------------------------------------

    def parse_to_text(self, file_bytes: bytes, **kwargs) -> str:
        """Return only the joined text (legacy ``str`` interface)."""
        result = self.parse(file_bytes, **kwargs)
        return result.text
