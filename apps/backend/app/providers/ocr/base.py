"""
OCR Provider Interface (Phase 34A).

Defines the abstract contract for OCR providers. The goal is to allow
multiple OCR backends (Tesseract today; PaddleOCR / Textract / Azure
Document Intelligence / Google Document AI in future phases) to plug
into the ingestion pipeline without rewriting parsers.

All OCR results are returned as :class:`OcrResult`, which captures text,
average confidence, per-page/per-image details, the provider name,
warnings, and arbitrary metadata (e.g. engine version, render DPI).

This module deliberately does NOT import any optional OCR SDKs.
Concrete implementations live in their own modules (e.g.
``tesseract_provider.py``) and must remain import-safe even when their
binary dependency is missing — they should expose a ``health_check()``
that reports availability and raise a typed error when used while
unavailable.

Security:
    Implementations must NEVER log full OCR text. They may log document
    IDs, image dimensions, status, and provider names only. OCR text
    flowing into the index pipeline is sanitized separately by the
    ingestion layer (see ``app.ingestion.ocr_cleaning`` and
    ``app.services.langsmith_tracing.redact_text``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class OcrPage:
    """Per-image or per-page OCR result.

    Attributes:
        page_index: Zero-based page or image index within the source.
        text: The OCR text for this page/image (raw — cleaning is the
            ingestion layer's responsibility, not the provider's).
        confidence: Average confidence reported by the engine for this
            page/image (0-100, where applicable). ``None`` if the engine
            did not report per-page confidence.
        width: Pixel width of the rendered image (if known).
        height: Pixel height of the rendered image (if known).
        language: Detected or configured language for this page.
        warnings: Provider-level warnings for this page (e.g. low
            contrast, missing orientation).
    """

    page_index: int
    text: str
    confidence: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    language: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class OcrResult:
    """Structured OCR output.

    Attributes:
        text: Concatenated text across all pages/images (joined by
            ``\\n\\n``). May be empty for image-only inputs with no
            detectable text — callers MUST check ``pages`` and
            ``warnings`` before declaring failure.
        confidence: Average confidence across pages (0-100) or ``None``.
        language: Language used for OCR (e.g. ``"eng"``).
        pages: One :class:`OcrPage` per processed page/image. Empty if
            the input was empty or unprocessable.
        provider: Stable provider identifier (e.g. ``"tesseract"``).
        extraction_method: Stable label describing how this OCR was
            produced (``"tesseract_image"``, ``"tesseract_pdf_page"``,
            ``"tesseract_docx_image"``).
        warnings: Provider-level warnings collected across the call.
        metadata: Provider-specific structured metadata (engine version,
            render DPI, etc.). MUST NOT contain secrets or full OCR
            content.
        success: True if OCR produced at least one page with text;
            False otherwise. Callers should still inspect ``pages`` and
            ``warnings`` to distinguish empty from failed.
        error: Provider error message, if any. ``None`` on success.
    """

    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    pages: List[OcrPage] = field(default_factory=list)
    provider: str = ""
    extraction_method: str = ""
    warnings: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class OcrProviderError(Exception):
    """Base class for OCR provider errors."""


class OcrProviderUnavailable(OcrProviderError):
    """Raised when the OCR backend binary/library is not installed or
    reachable. Recoverable — caller may skip OCR and continue with
    native text only."""


class OcrUnsupportedFormat(OcrProviderError):
    """Raised for MIME types/formats the provider cannot handle."""


class OcrTimeout(OcrProviderError):
    """Raised when OCR exceeds the configured timeout."""


class OcrInvalidInput(OcrProviderError):
    """Raised for corrupted or empty input that the provider cannot
    process."""


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class OcrProvider(ABC):
    """Abstract OCR provider.

    Implementations MUST be safe to instantiate even when the underlying
    binary is missing — they should expose their availability via
    ``health_check()`` and raise :class:`OcrProviderUnavailable` from
    ``extract_text`` / ``extract_text_from_pdf_page`` if they are used
    while unavailable.
    """

    #: Stable identifier used in settings / logs.
    name: str = "abstract"

    @abstractmethod
    def extract_text(
        self,
        image_bytes: bytes,
        mime_type: str,
        language: Optional[str] = None,
    ) -> OcrResult:
        """Run OCR on a single image.

        Args:
            image_bytes: Raw image bytes (PNG/JPEG/WEBP/TIFF/BMP/GIF).
            mime_type: MIME type (used to validate and for logging).
            language: Optional ISO-639-3/BCP-47 override (e.g. ``"eng"``).

        Returns:
            :class:`OcrResult`. Must always return a result, never raise
            for "OCR produced no text" — use ``success=False`` instead.

        Raises:
            OcrProviderUnavailable: Provider binary/library missing.
            OcrUnsupportedFormat: MIME type not supported.
            OcrInvalidInput: Bytes are corrupted.
            OcrTimeout: OCR exceeded timeout.
        """

    @abstractmethod
    def extract_text_from_pdf_page(
        self,
        pdf_bytes: bytes,
        page_index: int,
        dpi: int = 200,
        language: Optional[str] = None,
    ) -> OcrResult:
        """Render a single PDF page to image and OCR it.

        Implementations are expected to render at the requested DPI
        before OCR. The DPI is recorded in ``result.metadata["dpi"]``.

        Args:
            pdf_bytes: Raw PDF bytes.
            page_index: Zero-based page index.
            dpi: Render DPI (default 200).
            language: Optional language override.

        Returns:
            :class:`OcrResult` with a single :class:`OcrPage`.
        """

    @abstractmethod
    def health_check(self) -> dict:
        """Return a structured health snapshot for this provider.

        The returned dict MUST NOT include secrets. It is exposed via
        the admin status endpoint and may be logged. Recommended keys:

            - ``available`` (bool): whether the provider can serve
              requests right now.
            - ``provider`` (str): provider name.
            - ``engine_version`` (str | None): e.g. tesseract --version.
            - ``details`` (str | None): human-readable extra info.
        """

    def provider_info(self) -> dict:
        """Return non-secret provider metadata for the admin status page.

        Default implementation returns ``{"provider": self.name}``.
        Subclasses should extend with engine version, supported
        languages, etc.
        """
        return {"provider": self.name}

    # ------------------------------------------------------------------
    # Convenience helpers (non-abstract)
    # ------------------------------------------------------------------

    def supports_mime(self, mime_type: str) -> bool:
        """Return True if this provider can OCR ``mime_type``.

        Default implementation accepts common image types. Providers
        may override to be stricter.
        """
        if not mime_type:
            return False
        m = mime_type.lower().strip()
        return m in {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp",
            "image/tiff",
            "image/bmp",
            "image/gif",
            "application/pdf",  # some providers can render PDF pages
        }

    @staticmethod
    def _aggregate_confidence(pages: Iterable[OcrPage]) -> Optional[float]:
        """Compute the average confidence across pages.

        Returns None if no page reported a confidence value.
        """
        confs = [p.confidence for p in pages if p.confidence is not None]
        if not confs:
            return None
        return sum(confs) / len(confs)

    @staticmethod
    def _join_pages(pages: List[OcrPage]) -> str:
        """Concatenate pages with double-newline separators."""
        return "\n\n".join((p.text or "").strip() for p in pages if p.text is not None)
