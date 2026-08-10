"""
PDF Parser with optional OCR fallback for scanned pages (Phase 34A).

Behavior:
    1. Use pypdf to extract native text per page.
    2. For each page whose native text length is below
       ``OCR_PDF_PAGE_TEXT_MIN_CHARS`` (default 40) AND OCR is enabled,
       render that page to an image and run OCR.
    3. Concatenate native text and (when present) OCR text per page,
       preserving page numbers and reading order.

The parser never runs OCR on pages that already have meaningful native
text. This keeps the common case (text PDF) untouched.

Quality flags surfaced on ``ExtractionResult``:
    - ``native_text_chars``: total native chars extracted.
    - ``ocr_text_chars``: total OCR chars added.
    - ``pages_ocr_processed``: count of pages that went through OCR.
    - ``images_detected``: pages where OCR ran (counted as "images"
      for completeness accounting; not literal image attachments).
    - ``needs_human_review``: True when almost no text exists and OCR
      was disabled/failed, or when many pages produced no content.

The parser does NOT handle password-protected PDFs — pypdf will raise
its own error, which the upload endpoint translates to a 400.
"""

from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from typing import Any, List, Optional

from pypdf import PdfReader

from app.ingestion.parsers.base import (
    BaseParser,
    ExtractedImage,
    ExtractionResult,
)
from app.ingestion.ocr_cleaning import clean_ocr_text
from app.providers.ocr import OcrProvider, get_ocr_provider
from app.providers.ocr.base import OcrResult, OcrProviderUnavailable

logger = logging.getLogger(__name__)


def _hash_text(text: str) -> str:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


class PDFParser(BaseParser):
    """PDF parser with optional OCR fallback for scanned pages."""

    @property
    def supported_types(self) -> list[str]:
        return ["application/pdf"]

    def parse(
        self,
        file_bytes: bytes,
        *,
        ocr_provider: Optional[OcrProvider] = None,
        pdf_fallback: Optional[bool] = None,
        pdf_page_text_min_chars: Optional[int] = None,
        render_dpi: Optional[int] = None,
        original_filename: Optional[str] = None,
        **_: Any,
    ) -> ExtractionResult:
        provider = ocr_provider if ocr_provider is not None else get_ocr_provider()

        # Settings (with env fallbacks).
        if pdf_fallback is None:
            pdf_fallback = os.getenv("OCR_PDF_FALLBACK", "true").strip().lower() in (
                "true", "1", "yes", "on",
            )
        if pdf_page_text_min_chars is None:
            try:
                pdf_page_text_min_chars = int(os.getenv("OCR_PDF_PAGE_TEXT_MIN_CHARS", "40"))
            except (TypeError, ValueError):
                pdf_page_text_min_chars = 40
        if render_dpi is None:
            try:
                render_dpi = int(os.getenv("OCR_RENDER_DPI", "200"))
            except (TypeError, ValueError):
                render_dpi = 200

        # 1. Native extraction
        try:
            reader = PdfReader(BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Could not open PDF: {e}") from e

        page_texts: List[str] = []
        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception as e:
                logger.warning("pypdf extract_text failed for a page: %s", e)
                page_texts.append("")

        total_pages = len(page_texts)
        native_chars = sum(len(t) for t in page_texts)

        # 2. Decide which pages need OCR
        pages_needing_ocr: List[int] = []
        for i, t in enumerate(page_texts):
            if len(t.strip()) < pdf_page_text_min_chars:
                pages_needing_ocr.append(i)

        ocr_pages_results: dict = {}
        warnings: list = []
        pages_ocr_processed = 0
        ocr_provider_name: Optional[str] = None
        ocr_confidences: List[float] = []

        if pages_needing_ocr and pdf_fallback and provider is not None:
            for page_idx in pages_needing_ocr:
                try:
                    ocr_result: OcrResult = provider.extract_text_from_pdf_page(
                        file_bytes, page_idx, dpi=render_dpi
                    )
                except OcrProviderUnavailable as e:
                    warnings.append(
                        f"OCR provider unavailable for page {page_idx + 1}: {e}"
                    )
                    continue
                except Exception as e:
                    warnings.append(f"OCR failed for page {page_idx + 1}: {e}")
                    continue
                if not ocr_result.success or not (ocr_result.text or "").strip():
                    warnings.append(
                        f"OCR returned no text for page {page_idx + 1}"
                    )
                    continue
                cleaned = clean_ocr_text(ocr_result.text)
                if cleaned:
                    ocr_pages_results[page_idx] = cleaned
                    pages_ocr_processed += 1
                    if ocr_provider_name is None and ocr_result.provider:
                        ocr_provider_name = ocr_result.provider
                    if ocr_result.confidence is not None:
                        ocr_confidences.append(ocr_result.confidence)
        elif pages_needing_ocr and pdf_fallback and provider is None:
            warnings.append(
                "OCR fallback requested but OCR is disabled (OCR_ENABLED=false)"
            )

        # 3. Build joined text per page, preserving order.
        joined_chunks: List[str] = []
        images: List[ExtractedImage] = []
        ocr_text_chars = 0
        for i, native_t in enumerate(page_texts):
            native_clean = (native_t or "").strip()
            ocr_t = ocr_pages_results.get(i, "")
            page_block: List[str] = []
            page_block.append(f"--- Page {i + 1} ---")
            if native_clean:
                page_block.append(native_clean)
            if ocr_t:
                page_block.append("[OCR Text]")
                page_block.append(ocr_t)
                ocr_text_chars += len(ocr_t)
            joined_chunks.append("\n".join(page_block))

            # Record an ExtractedImage for every page that went through
            # OCR (rendered bytes reconstructed lazily by future Vision
            # phases — for now we keep the page number, OCR result, and
            # the original PDF bytes reference via document_version).
            if i in ocr_pages_results:
                ocr_clean = ocr_pages_results[i]
                # Confidence list is appended in lockstep with
                # ocr_pages_results — index by (len-1) to grab the
                # confidence for the page that was just added.
                page_conf_idx = len(ocr_pages_results) - 1
                page_conf = (
                    ocr_confidences[page_conf_idx]
                    if 0 <= page_conf_idx < len(ocr_confidences)
                    else None
                )
                images.append(
                    ExtractedImage(
                        raw_bytes=b"",  # original PDF page render can be regenerated
                        mime_type="image/png",
                        source_type="pdf_page_ocr",
                        page_number=i + 1,
                        sequence_number=i,
                        original_filename=original_filename,
                        storage_key=None,
                        ocr_text=ocr_clean,
                        ocr_provider=ocr_provider_name,
                        ocr_status="success",
                        ocr_confidence=page_conf,
                        ocr_text_hash=_hash_text(ocr_clean),
                        width=None,
                        height=None,
                        byte_size=None,
                    )
                )

        text = "\n\n".join(joined_chunks).strip()

        # Quality accounting.
        avg_conf = (
            sum(ocr_confidences) / len(ocr_confidences)
            if ocr_confidences
            else None
        )
        # needs_human_review: suspicious extraction.
        very_little_text = (native_chars + ocr_text_chars) < (10 * max(1, total_pages))
        many_pages_no_text = (
            len(pages_needing_ocr) >= max(1, total_pages // 2)
            and ocr_text_chars == 0
        )
        needs_human_review = bool(
            very_little_text
            or many_pages_no_text
            or (pages_needing_ocr and pdf_fallback and provider is None)
        )

        extraction_method = "pdf_pypdf"
        if pages_ocr_processed > 0:
            extraction_method = "pdf_pypdf_with_ocr_fallback"

        return ExtractionResult(
            text=text,
            extraction_method=extraction_method,
            native_text_chars=native_chars,
            ocr_text_chars=ocr_text_chars,
            images_detected=pages_ocr_processed,
            images_ocr_processed=pages_ocr_processed,
            pages_ocr_processed=pages_ocr_processed,
            ocr_average_confidence=avg_conf,
            ocr_provider=ocr_provider_name,
            ocr_disabled=(provider is None),
            needs_human_review=needs_human_review,
            warnings=warnings,
            images=images,
            metadata={
                "total_pages": total_pages,
                "pages_needing_ocr": len(pages_needing_ocr),
                "render_dpi": render_dpi,
                "pdf_page_text_min_chars": pdf_page_text_min_chars,
            },
        )
