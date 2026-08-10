"""
DOCX parser with OCR for embedded images (Phase 34A).

Behavior:
    1. Extract native paragraph text via python-docx (existing behavior).
    2. Walk the DOCX zip archive and extract every embedded image
       (``word/media/*``), preserving relative order when possible.
    3. For each image: preprocess, OCR, clean.
    4. Concatenate native paragraphs and per-image OCR blocks into a
       single joined text, preserving reading order using a per-image
       sequence number that follows the position of the image in the
       DOCX archive (the closest practical approximation to "in-line
       order" without re-implementing a full OOXML renderer).

Settings:
    - ``OCR_DOCX_IMAGES`` (default ``true``) — master switch.
    - ``OCR_DOCX_IMAGE_MAX_COUNT`` (default ``50``) — safety cap.

Quality flags:
    - ``images_detected`` — count of embedded images found.
    - ``images_ocr_processed`` — count that actually went through OCR.
    - ``needs_human_review`` — True when the document contained many
      images but native text is very small (likely screenshots
      carrying the real information).

Security:
    - Embedded image bytes are passed through preprocessing (Pillow)
      before OCR — original bytes are preserved on ``ExtractedImage``.
    - OCR text is deterministically cleaned via ``clean_ocr_text``;
      secret redaction is layered at the upload endpoint.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import zipfile
from typing import Any, List, Optional

from docx import Document

from app.ingestion.parsers.base import (
    BaseParser,
    ExtractedImage,
    ExtractionResult,
)
from app.ingestion.ocr_cleaning import clean_ocr_text
from app.ingestion.ocr_preprocessing import (
    is_image_mime_supported,
    preprocess_for_ocr,
    validate_image_bytes,
)
from app.providers.ocr import OcrProvider, get_ocr_provider
from app.providers.ocr.base import OcrProviderUnavailable

logger = logging.getLogger(__name__)


def _guess_image_mime(filename: str) -> str:
    f = (filename or "").lower()
    if f.endswith(".png"):
        return "image/png"
    if f.endswith(".jpg") or f.endswith(".jpeg"):
        return "image/jpeg"
    if f.endswith(".gif"):
        return "image/gif"
    if f.endswith(".bmp"):
        return "image/bmp"
    if f.endswith(".webp"):
        return "image/webp"
    if f.endswith(".tif") or f.endswith(".tiff"):
        return "image/tiff"
    # DOCX commonly uses PNG/JPEG for embedded media.
    return "image/png"


def _hash_text(text: str) -> str:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


class DOCXParser(BaseParser):
    """DOCX parser with embedded-image OCR."""

    @property
    def supported_types(self) -> list[str]:
        return [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]

    def parse(
        self,
        file_bytes: bytes,
        *,
        ocr_provider: Optional[OcrProvider] = None,
        ocr_docx_images: Optional[bool] = None,
        ocr_docx_image_max_count: Optional[int] = None,
        original_filename: Optional[str] = None,
        **_: Any,
    ) -> ExtractionResult:
        provider = ocr_provider if ocr_provider is not None else get_ocr_provider()

        if ocr_docx_images is None:
            ocr_docx_images = os.getenv("OCR_DOCX_IMAGES", "true").strip().lower() in (
                "true", "1", "yes", "on",
            )
        if ocr_docx_image_max_count is None:
            try:
                ocr_docx_image_max_count = int(os.getenv("OCR_DOCX_IMAGE_MAX_COUNT", "50"))
            except (TypeError, ValueError):
                ocr_docx_image_max_count = 50

        # 1. Native text
        try:
            doc = Document(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Could not open DOCX: {e}") from e

        paragraphs: List[str] = []
        try:
            for p in doc.paragraphs:
                if p.text:
                    paragraphs.append(p.text)
        except Exception as e:
            logger.warning("python-docx paragraph walk failed: %s", e)
        native_text = "\n".join(paragraphs).strip()
        native_chars = len(native_text)

        # 2. Extract embedded images
        warnings: list = []
        embedded: List[ExtractedImage] = []
        images_detected = 0
        images_ocr_processed = 0
        ocr_provider_name: Optional[str] = None
        ocr_confidences: List[float] = []
        ocr_text_chars = 0

        # 2a. Walk the zip in deterministic order so sequence_number is
        # consistent.
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                media_files = sorted(
                    n for n in zf.namelist()
                    if n.startswith("word/media/")
                )
                images_detected = len(media_files)
                if images_detected > ocr_docx_image_max_count:
                    warnings.append(
                        f"DOCX contains {images_detected} embedded images; "
                        f"OCR_DOCX_IMAGE_MAX_COUNT={ocr_docx_image_max_count} caps processing"
                    )
                for seq, name in enumerate(media_files[:ocr_docx_image_max_count]):
                    try:
                        img_bytes = zf.read(name)
                    except Exception as e:
                        warnings.append(f"Could not read embedded image {name}: {e}")
                        continue

                    mime = _guess_image_mime(name)

                    # Validate
                    ok, err = validate_image_bytes(
                        img_bytes, mime, max_bytes=15 * 1024 * 1024
                    )
                    if not ok:
                        warnings.append(
                            f"Skipping embedded image {name}: {err}"
                        )
                        # Record as a failed entry so the row exists for audit.
                        embedded.append(
                            ExtractedImage(
                                raw_bytes=img_bytes,
                                mime_type=mime,
                                source_type="docx_image_ocr",
                                page_number=None,
                                sequence_number=seq,
                                original_filename=name.rsplit("/", 1)[-1],
                                storage_key=None,
                                ocr_text="",
                                ocr_provider=None,
                                ocr_status="failed",
                                ocr_error=err,
                                ocr_text_hash=None,
                                width=None,
                                height=None,
                                byte_size=len(img_bytes),
                            )
                        )
                        continue

                    if not ocr_docx_images or provider is None:
                        status = "disabled" if provider is None else "skipped"
                        embedded.append(
                            ExtractedImage(
                                raw_bytes=img_bytes,
                                mime_type=mime,
                                source_type="docx_image_ocr",
                                sequence_number=seq,
                                original_filename=name.rsplit("/", 1)[-1],
                                ocr_text="",
                                ocr_provider=None,
                                ocr_status=status,
                                ocr_text_hash=None,
                                byte_size=len(img_bytes),
                            )
                        )
                        continue

                    # Preprocess + OCR.
                    try:
                        prep = preprocess_for_ocr(img_bytes)
                        pre_bytes = prep.image_bytes
                        prep_steps = prep.applied_steps
                        prep_w, prep_h = prep.width, prep.height
                    except Exception as e:
                        logger.warning(
                            "DOCX embedded image preprocessing failed: %s", e
                        )
                        pre_bytes = img_bytes
                        prep_steps = []
                        prep_w = prep_h = None

                    try:
                        ocr_result = provider.extract_text(pre_bytes, mime)
                    except OcrProviderUnavailable as e:
                        warnings.append(f"OCR unavailable for embedded image {name}: {e}")
                        embedded.append(
                            ExtractedImage(
                                raw_bytes=img_bytes,
                                mime_type=mime,
                                source_type="docx_image_ocr",
                                sequence_number=seq,
                                original_filename=name.rsplit("/", 1)[-1],
                                ocr_text="",
                                ocr_provider=None,
                                ocr_status="failed",
                                ocr_error=str(e),
                                ocr_text_hash=None,
                                width=prep_w,
                                height=prep_h,
                                byte_size=len(img_bytes),
                            )
                        )
                        continue
                    except Exception as e:
                        warnings.append(f"OCR failed for embedded image {name}: {e}")
                        embedded.append(
                            ExtractedImage(
                                raw_bytes=img_bytes,
                                mime_type=mime,
                                source_type="docx_image_ocr",
                                sequence_number=seq,
                                original_filename=name.rsplit("/", 1)[-1],
                                ocr_text="",
                                ocr_provider=None,
                                ocr_status="failed",
                                ocr_error=str(e),
                                ocr_text_hash=None,
                                width=prep_w,
                                height=prep_h,
                                byte_size=len(img_bytes),
                            )
                        )
                        continue

                    cleaned = clean_ocr_text(ocr_result.text or "")
                    if ocr_provider_name is None and ocr_result.provider:
                        ocr_provider_name = ocr_result.provider
                    if ocr_result.confidence is not None:
                        ocr_confidences.append(ocr_result.confidence)

                    if cleaned:
                        ocr_text_chars += len(cleaned)
                        images_ocr_processed += 1
                        status = "success"
                    else:
                        status = "empty"

                    embedded.append(
                        ExtractedImage(
                            raw_bytes=img_bytes,
                            mime_type=mime,
                            source_type="docx_image_ocr",
                            sequence_number=seq,
                            original_filename=name.rsplit("/", 1)[-1],
                            ocr_text=cleaned,
                            ocr_provider=ocr_result.provider,
                            ocr_status=status,
                            ocr_confidence=ocr_result.confidence,
                            ocr_text_hash=_hash_text(cleaned),
                            width=prep_w,
                            height=prep_h,
                            byte_size=len(img_bytes),
                        )
                    )
        except zipfile.BadZipFile as e:
            raise ValueError(f"DOCX is not a valid zip archive: {e}") from e
        except Exception as e:
            # Last-ditch: if we cannot enumerate media, log and continue.
            warnings.append(f"DOCX media enumeration failed: {e}")

        # 3. Build joined text in reading order. Native paragraphs
        #    first, then per-image OCR blocks sorted by sequence_number.
        chunks: List[str] = []
        if native_text:
            chunks.append(native_text)
        for img in sorted(embedded, key=lambda x: x.sequence_number):
            if img.ocr_text:
                chunks.append(
                    f"--- Embedded image {img.sequence_number + 1} "
                    f"({img.original_filename or 'image'}) ---\n"
                    f"[OCR Text]\n{img.ocr_text}"
                )

        text = "\n\n".join(chunks).strip()

        # Quality accounting
        avg_conf = (
            sum(ocr_confidences) / len(ocr_confidences)
            if ocr_confidences
            else None
        )
        very_little_text = (native_chars + ocr_text_chars) < 50
        many_images_no_text = (
            images_detected >= 3 and (ocr_text_chars + native_chars) < 200
        )
        needs_human_review = bool(
            very_little_text
            or many_images_no_text
            or (
                images_detected > 0
                and provider is None
            )
        )

        extraction_method = "docx_python_docx"
        if images_ocr_processed > 0:
            extraction_method = "docx_python_docx_with_ocr"

        return ExtractionResult(
            text=text,
            extraction_method=extraction_method,
            native_text_chars=native_chars,
            ocr_text_chars=ocr_text_chars,
            images_detected=images_detected,
            images_ocr_processed=images_ocr_processed,
            pages_ocr_processed=images_ocr_processed,
            ocr_average_confidence=avg_conf,
            ocr_provider=ocr_provider_name,
            ocr_disabled=(provider is None),
            needs_human_review=needs_human_review,
            warnings=warnings,
            images=embedded,
            metadata={
                "ocr_docx_image_max_count": ocr_docx_image_max_count,
            },
        )
