"""
Direct image parser (Phase 34A).

Handles uploaded image files (PNG/JPEG/WEBP/TIFF/BMP/GIF). The parser:
    1. Validates bytes against the MIME type using Pillow's decoder.
    2. Applies safe preprocessing (EXIF orientation, contrast,
       sharpening, upscaling) — original bytes are preserved and
       also stored in MinIO alongside the preprocessed render.
    3. Runs OCR via the configured :class:`OcrProvider`.
    4. Cleans the OCR text and sanitizes secrets via
       :func:`app.ingestion.ocr_cleaning.clean_ocr_text` (note: secret
       redaction is layered at the upload endpoint — this parser only
       does deterministic cleaning; the redaction happens AFTER, so
       that the same sanitizer applies to text and OCR alike).
    5. Returns an :class:`ExtractionResult` with one
       :class:`ExtractedImage`.

The parser NEVER raises when OCR produces no text — that is signaled
via ``ExtractionResult.warnings`` and a low-confidence / empty
``ocr_status``. The parser DOES raise on hard errors (corrupt input,
oversized image, OCR provider missing) so the upload endpoint can
decide between "fail the request" and "index partial content".

Security:
    - Never logs image bytes or OCR text.
    - OCR text is returned via ``ExtractionResult.text`` which is
      sanitized again at the upload endpoint via
      :func:`app.services.langsmith_tracing.redact_text`.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from typing import Any, Optional

from app.ingestion.parsers.base import (
    BaseParser,
    ExtractedImage,
    ExtractionResult,
)
from app.ingestion.ocr_preprocessing import (
    is_image_mime_supported,
    preprocess_for_ocr,
    validate_image_bytes,
)
from app.ingestion.ocr_cleaning import clean_ocr_text
from app.providers.ocr import (
    OcrProvider,
    OcrProviderUnavailable,
    get_ocr_provider,
)
from app.providers.ocr.base import OcrResult

logger = logging.getLogger(__name__)


# Headers used in the joined RAG text so future retrievers can present
# provenance. Mirrors the Phase 34A spec.
PROVENANCE_TEMPLATE = (
    "Source Type: Image\n"
    "OCR Text:\n{ocr_text}\n"
    "File: {original_name}\n"
    "Page: -\n"
)


def _ocr_text_hash(text: str) -> str:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


class ImageParser(BaseParser):
    """Parser for direct image uploads."""

    @property
    def supported_types(self) -> list[str]:
        return [
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp",
            "image/tiff",
            "image/bmp",
            "image/gif",
        ]

    def parse(
        self,
        file_bytes: bytes,
        *,
        original_filename: Optional[str] = None,
        original_storage_key: Optional[str] = None,
        ocr_provider: Optional[OcrProvider] = None,
        min_confidence: Optional[int] = None,
        image_max_size_mb: Optional[int] = None,
        **_: Any,
    ) -> ExtractionResult:
        # 1. Get OCR provider up-front (cheap; no I/O for Tesseract).
        provider = ocr_provider if ocr_provider is not None else get_ocr_provider()

        # 2. Determine MIME if not present.
        mime_type = ""
        for candidate in self.supported_types:
            # Caller passes mime via kwarg or via attribute on file_bytes
            # (we don't have access to the original MIME here, so rely
            # on validation below).
            mime_type = mime_type or candidate
        # We accept whatever the upload endpoint validated upstream; the
        # parser trusts the caller. We still verify by decoding.
        try:
            from PIL import Image  # type: ignore  # lazy
        except ImportError as e:  # pragma: no cover - environment issue
            raise RuntimeError(
                "Pillow is required for image parsing. Install with: pip install Pillow"
            ) from e

        # Decode to determine the actual MIME.
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            img = Image.open(io.BytesIO(file_bytes))
            fmt = (img.format or "").lower()
            mime_map = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "jpg": "image/jpeg",
                "webp": "image/webp",
                "tiff": "image/tiff",
                "bmp": "image/bmp",
                "gif": "image/gif",
            }
            mime_type = mime_map.get(fmt)
        except Exception as e:
            raise ValueError(f"Could not decode image: {e}") from e

        if not mime_type or not is_image_mime_supported(mime_type):
            raise ValueError(f"Unsupported image MIME type: {mime_type!r}")

        # 3. Validate against size limits.
        max_mb = image_max_size_mb if image_max_size_mb is not None else int(
            os.getenv("OCR_IMAGE_MAX_SIZE_MB", "15")
        )
        max_bytes = max_mb * 1024 * 1024
        ok, err = validate_image_bytes(file_bytes, mime_type, max_bytes)
        if not ok:
            raise ValueError(f"Image validation failed: {err}")

        width = img.size[0]
        height = img.size[1]

        # 4. Preprocess (returns PNG-encoded bytes).
        try:
            prep = preprocess_for_ocr(file_bytes)
        except Exception as e:
            logger.warning("Image preprocessing failed; using original bytes: %s", e)
            prep_image_bytes = file_bytes
            applied_steps: list = []
            prep_width, prep_height = width, height
        else:
            prep_image_bytes = prep.image_bytes
            applied_steps = prep.applied_steps
            prep_width, prep_height = prep.width, prep.height

        # 5. Run OCR.
        ocr_text = ""
        ocr_status = "pending"
        ocr_provider_name: Optional[str] = None
        ocr_confidence: Optional[float] = None
        ocr_error: Optional[str] = None
        warnings: list = []
        extraction_method = "image_no_ocr"
        ocr_text_hash_value: Optional[str] = None

        if provider is None:
            ocr_status = "disabled"
            warnings.append("OCR is disabled via OCR_ENABLED=false; image will not be indexed as text")
            # Even when OCR is disabled, we still produce a slim record
            # so future Vision phases can revisit the image.
            needs_human_review = True
        else:
            try:
                result: OcrResult = provider.extract_text(prep_image_bytes, mime_type)
            except OcrProviderUnavailable as e:
                ocr_status = "failed"
                ocr_error = str(e)
                warnings.append(f"OCR provider unavailable: {e}")
                needs_human_review = True
            except Exception as e:
                ocr_status = "failed"
                ocr_error = str(e)
                warnings.append(f"OCR failed: {e}")
                needs_human_review = True
            else:
                ocr_provider_name = result.provider or provider.name
                raw_text = result.text or ""
                ocr_text = clean_ocr_text(raw_text)
                ocr_text_hash_value = _ocr_text_hash(ocr_text)
                ocr_confidence = result.confidence
                extraction_method = (
                    result.extraction_method or "tesseract_image"
                )
                if result.warnings:
                    warnings.extend(result.warnings)

                threshold = (
                    min_confidence
                    if min_confidence is not None
                    else int(os.getenv("OCR_MIN_CONFIDENCE", "60"))
                )
                if not ocr_text:
                    ocr_status = "empty"
                    needs_human_review = True
                    warnings.append("OCR returned no text for this image")
                else:
                    if (
                        ocr_confidence is not None
                        and threshold > 0
                        and ocr_confidence < threshold
                    ):
                        ocr_status = "low_confidence"
                        warnings.append(
                            f"OCR confidence {ocr_confidence:.1f} below threshold {threshold}"
                        )
                    else:
                        ocr_status = "success"
                    needs_human_review = False

        # 6. Build the ExtractedImage metadata (raw_bytes is the
        # ORIGINAL bytes, not preprocessed — the original must remain
        # available for future Phase 34B Vision analysis).
        extracted = ExtractedImage(
            raw_bytes=file_bytes,
            mime_type=mime_type,
            source_type="direct_image",
            page_number=None,
            sequence_number=0,
            original_filename=original_filename,
            storage_key=original_storage_key,
            ocr_text=ocr_text,
            ocr_provider=ocr_provider_name,
            ocr_status=ocr_status,
            ocr_confidence=ocr_confidence,
            ocr_error=ocr_error,
            ocr_text_hash=ocr_text_hash_value,
            width=prep_width,
            height=prep_height,
            byte_size=len(file_bytes),
        )

        # 7. Compose the joined text for chunking.
        if ocr_text:
            text = PROVENANCE_TEMPLATE.format(
                ocr_text=ocr_text, original_name=(original_filename or "image")
            )
        else:
            text = ""

        if not warnings and ocr_status not in ("success",):
            warnings.append("No OCR content extracted")

        result_obj = ExtractionResult(
            text=text,
            extraction_method=extraction_method,
            native_text_chars=0,
            ocr_text_chars=len(ocr_text),
            images_detected=1,
            images_ocr_processed=1 if provider is not None else 0,
            pages_ocr_processed=0,
            ocr_average_confidence=ocr_confidence,
            ocr_provider=ocr_provider_name,
            ocr_disabled=(provider is None),
            needs_human_review=(provider is None) or (ocr_status in ("failed", "empty", "low_confidence")),
            warnings=warnings,
            images=[extracted],
            metadata={
                "width": prep_width,
                "height": prep_height,
                "byte_size": len(file_bytes),
                "preprocessing_steps": applied_steps,
                "original_mime_type": mime_type,
            },
        )
        return result_obj
