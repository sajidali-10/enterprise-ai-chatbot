"""
OCR Image Preprocessing (Phase 34A).

Safe, conservative preprocessing that improves Tesseract OCR accuracy
on real-world images (screenshots, scanned documents, phone photos)
without being destructive.

Design constraints (from Phase 34A spec):
    - Original image is NEVER modified. This module operates on a copy.
    - Preprocessing is opt-out via ``preprocess=False``.
    - Avoid transformations that empirically hurt OCR (e.g. heavy
      bilateral filtering that erases small text).
    - Each step is independently toggleable through settings so
      operations can dial behaviour without code changes.

Steps (in order):
    1. EXIF orientation correction — fixes photos that were rotated
       by the camera.
    2. Mode normalization — convert palette/RGBA/etc. to ``"RGB"`` or
       ``"L"`` so Tesseract receives a sane format.
    3. Grayscale — improves contrast for Tesseract; preserves color
       off by default (only used when explicitly enabled or when the
       image is already single-channel).
    4. Contrast enhancement — autocontrast stretching to expand the
       dynamic range.
    5. Sharpening — mild unsharp mask to crisp blurred text.
    6. Upscaling — if the image is below the safe OCR resolution
       (~150 DPI effective), it is upscaled using Lanczos. Never
       downscaled.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Target minimum effective DPI for OCR — empirical lower bound below
# which Tesseract accuracy collapses. 150 DPI on a 1 inch-tall letter
# page ≈ 1100 pixels of vertical text height. For arbitrary user images
# we instead use a pixel-area heuristic.
MIN_OCR_PIXEL_AREA = 800 * 600
UPSCALE_TARGET_PIXEL_AREA = 1600 * 1200


@dataclass
class PreprocessResult:
    """Outcome of :func:`preprocess_for_ocr`.

    Attributes:
        image_bytes: PNG-encoded bytes of the preprocessed image
            (suitable for pytesseract / Tesseract).
        width: Pixel width after preprocessing.
        height: Pixel height after preprocessing.
        applied_steps: Names of preprocessing steps that actually ran.
        skipped_reason: If preprocessing was skipped, the reason
            (``"disabled"``, ``"too_small_skipped"``, etc.). ``None``
            otherwise.
    """

    image_bytes: bytes
    width: int
    height: int
    applied_steps: list
    skipped_reason: Optional[str] = None


def _lazy_pil():
    try:
        from PIL import Image, ImageOps, ImageFilter  # type: ignore
    except ImportError as e:  # pragma: no cover - environment issue
        raise RuntimeError(
            "Pillow is required for OCR preprocessing. Install with: pip install Pillow"
        ) from e
    return Image, ImageOps, ImageFilter


def preprocess_for_ocr(
    image_bytes: bytes,
    *,
    apply_exif_orientation: bool = True,
    apply_grayscale: bool = True,
    apply_contrast: bool = True,
    apply_sharpening: bool = True,
    apply_upscaling: bool = True,
    upscale_target_area: int = UPSCALE_TARGET_PIXEL_AREA,
) -> PreprocessResult:
    """Run the safe preprocessing pipeline.

    Args:
        image_bytes: Raw image bytes.
        apply_exif_orientation: Apply ``ImageOps.exif_transpose``.
        apply_grayscale: Convert to ``"L"`` after EXIF.
        apply_contrast: ``ImageOps.autocontrast``.
        apply_sharpening: Mild ``ImageFilter.UnsharpMask``.
        apply_upscaling: If the image is below
            ``upscale_target_area`` pixels, upscale via Lanczos.
        upscale_target_area: Pixel-area threshold below which the
            image is upscaled. Never downscaled.

    Returns:
        :class:`PreprocessResult` with PNG-encoded bytes and
        metadata.
    """
    Image, ImageOps, ImageFilter = _lazy_pil()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as e:
        raise ValueError(f"Could not decode image bytes: {e}") from e

    applied: list = []

    if apply_exif_orientation:
        try:
            transposed = ImageOps.exif_transpose(image)
            if transposed is not None and transposed is not image:
                applied.append("exif_orientation")
                image = transposed
        except Exception:
            logger.debug("EXIF orientation step skipped", exc_info=True)

    # Normalize mode — Tesseract works best with RGB or L.
    if image.mode not in ("RGB", "L"):
        try:
            image = image.convert("RGB")
            applied.append("mode_normalize")
        except Exception:
            pass

    if apply_grayscale and image.mode != "L":
        try:
            image = image.convert("L")
            applied.append("grayscale")
        except Exception:
            pass

    if apply_contrast:
        try:
            before = image.getextrema() if hasattr(image, "getextrema") else None
            image = ImageOps.autocontrast(image, cutoff=1)
            after = image.getextrema() if hasattr(image, "getextrema") else None
            if before != after:
                applied.append("autocontrast")
        except Exception:
            logger.debug("autocontrast step skipped", exc_info=True)

    if apply_sharpening:
        try:
            sharpened = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
            applied.append("sharpen")
            image = sharpened
        except Exception:
            pass

    if apply_upscaling:
        w, h = image.size
        area = w * h
        if 0 < area < upscale_target_area:
            try:
                scale = (upscale_target_area / area) ** 0.5
                new_w = max(w + 1, int(round(w * scale)))
                new_h = max(h + 1, int(round(h * scale)))
                image = image.resize((new_w, new_h), Image.LANCZOS)
                applied.append(f"upscale_{w}x{h}_to_{new_w}x{new_h}")
            except Exception:
                pass

    # Encode as PNG (lossless) for Tesseract.
    buf = io.BytesIO()
    try:
        image.save(buf, format="PNG", optimize=False)
    except Exception as e:
        raise RuntimeError(f"Could not encode preprocessed image: {e}") from e

    width, height = image.size
    return PreprocessResult(
        image_bytes=buf.getvalue(),
        width=width,
        height=height,
        applied_steps=applied,
        skipped_reason=None,
    )


def is_image_mime_supported(mime_type: str) -> bool:
    """Return True if the MIME is one we can attempt to OCR."""
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
    }


def is_image_extension_supported(extension: str) -> bool:
    """Return True if the file extension is one we accept at upload."""
    if not extension:
        return False
    e = extension.lower().strip()
    if not e.startswith("."):
        e = "." + e
    return e in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
    }


def image_mime_from_extension(extension: str) -> Optional[str]:
    """Best-effort MIME-type resolution from extension."""
    if not extension:
        return None
    e = extension.lower().strip()
    if not e.startswith("."):
        e = "." + e
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
    }.get(e)


def validate_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    max_bytes: int,
    min_dim: int = 16,
    max_dim: int = 20000,
) -> Tuple[bool, str]:
    """Validate raw image bytes for OCR ingestion.

    Returns ``(ok, error_message)``. ``error_message`` is empty on
    success. Reasons checked:
        - bytes are non-empty
        - bytes do not exceed ``max_bytes``
        - Pillow can decode the bytes
        - reported MIME matches actual decoded format (best-effort)
        - image dimensions are within ``[min_dim, max_dim]``
    """
    Image, _, _ = _lazy_pil()
    if not image_bytes:
        return False, "empty image bytes"
    if len(image_bytes) > max_bytes:
        return False, f"image too large ({len(image_bytes)} > {max_bytes} bytes)"
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # full decode validation
        # verify() leaves the image unusable — reopen for dimensions.
        image = Image.open(io.BytesIO(image_bytes))
        w, h = image.size
        if w < min_dim or h < min_dim:
            return False, f"image too small ({w}x{h} < {min_dim}x{min_dim})"
        if w > max_dim or h > max_dim:
            return False, f"image too large ({w}x{h} > {max_dim}x{max_dim})"
        fmt = (image.format or "").lower()
        if mime_type and fmt:
            fmt_to_mime = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "jpg": "image/jpeg",
                "webp": "image/webp",
                "tiff": "image/tiff",
                "bmp": "image/bmp",
                "gif": "image/gif",
            }
            expected = fmt_to_mime.get(fmt)
            # Allow 'image/jpg' as alias of 'image/jpeg'.
            if expected and mime_type.lower() not in (expected, "image/jpg"):
                return False, f"MIME {mime_type!r} does not match decoded format {fmt!r}"
        return True, ""
    except Exception as e:
        return False, f"could not decode image: {e}"
