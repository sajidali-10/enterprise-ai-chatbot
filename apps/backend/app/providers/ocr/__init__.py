"""
OCR Provider Factory (Phase 34A).

Mirrors the existing provider-factory pattern used for LLMs, embeddings,
vector stores, etc. The active provider is selected via the
``OCR_PROVIDER`` env var (``"tesseract"`` by default).

Supported providers:
    - ``tesseract`` — local Tesseract OCR via pytesseract.

Future providers (placeholder list, NOT yet implemented):
    - ``paddleocr``   — planned
    - ``textract``    — planned (AWS)
    - ``azure_di``    — planned (Azure Document Intelligence)
    - ``google_docai`` — planned (Google Document AI)

Adding a new provider:
    1. Implement :class:`app.providers.ocr.base.OcrProvider` in a new module.
    2. Add the name to :data:`AVAILABLE_OCR_PROVIDERS`.
    3. Add a branch in :func:`get_ocr_provider`.

Disabled mode:
    When ``OCR_ENABLED=false`` the factory returns ``None`` and callers
    must short-circuit. The factory itself never raises for "OCR
    disabled" — that's a deployment choice.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from app.providers.ocr.base import (
    OcrPage,
    OcrProvider,
    OcrProviderError,
    OcrProviderUnavailable,
    OcrResult,
    OcrTimeout,
    OcrUnsupportedFormat,
    OcrInvalidInput,
)

logger = logging.getLogger(__name__)


AVAILABLE_OCR_PROVIDERS = ["tesseract"]

FUTURE_OCR_PROVIDERS = {
    "paddleocr": "planned",
    "textract": "planned",
    "azure_di": "planned",
    "google_docai": "planned",
    "none": "available",  # explicit disable
}


def _ocr_enabled() -> bool:
    val = os.getenv("OCR_ENABLED", "true").strip().lower()
    return val in ("true", "1", "yes", "on")


def _resolve(name: str, available: list[str], provider_kind: str) -> str:
    """Validate a provider name. Raises ValueError on unknown."""
    if name not in available:
        raise ValueError(
            f"Unknown {provider_kind} provider: '{name}'. "
            f"Available in this phase: {available}. "
            f"Future (not yet implemented): {sorted(FUTURE_OCR_PROVIDERS)}. "
            f"See app/providers/ocr/base.py for adding new providers."
        )
    return name


def get_ocr_provider() -> Optional[OcrProvider]:
    """Resolve the active OCR provider.

    Returns ``None`` when OCR is disabled (``OCR_ENABLED=false`` or
    ``OCR_PROVIDER=none``). Raises ``ValueError`` if the configured
    provider is unknown. Raises ``OcrProviderUnavailable`` if the
    configured provider's binary is missing — except for ``none``,
    which always returns ``None``.

    Provider instances are constructed lazily on each call. They are
    lightweight (no network I/O, no heavy model load in the
    Tesseract case) so caching is not necessary.
    """
    if not _ocr_enabled():
        logger.debug("OCR is disabled via OCR_ENABLED=false; returning None.")
        return None

    name = os.getenv("OCR_PROVIDER", "tesseract").strip().lower() or "tesseract"

    if name == "none":
        return None

    _resolve(name, AVAILABLE_OCR_PROVIDERS, "OCR")

    if name == "tesseract":
        # Import lazily so that the rest of the app can start even if
        # tesseract's dependencies (pytesseract / Pillow) are missing.
        from app.providers.ocr.tesseract_provider import TesseractOcrProvider
        provider = TesseractOcrProvider()
        # Do NOT raise here if the binary is missing — the admin
        # status endpoint should still be able to report "unavailable"
        # rather than crash. Callers must check health_check() before
        # using the provider.
        return provider

    raise ValueError(f"OCR provider '{name}' not implemented.")


def get_ocr_status() -> dict:
    """Return a safe, admin-readable OCR status summary.

    Does not include secrets. Safe to log.
    """
    enabled = _ocr_enabled()
    requested = os.getenv("OCR_PROVIDER", "tesseract").strip().lower() or "tesseract"
    info: dict = {
        "ocr_enabled": enabled,
        "ocr_provider": requested,
        "ocr_available_providers": AVAILABLE_OCR_PROVIDERS,
        "ocr_future_providers": dict(FUTURE_OCR_PROVIDERS),
        "ocr_health": {"available": False, "provider": requested, "engine_version": None, "details": "OCR disabled"},
        "ocr_settings": {
            "language": os.getenv("OCR_LANGUAGE", "eng"),
            "min_confidence": int(os.getenv("OCR_MIN_CONFIDENCE", "60")),
            "pdf_fallback": os.getenv("OCR_PDF_FALLBACK", "true").strip().lower() in ("true", "1", "yes", "on"),
            "pdf_page_text_min_chars": int(os.getenv("OCR_PDF_PAGE_TEXT_MIN_CHARS", "40")),
            "docx_images": os.getenv("OCR_DOCX_IMAGES", "true").strip().lower() in ("true", "1", "yes", "on"),
            "docx_image_max_count": int(os.getenv("OCR_DOCX_IMAGE_MAX_COUNT", "50")),
            "image_max_size_mb": int(os.getenv("OCR_IMAGE_MAX_SIZE_MB", "15")),
            "render_dpi": int(os.getenv("OCR_RENDER_DPI", "200")),
            "timeout_s": float(os.getenv("OCR_TIMEOUT_S", "120")),
            "upscaling_enabled": os.getenv("OCR_UPSCALING_ENABLED", "true").strip().lower() in ("true", "1", "yes", "on"),
        },
    }
    if not enabled:
        return info
    try:
        provider = get_ocr_provider()
        if provider is not None:
            info["ocr_health"] = provider.health_check()
            info["ocr_provider_info"] = provider.provider_info()
    except Exception as e:
        info["ocr_health"] = {
            "available": False,
            "provider": requested,
            "engine_version": None,
            "details": f"provider init failed: {e}",
        }
    return info


__all__ = [
    "AVAILABLE_OCR_PROVIDERS",
    "FUTURE_OCR_PROVIDERS",
    "OcrProvider",
    "OcrResult",
    "OcrPage",
    "OcrProviderError",
    "OcrProviderUnavailable",
    "OcrUnsupportedFormat",
    "OcrTimeout",
    "OcrInvalidInput",
    "get_ocr_provider",
    "get_ocr_status",
]
