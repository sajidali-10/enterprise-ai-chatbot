"""
Tesseract OCR Provider (Phase 34A).

Wraps the local Tesseract binary via ``pytesseract``. Designed to work
even when the Tesseract binary or pytesseract is missing — the
provider simply reports ``health_check()["available"] = False`` and
raises :class:`OcrProviderUnavailable` when used.

This implementation never logs OCR content. Only structural metadata
(provider name, dimensions, confidence, language) is allowed to leave
the process via ``health_check`` / ``provider_info``.

Tesseract supports many languages via language packs. Phase 34A only
installs ``eng``; additional languages can be installed via
``apt-get install tesseract-ocr-<lang>`` and configured through the
``OCR_LANGUAGE`` env var.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, List, Optional

from app.providers.ocr.base import (
    OcrPage,
    OcrProvider,
    OcrProviderUnavailable,
    OcrResult,
    OcrTimeout,
    OcrUnsupportedFormat,
    OcrInvalidInput,
)

logger = logging.getLogger(__name__)


# Lazy imports so the rest of the app can load even when Pillow /
# pytesseract / PyMuPDF are missing.
def _lazy_pil():
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:  # pragma: no cover - environment issue
        raise OcrProviderUnavailable(
            "Pillow is not installed; cannot run Tesseract OCR."
        ) from e
    return Image


def _lazy_pytesseract():
    try:
        import pytesseract  # type: ignore
    except ImportError as e:  # pragma: no cover - environment issue
        raise OcrProviderUnavailable(
            "pytesseract is not installed; cannot run Tesseract OCR."
        ) from e
    return pytesseract


def _lazy_fitz():
    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError as e:  # pragma: no cover - environment issue
        raise OcrProviderUnavailable(
            "PyMuPDF (fitz) is not installed; cannot OCR PDF pages."
        ) from e
    return fitz


def _is_tesseract_binary_available() -> bool:
    """Return True iff the ``tesseract`` binary is on PATH."""
    return shutil.which("tesseract") is not None


def _tesseract_version() -> Optional[str]:
    """Return the installed Tesseract version (e.g. ``"5.3.4"``) or None."""
    binary = shutil.which("tesseract")
    if not binary:
        return None
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=10
        )
        # First line looks like: "tesseract 5.3.4"
        line = (out.stdout or "").strip().splitlines()
        if line:
            return line[0].strip()
        return None
    except Exception:  # pragma: no cover - environment dependent
        return None


class TesseractOcrProvider(OcrProvider):
    """Tesseract OCR provider.

    Settings (read at call time from environment, defaulting to safe
    values if the helper ``ocr_settings`` is not provided):

    - ``OCR_LANGUAGE`` (default ``eng``): tesseract language code(s).
    - ``OCR_TIMEOUT_S`` (default ``120``): per-call timeout in seconds.
    - ``OCR_RENDER_DPI`` (default ``200``): PDF page render DPI.

    The provider never raises for "OCR produced no text" — that is
    returned as ``OcrResult(success=False, text="", ...)``. Failures
    that mean OCR cannot run at all (binary missing, Pillow missing,
    corrupt input that cannot be opened) are raised as typed
    exceptions so the ingestion layer can decide what to do.
    """

    name = "tesseract"

    def __init__(
        self,
        language: Optional[str] = None,
        timeout_s: Optional[float] = None,
        render_dpi: Optional[int] = None,
        tessdata_dir: Optional[str] = None,
    ) -> None:
        self._language = language or os.getenv("OCR_LANGUAGE", "eng")
        try:
            self._timeout_s = float(
                timeout_s if timeout_s is not None else os.getenv("OCR_TIMEOUT_S", "120")
            )
        except (TypeError, ValueError):
            self._timeout_s = 120.0
        try:
            self._render_dpi = int(
                render_dpi if render_dpi is not None else os.getenv("OCR_RENDER_DPI", "200")
            )
        except (TypeError, ValueError):
            self._render_dpi = 200
        self._tessdata_dir = tessdata_dir or os.getenv("TESSDATA_PREFIX") or None
        self._binary_version_cache: Optional[str] = None

    # ------------------------------------------------------------------
    # Health / info
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        info = {
            "available": _is_tesseract_binary_available(),
            "provider": self.name,
            "engine_version": None,
            "details": None,
        }
        ver = _tesseract_version()
        if ver:
            info["engine_version"] = ver
            info["details"] = "tesseract binary on PATH"
        else:
            info["details"] = "tesseract binary not found on PATH"
        # Try lazy imports to surface missing deps separately.
        try:
            _lazy_pil()
        except OcrProviderUnavailable as e:
            info["available"] = False
            info["details"] = str(e)
        try:
            _lazy_pytesseract()
        except OcrProviderUnavailable as e:
            info["available"] = False
            info["details"] = str(e)
        return info

    def provider_info(self) -> dict:
        info = {
            "provider": self.name,
            "language": self._language,
            "timeout_s": self._timeout_s,
            "render_dpi": self._render_dpi,
            "engine_version": _tesseract_version(),
        }
        return info

    # ------------------------------------------------------------------
    # Image OCR
    # ------------------------------------------------------------------

    def extract_text(
        self,
        image_bytes: bytes,
        mime_type: str,
        language: Optional[str] = None,
    ) -> OcrResult:
        if not _is_tesseract_binary_available():
            raise OcrProviderUnavailable(
                "tesseract binary is not installed. Install with "
                "'apt-get install -y tesseract-ocr tesseract-ocr-eng'."
            )
        if not self.supports_mime(mime_type) or mime_type == "application/pdf":
            raise OcrUnsupportedFormat(f"Tesseract OCR does not handle MIME {mime_type!r}")

        Image = _lazy_pil()
        pytesseract = _lazy_pytesseract()

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # force decode — surfaces corrupt input
        except Exception as e:
            raise OcrInvalidInput(f"Could not decode image bytes: {e}") from e

        # Normalize mode — Tesseract works best on RGB or L.
        try:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
        except Exception as e:
            raise OcrInvalidInput(f"Could not normalize image mode: {e}") from e

        width, height = image.size
        lang = (language or self._language or "eng")

        try:
            text, conf = self._image_to_data(pytesseract, image, lang)
        except RuntimeError as e:
            # pytesseract raises Tesseract subprocess errors
            msg = str(e)
            if "timeout" in msg.lower():
                raise OcrTimeout(msg) from e
            raise OcrProviderUnavailable(f"Tesseract failed: {e}") from e
        except Exception as e:
            raise OcrProviderUnavailable(f"Tesseract failed: {e}") from e

        page = OcrPage(
            page_index=0,
            text=text or "",
            confidence=conf,
            width=width,
            height=height,
            language=lang,
        )
        pages: List[OcrPage] = [page]
        warnings: List[str] = []
        if not text or not text.strip():
            warnings.append("OCR returned empty text")
        success = bool(text and text.strip())
        return OcrResult(
            text=text or "",
            confidence=conf,
            language=lang,
            pages=pages,
            provider=self.name,
            extraction_method="tesseract_image",
            warnings=warnings,
            metadata={
                "width": width,
                "height": height,
                "mime_type": mime_type,
            },
            success=success,
            error=None if success else "empty",
        )

    def _image_to_data(self, pytesseract, image, lang: str) -> tuple[str, Optional[float]]:
        """Run tesseract on a PIL image; return (text, avg_confidence)."""
        # image_to_data returns per-word info; aggregate to overall confidence.
        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                timeout=self._timeout_s,
                output_type=pytesseract.Output.DICT,
            )
            words = data.get("text", []) or []
            confs = data.get("conf", []) or []
            pieces: List[str] = []
            numeric_confs: List[float] = []
            for w, c in zip(words, confs):
                if not w:
                    continue
                pieces.append(w)
                try:
                    cval = float(c)
                except (TypeError, ValueError):
                    continue
                if cval >= 0:
                    numeric_confs.append(cval)
            text = " ".join(pieces).strip()
            avg_conf = (sum(numeric_confs) / len(numeric_confs)) if numeric_confs else None
            return text, avg_conf
        except pytesseract.TesseractError as e:
            # Some installs can't use image_to_data with all languages;
            # fall back to plain image_to_string.
            if "timeout" in str(e).lower():
                raise
            text = pytesseract.image_to_string(image, lang=lang, timeout=self._timeout_s)
            return (text or "").strip(), None

    # ------------------------------------------------------------------
    # PDF page OCR
    # ------------------------------------------------------------------

    def extract_text_from_pdf_page(
        self,
        pdf_bytes: bytes,
        page_index: int,
        dpi: int = 200,
        language: Optional[str] = None,
    ) -> OcrResult:
        if not _is_tesseract_binary_available():
            raise OcrProviderUnavailable(
                "tesseract binary is not installed. Install with "
                "'apt-get install -y tesseract-ocr tesseract-ocr-eng'."
            )
        fitz = _lazy_fitz()
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise OcrInvalidInput(f"Could not open PDF: {e}") from e

        try:
            if page_index < 0 or page_index >= doc.page_count:
                raise OcrInvalidInput(
                    f"PDF page index {page_index} out of range (0..{doc.page_count - 1})"
                )
            page = doc.load_page(page_index)
            try:
                zoom = max(1.0, dpi / 72.0)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                image_bytes = pix.tobytes("png")
                width, height = pix.width, pix.height
            except Exception as e:
                raise OcrProviderUnavailable(f"PDF render failed: {e}") from e
        finally:
            try:
                doc.close()
            except Exception:
                pass

        result = self.extract_text(image_bytes, mime_type="image/png", language=language)
        # Retag extraction_method for clarity.
        result.extraction_method = "tesseract_pdf_page"
        result.metadata["dpi"] = int(dpi)
        if result.pages:
            result.pages[0].page_index = int(page_index)
            result.pages[0].width = width
            result.pages[0].height = height
        return result
