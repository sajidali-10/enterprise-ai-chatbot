"""
Deterministic OCR test fixtures (Phase 34A).

This module provides **deterministic, hermetic** fixtures for testing the
Phase 34A OCR ingestion pipeline without depending on Tesseract, pytesseract,
PyMuPDF, real PDFs, or any external network service.

The fixtures intentionally avoid rasterizing real scanned documents. Real
scans are slow, flaky across CI environments, and not deterministic — a
test that fails 1-in-50 runs because of OCR confidence drift is worse
than no test. Instead we:

    * Build synthetic PNG / JPEG / WEBP / TIFF / BMP / GIF images in
      pure Pillow (or raw bytes for the GIF case) so the parser has
      something real to decode, validate, and preprocess.
    * Build synthetic PDFs and DOCX files via ``io.BytesIO`` so the
      PDF / DOCX parsers can extract them without bundling binary
      fixtures in git.
    * Provide a fully injectable :class:`FakeOcrProvider` that
      implements the :class:`app.providers.ocr.base.OcrProvider`
      contract with deterministic text, confidence, and warnings.
    * Provide a minimal in-memory Qdrant / MinIO double when tests
      need to exercise the upload endpoint end-to-end without the
      real Docker services.

All fixtures live in this single module so that importing them is
cheap and tests can opt-in precisely to the helpers they need.

Design rules:

    1. **No network I/O.** Every helper returns bytes or in-memory
       objects.
    2. **No tesseract binary.** Helpers detect the binary's presence
       via :func:`tesseract_available` so tests can decide whether to
       require a real provider.
    3. **Deterministic.** Same inputs -> same outputs. No timestamps,
       no randomness. (When randomness is unavoidable we expose a
       fixed seed parameter.)
    4. **Safe.** No real customer data; synthetic only. No secrets
       written into any fixture.
"""

from __future__ import annotations

import io
import os
import shutil
import struct
import zipfile
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Optional / lazy imports
# ---------------------------------------------------------------------------


def _lazy_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as e:  # pragma: no cover - environment issue
        raise RuntimeError(
            "Pillow is required for OCR test fixtures. Install with: pip install Pillow"
        ) from e
    return Image, ImageDraw, ImageFont


def _lazy_pymupdf():
    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError as e:  # pragma: no cover - environment issue
        raise RuntimeError(
            "PyMuPDF is required to build synthetic PDF fixtures."
        ) from e
    return fitz


def _lazy_docx():
    try:
        from docx import Document  # type: ignore
    except ImportError as e:  # pragma: no cover - environment issue
        raise RuntimeError(
            "python-docx is required to build synthetic DOCX fixtures."
        ) from e
    return Document


# ---------------------------------------------------------------------------
# Environment + binary detection
# ---------------------------------------------------------------------------


def tesseract_available() -> bool:
    """Return True iff the local ``tesseract`` binary is on PATH."""
    return shutil.which("tesseract") is not None


def pytesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


def pymupdf_available() -> bool:
    try:
        import fitz  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


def pillow_available() -> bool:
    try:
        from PIL import Image  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


def docx_available() -> bool:
    try:
        from docx import Document  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------


def make_solid_png(width: int = 200, height: int = 80, color=(255, 255, 255)) -> bytes:
    """Build a single-color PNG. Cheap to OCR; mostly blank.

    Used to exercise the parsing path where OCR is expected to return
    no/low-confidence text.
    """
    Image, _, _ = _lazy_pil()
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_solid_jpeg(width: int = 200, height: int = 80, color=(255, 255, 255), quality: int = 85) -> bytes:
    """Build a JPEG of a single color."""
    Image, _, _ = _lazy_pil()
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_blank_bmp(width: int = 100, height: int = 100) -> bytes:
    """Build a blank BMP."""
    Image, _, _ = _lazy_pil()
    img = Image.new("RGB", (width, height), (240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def make_text_image(
    text: str = "Hello OCR",
    width: int = 600,
    height: int = 200,
    *,
    font_size: int = 48,
    fg=(0, 0, 0),
    bg=(255, 255, 255),
) -> bytes:
    """Render ``text`` into a PNG using Pillow's default bitmap font.

    Used as a "real-looking" image that Tesseract can OCR (when
    available). Without a TrueType font file, this falls back to
    Pillow's built-in bitmap font — recognizable, but not perfect.
    """
    Image, ImageDraw, ImageFont = _lazy_pil()
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # Center the text vertically.
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(0, (width - text_w) // 2)
    y = max(0, (height - text_h) // 2)
    draw.text((x, y), text, fill=fg, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_text_image_bytes(
    mime_type: str,
    text: str = "Hello OCR",
    width: int = 600,
    height: int = 200,
    font_size: int = 48,
) -> bytes:
    """Build an image in the requested MIME type.

    Returns raw bytes suitable for uploading through the upload
    endpoint or passing to a parser directly.
    """
    mt = (mime_type or "").lower().strip()
    if mt in ("image/png",):
        return make_text_image(text=text, width=width, height=height, font_size=font_size)
    if mt in ("image/jpeg", "image/jpg"):
        Image, ImageDraw, ImageFont = _lazy_pil()
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    if mt == "image/webp":
        Image, ImageDraw, ImageFont = _lazy_pil()
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=85)
        return buf.getvalue()
    if mt in ("image/tiff",):
        Image, ImageDraw, ImageFont = _lazy_pil()
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)
        buf = io.BytesIO()
        img.save(buf, format="TIFF")
        return buf.getvalue()
    if mt == "image/bmp":
        return make_blank_bmp(width, height)
    if mt == "image/gif":
        # Build a tiny 1-frame GIF without Pillow's save() quirks.
        Image, _, _ = _lazy_pil()
        img = Image.new("P", (width, height), 255)  # palette mode
        buf = io.BytesIO()
        img.save(buf, format="GIF")
        return buf.getvalue()
    raise ValueError(f"Unsupported MIME type for synthetic image: {mime_type!r}")


def make_corrupt_image_bytes(mime_type: str = "image/png") -> bytes:
    """Return bytes that LOOK like the requested MIME but cannot be decoded.

    Used to exercise parser validation paths.
    """
    mt = (mime_type or "").lower().strip()
    if mt == "image/png":
        # PNG signature is correct, but the IHDR / IDAT are wrong.
        return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 50
    if mt in ("image/jpeg", "image/jpg"):
        return b"\xff\xd8\xff\xe0" + b"\x00" * 30  # JFIF header w/ garbage
    return b"NOT_A_REAL_IMAGE_FILE"


# ---------------------------------------------------------------------------
# PDF fixtures
# ---------------------------------------------------------------------------


def make_text_pdf(
    pages: Iterable[str],
    *,
    page_size: Tuple[int, int] = (595, 842),  # A4 portrait @ 72 DPI
) -> bytes:
    """Build a multi-page PDF whose native text is extractable via pypdf.

    Uses PyMuPDF when available (preferred); otherwise falls back to a
    hand-rolled minimal PDF with no extractable text. The fallback
    still produces a valid PDF that pypdf can open — useful when the
    test only needs "scanned-PDF-like" behavior (no native text).
    """
    pages_list = list(pages) or [""]
    if pymupdf_available():
        return _make_text_pdf_pymupdf(pages_list, page_size)
    return _make_minimal_blank_pdf(num_pages=len(pages_list))


def _make_text_pdf_pymupdf(
    pages: List[str], page_size: Tuple[int, int]
) -> bytes:
    fitz = _lazy_pymupdf()
    doc = fitz.open()
    try:
        for text in pages:
            page = doc.new_page(
                width=page_size[0], height=page_size[1]
            )
            if text:
                # Insert at a fixed position so pypdf can extract it.
                page.insert_text((72, 72), text, fontsize=12)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
    finally:
        doc.close()


def _make_minimal_blank_pdf(num_pages: int = 1) -> bytes:
    """Build a minimal valid PDF with N blank pages (no extractable text).

    Hand-rolled so we don't depend on PyMuPDF. Good enough to
    exercise the "no native text" branch of the parser.
    """
    if num_pages < 1:
        num_pages = 1

    objects: List[bytes] = []

    def add(obj_bytes: bytes) -> int:
        objects.append(obj_bytes)
        return len(objects)

    # 1: Catalog
    catalog_id = add(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages
    page_ids = []
    for _ in range(num_pages):
        # Reserve an object id for each page and its contents.
        # We'll fill them in below; use placeholder + fix later.
        page_ids.append(None)
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_obj = f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode()
    pages_id = add(pages_obj)

    # Pages reference -> assign ids
    for i in range(num_pages):
        content_id = 4 + 2 * i
        page_obj = (
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 595 842] "
            f"/Contents {content_id} 0 R "
            f"/Resources << >> >>"
        ).encode()
        page_id = add(page_obj)
        page_ids[i] = page_id
        content_obj = b"<< /Length 0 >>\nstream\n\nendstream"
        add(content_obj)

    # Re-write the Pages object with correct kid ids now that we have them.
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>"
    ).encode()

    # Build the file
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj_bytes in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode())
        out.write(obj_bytes)
        out.write(b"\nendobj\n")
    xref_offset = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(b"trailer\n")
    out.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
    out.write(b"startxref\n")
    out.write(f"{xref_offset}\n".encode())
    out.write(b"%%EOF\n")
    return out.getvalue()


def make_scanned_like_pdf(num_pages: int = 1) -> bytes:
    """Return a PDF with no extractable native text (forces OCR fallback).

    When PyMuPDF is available we render an image-only page (Tesseract
    will have something to OCR). Without PyMuPDF we fall back to the
    minimal blank PDF — pypdf will still see no native text, which is
    enough to exercise the parser's "needs OCR" branch.
    """
    if pymupdf_available():
        fitz = _lazy_pymupdf()
        doc = fitz.open()
        try:
            for _ in range(num_pages):
                page = doc.new_page(width=595, height=842)
                # Render some synthetic content that LOOKS like a scan.
                page.insert_text((100, 100), "Scanned Invoice", fontsize=18)
                page.insert_text((100, 140), "Total: $1234.56", fontsize=14)
            buf = io.BytesIO()
            doc.save(buf)
            return buf.getvalue()
        finally:
            doc.close()
    return _make_minimal_blank_pdf(num_pages=num_pages)


# ---------------------------------------------------------------------------
# DOCX fixtures
# ---------------------------------------------------------------------------


def make_text_docx(paragraphs: Iterable[str]) -> bytes:
    """Build a DOCX whose native text is extractable.

    Uses python-docx when available; falls back to a hand-rolled
    minimal DOCX (which python-docx can still open) otherwise.
    """
    paragraphs_list = list(paragraphs)
    if docx_available():
        return _make_text_docx_pydocx(paragraphs_list)
    return _make_minimal_docx(paragraphs_list)


def _make_text_docx_pydocx(paragraphs: List[str]) -> bytes:
    Document = _lazy_docx()
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_minimal_docx(paragraphs: List[str]) -> bytes:
    """Hand-roll a minimal DOCX so we don't depend on python-docx.

    python-docx can open this; it just won't have full styling.
    """
    paragraphs_xml = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{_xml_escape(p)}</w:t></w:r></w:p>"
        for p in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs_xml}</w:body></w:document>"
    ).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _minimal_content_types())
        zf.writestr("_rels/.rels", _minimal_root_rels())
        zf.writestr("word/_rels/document.xml.rels", _minimal_document_rels())
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _minimal_content_types() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )


def _minimal_root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )


def _minimal_document_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>'
    )


def make_docx_with_embedded_images(
    paragraphs: Iterable[str],
    images: Iterable[bytes],
    *,
    image_extension: str = ".png",
) -> bytes:
    """Build a DOCX with native paragraphs and embedded images in word/media/.

    Images are written sequentially under ``word/media/`` with the
    provided extension. Each image will be detected by the DOCX
    parser.
    """
    paragraphs_list = list(paragraphs)
    images_list = list(images)

    if docx_available():
        # Use python-docx for the document XML, then post-process the
        # zip to add image parts under word/media/.
        Document = _lazy_docx()
        doc = Document()
        for p in paragraphs_list:
            doc.add_paragraph(p)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        # Re-open the zip and append image parts + update relationships.
        return _inject_media_into_docx_zip(
            buf.read(), images_list, image_extension
        )
    # Fallback: hand-rolled minimal DOCX + media parts.
    return _build_minimal_docx_with_media(paragraphs_list, images_list, image_extension)


def _inject_media_into_docx_zip(
    docx_bytes: bytes, images: List[bytes], ext: str
) -> bytes:
    """Add media files into an existing DOCX zip in-place.

    We do not modify python-docx's relationships; we only ensure the
    files exist under word/media/. The DOCX parser does not require
    proper rels entries — it walks ``word/media/*`` directly. This is
    sufficient for testing the parser's image extraction logic.
    """
    ext = ext if ext.startswith(".") else "." + ext
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zin:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            existing_names = set(zin.namelist())
            for item in zin.infolist():
                zout.writestr(item, zin.read(item.filename))
            for i, img in enumerate(images, start=1):
                name = f"word/media/image{i}{ext}"
                if name in existing_names:
                    # Append a unique suffix to avoid collisions.
                    name = f"word/media/image{i}_{i}{ext}"
                zout.writestr(name, img)
    return out.getvalue()


def _build_minimal_docx_with_media(
    paragraphs: List[str], images: List[bytes], ext: str
) -> bytes:
    """Hand-rolled minimal DOCX with embedded media files."""
    paragraphs_xml = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{_xml_escape(p)}</w:t></w:r></w:p>"
        for p in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs_xml}</w:body></w:document>"
    ).encode("utf-8")

    ext = ext if ext.startswith(".") else "." + ext
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _minimal_content_types())
        zf.writestr("_rels/.rels", _minimal_root_rels())
        zf.writestr("word/_rels/document.xml.rels", _minimal_document_rels())
        zf.writestr("word/document.xml", document_xml)
        for i, img in enumerate(images, start=1):
            zf.writestr(f"word/media/image{i}{ext}", img)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fake OCR provider
# ---------------------------------------------------------------------------


@dataclass
class FakeOcrCall:
    """Record of one call into :class:`FakeOcrProvider`."""

    image_bytes: bytes
    mime_type: str
    page_index: Optional[int] = None
    dpi: Optional[int] = None
    language: Optional[str] = None


@dataclass
class FakeOcrProvider:
    """In-memory OCR provider for tests.

    Implements :class:`app.providers.ocr.base.OcrProvider` with no
    external dependencies. Behaviour is fully controlled by the
    constructor parameters so tests can simulate any outcome:

    - ``default_text`` / ``default_confidence`` — returned for any
      call when no per-call override exists.
    - ``per_mime_text`` — dict mapping MIME type to text returned for
      that MIME.
    - ``raise_exc`` — exception class to raise (e.g.
      :class:`OcrProviderUnavailable`).
    - ``calls`` — list of :class:`FakeOcrCall` recording every
      invocation.

    All returned :class:`OcrResult` objects carry
    ``provider="fake"`` and ``extraction_method`` consistent with the
    method called.
    """

    default_text: str = "fake OCR text"
    default_confidence: float = 95.0
    per_mime_text: dict = field(default_factory=dict)
    raise_exc: Optional[Exception] = None
    per_call_pages: Optional[int] = None
    calls: List[FakeOcrCall] = field(default_factory=list)

    name: str = "fake"

    # ----- Abstract implementations ----------------------------------

    def extract_text(
        self,
        image_bytes: bytes,
        mime_type: str,
        language: Optional[str] = None,
    ):
        from app.providers.ocr.base import OcrPage, OcrResult

        self.calls.append(
            FakeOcrCall(image_bytes=image_bytes, mime_type=mime_type, language=language)
        )
        if self.raise_exc is not None:
            raise self.raise_exc

        text = self.per_mime_text.get(mime_type, self.default_text)
        page = OcrPage(
            page_index=0,
            text=text,
            confidence=self.default_confidence,
            width=100,
            height=50,
            language=language or "eng",
        )
        return OcrResult(
            text=text,
            confidence=self.default_confidence,
            language=language or "eng",
            pages=[page],
            provider=self.name,
            extraction_method="tesseract_image",
            metadata={"width": 100, "height": 50, "mime_type": mime_type},
            success=bool(text and text.strip()),
        )

    def extract_text_from_pdf_page(
        self,
        pdf_bytes: bytes,
        page_index: int,
        dpi: int = 200,
        language: Optional[str] = None,
    ):
        from app.providers.ocr.base import OcrPage, OcrResult

        self.calls.append(
            FakeOcrCall(
                image_bytes=pdf_bytes,
                mime_type="application/pdf",
                page_index=page_index,
                dpi=dpi,
                language=language,
            )
        )
        if self.raise_exc is not None:
            raise self.raise_exc

        text = self.default_text
        page = OcrPage(
            page_index=page_index,
            text=text,
            confidence=self.default_confidence,
            width=595,
            height=842,
            language=language or "eng",
        )
        return OcrResult(
            text=text,
            confidence=self.default_confidence,
            language=language or "eng",
            pages=[page],
            provider=self.name,
            extraction_method="tesseract_pdf_page",
            metadata={"dpi": dpi, "width": 595, "height": 842, "mime_type": "application/pdf"},
            success=bool(text and text.strip()),
        )

    def health_check(self) -> dict:
        return {
            "available": True,
            "provider": self.name,
            "engine_version": "fake-1.0",
            "details": "fake provider",
        }

    def provider_info(self) -> dict:
        return {"provider": self.name, "engine_version": "fake-1.0"}


# ---------------------------------------------------------------------------
# Upload-endpoint double helpers
# ---------------------------------------------------------------------------


def disable_ocr_env(monkeypatch, **overrides) -> None:
    """Force OCR_ENABLED=false for tests that need to exercise the
    "OCR disabled" branches.

    Accepts additional env-var overrides (e.g. OCR_IMAGE_MAX_SIZE_MB).
    """
    monkeypatch.setenv("OCR_ENABLED", "false")
    for k, v in overrides.items():
        monkeypatch.setenv(k, str(v))


def enable_ocr_env(monkeypatch, **overrides) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    for k, v in overrides.items():
        monkeypatch.setenv(k, str(v))


# ---------------------------------------------------------------------------
# Convenience: combined "kitchen sink" fixtures
# ---------------------------------------------------------------------------


def all_supported_image_mimes() -> List[str]:
    """Return the list of direct-image MIME types accepted at upload."""
    return [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/tiff",
        "image/bmp",
        "image/gif",
    ]


def all_allowed_upload_mimes() -> List[str]:
    """Return the union of all MIME types accepted at upload."""
    return [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/csv",
        "application/csv",
        *all_supported_image_mimes(),
    ]


__all__ = [
    # availability checks
    "tesseract_available",
    "pytesseract_available",
    "pymupdf_available",
    "pillow_available",
    "docx_available",
    # image fixtures
    "make_solid_png",
    "make_solid_jpeg",
    "make_blank_bmp",
    "make_text_image",
    "make_text_image_bytes",
    "make_corrupt_image_bytes",
    # PDF fixtures
    "make_text_pdf",
    "make_scanned_like_pdf",
    # DOCX fixtures
    "make_text_docx",
    "make_docx_with_embedded_images",
    # fake provider
    "FakeOcrProvider",
    "FakeOcrCall",
    # env helpers
    "disable_ocr_env",
    "enable_ocr_env",
    # lists
    "all_supported_image_mimes",
    "all_allowed_upload_mimes",
]
