"""
Tests for Phase 34A — Enterprise OCR & Image Ingestion.

Coverage:
    * OCR provider behavior (factory, health, typed exceptions)
    * Preprocessing (validation, mode normalization, upscale decision,
      corrupt-input rejection)
    * Cleaning (deterministic text normalization, hyphen wrap, blanks,
      control-char strip)
    * Direct image parsing (success, low confidence, empty, disabled,
      oversized, unsupported MIME, corrupt bytes)
    * Scanned-PDF OCR fallback (native text only, OCR fallback when
      below threshold, disabled OCR, missing binary)
    * DOCX embedded-image OCR (paragraphs + images, OCR disabled,
      missing binary, media enumeration failure)
    * Upload endpoint handling (image MIME allow-list, image validation,
      direct image w/o OCR => 400, secret redaction via upload path)
    * Sanitization/security (filename path-traversal, MIME spoofing,
      non-image MIME rejected)
    * Failure behavior (provider missing, oversized image, corrupt image)

The tests are split into focused classes for readability. Most tests
use the in-memory :class:`tests.ocr_fixtures.FakeOcrProvider` so they
run deterministically without the Tesseract binary. A few tests are
skipped when Tesseract is not installed (clearly marked).
"""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.ingestion.ocr_cleaning import (
    CleanStats,
    clean_ocr_text,
    clean_ocr_text_with_stats,
)
from app.ingestion.ocr_preprocessing import (
    is_image_extension_supported,
    is_image_mime_supported,
    image_mime_from_extension,
    preprocess_for_ocr,
    validate_image_bytes,
)
from app.ingestion.parsers.base import ExtractionResult
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.image_parser import ImageParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.pipeline import get_parser, process_document_to_result
from app.providers.ocr import (
    AVAILABLE_OCR_PROVIDERS,
    get_ocr_provider,
    get_ocr_status,
)
from app.providers.ocr.base import (
    OcrInvalidInput,
    OcrPage,
    OcrProviderUnavailable,
    OcrResult,
    OcrTimeout,
    OcrUnsupportedFormat,
)
from app.providers.ocr.tesseract_provider import (
    TesseractOcrProvider,
    _is_tesseract_binary_available,
)

from tests.ocr_fixtures import (
    FakeOcrProvider,
    disable_ocr_env,
    enable_ocr_env,
    make_corrupt_image_bytes,
    make_docx_with_embedded_images,
    make_scanned_like_pdf,
    make_solid_jpeg,
    make_solid_png,
    make_text_docx,
    make_text_image,
    make_text_image_bytes,
    make_text_pdf,
    pymupdf_available,
    tesseract_available,
)


# ===========================================================================
# 1. OCR Provider factory + status
# ===========================================================================


class TestOcrProviderFactory:
    def test_get_ocr_provider_returns_provider_by_default(self, monkeypatch):
        enable_ocr_env(monkeypatch, OCR_PROVIDER="tesseract")
        provider = get_ocr_provider()
        # We don't require the binary to be present for this assertion —
        # the factory must not raise on init.
        assert provider is not None
        assert provider.name == "tesseract"

    def test_get_ocr_provider_returns_none_when_disabled(self, monkeypatch):
        disable_ocr_env(monkeypatch)
        assert get_ocr_provider() is None

    def test_get_ocr_provider_returns_none_for_explicit_none(self, monkeypatch):
        enable_ocr_env(monkeypatch, OCR_PROVIDER="none")
        assert get_ocr_provider() is None

    def test_get_ocr_provider_rejects_unknown_provider(self, monkeypatch):
        enable_ocr_env(monkeypatch, OCR_PROVIDER="not_a_real_provider")
        with pytest.raises(ValueError):
            get_ocr_provider()

    def test_available_ocr_providers_lists_tesseract(self):
        assert "tesseract" in AVAILABLE_OCR_PROVIDERS

    def test_get_ocr_status_safe_when_disabled(self, monkeypatch):
        disable_ocr_env(monkeypatch)
        status = get_ocr_status()
        assert status["ocr_enabled"] is False
        assert "ocr_health" in status
        # No secrets in the status payload.
        for value in status.values():
            assert "secret" not in str(value).lower()

    def test_get_ocr_status_returns_settings(self, monkeypatch):
        enable_ocr_env(
            monkeypatch,
            OCR_LANGUAGE="eng",
            OCR_MIN_CONFIDENCE="42",
            OCR_RENDER_DPI="250",
        )
        status = get_ocr_status()
        assert status["ocr_enabled"] is True
        settings = status["ocr_settings"]
        assert settings["language"] == "eng"
        assert settings["min_confidence"] == 42
        assert settings["render_dpi"] == 250


# ===========================================================================
# 2. OCR provider behavior (typed exceptions, health, supports_mime)
# ===========================================================================


class TestOcrProviderBehavior:
    def test_supports_mime_accepts_common_image_types(self):
        provider = TesseractOcrProvider()
        for m in [
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp",
            "image/tiff",
            "image/bmp",
            "image/gif",
            "application/pdf",
        ]:
            assert provider.supports_mime(m), m

    def test_supports_mime_rejects_unknown(self):
        provider = TesseractOcrProvider()
        assert provider.supports_mime("application/zip") is False
        assert provider.supports_mime("") is False
        assert provider.supports_mime(None) is False  # type: ignore[arg-type]

    def test_extract_text_raises_unsupported_format_for_pdf(self):
        provider = TesseractOcrProvider()
        with pytest.raises(OcrUnsupportedFormat):
            provider.extract_text(b"%PDF-1.4", "application/pdf")

    def test_extract_text_raises_unavailable_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(
            "app.providers.ocr.tesseract_provider._is_tesseract_binary_available",
            lambda: False,
        )
        provider = TesseractOcrProvider()
        with pytest.raises(OcrProviderUnavailable):
            provider.extract_text(b"\x89PNG\r\n\x1a\n", "image/png")

    def test_extract_text_raises_invalid_for_corrupt_bytes(self, monkeypatch):
        # Pretend the binary IS available so we can exercise validation.
        monkeypatch.setattr(
            "app.providers.ocr.tesseract_provider._is_tesseract_binary_available",
            lambda: True,
        )
        provider = TesseractOcrProvider()
        with pytest.raises((OcrInvalidInput, OcrProviderUnavailable)):
            provider.extract_text(make_corrupt_image_bytes("image/png"), "image/png")

    def test_extract_text_from_pdf_page_raises_invalid_for_corrupt_pdf(self, monkeypatch):
        monkeypatch.setattr(
            "app.providers.ocr.tesseract_provider._is_tesseract_binary_available",
            lambda: True,
        )
        provider = TesseractOcrProvider()
        with pytest.raises((OcrInvalidInput, OcrProviderUnavailable)):
            provider.extract_text_from_pdf_page(b"not a pdf", 0)

    def test_health_check_does_not_include_secrets(self):
        provider = TesseractOcrProvider()
        health = provider.health_check()
        for k, v in health.items():
            assert "secret" not in str(k).lower()
            assert "secret" not in str(v).lower()
        assert "provider" in health
        assert "available" in health
        assert isinstance(health["available"], bool)

    def test_provider_info_contains_expected_keys(self):
        provider = TesseractOcrProvider()
        info = provider.provider_info()
        assert info["provider"] == "tesseract"
        assert "language" in info
        assert "timeout_s" in info
        assert "render_dpi" in info


# ===========================================================================
# 3. OCR Cleaning
# ===========================================================================


class TestOcrCleaning:
    def test_clean_empty_returns_empty(self):
        assert clean_ocr_text("") == ""
        assert clean_ocr_text(None or "") == ""  # type: ignore[arg-type]

    def test_clean_strips_control_characters(self):
        text = "Hello\x00World\x07Test"
        cleaned = clean_ocr_text(text)
        assert "\x00" not in cleaned
        assert "\x07" not in cleaned
        assert "HelloWorldTest" in cleaned.replace(" ", "")

    def test_clean_normalizes_line_endings(self):
        text = "line1\r\nline2\rline3"
        cleaned = clean_ocr_text(text)
        assert "\r" not in cleaned
        assert "line1" in cleaned and "line2" in cleaned and "line3" in cleaned

    def test_clean_repairs_hyphen_wrap(self):
        text = "exam-\nple text"
        cleaned = clean_ocr_text(text)
        assert "example" in cleaned

    def test_clean_collapses_multiple_blank_lines(self):
        text = "para1\n\n\n\n\npara2"
        cleaned = clean_ocr_text(text)
        assert "\n\n\n" not in cleaned
        assert "para1" in cleaned and "para2" in cleaned

    def test_clean_collapses_runs_of_spaces(self):
        text = "word1     word2"
        cleaned = clean_ocr_text(text)
        assert "word1 word2" == cleaned

    def test_clean_preserves_single_newlines(self):
        text = "line1\nline2\nline3"
        cleaned = clean_ocr_text(text)
        # Single newlines preserved, but trailing whitespace trimmed.
        assert "line1\nline2\nline3" == cleaned

    def test_clean_with_stats(self):
        text = "exam-\nple   \n\n\n  text\x00"
        cleaned, stats = clean_ocr_text_with_stats(text)
        assert isinstance(cleaned, str)
        assert isinstance(stats, CleanStats)
        assert stats.chars_in == len(text)
        assert stats.control_chars_removed >= 1
        assert stats.hyphens_joined >= 1

    def test_clean_unicode_normalization(self):
        # "ﬁ" (U+FB01 ligature) should normalize to "fi".
        text = "ﬁle"
        cleaned = clean_ocr_text(text)
        assert cleaned == "file"

    def test_clean_is_deterministic(self):
        text = "Some  weird   OCR \n\n\n output\x00text"
        a = clean_ocr_text(text)
        b = clean_ocr_text(text)
        assert a == b


# ===========================================================================
# 4. OCR Preprocessing
# ===========================================================================


@pytest.mark.skipif(not __import__("os").path.exists, reason="")  # always run
class TestOcrPreprocessing:
    def test_is_image_mime_supported(self):
        assert is_image_mime_supported("image/png")
        assert is_image_mime_supported("image/jpeg")
        assert is_image_mime_supported("image/jpg")
        assert is_image_mime_supported("image/webp")
        assert is_image_mime_supported("image/tiff")
        assert is_image_mime_supported("image/bmp")
        assert is_image_mime_supported("image/gif")
        assert not is_image_mime_supported("application/pdf")
        assert not is_image_mime_supported("")
        assert not is_image_mime_supported("image/svg+xml")

    def test_is_image_extension_supported(self):
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"]:
            assert is_image_extension_supported(ext), ext
        assert not is_image_extension_supported(".exe")
        assert not is_image_extension_supported("")

    def test_image_mime_from_extension(self):
        assert image_mime_from_extension(".png") == "image/png"
        assert image_mime_from_extension("png") == "image/png"
        assert image_mime_from_extension(".jpg") == "image/jpeg"
        assert image_mime_from_extension(".jpeg") == "image/jpeg"
        assert image_mime_from_extension(".webp") == "image/webp"
        assert image_mime_from_extension(".unknown") is None
        assert image_mime_from_extension("") is None

    def test_validate_image_bytes_accepts_png(self):
        png = make_solid_png(200, 80)
        ok, err = validate_image_bytes(png, "image/png", max_bytes=1024 * 1024)
        assert ok, err

    def test_validate_image_bytes_rejects_empty(self):
        ok, err = validate_image_bytes(b"", "image/png", max_bytes=1024 * 1024)
        assert not ok
        assert "empty" in err.lower()

    def test_validate_image_bytes_rejects_too_large(self):
        ok, err = validate_image_bytes(b"x" * 100, "image/png", max_bytes=50)
        assert not ok
        assert "large" in err.lower()

    def test_validate_image_bytes_rejects_corrupt(self):
        ok, err = validate_image_bytes(
            make_corrupt_image_bytes("image/png"), "image/png", max_bytes=1024 * 1024
        )
        assert not ok

    def test_validate_image_bytes_rejects_too_small(self):
        # Tiny image (less than min_dim=16 default).
        png = make_solid_png(10, 10)
        ok, err = validate_image_bytes(png, "image/png", max_bytes=1024 * 1024)
        assert not ok
        assert "small" in err.lower() or "decode" in err.lower()

    def test_validate_image_bytes_rejects_mime_mismatch(self):
        # PNG bytes reported as JPEG.
        png = make_solid_png(50, 50)
        ok, err = validate_image_bytes(png, "image/jpeg", max_bytes=1024 * 1024)
        assert not ok
        assert "mime" in err.lower()

    def test_preprocess_returns_png(self):
        png = make_solid_png(300, 200)
        result = preprocess_for_ocr(png)
        assert result.image_bytes.startswith(b"\x89PNG")
        assert result.width > 0 and result.height > 0
        assert isinstance(result.applied_steps, list)

    def test_preprocess_upscales_small_images(self):
        # 200x100 = 20000 px, far below 1.6M target — should upscale.
        png = make_solid_png(200, 100)
        result = preprocess_for_ocr(png)
        assert any("upscale" in s for s in result.applied_steps)
        assert result.width >= 200 and result.height >= 100

    def test_preprocess_does_not_downscale(self):
        # Build a large image, ensure no downscaling happens.
        png = make_solid_png(2000, 1500)
        result = preprocess_for_ocr(png)
        assert result.width == 2000 and result.height == 1500
        assert not any("upscale" in s for s in result.applied_steps)

    def test_preprocess_disabled_returns_unchanged(self):
        png = make_solid_png(100, 80)
        result = preprocess_for_ocr(
            png,
            apply_exif_orientation=False,
            apply_grayscale=False,
            apply_contrast=False,
            apply_sharpening=False,
            apply_upscaling=False,
        )
        # No steps applied.
        assert result.applied_steps == []

    def test_preprocess_rejects_corrupt_bytes(self):
        with pytest.raises(ValueError):
            preprocess_for_ocr(make_corrupt_image_bytes("image/png"))


# ===========================================================================
# 5. Direct image parsing
# ===========================================================================


class TestDirectImageParsing:
    def _make_provider(self, **kwargs) -> FakeOcrProvider:
        return FakeOcrProvider(**kwargs)

    def test_image_parser_returns_extraction_result_with_ocr_text(self):
        parser = ImageParser()
        png = make_text_image("Hello OCR")
        result = parser.parse(
            png,
            original_filename="hello.png",
            ocr_provider=self._make_provider(default_text="Hello OCR"),
        )
        assert isinstance(result, ExtractionResult)
        assert result.text
        assert "Hello OCR" in result.text
        assert result.images_detected == 1
        assert result.images_ocr_processed == 1
        assert result.ocr_provider == "fake"
        assert result.ocr_disabled is False
        assert result.needs_human_review is False
        assert len(result.images) == 1
        img = result.images[0]
        assert img.source_type == "direct_image"
        assert img.ocr_status == "success"
        assert img.mime_type == "image/png"
        assert img.byte_size == len(png)

    def test_image_parser_low_confidence_flagged(self):
        parser = ImageParser()
        png = make_solid_png(100, 100)
        provider = self._make_provider(default_text="x", default_confidence=10.0)
        result = parser.parse(
            png, original_filename="x.png", ocr_provider=provider, min_confidence=80
        )
        assert result.ocr_disabled is False
        # Either needs_human_review is True or warnings include low confidence.
        assert result.needs_human_review is True
        assert any("confidence" in w.lower() for w in result.warnings)

    def test_image_parser_empty_text_marked_for_review(self):
        parser = ImageParser()
        png = make_solid_png(100, 100)
        provider = self._make_provider(default_text="")
        result = parser.parse(
            png, original_filename="blank.png", ocr_provider=provider
        )
        assert result.images_detected == 1
        assert result.ocr_text_chars == 0
        assert result.needs_human_review is True
        assert any("no text" in w.lower() for w in result.warnings)

    def test_image_parser_provider_unavailable_does_not_raise(self):
        parser = ImageParser()
        png = make_solid_png(100, 100)
        provider = self._make_provider(
            raise_exc=OcrProviderUnavailable("tesseract missing")
        )
        result = parser.parse(
            png, original_filename="missing.png", ocr_provider=provider
        )
        assert result.needs_human_review is True
        assert result.ocr_disabled is False
        assert any("unavailable" in w.lower() for w in result.warnings)
        assert result.images[0].ocr_status == "failed"

    def test_image_parser_generic_exception_caught(self):
        parser = ImageParser()
        png = make_solid_png(100, 100)
        provider = self._make_provider(raise_exc=RuntimeError("boom"))
        result = parser.parse(
            png, original_filename="boom.png", ocr_provider=provider
        )
        assert result.needs_human_review is True
        assert result.images[0].ocr_status == "failed"

    def test_image_parser_disabled_ocr(self, monkeypatch):
        parser = ImageParser()
        # Force the parser's fallback to also resolve to None.
        disable_ocr_env(monkeypatch)
        png = make_solid_png(100, 100)
        result = parser.parse(
            png, original_filename="x.png", ocr_provider=None
        )
        assert result.ocr_disabled is True
        assert result.needs_human_review is True
        assert result.images_detected == 1
        assert any("disabled" in w.lower() for w in result.warnings)

    def test_image_parser_uses_factory_when_no_provider(self, monkeypatch):
        # When no provider is passed AND OCR is disabled, the parser's
        # internal factory must return None (not raise).
        disable_ocr_env(monkeypatch)
        parser = ImageParser()
        png = make_solid_png(80, 80)
        result = parser.parse(png, original_filename="x.png")
        assert result.ocr_disabled is True

    def test_image_parser_rejects_corrupt_image(self):
        parser = ImageParser()
        with pytest.raises(ValueError):
            parser.parse(
                make_corrupt_image_bytes("image/png"),
                original_filename="bad.png",
                ocr_provider=self._make_provider(),
            )

    def test_image_parser_rejects_unsupported_mime(self):
        parser = ImageParser()
        png = make_solid_png(100, 100)
        # Force an unsupported MIME check by skipping mime_map; we send
        # bytes that don't decode to any supported format.
        with pytest.raises(ValueError):
            parser.parse(b"not an image", original_filename="bad.png", ocr_provider=self._make_provider())

    def test_image_parser_oversized_image_rejected(self):
        parser = ImageParser()
        # Build a real PNG; override image_max_size_mb to 0 to force failure.
        png = make_solid_png(100, 100)
        with pytest.raises(ValueError):
            parser.parse(
                png,
                original_filename="huge.png",
                ocr_provider=self._make_provider(),
                image_max_size_mb=0,
            )

    def test_image_parser_pipeline_dispatch(self):
        png = make_text_image("Pipeline Test")
        result = process_document_to_result(
            png, "image/png", original_filename="pipeline.png"
        )
        # Without OCR the pipeline should still produce an ExtractionResult
        # marked as needing review.
        assert isinstance(result, ExtractionResult)
        assert result.images_detected == 1

    def test_get_parser_dispatches_image(self):
        assert isinstance(get_parser("image/png"), ImageParser)
        assert isinstance(get_parser("image/jpeg"), ImageParser)
        assert isinstance(get_parser("application/pdf"), PDFParser)
        assert get_parser("application/zip") is None


# ===========================================================================
# 6. PDF parsing with OCR fallback
# ===========================================================================


class TestPdfOcrFallback:
    def _make_provider(self, **kwargs) -> FakeOcrProvider:
        return FakeOcrProvider(**kwargs)

    def test_pdf_with_native_text_does_not_call_ocr(self):
        # Build a PDF with substantial native text on each page so it
        # stays above the per-page threshold.
        pdf = make_text_pdf(
            [
                "This page has plenty of native text so pypdf can extract it "
                "and we can prove the parser does not fall back to OCR.",
                "Second page also has plenty of native text content to extract "
                "without any OCR fallback at all.",
            ]
        )
        parser = PDFParser()
        provider = self._make_provider()
        result = parser.parse(pdf, ocr_provider=provider)
        assert result.native_text_chars > 0
        # Provider was NOT called because native text is above threshold.
        assert provider.calls == []
        assert result.extraction_method == "pdf_pypdf"
        assert "pdf_pypdf_with_ocr_fallback" not in result.extraction_method

    def test_pdf_fallback_triggers_when_native_text_below_threshold(self):
        # Scanned-like PDF has no extractable text (or very little).
        pdf = make_scanned_like_pdf(num_pages=2)
        parser = PDFParser()
        provider = self._make_provider(default_text="OCR recovered text")
        result = parser.parse(
            pdf,
            ocr_provider=provider,
            pdf_page_text_min_chars=40,
            render_dpi=150,
        )
        assert result.pages_ocr_processed >= 1
        assert result.ocr_text_chars > 0
        assert "with_ocr_fallback" in result.extraction_method
        assert result.ocr_provider == "fake"
        assert len(provider.calls) == result.pages_ocr_processed

    def test_pdf_fallback_disabled_does_not_call_ocr(self):
        pdf = make_scanned_like_pdf(num_pages=1)
        parser = PDFParser()
        provider = self._make_provider()
        result = parser.parse(
            pdf, ocr_provider=provider, pdf_fallback=False
        )
        assert result.pages_ocr_processed == 0
        assert provider.calls == []
        assert any("disabled" in w.lower() for w in result.warnings) or not result.warnings

    def test_pdf_fallback_ocr_disabled_logs_warning(self, monkeypatch):
        # Disable OCR via env so the parser's internal factory returns
        # None and the disabled warning path is exercised.
        disable_ocr_env(monkeypatch)
        pdf = make_scanned_like_pdf(num_pages=1)
        parser = PDFParser()
        result = parser.parse(pdf, ocr_provider=None)
        assert result.ocr_disabled is True
        assert any("disabled" in w.lower() for w in result.warnings)

    def test_pdf_provider_unavailable_warns_and_continues(self):
        pdf = make_scanned_like_pdf(num_pages=2)
        parser = PDFParser()
        provider = self._make_provider(raise_exc=OcrProviderUnavailable("missing"))
        result = parser.parse(pdf, ocr_provider=provider)
        assert result.pages_ocr_processed == 0
        assert any("unavailable" in w.lower() for w in result.warnings)

    def test_pdf_provider_generic_error_warns_and_continues(self):
        pdf = make_scanned_like_pdf(num_pages=2)
        parser = PDFParser()
        provider = self._make_provider(raise_exc=RuntimeError("boom"))
        result = parser.parse(pdf, ocr_provider=provider)
        assert result.pages_ocr_processed == 0
        assert any(w for w in result.warnings)

    def test_pdf_invalid_bytes_raises(self):
        parser = PDFParser()
        with pytest.raises(ValueError):
            parser.parse(b"definitely not a pdf", ocr_provider=self._make_provider())

    def test_pdf_records_extracted_image_per_ocr_page(self):
        pdf = make_scanned_like_pdf(num_pages=2)
        parser = PDFParser()
        provider = self._make_provider(default_text="OCR text")
        result = parser.parse(pdf, ocr_provider=provider)
        assert len(result.images) == result.pages_ocr_processed
        for img in result.images:
            assert img.source_type == "pdf_page_ocr"
            assert img.page_number is not None
            assert img.ocr_status == "success"

    def test_pdf_needs_human_review_when_ocr_disabled_and_scanned(self, monkeypatch):
        disable_ocr_env(monkeypatch)
        pdf = make_scanned_like_pdf(num_pages=3)
        parser = PDFParser()
        result = parser.parse(pdf, ocr_provider=None)
        assert result.needs_human_review is True


# ===========================================================================
# 7. DOCX parsing with embedded-image OCR
# ===========================================================================


class TestDocxEmbeddedImageOcr:
    def _make_provider(self, **kwargs) -> FakeOcrProvider:
        return FakeOcrProvider(**kwargs)

    def test_docx_native_text_extracted(self):
        docx = make_text_docx(["Hello native", "Second paragraph"])
        parser = DOCXParser()
        result = parser.parse(docx, ocr_provider=None)
        assert "Hello native" in result.text
        assert "Second paragraph" in result.text
        assert result.native_text_chars > 0
        assert result.extraction_method == "docx_python_docx"

    def test_docx_embedded_images_ocr_when_enabled(self):
        png = make_text_image("embedded")
        docx = make_docx_with_embedded_images(
            ["Native paragraph"], [png], image_extension=".png"
        )
        parser = DOCXParser()
        provider = self._make_provider(default_text="embedded OCR")
        result = parser.parse(docx, ocr_provider=provider)
        assert result.images_detected == 1
        assert result.images_ocr_processed == 1
        assert result.ocr_text_chars > 0
        assert "with_ocr" in result.extraction_method
        # Joined text contains both native paragraph and embedded image OCR.
        assert "Native paragraph" in result.text
        assert "embedded OCR" in result.text
        # Image metadata recorded.
        assert len(result.images) == 1
        assert result.images[0].source_type == "docx_image_ocr"
        assert result.images[0].ocr_status == "success"

    def test_docx_embedded_images_skipped_when_disabled(self, monkeypatch):
        png = make_text_image("embedded")
        docx = make_docx_with_embedded_images(
            ["Native"], [png], image_extension=".png"
        )
        parser = DOCXParser()
        # Disable OCR env so the parser's internal get_ocr_provider()
        # fallback returns None too.
        disable_ocr_env(monkeypatch)
        result = parser.parse(docx, ocr_provider=None)
        assert result.images_detected == 1
        assert result.images_ocr_processed == 0
        assert result.ocr_text_chars == 0
        assert result.ocr_disabled is True

    def test_docx_embedded_images_disabled_via_flag(self):
        png = make_text_image("embedded")
        docx = make_docx_with_embedded_images(
            ["Native"], [png], image_extension=".png"
        )
        parser = DOCXParser()
        provider = self._make_provider(default_text="x")
        result = parser.parse(
            docx, ocr_provider=provider, ocr_docx_images=False
        )
        assert result.images_detected == 1
        assert result.images_ocr_processed == 0

    def test_docx_image_max_count_caps_processing(self, monkeypatch):
        images = [make_solid_png(80, 80) for _ in range(5)]
        docx = make_docx_with_embedded_images(
            ["Native"], images, image_extension=".png"
        )
        parser = DOCXParser()
        provider = self._make_provider(default_text="x")
        # Ensure OCR env is enabled (some test environments may have it
        # disabled from earlier monkeypatch.setenv calls).
        enable_ocr_env(monkeypatch)
        result = parser.parse(
            docx, ocr_provider=provider, ocr_docx_image_max_count=2
        )
        # 5 detected but only 2 processed.
        assert result.images_detected == 5
        assert result.images_ocr_processed == 2
        # Provider called at most max_count times.
        assert len([c for c in provider.calls]) <= 2

    def test_docx_invalid_bytes_raises(self):
        parser = DOCXParser()
        with pytest.raises(ValueError):
            parser.parse(b"not a docx", ocr_provider=self._make_provider())

    def test_docx_provider_unavailable_for_embedded_images(self):
        png = make_text_image("x")
        docx = make_docx_with_embedded_images(
            ["Native"], [png], image_extension=".png"
        )
        parser = DOCXParser()
        provider = self._make_provider(raise_exc=OcrProviderUnavailable("missing"))
        result = parser.parse(docx, ocr_provider=provider)
        assert result.images_ocr_processed == 0
        assert any("unavailable" in w.lower() for w in result.warnings)

    def test_docx_embedded_corrupt_image_recorded_as_failed(self):
        bad = make_corrupt_image_bytes("image/png")
        docx = make_docx_with_embedded_images(
            ["Native"], [bad], image_extension=".png"
        )
        parser = DOCXParser()
        provider = self._make_provider()
        result = parser.parse(docx, ocr_provider=provider)
        assert result.images_detected == 1
        # Corrupt bytes are skipped — provider is NOT called.
        assert result.images_ocr_processed == 0
        # A row is still recorded for audit.
        assert any(img.ocr_status == "failed" for img in result.images)

    def test_docx_with_no_images_works(self):
        docx = make_text_docx(["Just text"])
        parser = DOCXParser()
        result = parser.parse(docx, ocr_provider=self._make_provider())
        assert result.images_detected == 0
        assert result.images_ocr_processed == 0
        assert "Just text" in result.text


# ===========================================================================
# 8. Upload endpoint handling
# ===========================================================================


class TestUploadEndpointImageHandling:
    """End-to-end tests for image upload via the FastAPI endpoint.

    These tests stub MinIO, Qdrant, and the embedding provider so the
    endpoint runs in-process without the Docker stack. They verify
    Phase 34A-specific behaviors: MIME allow-list, OCR-disabled
    rejection, validation, and the new payload fields.
    """

    def _stub_outbound(self, monkeypatch):
        """Patch MinIO + Qdrant + embedding provider so the endpoint can run.

        Also patches ``get_ocr_provider`` at every module where it's
        imported (``app.api.documents`` plus the three parser modules)
        so the in-process tests don't depend on Tesseract being
        installed.
        """
        import app.api.documents as docs_mod
        from app.ingestion.parsers import (
            docx_parser as docx_parser_mod,
            image_parser as image_parser_mod,
            pdf_parser as pdf_parser_mod,
        )

        class FakeMinio:
            def put_object(self, *args, **kwargs):
                pass

            def remove_object(self, *args, **kwargs):
                pass

        fake_provider = FakeOcrProvider(default_text="OCR text")

        monkeypatch.setattr(docs_mod, "ensure_bucket_exists", lambda: None)
        monkeypatch.setattr(docs_mod, "get_minio_client", lambda: FakeMinio())
        monkeypatch.setattr(docs_mod, "ensure_collection", lambda dim: None)
        monkeypatch.setattr(docs_mod, "upsert_chunks", lambda chunks, meta: None)

        # Replace the embedding provider with a deterministic stub.
        from app.services.embeddings import EmbeddingProvider

        class FakeEmbed(EmbeddingProvider):
            dimension = 4
            def embed(self, texts):
                return [[0.1] * 4 for _ in texts]

        monkeypatch.setattr(docs_mod, "get_embedding_provider", lambda: FakeEmbed())

        # Patch OCR provider factory at every consumer module.
        monkeypatch.setattr(docs_mod, "get_ocr_provider", lambda: fake_provider)
        monkeypatch.setattr(image_parser_mod, "get_ocr_provider", lambda: fake_provider)
        monkeypatch.setattr(pdf_parser_mod, "get_ocr_provider", lambda: fake_provider)
        monkeypatch.setattr(docx_parser_mod, "get_ocr_provider", lambda: fake_provider)

    def test_image_upload_returns_200_with_phase34a_fields(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        # Admin auth_client is used to bypass upload visibility rules.
        png = make_text_image("Upload test")
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("hello.png", io.BytesIO(png), "image/png")},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["mime_type"] == "image/png"
        assert data["status"] == "indexed"
        assert data["images_persisted"] == 1
        assert data["extraction_method"]
        assert "ocr_disabled" in data
        assert data["ocr_disabled"] is False
        assert "needs_human_review" in data

    def test_image_upload_succeeds_for_each_supported_mime(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        for mime in ["image/png", "image/jpeg", "image/webp", "image/bmp", "image/gif"]:
            img = make_text_image_bytes(mime, "test")
            # PIL saving "image/gif" returns a valid GIF; the rest must too.
            res = auth_client.post(
                "/api/documents/upload",
                files={"file": (f"f.{mime.split('/')[-1]}", io.BytesIO(img), mime)},
            )
            assert res.status_code in (200, 201), (mime, res.text)

    def test_image_upload_rejects_ocr_disabled(
        self, auth_client, db_session, monkeypatch
    ):
        import app.api.documents as docs_mod
        monkeypatch.setattr(docs_mod, "get_ocr_provider", lambda: None)
        monkeypatch.setattr(docs_mod.settings, "OCR_ENABLED", False)
        monkeypatch.setattr(docs_mod, "ensure_bucket_exists", lambda: None)
        monkeypatch.setattr(docs_mod, "get_minio_client", lambda: MagicMock())
        png = make_text_image("no ocr")
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("x.png", io.BytesIO(png), "image/png")},
        )
        assert res.status_code == 400
        assert "OCR" in res.json()["detail"]

    def test_image_upload_rejects_unsupported_mime(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("x.exe", io.BytesIO(b"fake"), "application/x-msdownload")},
        )
        assert res.status_code == 415

    def test_image_upload_rejects_corrupt_bytes(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        bad = make_corrupt_image_bytes("image/png")
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("bad.png", io.BytesIO(bad), "image/png")},
        )
        # Pillow rejects the bytes → 415 (validation failure).
        assert res.status_code == 415

    def test_image_upload_rejects_mime_spoofing_pdf_as_png(
        self, auth_client, db_session, monkeypatch
    ):
        """A real PDF declared as image/png must be rejected by Pillow."""
        self._stub_outbound(monkeypatch)
        pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n"
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("spoof.png", io.BytesIO(pdf_bytes), "image/png")},
        )
        assert res.status_code == 415

    def test_image_upload_oversized_rejected(
        self, auth_client, db_session, monkeypatch
    ):
        import app.api.documents as docs_mod
        monkeypatch.setattr(docs_mod.settings, "OCR_IMAGE_MAX_SIZE_MB", 0)
        self._stub_outbound(monkeypatch)
        png = make_text_image("x")
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("huge.png", io.BytesIO(png), "image/png")},
        )
        assert res.status_code == 415

    def test_image_jpg_alias_normalised_to_jpeg(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        jpg = make_solid_jpeg(100, 100)
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("photo.jpg", io.BytesIO(jpg), "image/jpg")},
        )
        # Endpoint normalises "image/jpg" → "image/jpeg".
        assert res.status_code == 200, res.text
        assert res.json()["mime_type"] == "image/jpeg"

    def test_scanned_pdf_upload_runs_ocr_fallback(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        pdf = make_scanned_like_pdf(num_pages=1)
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("scan.pdf", io.BytesIO(pdf), "application/pdf")},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["mime_type"] == "application/pdf"
        assert data["extraction_method"]
        # With OCR fallback the extraction_method reflects that.
        if data["images_persisted"] > 0:
            assert "ocr" in data["extraction_method"].lower()

    def test_docx_with_image_upload_persists_image_rows(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        png = make_text_image("docx-image")
        docx = make_docx_with_embedded_images(
            ["Native text"], [png], image_extension=".png"
        )
        res = auth_client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "report.docx",
                    io.BytesIO(docx),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["images_persisted"] >= 1


# ===========================================================================
# 9. Sanitization & security
# ===========================================================================


class TestUploadSecurity:
    def _stub_outbound(self, monkeypatch):
        import app.api.documents as docs_mod
        from app.ingestion.parsers import (
            docx_parser as docx_parser_mod,
            image_parser as image_parser_mod,
            pdf_parser as pdf_parser_mod,
        )

        class FakeMinio:
            def put_object(self, *args, **kwargs):
                pass

        fake_provider = FakeOcrProvider(default_text="safe text")

        monkeypatch.setattr(docs_mod, "ensure_bucket_exists", lambda: None)
        monkeypatch.setattr(docs_mod, "get_minio_client", lambda: FakeMinio())
        monkeypatch.setattr(docs_mod, "ensure_collection", lambda dim: None)
        monkeypatch.setattr(docs_mod, "upsert_chunks", lambda chunks, meta: None)
        from app.services.embeddings import EmbeddingProvider

        class FakeEmbed(EmbeddingProvider):
            dimension = 4
            def embed(self, texts):
                return [[0.1] * 4 for _ in texts]

        monkeypatch.setattr(docs_mod, "get_embedding_provider", lambda: FakeEmbed())

        # Patch OCR provider factory at every consumer module.
        monkeypatch.setattr(docs_mod, "get_ocr_provider", lambda: fake_provider)
        monkeypatch.setattr(image_parser_mod, "get_ocr_provider", lambda: fake_provider)
        monkeypatch.setattr(pdf_parser_mod, "get_ocr_provider", lambda: fake_provider)
        monkeypatch.setattr(docx_parser_mod, "get_ocr_provider", lambda: fake_provider)

    def test_image_filename_path_traversal_blocked(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        png = make_text_image("safe")
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("../../etc/passwd.png", io.BytesIO(png), "image/png")},
        )
        # Either rejected for traversal (400) or sanitized to a safe name.
        assert res.status_code in (200, 400, 415)
        if res.status_code == 200:
            # Filename sanitised — must NOT contain '..'.
            assert ".." not in res.json()["original_name"]
            assert "/" not in res.json()["original_name"]

    def test_image_unsupported_extension_rejected(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        png = make_text_image("x")
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("image.svg", io.BytesIO(png), "image/svg+xml")},
        )
        # SVG is not in the allow-list → 415.
        assert res.status_code == 415

    def test_image_empty_filename_rejected(
        self, auth_client, db_session, monkeypatch
    ):
        self._stub_outbound(monkeypatch)
        png = make_text_image("x")
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("", io.BytesIO(png), "image/png")},
        )
        # Filename required → 400.
        assert res.status_code in (400, 415, 422)

    def test_image_upload_does_not_log_full_text(
        self, auth_client, db_session, monkeypatch, caplog
    ):
        """OCR-derived text must NOT appear in application logs verbatim."""
        import logging
        self._stub_outbound(monkeypatch)
        caplog.set_level(logging.WARNING)
        png = make_text_image("Secret-Document")
        res = auth_client.post(
            "/api/documents/upload",
            files={"file": ("doc.png", io.BytesIO(png), "image/png")},
        )
        assert res.status_code == 200, res.text
        for record in caplog.records:
            # Must not contain OCR content or large document text bodies.
            assert "Secret-Document" not in record.getMessage()


# ===========================================================================
# 10. Failure behavior
# ===========================================================================


class TestFailureBehavior:
    def test_image_parser_when_pillow_unavailable(self, monkeypatch):
        """If Pillow cannot be imported, parser raises a clear error."""
        # We don't import Pillow here — we patch the import to fail.
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PIL.Image" or name.startswith("PIL"):
                raise ImportError("Pillow not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        parser = ImageParser()
        with pytest.raises((ValueError, RuntimeError)):
            parser.parse(b"dummy", original_filename="x.png", ocr_provider=FakeOcrProvider())

    def test_ocr_provider_handles_empty_text(self):
        provider = FakeOcrProvider(default_text="", default_confidence=0.0)
        png = make_solid_png(50, 50)
        result = provider.extract_text(png, "image/png")
        assert result.success is False
        assert result.text == ""

    def test_ocr_status_returns_when_provider_factory_fails(self, monkeypatch):
        # Force factory to raise — get_ocr_status should still return a
        # safe payload (no secrets, no crash).
        enable_ocr_env(monkeypatch)

        # Patch the symbol in the providers.ocr package so callers
        # calling ``get_ocr_provider()`` raise.
        from app.providers.ocr import get_ocr_provider as factory_fn
        from app.providers.ocr import get_ocr_status

        original = factory_fn

        def boom():
            raise RuntimeError("simulated init failure")

        monkeypatch.setattr(
            "app.providers.ocr.get_ocr_provider", boom
        )
        status = get_ocr_status()
        # When init fails, status payload still exists and is safe.
        assert "ocr_health" in status
        assert status["ocr_health"]["available"] is False or status["ocr_health"].get("details")

    def test_extraction_result_summary_round_trip(self):
        from app.ingestion.parsers.base import ExtractedImage

        parser = ImageParser()
        png = make_text_image("summary test")
        provider = FakeOcrProvider(default_text="summary test", default_confidence=80)
        result = parser.parse(png, original_filename="summary.png", ocr_provider=provider)
        summary = result.as_summary_dict()
        assert summary["extraction_method"]
        assert summary["images_detected"] == 1
        assert summary["ocr_text_chars"] > 0
        # Should be JSON-serializable.
        import json
        json.dumps(summary)
