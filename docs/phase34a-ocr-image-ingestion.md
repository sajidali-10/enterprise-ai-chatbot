# Phase 34A — Enterprise OCR & Image Ingestion

> **Status:** Foundation phase. Adds OCR-backed ingestion for direct image
> uploads, scanned PDFs (page-level OCR fallback), and DOCX embedded images.
> Vision-based understanding (image captioning, object detection) is
> **deferred** to Phase 34B.

---

## 1. What this phase delivers

| Capability | Source | Status |
|---|---|---|
| Direct image upload (PNG / JPEG / WEBP / TIFF / BMP / GIF) | `image_parser.py` | ✅ |
| Scanned-PDF page-level OCR fallback | `pdf_parser.py` | ✅ |
| DOCX embedded-image OCR | `docx_parser.py` | ✅ |
| Tesseract OCR provider | `providers/ocr/tesseract_provider.py` | ✅ |
| Image preprocessing (EXIF, autocontrast, sharpen, upscale) | `ocr_preprocessing.py` | ✅ |
| Deterministic OCR-text cleaning | `ocr_cleaning.py` | ✅ |
| OCR settings via env vars | `app/core/config.py` | ✅ |
| Admin status endpoint surfaces OCR health | `api/admin_status.py` | ✅ |
| Per-image audit rows (`document_images`) | Alembic `012_phase34a_document_images` | ✅ |
| OCR-derived text redaction at the upload endpoint | `app/services/langsmith_tracing.redact_text` | ✅ |
| Vision descriptions / object detection / image embeddings | TBD | ⏳ Phase 34B |

## 2. Architecture

```
                          ┌──────────────────────┐
                          │  Upload Endpoint     │
                          │  /api/documents/upload│
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Parser Pipeline     │
                          │  (ingestion.pipeline)│
                          └──────────┬───────────┘
                                     │
                ┌─────────┬──────────┼──────────┬─────────┐
                ▼         ▼          ▼          ▼         ▼
           TextParser  PDFParser  DOCXParser  ImageParser
                                    │          │
                                    │          ▼
                                    │   preprocess_for_ocr (Pillow)
                                    │          │
                                    │          ▼
                                    │   get_ocr_provider() ──► TesseractOcrProvider
                                    │          │
                                    │          ▼
                                    │   clean_ocr_text (deterministic)
                                    │
                                    ▼
                          ExtractedImage rows → document_images
                          Joined text → DocumentVersion.extracted_text
                          Redacted text → Qdrant chunks (with OCR metadata)
```

### 2.1 Why a single `OcrProvider` interface?

The OCR subsystem mirrors the existing provider-factory pattern used for
LLMs, embeddings, and vector stores. A new backend (PaddleOCR, Textract,
Azure Document Intelligence, Google Document AI) plugs in by:

1. Implementing `app.providers.ocr.base.OcrProvider`.
2. Adding the name to `AVAILABLE_OCR_PROVIDERS` in
   `app/providers/ocr/__init__.py`.
3. Adding a branch to `get_ocr_provider()`.

No parser changes are needed.

### 2.2 Why deterministic OCR cleaning?

LLM-based cleaning is non-deterministic and slow. We use a small
character-level pipeline (`app/ingestion/ocr_cleaning.py`) that:

* normalizes Unicode (NFKC),
* strips control characters,
* repairs hyphen-wrapped lines,
* collapses runs of spaces and blank lines.

Secret redaction happens **after** cleaning, in the upload endpoint,
using the same `redact_text` helper that scrubs native document text.
This means OCR text and native text are treated identically — no
secret bypass via image content.

## 3. Settings

| Env var | Default | Meaning |
|---|---|---|
| `OCR_ENABLED` | `true` | Master switch. When `false`, image uploads return `400` and the upload endpoint refuses to index an image-only file. |
| `OCR_PROVIDER` | `tesseract` | Active provider. Only `tesseract` is implemented in Phase 34A. |
| `OCR_LANGUAGE` | `eng` | Tesseract language pack(s). Add packs at the OS level via `apt-get install tesseract-ocr-<lang>`. |
| `OCR_MIN_CONFIDENCE` | `60` | Below this, the OCR row is flagged `low_confidence` (still indexed, with a warning). |
| `OCR_PDF_FALLBACK` | `true` | Enables OCR for PDF pages whose native text is below `OCR_PDF_PAGE_TEXT_MIN_CHARS`. |
| `OCR_PDF_PAGE_TEXT_MIN_CHARS` | `40` | Per-page native-text threshold that triggers OCR. |
| `OCR_DOCX_IMAGES` | `true` | Master switch for embedded-image OCR in DOCX. |
| `OCR_DOCX_IMAGE_MAX_COUNT` | `50` | Safety cap on how many embedded images are OCR'd per document. |
| `OCR_IMAGE_MAX_SIZE_MB` | `15` | Maximum size of a single uploaded image. |
| `OCR_RENDER_DPI` | `200` | Render DPI when rasterizing a PDF page before OCR. |
| `OCR_TIMEOUT_S` | `120` | Per-call timeout for Tesseract. |
| `OCR_UPSCALING_ENABLED` | `true` | Whether `preprocess_for_ocr` upsamples sub-resolution images. |
| `TESSDATA_PREFIX` | unset | Optional override for the Tesseract data directory. |

All settings are read at request time from `app/core/config.py`. Changing
them requires no code change — only an env-var update and a backend
restart.

## 4. Data model

Phase 34A adds two pieces of database state via Alembic migration `012`:

### 4.1 `document_images` (new)

| Column | Type | Notes |
|---|---|---|
| `id` | int (PK) | |
| `document_id` | int (FK → documents.id) | |
| `document_version_id` | int (FK → document_versions.id, nullable) | |
| `storage_key` | varchar(512) | MinIO key (under `images/<doc_id>/...`). |
| `mime_type` | varchar(100) | |
| `source_type` | varchar(40) | `direct_image` \| `pdf_page_ocr` \| `docx_image_ocr` |
| `page_number` | int, nullable | 1-based page index for `pdf_page_ocr`. |
| `sequence_number` | int | Per-document ordering. |
| `original_filename` | varchar(512), nullable | |
| `ocr_provider` | varchar(40), nullable | `tesseract` \| null. |
| `ocr_status` | varchar(32) | `pending` \| `success` \| `failed` \| `disabled` \| `empty` \| `low_confidence` |
| `ocr_confidence` | int, nullable | 0–100. |
| `ocr_text_hash` | varchar(64), nullable | SHA-256 prefix of cleaned OCR text. |
| `ocr_error` | text, nullable | |
| `width` / `height` | int, nullable | Pixel dimensions after preprocessing. |
| `byte_size` | int, nullable | Original byte size. |

### 4.2 `document_versions.extraction_summary` (new, nullable)

JSON-text column capturing the parser's `ExtractionResult` summary
(extraction_method, native vs OCR char counts, completeness flag,
warnings). Existing rows keep `NULL`.

### 4.3 Phase 34B forward compatibility

The Phase 34A schema is intentionally additive. Phase 34B will add
nullable Vision columns to `document_images` (vision_provider,
vision_description, vision_tags, image_type, detected_entities,
objects, vision_confidence) **without** changing the existing
columns or rows.

## 5. Operational behavior

### 5.1 Direct image upload (PNG / JPEG / WEBP / TIFF / BMP / GIF)

1. The upload endpoint validates the MIME type against `ALLOWED_TYPES`
   (now includes the six image types).
2. It re-decodes the bytes with Pillow to catch MIME spoofing
   (e.g. a PDF renamed to `.png`).
3. The parser runs `preprocess_for_ocr`, then `get_ocr_provider`,
   then `clean_ocr_text`.
4. Original bytes are stored under `images/<doc_id>/<uuid>.<ext>` in
   MinIO **in addition to** the main upload. The original image must
   survive so Phase 34B Vision analysis can run on it later.
5. The OCR row is recorded in `document_images` with the OCR status.
6. The joined text (with the standard `Source Type: Image` /
   `OCR Text:` provenance header) is sanitized via `redact_text`
   and then indexed in Qdrant with `content_type=image_ocr`.

If `OCR_ENABLED=false`, the upload endpoint returns `400
(Image upload requires OCR, but OCR is disabled)`.

### 5.2 Scanned PDF fallback

* `pypdf` extracts native text per page.
* Pages whose native text length is below `OCR_PDF_PAGE_TEXT_MIN_CHARS`
  are rendered at `OCR_RENDER_DPI` DPI and OCR'd.
* The parser concatenates native + OCR text per page, preserving
  reading order.
* One `document_images` row is recorded per OCR'd page with
  `source_type=pdf_page_ocr`.

### 5.3 DOCX embedded-image OCR

* `python-docx` extracts native paragraph text.
* The DOCX zip is walked for files under `word/media/`.
* Each embedded image is validated, preprocessed, OCR'd, and cleaned.
* The joined output puts native paragraphs first, then per-image OCR
  blocks sorted by `sequence_number`.

### 5.4 What happens when OCR is unavailable?

The Tesseract binary, Pillow, or PyMuPDF can each be missing
independently. The OCR subsystem is **fail-open** for the file format
parsers (text, PDF native, DOCX native paragraphs keep working) but
**fail-closed** for direct image uploads (the upload endpoint returns
a clear 4xx error rather than silently indexing an image with no
text).

The admin status endpoint exposes OCR health at
`/api/admin/system/status` under the `ocr` key.

## 6. Security

* OCR-derived text is sanitized with the same `redact_text` helper as
  native text — secrets cannot bypass redaction by being rendered as
  an image.
* Image MIME spoofing (e.g. `.exe` renamed to `.png`) is caught by
  Pillow re-decoding the bytes.
* Original uploaded bytes are stored under a UUID key in MinIO.
  Path traversal in `original_name` is sanitized via
  `_sanitize_filename`.
* OCR providers never log OCR text content. Only structural metadata
  (provider name, dimensions, confidence, language) is allowed to
  leave the process via `health_check` / `provider_info`.
* The admin status endpoint never returns API keys, tokens, or full
  env values.

## 7. Testing

The Phase 34A test suite lives in
`apps/backend/tests/test_phase34a_ocr.py` and uses the deterministic
fixtures in `apps/backend/tests/ocr_fixtures.py`.

Run the OCR suite:

```bash
cd apps/backend
.venv/bin/python -m pytest tests/test_phase34a_ocr.py -v
```

The fixtures build synthetic PNG / JPEG / WEBP / TIFF / BMP / GIF /
PDF / DOCX files in-process, plus a `FakeOcrProvider` that records
calls and returns deterministic text — so the suite does not depend on
the Tesseract binary being installed. (Tests that exercise real OCR
paths are marked clearly.)

## 8. Rollout / rollback

Phase 34A is **additive only** — no existing tables, columns, API
endpoints, or contracts change. To roll back, run
`alembic downgrade -1` to drop `document_images` and the
`extraction_summary` column; the application code still works
because the column is nullable and the table is referenced only from
Phase 34A code paths.

To disable OCR without a code change, set `OCR_ENABLED=false`.
Direct image uploads will return a clear 4xx error and all other
document types continue to work.

## 9. Limitations and known gaps

* English language pack only — additional languages require
  `apt-get install tesseract-ocr-<lang>` in the Dockerfile.
* No image-captioning, object-detection, or visual-question-answering
  (Phase 34B).
* DOCX embedded images are OCR'd, but image positioning vs. surrounding
  text is approximated via `sequence_number` (the closest practical
  signal without re-implementing OOXML layout).
* Performance: Tesseract is CPU-bound and slow on large images. The
  pipeline does not currently parallelize across pages; a future
  optimization can fan out OCR calls per page or per image.
