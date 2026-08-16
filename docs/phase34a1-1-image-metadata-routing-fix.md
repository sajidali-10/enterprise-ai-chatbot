# Phase 34A.1.1 — Image Metadata & Source Routing Fix

**Branch:** `phase-34a1-1-image-metadata-routing-fix`
**Tag:** `phase-34a1-1-image-metadata-routing-fix`
**Phase scope:** corrective patch only. Does NOT implement Phase 34A.2 or Vision AI.

---

## 1. Confirmed root cause

Live E2E exposed two coupled defects inside Phase 34A.1:

1. **Qdrant payload was silently dropping the structured OCR metadata
   the upload endpoint already passed in.** `qdrant_service.upsert_chunks`
   hard-coded its own payload builder and ignored every
   `content_type` / `image_id` / `ocr_provider` / `ocr_confidence`
   field the upload endpoint wrote. As a result, OCR-derived chunks
   indexed BEFORE Phase 34A.1.1 were stored with only the legacy
   `Source Type: Image` body prefix and the bare filename — nothing
   machine-readable.

2. **Hybrid retrieval stripped structured metadata out of the chunk
   dicts it forwarded downstream.** `hybrid_retriever` extracted only
   8 fields from each Qdrant payload. `source_type`, `content_type`,
   `image_id`, `ocr_provider`, `ocr_confidence`, `page_number`,
   `mime_type`, and `is_ocr` never reached `app.rag.image_routing`.

3. **`select_image_aware_chunks` branched on `has_identifiers()` instead
   of `query_type`.** This misclassified any image-referencing question
   that lacked an extracted numeric code as image_content, but
   misclassified troubleshooting-with-image-context questions
   ("How do I troubleshoot the error shown in the image?") the same
   way. The function never reached Case B.

4. **`query_analysis` treated weak hints like "what does" as
   troubleshooting intent, so "What does the screenshot I uploaded
   say?" was classified as `error_lookup` instead of `image_content`.**
   The image_routing layer then routed it as a knowledge lookup that
   pulled in unrelated KB sources.

The combined effect: an `image_content` query for "What error code is
shown in the image I uploaded?" retrieved the OCR chunk plus arbitrary
native-text KB sources (admin_test.txt, Programmer's Guide) because the
router had no structured signal to distinguish OCR chunks from native
text, and the query type classification pulled the question toward
the broad-KB path.

---

## 2. Files changed

| File | Change |
|------|--------|
| `apps/backend/app/services/vector/qdrant_service.py` | `upsert_chunks` now writes the full structured OCR/source metadata (`source_type`, `content_type`, `image_id`, `ocr_provider`, `ocr_confidence`, `page_number`, `mime_type`, `is_ocr`); strips explicit nulls; infers `source_type` from `image_id` when missing; new `get_point()` helper for diagnostics. |
| `apps/backend/app/rag/hybrid_retriever.py` | `qdrant_vector_search` extracts the new payload fields and forwards them on each chunk dict. |
| `apps/backend/app/rag/image_routing.py` | Adds `_looks_like_legacy_ocr_chunk` (conservative content-prefix + image-extension detector) and `_is_image_source_with_fallback`. `is_image_source_chunk` now combines structured and legacy signals. `select_image_aware_chunks` branches on `query_type == "image_content"` and scopes to explicit `image_context` / `document_id` first, falling back to `recent_images`, then to a single best-effort OCR chunk. Adds `LEGACY_OCR_FALLBACK_ENABLED` env-tunable toggle. |
| `apps/backend/app/rag/query_analysis.py` | Splits troubleshooting detection into `_detect_troubleshooting_intent` (broad, used for diagnostics) and `_detect_strong_troubleshooting_intent` (only unambiguous verbs). Classification uses the strong detector when an image is referenced so image-content questions stay `image_content`. |
| `apps/backend/app/api/documents.py` | `upload_document` now writes `source_type`, `content_type`, `mime_type`, `ocr_provider`, `ocr_confidence`, `page`, `image_id` into the Qdrant payload. `index_document` (manual reindex) reads the linked `DocumentImage` row and writes the same structured fields back into Qdrant for OCR-derived documents. |
| `apps/backend/tests/test_phase34a1_1_image_metadata_routing.py` | New: 29 regression tests covering structured payload for each source_type, legacy fallback, image_content scoping, troubleshooting passthrough, citations-from-surviving-chunks. |
| `apps/backend/tests/test_phase34a1_image_routing.py` | Updated two tests that asserted the pre-fix, pre-spec behaviour (return ALL OCR chunks) to assert the spec-compliant behaviour (scope to the explicit image_context). |

---

## 3. Qdrant payload — before / after

### Before (Phase 34A.1)

```json
{
  "chunk_id": 1107,
  "document_id": 55,
  "document_version_id": 51,
  "chunk_index": 0,
  "content": "Source Type: Image\nOCR Text:\nFailed Reason: 902 - Message delivery failed: rejected-forbidden-country ...",
  "source_file_name": "Screenshot 2026-08-05 174611.png",
  "title": "Screenshot 2026-08-05 174611.png",
  "section_heading": null
}
```

No `source_type`, no `image_id`, no `ocr_provider`, no `is_ocr`. The
router had to fall back to scanning the body for `Source Type: Image`.

### After (Phase 34A.1.1) — reindexed OCR document

```json
{
  "chunk_id": 1107,
  "document_id": 55,
  "document_version_id": 51,
  "chunk_index": 0,
  "content": "Source Type: Image\nOCR Text:\nFailed Reason: 902 - Message delivery failed: rejected-forbidden-country ...",
  "source_file_name": "Screenshot 2026-08-05 174611.png",
  "title": "Screenshot 2026-08-05 174611.png",
  "source_type": "image_ocr",
  "content_type": "image_ocr",
  "mime_type": "image/png",
  "image_id": 7,
  "ocr_provider": "tesseract",
  "ocr_confidence": 92,
  "is_ocr": true
}
```

### Other source_types after the fix

| Source | `source_type` | `is_ocr` | `image_id` |
|--------|---------------|----------|------------|
| Direct image upload (PNG/JPG/...) | `image_ocr` | `true` | linked `DocumentImage.id` |
| PDF page OCR | `pdf_ocr` | `true` | linked `DocumentImage.id` |
| DOCX embedded image OCR | `docx_image_ocr` | `true` | linked `DocumentImage.id` |
| Native text document | `native_text` | `false` | absent |

---

## 4. Duplicate investigation

The live E2E evidence shows:

* `Screenshot 2026-08-05 174611.png` exists as `document_id` 55 **and** 56.
* `admin_test.txt` has five indexed copies.

**Finding:** these are **repeated user uploads**, not an indexing bug.

* `app/api/documents.py::upload_document` always creates a fresh
  `Document` row. There is no content-hash-based deduplication or
  "you already uploaded this" check. Every upload is independent.
* `qdrant_service.upsert_chunks` uses `chunk_id` as the point ID, so
  two separate uploads produce two independent Qdrant points (each
  with its own `document_id`). This is the desired behaviour — the
  user may have legitimately updated or re-uploaded the file.
* The five `admin_test.txt` copies have **five distinct
  `document_id`s** with the same `original_name` and similar
  `content_hash`. There is no idempotency layer because Phase 34A
  did not introduce one and Phase 34A.1.1 is a corrective patch only.

**Action:** No automatic deletion. The duplicates are NOT a defect
introduced or exacerbated by Phase 34A.1.1. Operators may de-duplicate
manually via the admin documents endpoint if desired.

**Why this matters for the live scenario:** when the user asks an
image-content question with explicit `image_context`, the new
`image_content_scoped_explicit` branch in `select_image_aware_chunks`
restricts citations to chunks from THAT specific document_id. Stale
historical duplicates (e.g. `document_id` 56 from a previous upload)
are dropped, so the user sees only the answer for the image they
just uploaded.

---

## 5. Reindex instructions

**Goal:** backfill the structured `source_type` / `image_id` /
`ocr_provider` payload on existing OCR-derived documents without
deleting the Qdrant collection.

The `POST /api/documents/{document_id}/index` endpoint already
per-document-reindexes a single document. Phase 34A.1.1 extends
that endpoint to read the linked `DocumentImage` row and write the
structured metadata back into Qdrant for OCR-derived documents.

### Per-document reindex (recommended)

For each OCR/image document that needs backfilling, call:

```bash
# As admin
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/documents/$DOCUMENT_ID/index
```

This:

1. Re-chunks the existing `extracted_text` on the latest
   `DocumentVersion`.
2. Re-embeds with the current embedding provider.
3. Reads the linked `DocumentImage` row to recover
   `source_type` / `mime_type` / `ocr_provider` / `ocr_confidence` /
   `page_number`.
4. `upsert_chunks` writes the structured payload to Qdrant, replacing
   the old points for that document (the `chunk_id` IDs are stable,
   so Qdrant performs an in-place update — no new points).
5. Marks the document `indexed`.

### Targeted list (live system)

Documents that should be reindexed with this fix:

* `document_id` 55 — `Screenshot 2026-08-05 174611.png` (OCR screenshot)
* `document_id` 56 — `Screenshot 2026-08-05 174611.png` (duplicate upload)
* Any document whose `DocumentImage.source_type IN ('direct_image',
  'pdf_page_ocr', 'docx_image_ocr')`.

Native-text documents (e.g. `admin_test.txt`, `Programmer's Guide.txt`)
do **not** need reindexing — their `source_type` is already `native_text`
by default (or will be set when reindexed).

### Bulk option (NOT recommended by default)

`POST /api/documents/reindex-all` exists for collection-wide reindexing
(it clears the Qdrant collection and re-embeds every document). It is
NOT recommended for this fix because:

* It clears the entire collection — brief downtime.
* It re-embeds every document, not just the OCR ones.
* Per-document reindex is faster and surgical.

### Legacy fallback

While reindexing is in progress, the new
`_looks_like_legacy_ocr_chunk` detector in `image_routing.py`
classifies legacy OCR chunks by content prefix and filename pattern,
so existing indexed OCR documents remain routable immediately after
deploy. Disable with `RAG_LEGACY_OCR_FALLBACK_ENABLED=false` once all
OCR documents are reindexed.

---

## 6. Live E2E steps

After deploying Phase 34A.1.1 and reindexing OCR documents:

1. Open the chat UI and upload the OCR screenshot of "Error 902 -
   rejected-forbidden-country".
2. Send: `What error code is shown in the image I uploaded?`
3. Expected:
   * Answer: `902`
   * Sources: `Screenshot 2026-08-05 174611.png`
   * NO `admin_test.txt` source card.
   * NO unrelated `Programmer's Guide` source card.
4. Send: `How do I troubleshoot error 902?`
   * Expected:
     * Broader KB search.
     * Exact-match boost for `902` brings the OCR chunk forward.
     * Relevant support/programmer documentation may appear.
     * `admin_test.txt` remains filtered (its content does not match
       the technical term `rejected-forbidden-country`).
5. Confirm via debug payload:
   * `retrieval_mode == "image_aware"` for the image_content query.
   * `image_routing.routing_mode == "image_content_scoped_explicit"`.
   * `grounding.citation_count >= 1`.
   * `source_file_names == ["Screenshot 2026-08-05 174611.png"]`.

---

## 7. Tests

```
apps/backend/tests/test_phase34a1_1_image_metadata_routing.py   # new — 29 tests
apps/backend/tests/test_phase34a1_image_routing.py              # 2 assertions updated
apps/backend/tests/test_phase34a1_query_analysis.py             # unchanged (still passes)
apps/backend/tests/test_phase34a_ocr.py                         # unchanged (still passes)
apps/backend/tests/test_phase34a1_relevance_filter.py           # unchanged (still passes)
apps/backend/tests/test_phase34a1_startup.py                    # unchanged (still passes)
```

Coverage:

| Concern | Test |
|---------|------|
| Structured `source_type=image_ocr` payload | `TestQdrantPayloadSchema::test_image_ocr_payload_carries_structured_metadata` |
| Structured `pdf_ocr` | `test_pdf_ocr_payload_carries_structured_metadata` |
| Structured `docx_image_ocr` | `test_docx_image_ocr_payload_carries_structured_metadata` |
| Structured `native_text` | `test_native_text_payload_is_ocr_false` |
| Explicit nulls stripped | `test_payload_strips_explicit_nulls` |
| `source_type` inferred from `image_id` | `test_inferred_source_type_from_image_id` |
| `is_image_source_chunk` legacy fallback | `TestIsImageSourceChunk` (6 cases) |
| Image-content filters native_text | `TestImageContentRouting::test_filters_native_text_chunks_for_image_content_query` |
| Explicit `image_context` scopes to one chunk | `test_explicit_image_context_scopes_to_matching_chunks` |
| Explicit `document_id` scopes to one chunk | `test_explicit_document_context_scopes_to_matching_chunks` |
| `recent_images` fallback | `test_recent_images_fallback_when_no_explicit_match` |
| No OCR chunks → passthrough | `test_no_ocr_chunks_at_all_returns_no_scoping` |
| Legacy OCR chunk included via fallback | `test_legacy_image_chunk_is_included_via_fallback` |
| Troubleshooting query keeps all chunks | `test_troubleshooting_query_keeps_all_chunks` |
| Troubleshooting with image context keeps all | `test_troubleshooting_with_image_context_keeps_all_chunks` |
| `admin_test.txt` survives when relevant | `test_unrelated_admin_test_still_survives_when_relevant` |
| Citations only use final surviving chunks | `TestFinalSourceFilter` (3 cases) |
| Native-text RAG path unchanged | `test_existing_native_text_rag_path_unchanged` |
| Legacy fallback toggle | `TestLegacyFallback` |

Result: `188 passed` (Phase 34A OCR + 34A.1 retrieval tests + new 34A.1.1 regression tests).

---

## 8. Out of scope (deferred)

The following were explicitly listed as out of scope for Phase 34A.1.1
and are NOT implemented here:

* Vision AI (CLIP / image embeddings / vision-language model).
* Chat paperclip UI for inline image preview.
* Voice input/output.
* New vector database (Qdrant is unchanged).
* Broad architecture rewrite.

---

## 9. Commit & tag

```
git checkout -b phase-34a1-1-image-metadata-routing-fix
git add \
  apps/backend/app/services/vector/qdrant_service.py \
  apps/backend/app/rag/hybrid_retriever.py \
  apps/backend/app/rag/image_routing.py \
  apps/backend/app/rag/query_analysis.py \
  apps/backend/app/api/documents.py \
  apps/backend/tests/test_phase34a1_image_routing.py \
  apps/backend/tests/test_phase34a1_1_image_metadata_routing.py \
  docs/phase34a1-1-image-metadata-routing-fix.md

git commit -m "fix(rag): correct OCR image metadata and source routing"
git tag -a phase-34a1-1-image-metadata-routing-fix -m "Phase 34A.1.1 — image metadata + source routing fix"
```
