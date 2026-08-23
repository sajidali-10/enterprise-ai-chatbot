# Phase 34C -- Persistent Multimodal Knowledge

> Durable, searchable representation of previously uploaded images.
> Reuses the Phase 34B Vision cache; does not call Vision or LLM
> providers during indexing.

---

## 1. Goals

Phase 34C turns persisted Phase 34A OCR + Phase 34B Vision analysis
into a **searchable image knowledge layer** inside the existing
Qdrant collection. Previously uploaded screenshots and dashboards
become retrievable as semantic evidence when the user later asks:

* "Which screenshot showed Server B failing?"
* "Find a dashboard showing a large traffic spike."
* "Do we have an image showing error 902?"

without explicitly attaching the image.

Phase 34C does NOT replace Phase 34B, OCR, or the chat flow. It is an
**additive overlay** with a single kill-switch:

```bash
MULTIMODAL_KNOWLEDGE_ENABLED=false
```

When the flag is off, every Phase 34C code path short-circuits to a
no-op and Phase 34A / 34B / 34A.1 / 34A.2 behaviour is unchanged.

---

## 2. Architecture

```
Image upload
   |
   v
Phase 34A OCR                  (unchanged)
   |
   v
Phase 34B Vision Intelligence  (unchanged)
   |
   v
Persisted columns on DocumentImage  (Phase 34B, migration 013)
   |
   v
+-----------------------------+    +-------------------------+
| Phase 34C indexer           |    | Backfill CLI            |
|                             |    |                         |
| load_image_record(db, id)   |    | python -m app.services  |
| build_knowledge_text(rec)   |    | .multimodal.scripts     |
| compute_point_id(d, i, v)   |    | .backfill               |
| get_embedding_provider()    |    |                         |
| qdrant_client.upsert(point) |    +-------------------------+
+-----------------------------+
   |
   v
Qdrant collection `documents`, new source_type = `image_knowledge`
   |
   v
+-----------------------------+
| Retrieval overlay           |
|                             |
| integrate_image_knowledge() |
|   * vector search image     |
|   * boost on image-intent   |
|   * explicit demote for     |
|     product-meaning queries |
|   * cap to MAX_IMAGE_SOURCES|
+-----------------------------+
   |
   v
Hybrid retriever result + RBAC filter + citations
   |
   v
Grounded answer + image citation
```

### Why a single Qdrant collection?

Phase 34A.1.1 already distinguishes chunks by their `source_type`
payload field (`image_ocr | pdf_ocr | docx_image_ocr | native_text`).
Adding a new value (`image_knowledge`) is a strictly additive change:

* same vector dimension (384),
* same cosine distance,
* same `document_id`-based RBAC filter (`filter_documents_by_permission`
  applies unchanged to image knowledge points),
* same reindex / deletion flow.

A dedicated collection would require operators to maintain two
collections, two cosine indexes, and a cross-collection join for
retrieval. Phase 34C avoids that operational complexity.

### Why reuse the existing text embedding provider?

The knowledge representation is **text** (OCR + Vision description +
entities + states + tags). Embedding it via the existing
`get_embedding_provider()` (mock | local sentence-transformers |
openai-compatible) keeps the indexing path identical to every other
chunk and avoids a new dependency or a new model to operate.

Native image embeddings (CLIP / BLIP-2) are explicitly deferred to a
later 34C.x or 34D phase. See §14.

---

## 3. ImageKnowledgeRecord

```python
@dataclass
class ImageKnowledgeRecord:
    document_id: int
    image_id: int
    owner_user_id: Optional[int]
    visibility: str                      # private | shared | global
    document_version_id: Optional[int]
    source_type: str = "image_knowledge"
    original_filename: Optional[str]
    mime_type: Optional[str]
    image_type: Optional[str]            # dashboard | application_ui | ...
    ocr_text: str
    ocr_confidence: Optional[int]
    ocr_status: Optional[str]
    vision_description: str
    visual_findings: List[str]
    detected_entities: List[str]
    visual_states: List[str]
    vision_tags: List[str]
    vision_provider: str
    vision_model: str
    vision_processed_at: Optional[str]
    vision_status: Optional[str]
    is_ocr_only: bool
    knowledge_schema_version: int = 1
    knowledge_text: str                  # the canonical text used for embedding
    embedding_text: str                  # the length-capped variant actually sent
```

`load_image_record(db, image_id, schema_version=...)` builds the
record from the persisted `DocumentImage` columns (Phase 34A OCR +
Phase 34B Vision) and from the OCR chunks already in Qdrant (Phase 34A
stores OCR text in chunk payloads, not on the row itself).

### Knowledge text format

The `knowledge_text` builder (`build_knowledge_text`) is a pure
deterministic function. Same record -> same text. Sections with no
content are omitted entirely (NOT left blank) so the text is compact
and stable across Vision enrichment transitions.

```
Source image: <filename>
Image type: <image_type>

OCR:
<ocr_text>

Visual description:
<vision_description>

Visual findings:
- <finding>
- <finding>

Entities: <comma-joined entities>
Visual states: <comma-joined states>
Tags: <comma-joined tags>
```

When `vision_status != "success"`, the Vision sections are omitted and
`is_ocr_only=True`. When neither OCR nor Vision contributed anything
the text becomes `(no extractable image content)` so the embedding
still has a non-zero vector.

Length cap: `MULTIMODAL_KNOWLEDGE_TEXT_MAX_CHARS` (default **1800**).
Build-time OCR clipping also keeps the OCR section <= 900 chars and
each finding <= 200 chars. The runtime cap is the binding ceiling.

---

## 4. Qdrant payload schema

Every image knowledge point carries:

| Field | Purpose |
| --- | --- |
| `source_type` | `"image_knowledge"` (canonical; distinct from `image_ocr`) |
| `content_type` | Alias of `source_type` for downstream code that reads `content_type` first |
| `document_id` | Citation + RBAC anchor |
| `image_id` | Back-reference to `DocumentImage` |
| `document_version_id` | Reindex lifecycle |
| `owner_user_id` | RBAC (visibility / ownership) |
| `visibility` | RBAC (`private | shared | global`) |
| `source_file_name` | Human-readable citation label |
| `title` | Same value as `source_file_name` (citation renderer expects this key) |
| `mime_type` | From the upload |
| `image_type` | `vision_image_type` when Vision succeeded; else omitted |
| `is_ocr` | Always `true` -- OCR is the indexing precondition |
| `has_vision` | `true` when Phase 34B Vision enriched this image |
| `vision_provider` | `"mock"`, `"openai-compatible"`, ... |
| `vision_model` | Provider-specific model identifier |
| `vision_processed_at` | ISO timestamp |
| `ocr_status` / `ocr_confidence` | Provenance |
| `knowledge_schema_version` | Bump to invalidate all points |
| `knowledge_text` | The canonical searchable text |
| `content` | Same value (the retrieval overlay searches on this) |
| `embedding_model` | For ops debugging; never a secret |
| `embedding_dimension` | Sanity check on provider dimension drift |

No secrets. No raw image bytes. No OCR text in LangSmith unless the
operator has explicitly enabled `LANGSMITH_LOG_RETRIEVED_CONTEXT`.

---

## 5. Idempotency

Point id is `compute_point_id(document_id, image_id, schema_version)`
-- a deterministic positive integer hash over a stable namespace
string. Re-running indexing for the same image hits the same point id
and overwrites the payload in place. **No duplicates.**

The function does NOT depend on timestamps or call counts. The same
`(document_id, image_id, schema_version)` always produces the same
point id.

`MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION` (default 1) is embedded in the
hash, so bumping it invalidates every previously indexed image
knowledge point. The next backfill / reindex writes fresh points with
the new id and leaves stale ones in place; a `reindex-all` removes them
when the collection is recreated.

---

## 6. Retrieval overlay

`integrate_image_knowledge(query, base_chunks, analysis, image_context=None)`

Single integration point in `app/rag/retriever.py`. Called after the
hybrid retriever + relevance thresholding, BEFORE RBAC. The
`retrieve_chunks_with_auth` permission filter applies to image
knowledge chunks identically to KB chunks.

Behaviour:

1. If `MULTIMODAL_KNOWLEDGE_ENABLED=false`, return `base_chunks`
   unchanged.
2. If the query is empty, return `base_chunks` unchanged.
3. Embed the (rewritten) query and vector-search Qdrant for points
   where `source_type = "image_knowledge"`. Limit: `RETRIEVAL_VECTOR_TOP_K`
   capped to 20.
4. Merge candidates with `base_chunks`. Candidates that duplicate an
   existing image in `base_chunks` (same `document_id` + `image_id`)
   are dropped to avoid double-citing.
5. **Image-intent boost** (`MULTIMODAL_IMAGE_INTENT_BOOST`, default
   0.25). Multiplicative on image-knowledge scores when the query
   carries an image-intent phrase:
   * historical patterns (added in Phase 34C): "which screenshot",
     "show me the dashboard", "find a dashboard showing ...",
     "do we have an image ...", "previously uploaded screenshot",
     "image where", etc.
   * in-scope image references from `query_analysis`
   * explicit `image_context`
6. **Authority demote** for product-meaning queries. When the query
   asks for authoritative product knowledge (e.g. "What does error
   902 mean?") and there is no image-intent phrase, image knowledge
   chunks are sorted to the bottom of the list via a stable secondary
   sort on `_knowledge_kind`. The first citation remains the KB chunk.
7. **Hard cap** (`MULTIMODAL_MAX_IMAGE_SOURCES`, default 2). Excess
   image knowledge chunks are dropped from the tail.

The overlay NEVER raises on transient errors. Failures are recorded in
the metadata dict under `multimodal.error` and the base chunks are
returned unchanged.

### Metadata returned to LangSmith + response debug

```python
{
    "enabled": bool,
    "knowledge_kind": "image_knowledge",
    "multimodal_candidates_retrieved": int,
    "image_knowledge_candidates": int,
    "image_knowledge_selected": int,
    "image_knowledge_boost_applied": float,
    "image_knowledge_authority_demote_applied": bool,
    "image_knowledge_source_ids": list[int],
    "image_knowledge_dropped_by_cap": int,
    "multimodal_index_version": int,
    "image_intent_matched": bool,
    "product_meaning_matched": bool,
}
```

Operators can opt into seeing this in the chat response by setting
`RETRIEVAL_SHOW_DEBUG=true`.

---

## 7. RBAC

Image knowledge points carry `document_id`, `owner_user_id`, and
`visibility` in their payload. The existing
`app.security.permissions.filter_documents_by_permission` helper
filters the merged retrieval result by `document_id`, which means the
RBAC gate applies to image knowledge automatically -- no Phase 34C
duplication of the permission logic.

Admins see every image knowledge point. Regular users see only points
on documents they can read. The Phase 34C test suite includes an
explicit RBAC isolation test (`test_phase34c_rbac.py`).

---

## 8. Lifecycle

| Event | Phase 34C action |
| --- | --- |
| `app.vision.persistence.persist_vision_result` writes a successful Vision row | `multimodal_lifecycle.refresh_image_knowledge(db, image_id)` rebuilds + upserts the point. Same point id -> in-place overwrite. |
| Document delete (`DELETE /api/documents/{id}`) | `multimodal_lifecycle.delete_image_knowledge_for_document(id)` runs alongside `qdrant_service.delete_vectors_by_document_id`. |
| Document reindex (`POST /api/documents/{id}/index`) | `multimodal_lifecycle.reindex_document_images(db, id)` rebuilds every image knowledge point for the document in the `finally` block. |
| Image delete (Phase 34A path) | `multimodal_lifecycle.delete_image_knowledge_for_image(image_id)` removes the matching point. |
| Operator runs the backfill CLI | `multimodal_lifecycle.reindex_document_images` over the requested scope. |

Every hook is fault-tolerant: a Qdrant failure during indexing is
logged and counted in the lifecycle summary, never raised into the
caller.

---

## 9. Backfill CLI

```bash
# Dry-run (no writes):
python -m app.services.multimodal.scripts.backfill --dry-run

# Backfill a single document:
python -m app.services.multimodal.scripts.backfill --document-id 42

# Backfill up to N images:
python -m app.services.multimodal.scripts.backfill --limit 200 --batch-size 25

# Permission-scoped backfill (only documents the user can read):
python -m app.services.multimodal.scripts.backfill --user-id 7

# Include images whose OCR is empty (default: skip):
python -m app.services.multimodal.scripts.backfill --include-ocr-empty
```

The CLI is:

* **Idempotent** -- re-running writes the same point id and overwrites
  the payload.
* **Restartable** -- exit codes are 0 (no errors) / 1 (errors). No
  state is lost between runs.
* **Observable** -- prints a single `BACKFILL_SUMMARY {...}` line per
  invocation. Never logs OCR text, Vision descriptions, or image
  bytes.
* **Permission-safe** -- when `--user-id` is supplied, only documents
  that user can read are indexed.
* **Never auto-run** -- operators invoke it explicitly. No startup
  side-effects.

---

## 10. Citations

`app/rag/citations.format_citations` recognises `source_type ==
"image_knowledge"` and enriches the citation dict with:

```python
{
    "citation_kind": "image_knowledge",
    "image_id": int,
    "image_type": str,
    "vision_provider": str,
    "vision_model": str,
    "has_vision": bool,
    "knowledge_schema_version": int,
    "document_id": int,
    "source_file_name": str,    # original image filename
    ...
}
```

The `[N]` citation marker format is unchanged. The distinction lives
in the structured fields of each citation dict. KB citations keep
their existing shape -- only image knowledge citations get the extra
metadata block.

The snippet for image citations is the (clipped) knowledge text, so
the user sees the human-readable observation. Internal Qdrant point
ids are never exposed.

---

## 11. Grounding

Phase 34C preserves the Phase 34B distinction:

* **IMAGE OBSERVATION** -- "The screenshot shows Server B as Failed."
  Supported by image knowledge citations.
* **PRODUCT KNOWLEDGE** -- "Server B failed because service X crashed."
  Requires KB citations.

The authority demote (default `MULTIMODAL_IMAGE_INTENT_BOOST=0.25` +
`MULTIMODAL_MAX_IMAGE_SOURCES=2` + explicit demote when
`product_meaning_matched`) ensures KB documentation outranks image
knowledge for product-meaning queries. Image knowledge may appear as
supporting evidence, but never outranks authoritative KB.

---

## 12. Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `MULTIMODAL_KNOWLEDGE_ENABLED` | `true` | Master kill-switch. When `false`, indexing + retrieval overlay are inert. |
| `MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION` | `1` | Embedded in the deterministic point id. Bump to invalidate. |
| `MULTIMODAL_IMAGE_INTENT_BOOST` | `0.25` | Multiplicative boost on image-knowledge scores when the query is image-intent. |
| `MULTIMODAL_MAX_IMAGE_SOURCES` | `2` | Hard cap on image knowledge sources per answer. |
| `MULTIMODAL_KNOWLEDGE_TEXT_MAX_CHARS` | `1800` | Length cap on knowledge text sent to the embedding provider. |

Defaults are safe: enabling Phase 34C requires no other config change.
Rollback is `MULTIMODAL_KNOWLEDGE_ENABLED=false`.

---

## 13. Tests

### Unit tests (hermetic)

| File | Scope |
| --- | --- |
| `test_phase34c_knowledge_record.py` | Deterministic text builder, point id stability, length cap. |
| `test_phase34c_indexer.py` | Idempotent upsert, payload schema, RBAC anchors, kill-switch. |
| `test_phase34c_lifecycle.py` | Document-delete cleanup, image-delete cleanup, Vision-enrichment in-place update, reindex idempotency, VISION_ENABLED does not block OCR-only indexing. |
| `test_phase34c_retrieval.py` | Historical image-intent detection, product-meaning detection, image-intent boost, authority demote, cap, duplicate prevention. |
| `test_phase34c_rbac.py` | Cross-user isolation, admin bypass, payload carries owner_user_id + visibility. |
| `test_phase34c_citations.py` | Image knowledge citations carry the structured metadata block; KB citations unchanged; mixed answer keeps source separation. |
| `test_phase34c_backfill.py` | CLI surface: argument parser, disabled short-circuit, dry-run summary, wiring contract. |

### Live E2E (against running Qdrant + backend)

`tests/test_phase34c_e2e_live.py` plus a companion `live_e2e_phase34c.py`
script driven against the Docker Compose stack. All six scenarios pass:

* **A** -- "Which screenshot showed Server B failing?" retrieves the
  indexed screenshot.
* **B** -- "Do we have a dashboard showing a large traffic spike?"
  retrieves the dashboard.
* **C** -- "Which screenshot showed error 902?" retrieves the
  screenshot containing 902.
* **D** -- "What does error 902 mean?" -- KB first, image knowledge
  demoted to the tail.
* **E** -- Private image owned by User A; User B's permission filter
  denies access.
* **F** -- OCR-only -> Vision-enriched updates the same point id; no
  duplicate; `has_vision` flips to `true`.

### Regression

Phase 34A / 34A.1 / 34B regression suites pass with no changes. The
six pre-existing failures in `test_phase34a2_chat_image_attachment.py`
are present on the base commit `30babf4` (verified by stashing the
Phase 34C changes and re-running) and are NOT caused by Phase 34C.

---

## 14. Rollback

```bash
# 1. Stop the backend.
docker compose stop backend

# 2. Set the kill-switch.
# Add to .env:
# MULTIMODAL_KNOWLEDGE_ENABLED=false

# 3. Restart.
docker compose up -d backend
```

When `MULTIMODAL_KNOWLEDGE_ENABLED=false`:

* `upsert_image_knowledge` returns `IndexResult(status="disabled")`.
* `delete_image_knowledge*` returns `DeleteResult(deleted=0)`.
* `integrate_image_knowledge` returns the base chunks unchanged.
* `reindex_*` returns immediately.
* The backfill CLI reports `disabled=True` and exits.

Existing image knowledge points remain in Qdrant but are never
queried. To remove them, run:

```bash
# Operator can manually purge via the Qdrant HTTP API or via:
python -c "
from app.services.multimodal.indexer import delete_image_knowledge_by_document_id
print(delete_image_knowledge_by_document_id(<doc_id>).deleted)
"
```

---

## 15. Future work (deferred to 34C.x / 34D)

* **Native image embeddings (CLIP / BLIP-2)** -- add a parallel
  vector collection that embeds the original image pixels. The
  Phase 34C text-semantic index satisfies the Phase 34C acceptance
  criteria; native image embeddings are only justified if semantic
  retrieval alone proves insufficient for diagrams and infographics.
* **Vision reranking** -- use a cross-encoder to rerank image
  knowledge candidates against the user's question.
* **Image-citation rendering** -- add a thumbnail preview in the
  chat UI for `[N]` citations where `citation_kind == "image_knowledge"`.

None of these are required for Phase 34C.

---

## 16. File map

```
apps/backend/app/services/multimodal/
    __init__.py
    config.py               # constants (source type, defaults)
    knowledge_record.py     # ImageKnowledgeRecord + deterministic builder
    indexer.py              # idempotent Qdrant upsert / delete
    retrieval.py            # retrieval overlay (boost + authority demote)
    lifecycle.py            # delete / refresh / reindex hooks
    scripts/__init__.py
    scripts/backfill.py     # explicit, idempotent admin backfill CLI

apps/backend/tests/
    test_phase34c_knowledge_record.py
    test_phase34c_indexer.py
    test_phase34c_lifecycle.py
    test_phase34c_retrieval.py
    test_phase34c_rbac.py
    test_phase34c_citations.py
    test_phase34c_backfill.py
    test_phase34c_e2e_live.py

apps/backend/app/core/config.py                       # + MULTIMODAL_* settings
apps/backend/app/rag/query_analysis.py                # + historical image-intent patterns + is_product_meaning_query
apps/backend/app/rag/retriever.py                     # + integrate_image_knowledge overlay hook
apps/backend/app/rag/citations.py                     # + image_knowledge citation metadata
apps/backend/app/vision/persistence.py                # + refresh_image_knowledge after persist
apps/backend/app/api/documents.py                     # + delete_image_knowledge + reindex hook

docs/phase34c-persistent-multimodal-knowledge.md      # this document
```
