# Phase 34A.1 — RAG Retrieval & DB Migration Hardening

> **Status:** Shipped on branch `phase-34a1-rag-retrieval-hardening`.
> Tag: `phase-34a1-rag-retrieval-hardening`.
> Builds on Phase 34A (`phase-34a-enterprise-ocr`, `4dc939b`).
> All additive. No existing tables, columns, endpoints, or contracts
> are removed or renamed.

## 1. Motivation

Phase 34A verified that direct image OCR answers are correct, but
exposed two concrete gaps:

1. **Retrieval quality** — for "What error code is shown in the image
   I uploaded?" the system surfaced the correct uploaded image AND
   unrelated low-confidence KB sources (e.g. `admin_test.txt`).
2. **Production schema policy** — `Base.metadata.create_all(bind=engine)`
   in `app/main.py` silently created the new `document_images` table
   before Alembic migration `012` ran, producing a `DuplicateTable`
   conflict at deploy time.

This phase fixes both without rewriting the existing RAG architecture
(no new vector DB, no LLM-based identifier extraction, no vision
models).

## 2. Architecture

```
                              chat /api/chat
                                     │
                                     ▼
                       analyze_query(query)   ← deterministic, no LLM
                                     │
                                     ▼
                       retrieve_chunks_with_auth(query, ...)
                                     │
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
   retrieve_chunks_hybrid                       keyword_search (PG)
        │  (vector + keyword fusion)
        ▼
   exact_match_boost(chunks, analysis)   ← new
        │  multiplicative boost per matched error code / term
        ▼
   hybrid_mmr diversity filter (Phase 30E)
        │
        ▼
   apply_relevance_threshold(min_score, max_results)   ← new
        │  drops chunks below RAG_MIN_RELEVANCE_FLOOR
        │  caps at RAG_MAX_FINAL_SOURCES
        ▼
   select_image_aware_chunks(chunks, analysis, image_context)   ← new
        │  scopes to OCR-derived chunks when query is image-content
        ▼
   generate_answer_with_rag_audit → grounding → LLM → citations
```

### 2.1 What changed and what didn't

| Layer | Status | Notes |
|---|---|---|
| Qdrant vector store | unchanged | Same collection, same payload |
| Embedding provider | unchanged | Same provider / model |
| Hybrid scoring (Phase 30E) | unchanged | Still vector + keyword fusion |
| MMR diversity filter | unchanged | Still applied after hybrid scoring |
| Citation rendering | unchanged | Same frontend/API contract |
| LLM prompt | unchanged | Same evidence-level-aware prompt |
| **Query analysis (new)** | added | Deterministic regex extractor |
| **Exact-match boost (new)** | added | Multiplicative re-ranking signal |
| **Relevance floor + cap (new)** | added | Drops & trims before LLM call |
| **Image-aware routing (new)** | added | Scopes retrieval to OCR chunks |
| **Startup `create_all` gate (new)** | added | Production-safe by default |

## 3. Identifier extraction (deterministic, no LLM)

Implemented in `app/rag/query_analysis.py`. Recognises:

| Pattern | Examples |
|---|---|
| Oracle | `ORA-12541`, `ORA-xxxxx` |
| SQLSTATE | `SQLSTATE 08001`, `SQLSTATE 22P02` |
| HTTP status | `HTTP 404`, `HTTP 500` |
| Hex / Win32 | `0x80070005` |
| `ERR_*` constants | `ERR_CONNECTION_REFUSED` |
| Vendor `XX-####` | `HL-1053` |
| Context-gated numerics | `error 902`, `code 18456`, `errno 500`, `return code 1053`, `trigger 902`, `got 18456` |

### 3.1 Conservative design

- Bare numerics like "2024", "page 42", "received 1000 messages"
  are NOT auto-classified as error codes. A context word
  (`error`, `code`, `errno`, `errcode`, `status`, `return code`,
  `trigger`, `threw`, `showing`, `see`, `saw`, `got`, `reported`)
  must be adjacent.
- Numeric-only codes use a word-boundary regex so `902` does NOT
  match `19021`.
- ORA- and HL- codes use their non-digit prefix for safe matching.
- HTTP and SQLSTATE codes preserve their canonical space form
  (`HTTP 404`, `SQLSTATE 08001`) so direct substring comparison
  against chunk content works without re-formatting.

### 3.2 Intent classification

| Query | query_type | references_uploaded_image |
|---|---|---|
| `What error code is shown in the image I uploaded?` | `image_content` | True |
| `What does error 902 mean?` | `error_lookup` | False |
| `How do I troubleshoot error 902?` | `error_lookup` | False |
| `ORA-12541 please help` | `error_lookup` | False |
| `How do I troubleshoot the error shown in the screenshot?` | `error_lookup` | True |
| `summarise the document` | `general` | False |

When `query_type == image_content`, retrieval is scoped to
OCR-derived chunks. When `error_lookup`, retrieval broadens to the
full KB and exact-match reranking lifts the chunk containing the
identifier to the top.

## 4. Hybrid retrieval (exact-match reranking)

`app/rag/exact_match.py` adds an optional `query_analysis` argument
to `retrieve_chunks_hybrid` and `retrieve_with_strategy`. When the
analysis reports identifiers, each chunk's fused score is multiplied
by `(1 + boost)`:

```
final_score = fused_score × (1 + exact_match_boost × matched_codes
                                       + exact_term_boost  × matched_terms)
```

Defaults (env-configurable):

| Setting | Default | Effect |
|---|---|---|
| `RAG_EXACT_MATCH_BOOST` | `0.30` | +30% per matched error code |
| `RAG_EXACT_TERM_BOOST` | `0.15` | +15% per matched technical term |
| Cap (internal) | `0.75` | Prevents accidental over-boosting |

Example:

| Chunk | Base score | Contains "902" | Contains "rejected-forbidden-country" | Final |
|---|---|---|---|---|
| `Error 902 - rejected-forbidden-country` | 0.65 | ✓ | ✓ | 0.65 × (1 + 0.30 + 0.15) = **0.943** |
| `totally unrelated article` | 0.76 | ✗ | ✗ | **0.76** |

The exact-match chunk ranks first even though the unrelated chunk
has a higher base vector score.

Diagnostics on each chunk: `_exact_match_count`, `_exact_match_boost`,
`_matched_codes`, `_matched_terms` — safe for observability, no
sensitive content.

## 5. Minimum relevance filtering

`app/rag/relevance_filter.py` applies:

```
1. Drop chunks with score < RAG_MIN_RELEVANCE_FLOOR
2. Sort remaining chunks by score descending
3. Keep at most RAG_MAX_FINAL_SOURCES chunks
```

| Setting | Default | Meaning |
|---|---|---|
| `RAG_MIN_RELEVANCE_FLOOR` | `0.10` | Chunks below this fused score are dropped before the LLM call |
| `RAG_MAX_FINAL_SOURCES` | `6` | Maximum sources presented to the LLM |
| `RAG_MAX_RETRIEVED_CANDIDATES` | `30` | Maximum candidates returned by initial hybrid retrieval |

When no chunk meets the floor, the existing grounding layer returns
the existing fallback message ("no sufficiently relevant
knowledge-base source was found") — no fabricated sources.

## 6. Image-aware query routing

`app/rag/image_routing.py` distinguishes two intents:

### A. IMAGE CONTENT QUESTION
> "What error code is shown in the image I uploaded?"
> "What does the screenshot say?"
> "What message is visible in the latest screenshot?"

Routing: return ONLY chunks with `source_type` ∈
`{image_ocr, pdf_page_ocr, docx_image_ocr, image, pdf_page, docx_image}`.
When an `image_context` is supplied, the chunks matching
`image_id` / `document_id` are surfaced first but ALL image-source
chunks are kept. Unrelated KB sources (admin scratch, marketing copy,
etc.) are dropped.

### B. TROUBLESHOOTING / KNOWLEDGE QUESTION
> "What does error 902 mean?"
> "How do I troubleshoot the error shown in the screenshot?"
> "Has this error occurred in our documentation?"

Routing: keep ALL chunks. The exact-match booster inside
`hybrid_retriever` ranks the OCR chunk containing the identifier to
the top, alongside the relevant KB documentation.

No vision model is invoked. No external API calls. Routing decisions
come entirely from the deterministic query analysis.

## 7. Database migration hardening

### 7.1 Problem

Before Phase 34A.1, `app/main.py` ran
`Base.metadata.create_all(bind=engine)` unconditionally at startup.
When Phase 34A deployed, this auto-created `document_images` before
Alembic migration `012` ran, causing a `DuplicateTable` failure.

### 7.2 Fix

`app/main.py` now wraps `create_all` in a gate:

```python
def _should_autocreate_schema() -> bool:
    if settings.DB_AUTO_CREATE_SCHEMA:        # explicit opt-in
        return True
    url = (settings.DATABASE_URL or "").lower()
    if url.startswith("sqlite"):              # test harness only
        return True
    return False                             # production: no

@app.on_event("startup")
def startup():
    validate_startup()
    if _should_autocreate_schema():
        try:
            Base.metadata.create_all(bind=engine)
        except Exception:
            logger.exception("...")
    else:
        logger.info("DB_AUTO_CREATE_SCHEMA=false; relying on Alembic.")
    _bootstrap_admin_user()
```

### 7.3 Production deployment policy

```
# 1. Apply migrations
alembic upgrade head

# 2. Start / restart the application
#    (DB_AUTO_CREATE_SCHEMA defaults to false; create_all is a no-op.)
```

The current production head is `012`; this phase introduces NO new
migration. Existing migration history is unchanged.

### 7.4 Test harness policy

Tests use `sqlite:///./test.db` and the `db_session` fixture in
`tests/conftest.py` already calls `Base.metadata.create_all(bind=engine)`
per test. The startup gate additionally permits `create_all` when
`DATABASE_URL` starts with `sqlite`, providing a second layer of
safety for any test that bypasses the fixture.

Setting `DB_AUTO_CREATE_SCHEMA=true` overrides the gate for any
operator who wants the legacy behaviour on a non-sqlite DB (e.g. a
local dev sandbox).

## 8. Configuration

`app/core/config.py` exposes:

| Setting | Env var | Default | Purpose |
|---|---|---|---|
| `RAG_MAX_RETRIEVED_CANDIDATES` | `RAG_MAX_RETRIEVED_CANDIDATES` | `30` | Cap on hybrid retrieval candidates |
| `RAG_MAX_FINAL_SOURCES` | `RAG_MAX_FINAL_SOURCES` | `6` | Cap on chunks forwarded to LLM |
| `RAG_EXACT_MATCH_BOOST` | `RAG_EXACT_MATCH_BOOST` | `0.30` | Per error-code hit |
| `RAG_EXACT_TERM_BOOST` | `RAG_EXACT_TERM_BOOST` | `0.15` | Per technical-term hit |
| `RAG_IMAGE_AWARE_ROUTING_ENABLED` | `RAG_IMAGE_AWARE_ROUTING_ENABLED` | `true` | Master switch for image routing |
| `RAG_MIN_RELEVANCE_FLOOR` | `RAG_MIN_RELEVANCE_FLOOR` | `0.10` | Min fused score to survive |
| `DB_AUTO_CREATE_SCHEMA` | `DB_AUTO_CREATE_SCHEMA` | `false` | Production-safe schema policy |

All settings are documented in `.env.example`.

## 9. Observability

Each retrieval now returns diagnostics under
`metadata.relevance_filter`, `metadata.image_routing`,
`metadata.query_analysis`, `metadata.exact_match_count`, and
`metadata.retrieval_mode`. Counters (no content):

```
candidate_count      — chunks returned by hybrid retrieval
filtered_count       — chunks dropped by relevance floor / cap
final_source_count   — chunks presented to the LLM
exact_match_count    — chunks boosted by an exact identifier match
image_context_used   — True if image_context was applied
retrieval_mode       — "image_aware" | "standard"
```

When LangSmith is enabled, these counters are attached to the
existing `rag_retrieval` span via `set_meta()` so the trace shows
the retrieval decision without exposing raw document content.

## 10. Testing

| Suite | File | Result |
|---|---|---|
| Deterministic query analysis | `tests/test_phase34a1_query_analysis.py` | 30 passed |
| Exact-match boost | `tests/test_phase34a1_exact_match.py` | 14 passed |
| Relevance thresholding | `tests/test_phase34a1_relevance_filter.py` | 13 passed |
| Image-aware routing | `tests/test_phase34a1_image_routing.py` | 17 passed |
| Startup / migration policy | `tests/test_phase34a1_startup.py` | 6 passed |
| Phase 34A OCR regression | `tests/test_phase34a_ocr.py` | all passed |
| Hybrid retrieval regression | `tests/test_hybrid_retrieval.py` | all passed |
| Retrieval strategy regression | `tests/test_retrieval_strategy.py` | all passed |
| Core RAG regression | `tests/test_rag.py` | all passed |
| RBAC + security + audit regression | `tests/test_rbac_integration.py`, `test_security.py`, `test_audit.py` | all passed |
| Citations + grounding regression | `tests/test_citations.py`, `test_grounding.py`, `test_rag_access.py` | all passed |
| LangSmith / embeddings / admin | `tests/test_langsmith_tracing.py`, `test_embeddings.py`, `test_admin_rag_config.py`, `test_chunking.py`, `test_indexing.py` | all passed |

`test_chat.py` and `test_compare_rag_modes.py` require a live Qdrant
service (`[Errno -3] Temporary failure in name resolution` in the
sandbox) and were deferred to the E2E environment.

## 11. Spec scenario coverage

| # | Scenario | Coverage |
|---|---|---|
| 1 | Image-content question → OCR chunk prioritised, unrelated excluded | `test_phase34a1_image_routing.py::test_image_content_question_scopes_to_ocr_chunks` |
| 2 | "How do I troubleshoot error 902?" → exact 902 doc boosted | `test_phase34a1_exact_match.py::test_boost_reranks_exact_match_above_higher_vector_score` |
| 3 | Higher-vector but no-match chunk ranks below exact match | `test_phase34a1_exact_match.py::test_boost_reranks_exact_match_above_higher_vector_score` |
| 4 | admin_test-like unrelated content excluded | `test_phase34a1_relevance_filter.py::test_scenario_unrelated_admin_test_excluded` |
| 5 | No KB doc meets threshold → no fabricated source | `test_phase34a1_relevance_filter.py::test_scenario_no_results_relevant` |
| 6 | OCR / image source citation preserved | `test_phase34a1_image_routing.py::test_image_content_question_scopes_to_ocr_chunks` |
| 7 | Normal non-image RAG query still works | `test_phase34a1_query_analysis.py::test_query_type_general` + retrieval regression suites |
| 8 | RBAC filtering still occurs | `test_rbac_integration.py` (regression, all passed) |
| 9 | Production startup does NOT use create_all | `test_phase34a1_startup.py::test_production_url_does_not_autocreate_schema` |
| 10 | Test DB init still works | `test_phase34a1_startup.py::test_sqlite_test_url_still_autocreates` |

## 12. Rollback

Every new module is purely additive on the retrieval path; existing
behaviour is gated by feature flags.

| Roll back | How |
|---|---|
| Disable image-aware routing | `RAG_IMAGE_AWARE_ROUTING_ENABLED=false` |
| Disable exact-match boost | Leave `analyze_query()` (it still runs) — its output has no effect when no identifiers are found in the query. To fully neutralise, set `RAG_EXACT_MATCH_BOOST=0` and `RAG_EXACT_TERM_BOOST=0`. |
| Re-enable production `create_all` (NOT recommended) | `DB_AUTO_CREATE_SCHEMA=true` |
| Revert the entire phase | `git revert` the commit and re-deploy — no DB migration was added, so no schema rollback is required. |

## 13. Known limitations

- **Conservative identifier detection.** Code patterns that don't
  match any of the listed shapes are not flagged. This is intentional
  to keep false positives low.
- **English-only regex patterns.** Code-shape detection is language-
  agnostic; troubleshooting-hint detection uses English vocabulary.
- **Single `image_context`.** When multiple images are uploaded in a
  session, only the most-recent (or supplied) `image_id` biases the
  ranking — but all image-source chunks remain available.
- **No vision model.** OCR text is the only input. Phase 34B will
  add image captioning / object detection / image embeddings.
- **Qdrant-dependent tests require a live cluster.** The integration
  suites (`test_chat.py`, `test_compare_rag_modes.py`,
  `test_rag_llm_integration.py`) need Qdrant DNS resolution; the
  sandbox environment lacks this so they were deferred to E2E.
