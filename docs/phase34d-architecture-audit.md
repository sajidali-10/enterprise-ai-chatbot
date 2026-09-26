# Phase 34D — Advanced Visual Understanding: Architecture Audit

> **Status:** Read-only audit complete. Plan approved with two architecture adjustments (provider reuse + Redis-backed cache). Implementation in progress.
> **Branch:** `phase-34d-advanced-visual-understanding`
> **Base commit:** `7aca63f`
> **Audit date:** 2026-09-25
> **Approval date:** 2026-09-26

This document records what is already in place on top of which Phase 34D
will build, plus the approved architecture adjustments for Phase 34D.
The plan is in §9; the approved changes are summarised in §0A.

---

## 0A. Approved architecture adjustments (2026-09-26)

Two corrections were applied before implementation:

### 0A.1 Reuse the Phase 34B VisionProvider abstraction

The new `app/services/advanced_vision/` package contains:

- task taxonomy (`AdvancedVisualTask` enum) + deterministic classifier
- task-specific prompt templates
- `VisualReasoningResult` schema
- advanced orchestrator (RBAC → cache → provider → evidence)
- Redis cache integration
- comparison logic

It does **not** contain a duplicate provider framework. The package
extends the existing `app/services/vision/base.py:VisionProvider`
protocol with an optional `task` parameter (and an optional `images`
plural list for comparison). The existing `MockVisionProvider` and
`OpenAICompatibleVisionProvider` are updated to accept these new
parameters; Phase 34B callers that omit them see no behavior change.

Provider configuration (`VISION_PROVIDER`, `VISION_BASE_URL`,
`VISION_MODEL`, `VISION_TIMEOUT_SECONDS`, `VISION_MAX_RETRIES`,
`VISION_API_KEY`), authentication, timeout/retry behavior, and error
handling therefore remain centralised in the Phase 34B provider
framework. Phase 34D does not introduce a parallel `ADVANCED_VISION_*`
provider layer.

### 0A.2 Redis-backed production-safe cache

Phase 34D's task-specific advanced Vision cache is backed by Redis
(already wired in `app/core/rate_limit.py:_get_redis_client()`). The
authoritative cache is Redis. A tiny in-process LRU may exist only as
an optimization (single-process hot path), never as the authoritative
store.

Cache key (suggested by the brief, adapted):

```
advanced_vision:{schema_version}:{provider}:{model}:{task_type}:{image_content_hash}
```

For comparison (image A → image B):

```
advanced_vision:{schema_version}:{provider}:{model}:{IMAGE_COMPARISON}:{image_content_hash_a}:{image_content_hash_b}
```

The comparison key preserves A→B order (NOT lexicographically sorted).
Rationale: order matters for "what changed between these two
screenshots?" — image A is the *before*, image B is the *after*. A
swapped cache hit would silently invert the answer.

Requirements:

| Requirement | Implementation |
| --- | --- |
| Configurable TTL | `ADVANCED_VISION_CACHE_TTL_SECONDS` (default 3600) |
| Safe serialization of `VisualReasoningResult` | JSON via a `to_dict()` / `from_dict()` pair with hard length caps and value validation |
| Cache miss/failure never breaks chat | All Redis calls wrapped in try/except; failure → log + return cache-miss → fall through to provider call or Phase 34B OCR-only |
| Redis outage degrades safely | Same path as cache miss; orchestrator never raises; Phase 34B OCR-only path remains the safety net |
| No image bytes / secrets in cache | Only `VisualReasoningResult.to_dict()` JSON — which already excludes `image_bytes`, raw provider payload, API keys, and provider-internal fields |

### 0A.3 Other constraints reaffirmed

- `ADVANCED_VISION_ENABLED=false` by default.
- Authorization (`can_access_document`) is checked BEFORE the MinIO
  fetch AND BEFORE any provider call. Both gates are mandatory.
- OCR remains the source for exact visible text. Advanced Vision is
  for layout / state / trend / relationship / comparison only.
- Product meaning / root cause / troubleshooting remain KB-grounded.
- Phase 34C retrieval is not redesigned.
- No CLIP / native image embeddings.
- No Alembic migration 014 in Phase 34D — Redis is the cache, no DB
  schema change required. If implementation uncovers a genuine
  persistence need (e.g. compliance audit trail), a dedicated table
  is added as Phase 34D.1 with migration `014_phase34d_*`.
- Image comparison backend capability stays in Phase 34D (cleanly
  implementable with two authorized image IDs).
- Frontend multi-image UX is deferred to Phase 34D.1.

---

## 1. Runtime configuration observed

Read from `.env` and `app/core/config.py`:

| Setting | Value | Source |
| --- | --- | --- |
| `LLM_PROVIDER` | `openrouter` | `.env` |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | `.env` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | `.env` |
| `EMBEDDING_PROVIDER` | `local` | `.env` |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | `.env` |
| `EMBEDDING_DIMENSION` | `384` | `.env` |
| `REDIS_HOST` | `redis` (default; not overridden in `.env`) | `config.py` |
| `REDIS_PORT` | `6379` (default) | `config.py` |
| `VISION_ENABLED` | `false` (default; not set in `.env`) | `config.py` |
| `VISION_PROVIDER` | `mock` (default) | `config.py` |
| `MULTIMODAL_KNOWLEDGE_ENABLED` | `true` (default) | `config.py` |
| `OCR_ENABLED` | `true` | `.env` |
| `OCR_PROVIDER` | `tesseract` | `.env` |

Phase 34B Vision is currently **off** in the deployed env. No Vision
provider calls happen until an operator explicitly sets
`VISION_ENABLED=true`. Phase 34C image_knowledge indexing still runs
because it only reads persisted columns — it does not call Vision.

Phase 34D `ADVANCED_VISION_ENABLED=false` (default) means Phase 34D
is fully inert until an operator explicitly opts in.

---

## 2. Phase 34B VisionResult schema (current)

`apps/backend/app/services/vision/base.py`:

```python
VISION_SCHEMA_VERSION = 1

@dataclass
class VisionResult:
    description: str = ""
    image_type: str = "unknown"
    visual_findings: List[str]
    detected_entities: List[str]
    visual_states: List[str]
    tags: List[str]
    confidence: Optional[int]   # 0..100
    provider: str
    model: str
    processing_time_ms: int
    schema_version: int = VISION_SCHEMA_VERSION
    raw: Optional[Dict[str, Any]] = None

class VisionProvider(Protocol):
    name: str
    model: str
    def analyze_image(
        self, *, image_bytes, mime_type, prompt, ocr_text="",
        context_hint="", max_tokens=600, timeout_s=None,
    ) -> VisionResult: ...
```

Built-in providers (`app/services/vision/`):

| Name | Network | Notes |
| --- | --- | --- |
| `mock` | none | Deterministic offline. Synthesises from OCR + filename cues. |
| `openai-compatible` | HTTPS | OpenAI Chat Completions Vision (OpenAI / OpenRouter / Azure / vLLM). |

Errors: `VisionProviderError`,
`VisionProviderUnavailableError`,
`VisionProviderTimeoutError`. Provider failures fall back to OCR-only
evidence; the orchestrator never raises.

Cache key (`build_cache_key`):

```
sha256(f"{image_id}|{provider}|{model}|{schema_version}|{salt}")[:40]
```

Phase 34D extends this contract with an optional `task` parameter on
`analyze_image` and an optional `images` parameter (plural, for
comparison). All existing Phase 34B callers continue to compile and
behave identically.

---

## 3. Phase 34B orchestration and routing

`apps/backend/app/vision/orchestrator.py` — `process_image_for_question(...)`:

1. Load `DocumentImage` row (best-effort).
2. `decide_processing_mode(...)` — deterministic router, no LLM.
3. Cache lookup → if hit, reuse `VisionResult`.
4. Provider call → only when justified + cache miss + provider configured.
5. Persistence (success) / failure (recorded).
6. `build_image_evidence(...)` — combines OCR + VisionResult into
   `ImageEvidence` with hard length caps.

The orchestrator never raises. RBAC is the caller's responsibility —
the orchestrator trusts the chat endpoint, which already gates
`image_context.document_id` / `image_id` against `can_access_document`.
Phase 34D adds a second RBAC gate inside the advanced orchestrator
because the advanced path can receive an image the standard chat
flow did not validate (e.g. resolved historical image).

Routing tiers (`app/vision/router.py`):

```
Tier 0: vision disabled              -> OCR_ONLY (vision_disabled)
Tier 1: ocr_status failed/disabled    -> VISION_FALLBACK (ocr_error)
Tier 2: text_len < threshold + visual -> VISION_FALLBACK
Tier 2b:text_len < threshold + non-visual -> OCR_ONLY (identifier)
Tier 3: visual_intent                -> OCR_PLUS_VISION
Tier 4: ocr_confidence < threshold    -> OCR_PLUS_VISION
Tier 5: default                      -> OCR_ONLY
```

`detect_visual_intent(question)` is conservative; identifier-lookup
phrases always force OCR_ONLY. `infer_image_type_hint(ocr_text)` is
observability-only.

---

## 4. Phase 34B evidence and integration

`apps/backend/app/vision/evidence.py`:

- `ImageEvidence` dataclass with hard caps
  (MAX_DESCRIPTION_CHARS=800, MAX_FINDINGS=6, MAX_ENTITIES=12,
  MAX_STATES=6, MAX_TAGS=8).
- `to_prompt_section()` renders the evidence as a single labelled
  prompt block.
- `build_image_evidence(...)` is non-throwing.

`apps/backend/app/vision/integration.py`:

- `maybe_run_vision(...)` is called from `answer_generator._run_rag_with_chunks*`.
- Builds a synthetic `vision_synthetic` chunk prepended to the chunk
  list. The chunk's `chunk_id` is a deterministic SHA-256 prefix of
  `(document_id, image_id, provider, model)` — no timestamp — so
  cache hits and misses produce the same `[N]` marker.
- The synthetic chunk carries the OCR document's `source_file_name`
  so existing citation rendering continues to work.

Phase 34D extends `ImageEvidence` with an optional
`advanced_reasoning: Optional[VisualReasoningResult]` field. When
advanced Vision is disabled, the field is `None` and the existing
prompt rendering is byte-identical.

---

## 5. Current vision prompt (Phase 34B)

`OpenAICompatibleVisionProvider._SYSTEM_PROMPT`:

```
"You are analyzing an enterprise technical screenshot. Describe
visually observable facts only. Do not infer undocumented product
behavior or troubleshooting steps. Preserve visible error codes,
labels, status indicators, relationships, graph trends, selected
controls, and warnings exactly as they appear. Mark anything you
cannot see clearly as 'uncertain'. Avoid unnecessary verbose
descriptions. Respond with STRICT JSON of the form:
{"description": ..., "image_type": ..., "visual_findings": [...],
"detected_entities": [...], "visual_states": [...], "tags": [...],
"confidence": int}."
```

User prompt template appends:

- context hint
- user question
- truncated OCR body (≤2000 chars)

There is currently ONE prompt for all visual reasoning tasks. Phase 34D
introduces task-specific prompts (UI_STATE_ANALYSIS, CHART_ANALYSIS,
DIAGRAM_ANALYSIS, TABLE_VISUAL_ANALYSIS, IMAGE_COMPARISON,
GENERAL_VISUAL). The system prompt stays as the safety wrapper; the
user prompt becomes task-aware.

---

## 6. Phase 34C image_knowledge representation

`apps/backend/app/services/multimodal/`:

- `knowledge_record.py`: `ImageKnowledgeRecord` (OCR + Vision columns),
  `build_knowledge_text()` pure deterministic builder,
  `compute_point_id()` deterministic positive integer via SHA-256,
  `load_image_record(db, image_id)`.
- `indexer.py`: `upsert_image_knowledge()`, `delete_image_knowledge*()` —
  idempotent in-place overwrite via the deterministic point id.
- `retrieval.py`: `integrate_image_knowledge(...)` overlay after
  hybrid retrieval; image-intent boost (default 0.25); authority
  demote for product-meaning queries; hard cap
  (`MULTIMODAL_MAX_IMAGE_SOURCES=2`); `has_historical_image_intent()`
  regex detector.
- `lifecycle.py`: `refresh_image_knowledge(db, image_id)` (called from
  `vision.persistence.persist_vision_result`), delete / reindex hooks.
  All fault-tolerant.
- `scripts/backfill.py`: explicit idempotent CLI.

`MULTIMODAL_KNOWLEDGE_ENABLED=false` short-circuits the entire
Phase 34C subsystem. Phase 34D does not modify any of these modules.

---

## 7. query_analysis, answer_generator, grounding, citations

`apps/backend/app/rag/query_analysis.py`:

- `analyze_query()` returns `QueryAnalysis(query_type, error_codes,
  technical_terms, references_uploaded_image, raw_signals)`.
- `query_type` ∈ `{general, error_lookup, image_content}`.
- Identifier extractors: ORA-, SQLSTATE, HTTP nnn, 0x hex, ERR_*,
  HL-nnn, context-gated bare numerics ("error 902", "code 18456").
- Image-intent patterns: in-scope ("the image I uploaded", "in the
  attached image"). Historical patterns ("which screenshot showed
  error 902?") are kept separate so they boost retrieval without
  flipping the query type.
- `is_product_meaning_query(analysis)` — used by Phase 34C demote.
- Phase 34D adds task-classification hints to `raw_signals` (no
  behavior change to existing query_type).

`apps/backend/app/rag/answer_generator.py`:

- `generate_answer_with_rag()` and `generate_answer_with_rag_audit()`.
- Image pre-resolution fast path → `_run_rag_with_chunks*`.
- Vision orchestrator runs BEFORE grounding so the synthetic vision
  chunk participates in topic relevance / evidence-level checks.
- Citation retry + content-overlap attachment.
- `maybe_synthesize_image_observation_answer()` (Phase 34C.1) for
  LLM-refusal recovery on historical-image observation queries.

`apps/backend/app/rag/grounding.py`:

- `apply_grounding_checks()` — evidence-level decision (STRONG/MEDIUM/WEAK).
- Phase 34C.1 bypass for image-observation queries (topic + answer
  checks skipped).
- Phase 34A.1.2 bypass for resolved image_content sources (source
  relationship is the relevance signal).

`apps/backend/app/rag/image_observation_helpers.py` /
`image_observation_synthesis.py`:

- Deterministic classifier (`classify_image_observation_query`).
- Path A (exact identifier match) and Path B (strong semantic
  dominance).
- Post-LLM refusal recovery only — substantive LLM answers pass
  through untouched.

`apps/backend/app/rag/citations.py`:

- `format_citations()` adds structured fields for
  `citation_kind="image_knowledge"` (`image_id`, `image_type`,
  `vision_provider`, `vision_model`, `has_vision`,
  `knowledge_schema_version`, `document_id`).
- `[N]` marker format is unchanged; the distinction lives in the
  structured fields.

---

## 8. Chat request schema and RBAC

`apps/backend/app/schemas/chat.py`:

- `ChatRequest`: `message`, `mode`, `session_id`, `image_context`
  (Optional[dict] — currently untyped).
- `ChatResponse.vision`: safe Phase 34B vision metadata.

`apps/backend/app/api/chat.py`:

- Forwards `image_context` to `generate_answer_with_rag_audit()` /
  `generate_answer_with_rag()`.
- Surface `response.vision` from `retrieval_metadata["vision"]`.

`apps/backend/app/rag/image_resolver.py`:

- `extract_explicit_target(image_context)` → `ResolvedImage`.
- `resolve_recent_image(db, auth, limit=25)` → RBAC-filtered most-recent
  OCR-bearing image.
- `recent_targets_from_image_context()` reads `image_context.recent_images`.
- Phase 34D adds `extract_comparison_targets(image_context)` reading
  `image_context.comparison = [{"document_id", "image_id"}, ...]`
  (length 2 for the comparison path).

`apps/backend/app/rag/image_routing.py`:

- `select_image_content_chunks(...)` — Phase 34A.1.2 fast path.
- `select_image_aware_chunks(...)` — Phase 34A.1 routing.
- The image-routing layer only ever operates on a single resolved
  target. Comparison is not routed through this layer — it goes
  through a dedicated `app/services/advanced_vision/comparison.py`
  helper.

RBAC: `app/security/permissions.filter_documents_by_permission` and
`can_access_document` are the canonical gates. Phase 34D's
advanced orchestrator re-checks `can_access_document` BEFORE the
MinIO fetch AND BEFORE the provider call — defense in depth.

---

## 9. Frontend attachment surface

`apps/frontend/app/chat/page.tsx`:

- `chatAttachment` context holds ONE attachment at a time.
- On submit, the frontend uploads the file, then sends
  `body.image_context = { document_id, image_id }`.
- No multi-image selection UI. The backend `image_context` schema
  is an untyped dict, so it can technically accept arrays, but the
  frontend does not produce them today.

Multi-image frontend support is **deferred to Phase 34D.1** per the
brief. The backend `image_context.comparison = [{document_id,
image_id}, ...]` shape is reserved for that future UI. In Phase 34D
the backend is exercised via direct API calls / E2E test scripts,
not the chat UI.

---

## 10. Observability

LangSmith tracing (`app/services/langsmith_tracing.py`, env-gated):

- `LANGSMITH_TRACING=false` (default) — zero external calls.
- Span hierarchy: `chat_request` > `rag_retrieval` /
  `evidence_grounding` / `prompt_building` / `llm_call` /
  `citation_processing` / `final_response`.
- `sanitize_metadata()` redacts API keys, JWTs, bearer tokens,
  passwords, connection strings, emails, phone numbers, private key
  blocks.

Phase 34D surfaces these new safe fields under `retrieval_metadata.advanced_vision`:

- `task_type`
- `advanced_vision_called`
- `advanced_vision_cache_hit`
- `advanced_vision_trigger_reason`
- `advanced_vision_provider`
- `advanced_vision_model`
- `advanced_vision_latency_ms`
- `advanced_vision_image_count`
- `advanced_vision_image_ids`
- `advanced_vision_error` (when failed)

These mirror the Phase 34B `vision.*` fields for consistency. They
appear in `ChatResponse.vision` (when `advanced_vision_called=True`)
and in the `chat_request` LangSmith span metadata. **No image
bytes, JWTs, or API keys are logged.**

---

## 11. Alembic state

Last migration: `013_phase34b_vision_columns.py` (Phase 34B Vision
columns on `document_images`). The current head is revision `013`.

Phase 34D does **not** add migration 014. The Redis-backed cache is
external to the application DB; no schema change is required. If
implementation uncovers a genuine persistence need (e.g. compliance
audit trail), migration `014_phase34d_*` is added in Phase 34D.1
and only at that point.

---

## 12. What does NOT exist today

The brief's Phase 34D objectives map to capabilities the current
codebase does NOT provide:

- ❌ No task-aware visual classifier (UI vs chart vs diagram vs
  table vs comparison).
- ❌ No task-specific Vision prompts.
- ❌ No structured `VisualReasoningResult` with `ui_states[]`,
  `chart_findings[]`, `diagram_findings[]`, `relationships[]`,
  `anomalies[]`, `comparison_changes[]`.
- ❌ No visual task routing separate from Phase 34B's generic
  visual-intent detector.
- ❌ No image-comparison path.
- ❌ No multi-image chat attachment support.
- ❌ No Redis-backed task-specific cache.

The existing Phase 34B path answers the brief's "Which screenshot
showed error 902?" but does NOT answer the brief's "Which server
appears failed?" or "What trend does this chart show?".

---

## 13. Phase 34D cache strategy (Redis-backed, production-safe)

### 13.1 Cache layout

```
Key:   advanced_vision:{schema_version}:{provider}:{model}:{task_type}:{image_content_hash}
Value: JSON-serialised VisualReasoningResult (to_dict())
TTL:   ADVANCED_VISION_CACHE_TTL_SECONDS (default 3600)
```

For comparison (order preserved — A is before, B is after):

```
Key:   advanced_vision:{schema_version}:{provider}:{model}:{IMAGE_COMPARISON}:{image_content_hash_a}:{image_content_hash_b}
```

`image_content_hash` = first 16 bytes of SHA-256 of the raw image
bytes (hex-encoded). This distinguishes re-uploads of the same
filename with different content without paying for a full hash on
every call.

### 13.2 Why a content hash instead of `image_id`

`image_id` is stable but the bytes behind it can change between
deployments (re-upload, re-ingestion). The content hash guarantees
cache validity against actual image bytes; using `image_id` alone
could let stale results survive legitimate content changes.

### 13.3 In-process LRU as optimization only

A small bounded LRU (`maxsize=64`, opt-in via
`ADVANCED_VISION_PROCESS_CACHE_ENABLED=true`) may be layered on top
of Redis as a single-process hot-path optimization. It is **never**
authoritative:

- On every cache lookup the orchestrator queries Redis first.
- On every cache write the orchestrator writes Redis first; the
  in-process cache is updated opportunistically.
- On Redis outage the in-process cache is also bypassed — the
  orchestrator proceeds to the provider call. There is no scenario
  in which the in-process cache substitutes for Redis.

### 13.4 Cache miss / failure handling

Every Redis call is wrapped:

```python
try:
    blob = redis.get(key)
except Exception as exc:
    logger.debug("advanced_vision: redis read failed: %s", exc)
    return CacheResult(hit=False, error=str(exc)[:200])
```

Failure paths:

| Failure | Behavior |
| --- | --- |
| Redis unreachable | Cache miss → provider call → result cached on success |
| Redis timeout | Same |
| Corrupt cached JSON | Cache miss → provider call → old value overwritten |
| Provider call fails | OCR-only / Phase 34B evidence retained; nothing cached |
| `ADVANCED_VISION_CACHE_ENABLED=false` | Always miss → provider call |

### 13.5 What is stored

Only `VisualReasoningResult.to_dict()` JSON. By construction:

- No `image_bytes` (the bytes are never read into the result).
- No `prompt` (the provider's input is not stored).
- No `raw` (provider internals are not stored).
- No API keys (the provider never returns them; we never serialize them).

The orchestrator validates the deserialised result before use (length
caps, required fields). Malformed JSON is treated as cache miss.

---

## 14. Image-comparison feasibility

The brief asks for backend comparison between two authorized image IDs
first, then historical-image comparison, then general frontend
multi-image only if clean.

### 14.1 Backend comparison (Phase 34D, in scope)

- `image_resolver.extract_comparison_targets(image_context)` reads
  `image_context.comparison = [{"document_id", "image_id"}, ...]`.
- Each image must independently pass `can_access_document`. If either
  fails, the comparison request is denied — NO image bytes are sent
  to the provider, NO filename is returned, NO error leaks the
  existence of the other image.
- Provider call uses the `IMAGE_COMPARISON` task with a comparison-
  specific prompt.
- Cache key preserves A→B order (NOT lexicographically sorted):
  `advanced_vision:{schema_version}:{provider}:{model}:IMAGE_COMPARISON:{hash_a}:{hash_b}`.
  Order matters because image A is the *before* state and image B
  is the *after* state; swapping them would silently invert the
  answer.
- Citations include BOTH image IDs.

### 14.2 Historical-image comparison (Phase 34D, in scope)

- Same path but with `comparison_recent=true`: the orchestrator
  queries Phase 34C's image_knowledge to find two authoritative
  candidates the user has previously uploaded. The user does not
  need to attach either image.
- Resolved images go through the same RBAC → MinIO fetch → cache →
  provider flow.
- Cache key is `(hash_a, hash_b)` of the resolved image bytes.

### 14.3 Frontend multi-image UI (Phase 34D.1, deferred)

- The current frontend only attaches one image. A multi-image
  selection UI would require a new attachment context, preview
  carousel, ordering, and removal logic. Per the brief, this
  expands scope and is moved to Phase 34D.1.
- The backend already accepts arbitrary JSON in
  `image_context.comparison`, so a Phase 34D.1 frontend can
  integrate without backend changes.

---

## 15. Phase 34D implementation plan

### 15.1 Files that change

New files:

| Path | Purpose |
| --- | --- |
| `apps/backend/app/services/advanced_vision/__init__.py` | Package surface |
| `apps/backend/app/services/advanced_vision/base.py` | `AdvancedVisualTask` enum, `VisualReasoningResult` dataclass, `ADVANCED_VISION_SCHEMA_VERSION`, cache key builder, JSON-safe `to_dict()` / `from_dict()` |
| `apps/backend/app/services/advanced_vision/prompts.py` | Task-specific system + user prompt templates |
| `apps/backend/app/services/advanced_vision/router.py` | Deterministic `classify_visual_task()` regex classifier |
| `apps/backend/app/services/advanced_vision/cache.py` | Redis-backed cache + optional in-process LRU helper |
| `apps/backend/app/services/advanced_vision/orchestrator.py` | `run_advanced_visual_reasoning()`: RBAC → cache → provider → evidence |
| `apps/backend/app/services/advanced_vision/comparison.py` | `compare_visual_state(image_a, image_b)` helper |
| `apps/backend/app/services/advanced_vision/_image_fetch.py` | `fetch_authorized_image_bytes(db, image_id, auth)` (RBAC → MinIO) |
| `apps/backend/tests/test_phase34d_task_classifier.py` | Unit tests for `classify_visual_task` |
| `apps/backend/tests/test_phase34d_prompts.py` | Prompt template structure + grounding rules |
| `apps/backend/tests/test_phase34d_cache.py` | Cache key determinism, hit/miss, Redis outage, TTL, serialization |
| `apps/backend/tests/test_phase34d_orchestrator.py` | RBAC, cache, provider failure, evidence caps |
| `apps/backend/tests/test_phase34d_ui_analysis.py` | UI state semantics |
| `apps/backend/tests/test_phase34d_chart_analysis.py` | Chart findings semantics |
| `apps/backend/tests/test_phase34d_diagram_analysis.py` | Diagram relationships |
| `apps/backend/tests/test_phase34d_table_analysis.py` | Table findings |
| `apps/backend/tests/test_phase34d_comparison.py` | Two-image comparison + RBAC + cache key (order preserved) |
| `apps/backend/tests/test_phase34d_kill_switch.py` | `ADVANCED_VISION_ENABLED=false` preserves 34B |
| `apps/backend/tests/test_phase34d_rbac.py` | Unauthorized images rejected before MinIO fetch and before provider |
| `apps/backend/tests/test_phase34d_citations.py` | Comparison cites both images; single-image cites one |
| `apps/backend/tests/test_phase34d_e2e_live.py` | Live E2E A-H scenarios |
| `docs/phase34d-advanced-visual-understanding.md` | Phase documentation |

Modified files:

| Path | Change |
| --- | --- |
| `apps/backend/app/services/vision/base.py` | Extend `VisionProvider.analyze_image()` with optional `task` and `images` parameters (backward-compatible default `None`) |
| `apps/backend/app/services/vision/openai_compatible_provider.py` | Accept optional `task` and `images` parameters; pass to prompt builder; image list becomes a list of base64 data URLs for comparison |
| `apps/backend/app/services/vision/mock_provider.py` | Accept optional `task` and `images` parameters; no-op branches for backward compatibility |
| `apps/backend/app/services/vision/factory.py` | No change (Phase 34D reuses the existing factory) |
| `apps/backend/app/vision/orchestrator.py` | Optional hook to call advanced vision when task classifier fires and Phase 34B has already produced a `VisionResult` (or when comparing two images) |
| `apps/backend/app/vision/integration.py` | Accept extended `image_context` with `comparison` field; surface advanced vision metadata on the synthetic chunk |
| `apps/backend/app/vision/evidence.py` | Extend `ImageEvidence` with optional `advanced_reasoning: Optional[VisualReasoningResult]` field (None when disabled — prompt rendering byte-identical to Phase 34B) |
| `apps/backend/app/rag/answer_generator.py` | Call advanced vision orchestrator when task classification + visual intent fire; preserve Phase 34B path when disabled |
| `apps/backend/app/rag/image_resolver.py` | Add `extract_comparison_targets(image_context)` |
| `apps/backend/app/api/chat.py` | Document extended `image_context` (no schema change — still `dict`) |
| `apps/backend/app/schemas/chat.py` | Update `image_context` docstring with comparison shape |
| `apps/backend/app/core/config.py` | Add `ADVANCED_VISION_*` settings |
| `.env.example` | Add `ADVANCED_VISION_*` settings with safe defaults |
| `docs/PHASE_STATUS.md` | Mark Phase 34D as in-flight |

### 15.2 Configuration additions

```
ADVANCED_VISION_ENABLED=false                # master kill switch (default OFF)
ADVANCED_VISION_ROUTER_ENABLED=true          # router on/off independently
ADVANCED_VISION_SCHEMA_VERSION=1             # bump to invalidate Redis cache
ADVANCED_VISION_PROMPT_VERSION=1             # bump to refresh task prompts
ADVANCED_VISION_TIMEOUT_SECONDS=30           # inherited from VISION_TIMEOUT_SECONDS if 0
ADVANCED_VISION_MAX_IMAGE_BYTES=8388608      # inherited from VISION_MAX_IMAGE_BYTES if 0
ADVANCED_VISION_MAX_IMAGES_PER_REQUEST=2     # cap for comparison
ADVANCED_VISION_CACHE_ENABLED=true
ADVANCED_VISION_CACHE_TTL_SECONDS=3600
ADVANCED_VISION_CACHE_NAMESPACE=advanced_vision
ADVANCED_VISION_PROCESS_CACHE_ENABLED=false  # optional in-process LRU; default OFF
ADVANCED_VISION_PROCESS_CACHE_MAXSIZE=64
```

Provider configuration continues to use the existing Phase 34B
variables: `VISION_PROVIDER`, `VISION_BASE_URL`, `VISION_MODEL`,
`VISION_TIMEOUT_SECONDS`, `VISION_MAX_RETRIES`, `VISION_API_KEY`.

### 15.3 Task taxonomy

```
AdvancedVisualTask (enum, lowercase values for JSON stability)
  general_visual             # default when no stronger signal
  ui_state_analysis          # "which button is disabled?"
  chart_analysis             # "what trend does this chart show?"
  diagram_analysis           # "what connects A to B?"
  table_visual_analysis      # "which row is highlighted?"
  image_comparison           # two-image flow (hash_a, hash_b in cache key)
```

`classify_visual_task(question, image_count=1)` is deterministic,
regex-based, conservative. Identifier-only queries return
`general_visual` (OCR handles them, advanced Vision adds nothing).

### 15.4 VisualReasoningResult schema

```python
@dataclass
class VisualReasoningResult:
    task_type: str                       # AdvancedVisualTask value
    summary: str                         # ≤ 280 chars
    observations: List[str]              # ≤ 8, each ≤ 200 chars
    anomalies: List[str]                 # ≤ 6, each ≤ 200 chars
    relationships: List[Dict[str, str]]  # [{"from": "...", "to": "...", "label": "..."}]
    ui_states: List[Dict[str, Any]]      # [{"kind": "disabled-control", "label": "Apply", "evidence": "..."}]
    chart_findings: List[Dict[str, Any]] # [{"kind": "spike", "location": "14:00", "evidence": "..."}]
    diagram_findings: List[Dict[str, Any]]  # [{"nodes": [...], "edges": [...]}]
    table_findings: List[Dict[str, Any]] # [{"headers": [...], "abnormal_row_index": 3}]
    comparison_changes: List[Dict[str, Any]]  # [{"subject": "Server B", "before": "Healthy", "after": "Failed"}]
    entities: List[str]                  # ≤ 12
    confidence: Optional[int]            # 0..100
    provider: str
    model: str
    processing_time_ms: int
    schema_version: int
    image_ids: List[int]                 # 1 for single-image tasks, 2 for IMAGE_COMPARISON
    evidence_refs: List[str]             # internal refs, not forwarded to LLM

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisualReasoningResult": ...
```

Hard caps on every list. The provider is instructed to mark uncertainty
explicitly ("appears", "approximately", "uncertain") rather than invent.

### 15.5 Task-specific prompt rules

Every prompt template MUST explicitly forbid:

- inventing product-specific behavior, troubleshooting steps, or
  root cause analysis;
- inferring hidden state (e.g. "the service crashed");
- quoting content not visible in the image;
- producing numeric precision beyond visual confidence;
- mapping identifiers to authoritative meanings (that's KB work).

Each prompt asks for STRICT JSON matching `VisualReasoningResult`.
The provider parser is lenient (the Phase 34B `_parse_json_lenient`
pattern is reused) — malformed responses degrade to a safe
`VisualReasoningResult(task_type=general_visual, summary="provider
returned unparseable response")` so the LLM still gets a citation.

### 15.6 Provider design (extended, not duplicated)

The existing Phase 34B provider protocol is extended with optional
parameters:

```python
class VisionProvider(Protocol):
    name: str
    model: str
    def analyze_image(
        self, *,
        image_bytes: bytes,                # required, single image
        mime_type: str,                    # required
        prompt: str,
        ocr_text: str = "",
        context_hint: str = "",
        max_tokens: int = 600,
        timeout_s: Optional[float] = None,
        # Phase 34D additions (optional, backward-compatible default None):
        task: Optional[str] = None,        # AdvancedVisualTask value
        images: Optional[List[Tuple[bytes, str]]] = None,  # for IMAGE_COMPARISON
    ) -> VisionResult: ...
```

Phase 34D's `MockVisionProvider` and `OpenAICompatibleVisionProvider`
accept the new parameters. When `task` is `None`, behavior is
byte-identical to Phase 34B. When `task` is set, the prompt builder
uses the task-specific template. When `images` is provided (length 2),
the prompt builder uses the IMAGE_COMPARISON template and both image
data URLs are sent.

Phase 34D maps the structured `VisualReasoningResult` into
`VisionResult` (so the rest of the Phase 34B pipeline is unchanged):
the result's `summary` becomes `description`; `observations` +
`chart_findings` + `ui_states` etc. become `visual_findings`; the
raw `VisualReasoningResult` JSON is stored under a Phase 34D-specific
metadata field for downstream rendering.

### 15.7 Routing logic (extended)

```
INPUT: question, ocr_text, image_count, image_context

1. ADVANCED_VISION_ENABLED=false -> skip entirely (kill switch)
2. ADVANCED_VISION_ROUTER_ENABLED=false -> skip (router off)
3. image_count > ADVANCED_VISION_MAX_IMAGES_PER_REQUEST -> skip (safety cap)
4. classify_visual_task(question, image_count) -> task_type
5. if task_type == general_visual and no specific signal -> skip
6. if the question is identifier-only (delegated to Phase 34B's
   identifier detector) -> skip (OCR suffices)
7. if the question is product-meaning / troubleshooting -> skip
   (KB authority preserved; advanced vision is observation-only)
8. else -> call run_advanced_visual_reasoning()
```

The advanced router NEVER replaces Phase 34B. The Phase 34B router
runs first; if it returns `OCR_ONLY`, advanced vision is not called
unless the question text carries a strong task-specific signal
(chart words, diagram words, comparison words).

OCR-first is preserved: when OCR is sufficient for an identifier
lookup, advanced vision does NOT run. This is enforced by routing
tier 6 — identifier-only questions with high OCR confidence skip
advanced vision even when `ADVANCED_VISION_ENABLED=true`.

### 15.8 RBAC enforcement (two gates)

Gate 1 — BEFORE MinIO fetch:

```python
authorized = can_access_document(auth, image.document_id)
if not authorized:
    return Failure(reason="unauthorized", image_id=image_id)
bytes_ = fetch_minio_bytes(image.storage_key)
```

Gate 2 — BEFORE provider call (defense in depth, especially for
historical images that the chat endpoint did not directly authorize):

```python
# re-check immediately before calling the provider
if not can_access_document(auth, image.document_id):
    return Failure(reason="unauthorized_pre_provider", image_id=image_id)
result = provider.analyze_image(image_bytes=bytes_, ...)
```

For comparison, both gates run for BOTH images. If either image fails
either gate, the comparison request returns `Failure` with NO image
bytes, NO filenames, NO comparison result, NO existence leak.

### 15.9 Citation behavior

- Single-image task → the synthetic chunk's existing `[N]` marker is
  reused; `citation.citation_kind = "advanced_vision"` is added
  alongside the existing `vision` metadata fields.
- Comparison → TWO `[N]` markers (one per image); the synthesized
  answer cites both.
- For each image: `citation.image_id`, `citation.source_file_name`,
  `citation.vision_task_type`, `citation.confidence`.

### 15.10 Implementation sequence

1. **Foundation (no provider calls):**
   1. Add `ADVANCED_VISION_*` settings to `app/core/config.py` and
      `.env.example`.
   2. Add `AdvancedVisualTask` enum + `VisualReasoningResult` schema +
      `to_dict()` / `from_dict()` JSON helpers.
   3. Add task-classifier (`classify_visual_task`) with deterministic
      regex.
   4. Add prompts module with 6 task templates + strict grounding rules.
2. **Cache + provider extension:**
   5. Add Redis-backed cache with deterministic key.
   6. Extend `VisionProvider.analyze_image()` with optional `task`
      and `images` parameters.
   7. Update `OpenAICompatibleVisionProvider` and `MockVisionProvider`
      to accept the new parameters.
   8. Add `_image_fetch.fetch_authorized_image_bytes()` (RBAC → MinIO).
3. **Orchestrator + comparison:**
   9. Add advanced orchestrator: RBAC → cache → provider → evidence.
   10. Add comparison helper (two-image flow, order-preserving cache key).
   11. Extend `image_resolver.extract_comparison_targets()`.
4. **Integration:**
   12. Wire orchestrator into `app/vision/orchestrator.py`.
   13. Wire `app/rag/answer_generator.py` to call advanced vision
       when task classifier fires.
   14. Extend `ImageEvidence` and `vision/integration.py` to surface
       `advanced_reasoning` on the synthetic chunk.
   15. Update `app/api/chat.py` and `app/schemas/chat.py` docs to
       describe the extended `image_context` shape.
5. **Tests + docs:**
   16. Write unit tests (≥ 24 scenarios per brief).
   17. Write live E2E A-H.
   18. Regression: full Phase 34A/34B/34C/34C.1 suites must pass.
   19. Write `docs/phase34d-advanced-visual-understanding.md`.
6. **Validation:**
   20. `alembic current` shows `013` head (no migration added).
   21. Backend health check passes.
   22. No secrets / debug endpoints / test-only credentials remain.

### 15.11 Risks

| Risk | Mitigation |
| --- | --- |
| Provider returns malformed JSON | Lenient parser (reuses Phase 34B pattern); degrade to `summary="provider returned unparseable response"` |
| Provider invents root cause | Strict grounding rules in every prompt; OCR-first + cap on advanced vision |
| Redis outage | `try/except` around every Redis call; orchestrator treats outage as cache miss and proceeds to provider call |
| Corrupt cached JSON | Length / type validation in `from_dict()`; treated as cache miss |
| Schema drift breaks cache | `ADVANCED_VISION_SCHEMA_VERSION` baked into cache key; bump to invalidate |
| Multi-image auth leak | Every image independently passes `can_access_document` BEFORE MinIO fetch AND BEFORE provider call |
| Comparison key inversion | A→B order preserved (NOT lexicographically sorted) in cache key |
| Provider cost spike | Router gates by task classifier; OCR-first; cache; cap on images per request; `ADVANCED_VISION_ENABLED=false` kill switch |
| Phase 34B regression | New `task` and `images` parameters on `VisionProvider.analyze_image()` are optional with `None` defaults; existing Phase 34B call sites unchanged |
| Frontend scope creep | Multi-image UI deferred to Phase 34D.1 |
| LangSmith payload growth | Advanced vision metadata is small + bounded; no image bytes, no OCR body, no raw provider payload |
| Migration drift | Skipping migration 014 keeps the head stable; Phase 34D.1 can add a clean dedicated table if needed |

### 15.12 Test strategy

**Unit (hermetic, ≥24 scenarios from the brief):**

1. Text-only OCR question does NOT call advanced Vision.
2. UI-state query routes to `ui_state_analysis`.
3. Chart query routes to `chart_analysis`.
4. Diagram query routes to `diagram_analysis`.
5. Comparison query routes to `image_comparison` (when 2 images).
6. Product-meaning query does NOT rely on advanced Vision alone.
7. UI result correctly identifies failed/red status.
8. Disabled control represented correctly.
9. Chart spike represented without invented root cause.
10. Diagram relationship represented correctly.
11. Table abnormal row represented correctly.
12. Provider structured result parsed safely.
13. Provider malformed result fails safely.
14. Provider timeout fails safely.
15. Cache prevents duplicate provider call.
16. Cache key changes when task type changes.
17. Cache key preserves A→B order for comparison.
18. Unauthorized image rejected before MinIO fetch.
19. Unauthorized image rejected before provider call (defense in depth).
20. Comparison fails safely if either image unauthorized.
21. Citations contain correct image IDs.
22. Comparison citations include both images.
23. Kill switch (`ADVANCED_VISION_ENABLED=false`) preserves Phase 34B behavior.
24. No advanced Vision call for ordinary text RAG.
25. Historical image retrieved by Phase 34C can be advanced-analyzed.
26. Deterministic Phase 34C.1 synthesis remains unchanged.
27. Redis outage degrades safely (cache miss → provider call).
28. Corrupt cached JSON degrades safely (cache miss → provider call).
29. `VisualReasoningResult.to_dict()` excludes image bytes and raw payloads.
30. `VisualReasoningResult.from_dict()` validates length caps.

**Live E2E (against the Docker stack):**

- **TEST A** — UI screenshot: server statuses + disabled Apply button → "What visually looks wrong?" → identifies Failed server + disabled control, no root cause, image citation.
- **TEST B** — Chart screenshot: spike around 14:00 → "What trend or anomaly do you see?" → spike + decline, no fabricated cause, image citation.
- **TEST C** — Diagram: Internet → Nginx → Backend → PostgreSQL → "What flow does this diagram show?" → visible relationships, no invented ports/protocols, image citation.
- **TEST D** — OCR guard: 902 screenshot → "What error code is shown?" → `advanced_vision_called=false`.
- **TEST E** — Product authority: 902 screenshot → "What does error 902 mean and how should I fix it?" → KB first, image citation identifies 902 but does not invent meaning/fix.
- **TEST F** — Cache: same visual-analysis query twice → first `advanced_vision_called=true`, second `advanced_vision_called=false`, `advanced_vision_cache_hit=true`.
- **TEST G** — Failure resilience: forced provider failure → backend healthy, OCR/persisted evidence retained, no fabricated visual answer, no 500.
- **TEST H** — RBAC: private image owned by User A → User B's visual question → no provider call, no MinIO fetch, zero information leakage.

**Optional comparison E2E (Phase 34D, backend only; frontend multi-image deferred to 34D.1):**

- Two authorized images via `image_context.comparison = [{document_id, image_id}, ...]`.
- "What changed between these screenshots?" → Server B Healthy → Failed, both citations present.
- Cache key order-preserved (A,B ≠ B,A; verified by asserting that swapping inputs causes a cache miss).

**Regression:**

- Phase 34A OCR suite.
- Phase 34A.1 retrieval.
- Phase 34A.2 attachment.
- Phase 34B Vision suite.
- Phase 34C image knowledge suite.
- Phase 34C.1 deterministic synthesis suite.

All existing tests must pass unchanged.

**Observability assertions:**

- `vision_meta.advanced_vision_task_type` matches classifier output.
- `vision_meta.advanced_vision_called` reflects actual provider call.
- `vision_meta.advanced_vision_cache_hit` reflects actual cache lookup.
- `vision_meta.advanced_vision_latency_ms` is non-negative.
- No image bytes, JWTs, or API keys in any log line.

---

## 16. Acceptance check-list (mapping to brief)

| Brief criterion | Implementation mapping |
| --- | --- |
| A. OCR-only questions still avoid Vision | Phase 34B router unchanged; advanced vision gated behind explicit task signal + identifier-lookup check |
| B. UI-state reasoning works | `ui_state_analysis` task + prompt + mock outputs |
| C. Chart reasoning works | `chart_analysis` task + prompt + mock outputs |
| D. Diagram reasoning works | `diagram_analysis` task + prompt + mock outputs |
| E. Advanced Vision never invents product root cause | Strict grounding rules in every prompt; OCR-first; KB authority preserved |
| F. KB remains authoritative | Phase 34C.1 demote + Phase 34B provider prefix unchanged |
| G. RBAC enforced before image bytes reach Vision | Two-step RBAC: chat endpoint gates, orchestrator re-checks before MinIO fetch AND before provider call |
| H. Image citations correct | Single-image task: 1 citation; comparison: 2 citations |
| I. Advanced Vision cache prevents repeated calls | Redis-backed cache keyed by `(schema_version, provider, model, task_type, image_content_hash[, hash_b])` |
| J. Provider failure does not break backend/chat | Provider errors caught and degraded; OCR/persisted evidence retained; no 500 |
| K. Phase 34C historical image retrieval can feed advanced reasoning | `run_advanced_visual_reasoning()` accepts image_id and bypasses chat attachment flow |
| L. Phase 34C.1 deterministic synthesis still passes | Regression suite; advanced vision is additive; refusal recovery untouched |
| M. Kill switch restores previous behavior | `ADVANCED_VISION_ENABLED=false` short-circuits the layer |
| N. Live E2E A-H pass | Test scenarios per §15.12 |
| O. Backend healthy | Health check passes after every change |
| P. Alembic current/head verified | Head stays at `013` (no migration added) |
| Q. No secrets / debug endpoints / test-only credentials remain | Review pass; `.env` keys never logged; LangSmith sanitization active |
| R. Regression suite has no new failures | All existing Phase 34A/B/C/C.1 tests pass |

---

## 17. Rollback

```
ADVANCED_VISION_ENABLED=false   # restores Phase 34B / 34C / 34C.1 behaviour
```

No code revert needed. No DB migration to downgrade. Redis cache
evaporates on TTL expiry (or via `redis-cli del advanced_vision:*`).
Phase 34D's kill switch is the primary rollback path; the worst case
(provider causes issues) is one env-var flip and a backend restart.

If a code revert is ever required:

```bash
git checkout 7aca63f -- apps/backend/app/services/advanced_vision/ \
                        apps/backend/app/vision/orchestrator.py \
                        apps/backend/app/vision/integration.py \
                        apps/backend/app/vision/evidence.py \
                        apps/backend/app/services/vision/base.py \
                        apps/backend/app/services/vision/openai_compatible_provider.py \
                        apps/backend/app/services/vision/mock_provider.py \
                        apps/backend/app/rag/answer_generator.py \
                        apps/backend/app/rag/image_resolver.py \
                        apps/backend/app/core/config.py
```

No Alembic downgrade is needed because no migration was added.

---

## 18. Summary

Phase 34D is feasible as an additive layer over Phase 34B + 34C
without:

- any new migration (Redis is the cache);
- any new vector collection;
- any LLM-based routing decision;
- any change to OCR-first discipline;
- any duplicate provider framework (extends Phase 34B's existing one);
- any change to Phase 34B's existing provider / cache / prompt;
- any frontend change (multi-image UI deferred to 34D.1).

The architecture audit confirms the existing seams are in the right
places: the provider abstraction is extensible with optional
parameters, the Redis service is already wired (rate_limit), the
synthetic-chunk integration is type-safe, and RBAC is already
enforced upstream. Phase 34D adds a structured, task-aware reasoning
layer on top while preserving every Phase 34A → 34C.1 guarantee.

**Implementation in progress.**
