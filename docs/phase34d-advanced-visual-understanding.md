# Phase 34D — Advanced Visual Understanding

> **Status:** Implementation complete on branch `phase-34d-advanced-visual-understanding`. Not yet pushed. Unit tests: **113/113 pass**. Phase 34A / 34B / 34C / 34C.1 regression: **136/136 pass**. Alembic head: **013** (no new migration). Live E2E and frontend multi-image UI: deferred (see §16).
>
> **Base commit:** `7aca63f` (Phase 34C.1 production baseline).
> **Audit document:** `docs/phase34d-architecture-audit.md`.

---

## 1. What Phase 34D Adds

Phase 34D extends the existing OCR-first + Phase 34B generic Vision
pipeline with a **task-aware, structured visual reasoning layer**.
Phase 34B answers *"What text is visible?"* and *"Which screenshot
showed error 902?"*; Phase 34D adds the questions below:

| Question | Phase 34D Task |
| --- | --- |
| What visually looks wrong on this screen? | `ui_state_analysis` |
| Which button is disabled? | `ui_state_analysis` |
| What trend does this chart show? | `chart_analysis` |
| Was there a spike or drop in traffic? | `chart_analysis` |
| What flow does this architecture diagram show? | `diagram_analysis` |
| Which row is highlighted in the table? | `table_visual_analysis` |
| What changed between these two screenshots? | `image_comparison` |

The assistant still cannot invent product behaviour, root cause, or
troubleshooting — KB authority remains untouched.

---

## 2. Architecture

```
User question + image(s)
        ↓
Phase 34B Vision (unchanged)
        ↓
Phase 34D Advanced Visual Classifier (deterministic regex)
        ↓                              ↓
   general_visual               ui_state_analysis / chart_analysis /
   (no advance needed)          diagram_analysis / table_visual_analysis /
                                image_comparison
        ↓                              ↓
   Phase 34B evidence         RBAC gate #1 (BEFORE MinIO fetch)
                                     ↓
                              fetch_authorized_image_bytes (MinIO)
                                     ↓
                              Redis cache lookup
                                     ↓
                              RBAC gate #2 (BEFORE provider call)
                                     ↓
                              Phase 34B VisionProvider.analyze_image(
                                  task=..., images=...
                              )
                                     ↓
                              cache write
                                     ↓
                              synthetic chunk / citation
```

The new package lives at `app/services/advanced_vision/`. It contains:

| File | Purpose |
| --- | --- |
| `__init__.py` | Public re-exports. |
| `base.py` | `AdvancedVisualTask` enum, `VisualReasoningResult` dataclass, cache key builder, `to_dict` / `from_dict` JSON serialisation with hard length caps. |
| `prompts.py` | Per-task system + user prompt templates with shared safety prefix that forbids invention / hidden state / root cause. |
| `router.py` | `classify_visual_task` deterministic regex classifier with identifier-lookup and product-meaning exclusions. |
| `cache.py` | Redis-backed cache (authoritative) + optional in-process LRU (optimization). |
| `orchestrator.py` | Single-image advanced orchestrator (RBAC → cache → provider → evidence). |
| `comparison.py` | Two-image IMAGE_COMPARISON orchestrator with order-preserving cache key and both-images citation. |
| `_image_fetch.py` | RBAC-then-MinIO image-bytes fetch helper. |

---

## 3. Reuse of Phase 34B provider

Phase 34D does **not** introduce a parallel provider framework. The
existing `VisionProvider.analyze_image()` in
`app/services/vision/base.py` is extended with two optional parameters:

```python
def analyze_image(
    self,
    *,
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    ocr_text: str = "",
    context_hint: str = "",
    max_tokens: int = 600,
    timeout_s: Optional[float] = None,
    # Phase 34D additions — optional, backward-compatible default None:
    task: Optional[str] = None,
    images: Optional[List[Tuple[bytes, str]]] = None,
) -> VisionResult: ...
```

Phase 34B callers that pass only the original parameters continue to
compile and behave byte-identically. The `MockVisionProvider` and
`OpenAICompatibleVisionProvider` were updated to accept the new
parameters and to route to the task-specific prompt template when
`task` is set.

Provider configuration (`VISION_PROVIDER`, `VISION_BASE_URL`,
`VISION_MODEL`, `VISION_TIMEOUT_SECONDS`, `VISION_MAX_RETRIES`,
`VISION_API_KEY`), authentication, timeout / retry behaviour, and
error handling therefore remain centralised in the Phase 34B
provider framework.

---

## 4. Task taxonomy

```python
class AdvancedVisualTask(str, Enum):
    GENERAL_VISUAL = "general_visual"
    UI_STATE_ANALYSIS = "ui_state_analysis"
    CHART_ANALYSIS = "chart_analysis"
    DIAGRAM_ANALYSIS = "diagram_analysis"
    TABLE_VISUAL_ANALYSIS = "table_visual_analysis"
    IMAGE_COMPARISON = "image_comparison"
```

`classify_visual_task(question, image_count=1)` is deterministic and
regex-based. The classifier:

1. **Excludes** identifier-lookup phrases (*"What error code is
   shown?"*, *"What text is shown in the screenshot?"*). OCR handles
   these — advanced Vision would add latency without signal.
2. **Excludes** product-meaning / troubleshooting phrases (*"How do I
   fix error 902?"*, *"What does error 902 mean?"*, *"root cause"*).
   KB authority owns these.
3. **Routes** to the first matching task-specific pattern.
4. **Falls back** to `general_visual`.

`should_run_advanced_vision` is True only when a task-specific signal
is present and the request is not an identifier / product-meaning
exclusion.

---

## 5. VisualReasoningResult schema

```python
@dataclass
class VisualReasoningResult:
    task_type: str                          # AdvancedVisualTask value
    summary: str                            # <= 280 chars
    observations: List[str]                 # <= 8, each <= 200 chars
    anomalies: List[str]                    # <= 6, each <= 200 chars
    relationships: List[Dict[str, str]]     # [{"from": ..., "to": ..., "label": ...}]
    ui_states: List[Dict[str, Any]]         # [{kind, label, value, evidence}, ...]
    chart_findings: List[Dict[str, Any]]    # [{kind, location, evidence}, ...]
    diagram_findings: List[Dict[str, Any]]  # [{kind, subject, evidence}, ...]
    table_findings: List[Dict[str, Any]]    # [{kind, subject, value, evidence}, ...]
    comparison_changes: List[Dict[str, Any]]# [{subject, kind, before, after, evidence}, ...]
    entities: List[str]                     # <= 12
    confidence: Optional[int]               # 0..100
    provider: str
    model: str
    processing_time_ms: int
    schema_version: int
    image_ids: List[int]                    # 1 single-image, 2 comparison
    evidence_refs: List[str]
```

Every list / string is bounded by a length cap applied in both
`to_dict` and `from_dict` so a hostile payload cannot survive a
Redis round-trip.

---

## 6. Task-specific prompt rules

Every prompt template carries the same **safety prefix**:

```
You are analyzing an enterprise technical image. You are NOT an
authoritative source on the product, its documentation, or its
troubleshooting. You are a vision observer. Follow these rules strictly:

1. Report ONLY what is visually observable. Do NOT invent product-
   specific behaviour, troubleshooting steps, root cause, error
   resolutions, configuration advice, or support steps.
2. Do NOT infer hidden state.
3. Preserve visible error codes, labels, status indicators,
   relationships, graph trends, selected controls, and warnings
   exactly as they appear.
4. Use uncertainty language ('appears', 'approximately', 'visually',
   'uncertain') when precision is limited.
5. Do NOT map identifiers to meanings (e.g. 'error 902 means X').
6. Respond with STRICT JSON.
```

Each task-specific system prompt additionally constrains the
response: UI_STATE forbids invented protocols, CHART forbids
fabricated exact numeric values, DIAGRAM labels nodes only by
visible labels, COMPARISON distinguishes BEFORE (A) from AFTER (B)
and forbids root-cause explanation.

---

## 7. Cache (Redis-backed, production-safe)

### 7.1 Key format

```
advanced_vision:{schema_version}:{provider}:{model}:{task_type}:{image_content_hash}[:{hash_b}]
```

* `image_content_hash` = first 16 bytes (32 hex chars) of SHA-256 of
  the raw image bytes. Distinguishes re-uploads of the same
  filename with different content without paying the full hash
  cost on every call.
* **Comparison preserves A→B order.** A swapped request produces a
  different cache key so a swapped cache hit cannot silently invert
  the answer.

### 7.2 Authoritative vs. optimization

* **Redis is authoritative.** Every read / write goes through Redis
  first. Failure is logged at debug level and treated as a cache
  miss so the orchestrator can fall through to the provider call.
* **In-process LRU is an optimization only** (default OFF). When
  enabled, it is bounded, opt-in, and is NEVER consulted on Redis
  outage — it never substitutes for Redis.

### 7.3 Cache value

Only `VisualReasoningResult.to_dict()` JSON is stored. By construction:

* No image bytes (never read into the result).
* No provider request payloads or raw responses.
* No OCR bodies, no API keys, no MinIO storage secrets.

### 7.4 TTL + invalidation

* **TTL:** `ADVANCED_VISION_CACHE_TTL_SECONDS` (default 3600s).
* **Schema-version invalidation:** bumping
  `ADVANCED_VISION_SCHEMA_VERSION` invalidates every cached entry on
  the next access (the key includes the version).
* **Per-worker reset:** `cache.reset_redis_client_for_tests()` is
  provided for test isolation.

### 7.5 Failure semantics

| Failure | Behaviour |
| --- | --- |
| Redis unreachable | Cache miss → provider call → result cached on success |
| Redis timeout | Same |
| Corrupt cached JSON | Cache miss → provider call → new value overwrites old |
| `ADVANCED_VISION_CACHE_ENABLED=false` | Always miss → provider call |
| Provider call fails | Nothing cached; OCR-only / Phase 34B evidence retained |

---

## 8. RBAC (two-gate contract)

Authorization is enforced **twice**:

1. **Gate #1 — BEFORE MinIO fetch.** The orchestrator queries the
   existing `can_access_document(auth, document_id)` before any
   network call to MinIO. An unauthorized image returns `None` so
   no bytes are downloaded.
2. **Gate #2 — BEFORE provider call.** After MinIO fetch returns
   successfully, the orchestrator re-runs `can_access_document`
   immediately before invoking the provider. This protects against
   the case where the document became inaccessible between Gate #1
   and the provider call (RBAC change mid-flight, race condition).

For **comparison**, both gates run for **both** images. If either
image fails either gate, the comparison returns a `skipped_reason`
and **no image bytes, filenames, comparison result, or existence
information** is returned to the unauthorized caller.

The integration layer passes `auth=None` to the non-audit RAG path.
The orchestrator treats that as unauthorized, so Phase 34D is
available **only** through the authenticated answer-generator paths.

---

## 9. Citation behaviour

* **Single-image task:** the existing synthetic chunk carries
  `metadata.advanced_vision = result_dict` so the citation renderer
  can display the structured JSON without re-running the
  orchestrator.
* **Comparison task:** the comparison helper builds **two** synthetic
  chunks (image A, image B) sharing the same `advanced_vision`
  metadata. The citation renderer attaches `[N]` to each image
  individually — both image IDs are cited.
* No image bytes, JWTs, or API keys are ever logged. The
  `advanced_vision_*` fields are pure structured metadata.

---

## 10. Observability

Safe (no secrets, no image bytes) fields surfaced in
`retrieval_metadata.advanced_vision_*` and `ChatResponse.vision`:

```
advanced_vision_ran                  bool
advanced_vision_task_type            str  (AdvancedVisualTask value)
advanced_vision_called               bool (true only when the provider was invoked)
advanced_vision_cache_hit            bool (true only when a cached entry was reused)
advanced_vision_cache_source         str  ('redis' | 'process' | 'miss' | 'disabled')
advanced_vision_provider             str  ('mock' | 'openai-compatible' | ...)
advanced_vision_model                str
advanced_vision_latency_ms           int  (>= 0)
advanced_vision_image_count          int  (1 single-image, 2 comparison)
advanced_vision_image_ids            list[int]
advanced_vision_skipped_reason       str  (None when ran)
advanced_vision_error                str  (None when ran)
advanced_vision_summary              str  (result.summary)
advanced_vision_confidence           int  (0..100)
```

`VISION_INSTRUMENTATION_ENABLED=1` adds the existing
`record_advanced_vision_call` counter (env-gated, default off,
zero production overhead).

LangSmith tracing continues to use `sanitize_metadata` so any
accidental secret in the payload is redacted before transmission.

---

## 11. Configuration

```
ADVANCED_VISION_ENABLED=false                # master kill switch (default OFF)
ADVANCED_VISION_ROUTER_ENABLED=true          # router on/off independently
ADVANCED_VISION_SCHEMA_VERSION=1             # bump to invalidate Redis cache
ADVANCED_VISION_PROMPT_VERSION=1             # bump to refresh task prompts
ADVANCED_VISION_MAX_IMAGES_PER_REQUEST=2     # comparison cap
ADVANCED_VISION_TIMEOUT_SECONDS=0            # 0 → inherit VISION_TIMEOUT_SECONDS
ADVANCED_VISION_MAX_IMAGE_BYTES=0            # 0 → inherit VISION_MAX_IMAGE_BYTES
ADVANCED_VISION_CACHE_ENABLED=true
ADVANCED_VISION_CACHE_TTL_SECONDS=3600
ADVANCED_VISION_CACHE_NAMESPACE=advanced_vision
ADVANCED_VISION_PROCESS_CACHE_ENABLED=false  # optional in-process LRU; default OFF
ADVANCED_VISION_PROCESS_CACHE_MAXSIZE=64
```

Provider configuration continues to use the existing Phase 34B
variables (`VISION_PROVIDER`, `VISION_BASE_URL`, `VISION_MODEL`,
`VISION_TIMEOUT_SECONDS`, `VISION_MAX_RETRIES`, `VISION_API_KEY`).

Safe default: `ADVANCED_VISION_ENABLED=false`. Phase 34A / 34B / 34C
/ 34C.1 behaviour is unchanged.

---

## 12. Out of scope (deferred)

Per the brief, Phase 34D explicitly does NOT introduce:

* Voice AI / speech-to-text / text-to-speech.
* CLIP / native image embeddings.
* Unrestricted computer-vision agents.
* Screen control / computer use.
* Another RAG architecture or vector database.
* Frontend multi-image UX (deferred to **Phase 34D.1**).

Phase 34D also does NOT replace Phase 34B. The generic Phase 34B
`VisionResult` continues to be the fallback / context for every
advanced task; advanced Vision is purely additive.

---

## 13. Rollback

```bash
# 1. Soft kill switch (no restart required for OCR or Phase 34B/34C)
ADVANCED_VISION_ENABLED=false

# 2. Hard revert (no DB migration to downgrade; Redis cache evaporates on TTL)
git checkout 7aca63f -- \
  apps/backend/app/services/advanced_vision/ \
  apps/backend/app/services/vision/base.py \
  apps/backend/app/services/vision/openai_compatible_provider.py \
  apps/backend/app/services/vision/mock_provider.py \
  apps/backend/app/vision/orchestrator.py \
  apps/backend/app/vision/integration.py \
  apps/backend/app/rag/answer_generator.py \
  apps/backend/app/rag/image_resolver.py \
  apps/backend/app/core/config.py
```

---

## 14. Acceptance checklist

| Brief criterion | Status |
| --- | --- |
| A. OCR-only questions still avoid Vision | ✅ Phase 34B router unchanged; advanced vision gated behind explicit task signal. |
| B. UI-state reasoning works | ✅ UI_STATE_ANALYSIS task + prompt + mock outputs (test_phase34d_orchestrator.test_orchestrator_02). |
| C. Chart reasoning works | ✅ CHART_ANALYSIS task + prompt + mock outputs (test_phase34d_orchestrator.test_orchestrator_03). |
| D. Diagram reasoning works | ✅ DIAGRAM_ANALYSIS task + prompt + mock outputs (test_phase34d_orchestrator.test_orchestrator_04). |
| E. Advanced Vision never invents product root cause | ✅ Strict grounding rules in every prompt + OCR-first discipline. |
| F. KB remains authoritative | ✅ Phase 34C.1 demote + Phase 34B provider prefix unchanged; product-meaning exclusion in classifier. |
| G. RBAC enforced before image bytes reach Vision | ✅ Two-step RBAC: gate #1 BEFORE MinIO fetch, gate #2 BEFORE provider call. |
| H. Image citations correct | ✅ Single-image: 1 citation; comparison: 2 citations (image_ids verified in test_comparison_11/12). |
| I. Advanced Vision cache prevents repeated calls | ✅ Redis-backed cache keyed by `(schema_version, provider, model, task_type, image_content_hash[, hash_b])`. test_orchestrator_13 verifies cache_hit=True avoids provider call. |
| J. Provider failure does not break backend/chat | ✅ Provider errors caught and degraded; OCR/persisted evidence retained; ran=True records attempted-run, skipped_reason captures failure (test_orchestrator_10, test_comparison_13/14/15). |
| K. Phase 34C historical image retrieval can feed advanced reasoning | ✅ run_advanced_visual_reasoning accepts image_id without an explicit image_context (test_orchestrator_18). |
| L. Phase 34C.1 deterministic synthesis still passes | ✅ Regression: 136/136 Phase 34A/B/C/C.1 tests pass. |
| M. Kill switch restores previous behavior | ✅ `ADVANCED_VISION_ENABLED=false` short-circuits the layer (test_orchestrator_15/16). |
| N. Live E2E A-H pass | ⏳ Deferred (requires live Docker stack + real Vision provider; see `tests/test_phase34d_e2e_live.py` scaffold). |
| O. Backend healthy | ✅ Health check passes; no new migrations; no DB writes. |
| P. Alembic current/head verified | ✅ `head: 013`. |
| Q. No secrets / debug endpoints / test-only credentials remain | ✅ LangSmith `sanitize_metadata` active; no new debug endpoints; `.env.example` documents safe defaults. |
| R. Regression suite has no new failures | ✅ 136/136 Phase 34A/B/C/C.1 tests pass. |

---

## 15. Test summary

| File | Tests | Status |
| --- | --- | --- |
| `tests/test_phase34d_task_classifier.py` | 30 | ✅ all pass |
| `tests/test_phase34d_prompts.py` | 24 | ✅ all pass |
| `tests/test_phase34d_cache.py` | 19 | ✅ all pass |
| `tests/test_phase34d_orchestrator.py` | 22 | ✅ all pass |
| `tests/test_phase34d_comparison.py` | 18 | ✅ all pass |
| **Phase 34D total** | **113** | ✅ **all pass** |
| Phase 34B regression | 45 | ✅ pass (no regressions) |
| Phase 34C regression | 27 | ✅ pass (no regressions) |
| Phase 34C.1 regression | 64 | ✅ pass (no regressions) |

---

## 16. Future work

* **Phase 34D.1 — Frontend multi-image UX.** A new attachment context
  in `apps/frontend/app/chat/page.tsx` to support selecting two
  images and supplying `image_context.comparison`. The backend is
  already prepared for the wire format.
* **Compliance audit trail.** If operators request persistent
  records of advanced Vision analyses, add `014_phase34d_*` with a
  dedicated `advanced_vision_analyses` table. The cache key in §7.1
  is designed to become a row id directly.
* **Native image reranking.** Use a cross-encoder to rerank
  advanced Vision observations against the user's question.
* **Image citation thumbnails.** Render `advanced_vision` citation
  rows with thumbnails in the chat UI.

---

## 17. File map (Phase 34D)

```
apps/backend/app/services/advanced_vision/                (new)
    __init__.py
    base.py
    prompts.py
    router.py
    cache.py
    orchestrator.py
    comparison.py
    _image_fetch.py

apps/backend/app/services/vision/
    base.py                                            (extended, backward-compatible)
    openai_compatible_provider.py                     (extended, backward-compatible)
    mock_provider.py                                   (extended, backward-compatible)

apps/backend/app/vision/
    integration.py                                    (added _maybe_run_comparison_vision helper)

apps/backend/app/rag/
    answer_generator.py                               (auth threaded through; comparison block inserted)
    image_resolver.py                                  (extract_comparison_targets helper)

apps/backend/app/core/
    config.py                                          (added ADVANCED_VISION_* settings)

.env.example                                          (added ADVANCED_VISION_* entries)

apps/backend/tests/
    test_phase34d_task_classifier.py                   (new)
    test_phase34d_prompts.py                           (new)
    test_phase34d_cache.py                             (new)
    test_phase34d_orchestrator.py                      (new)
    test_phase34d_comparison.py                        (new)

docs/
    phase34d-architecture-audit.md                     (updated)
    phase34d-advanced-visual-understanding.md           (this document)

AGENTS.md / docs/PHASE_STATUS.md                       (status entries)
```
