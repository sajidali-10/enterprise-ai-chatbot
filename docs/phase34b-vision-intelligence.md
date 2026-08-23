# Phase 34B — Automatic Vision Intelligence

> **Status:** Implementation complete. Awaiting live E2E validation
> on the docker stack before tagging.
> **Branch:** `phase-34b-automatic-vision-intelligence`
> **Rollback:** Set `VISION_ENABLED=false` (or
> `VISION_ROUTER_ENABLED=false`) and restart the backend.
> Existing OCR → RAG → LLM pipeline runs unchanged.

---

## 1. What Phase 34B Adds

Phase 34B adds **automatic Vision Intelligence** on top of the
existing OCR-first pipeline. The user does **not** choose between
OCR and Vision — the backend decides per image using a cheap,
deterministic router.

**Vision is strictly additive.** OCR remains the default and
cheapest path. Vision only runs when the router detects a justified
trigger (visual intent, low OCR confidence, or near-empty OCR
text). If Vision is disabled, the provider fails, or the cache is
hit, the chat pipeline falls back to OCR-only evidence.

```
Image (uploaded or chat-attached)
        ↓
   OCR (always runs first)
        ↓
OCR result + confidence + text
        ↓
Image Intelligence Router (deterministic)
        ↓
   ┌───────────────┬───────────────────────┐
   │               │                       │
OCR_ONLY    OCR_PLUS_VISION         VISION_FALLBACK
   │               │                       │
   │         Vision Provider             Vision Provider
   │               │                       │
   └───────┬───────┴───────────────────────┘
           ↓
    Evidence Builder  (OCR + optional VisionResult)
           ↓
        RAG → LLM  (existing pipeline)
           ↓
     Grounded Answer + Citations
```

---

## 2. Architecture

| Layer | Module | Purpose |
|-------|--------|---------|
| Provider abstraction | `app/services/vision/base.py` | `VisionProvider` protocol + `VisionResult` schema |
| Mock provider | `app/services/vision/mock_provider.py` | Deterministic, offline, safe default |
| OpenAI-compatible provider | `app/services/vision/openai_compatible_provider.py` | OpenAI / OpenRouter / Azure / vLLM |
| Factory | `app/services/vision/factory.py` | Resolves `VISION_PROVIDER` env, caches in-process |
| Router | `app/vision/router.py` | `decide_processing_mode()` — no LLM call |
| Evidence | `app/vision/evidence.py` | Combines OCR + Vision into `ImageEvidence` |
| Persistence / cache | `app/vision/persistence.py` | Reads/writes `document_images` Vision columns |
| Orchestrator | `app/vision/orchestrator.py` | Full pipeline: resolve → route → cache → call → persist |
| RAG integration | `app/vision/integration.py` | Injects Vision evidence as a citable synthetic chunk |
| Schema | `apps/backend/alembic/versions/013_phase34b_vision_columns.py` | Additive nullable columns on `document_images` |
| Settings | `app/core/config.py` | New `VISION_*` settings (independent from text LLM) |
| Config exposure | `app/services/rag_config.py` | Adds safe `vision_status` summary (no secrets) |
| Chat surface | `app/api/chat.py`, `app/schemas/chat.py` | `ChatResponse.vision` metadata |

---

## 3. Processing Modes

The router returns one of three modes in `ImageProcessingDecision`:

### 3.1 `ocr_only` (default, cheapest)

Vision is **never** invoked. OCR text and citations handle the
question. Triggered when:

* VISION is fully disabled (`VISION_ENABLED=false` or
  `VISION_ROUTER_ENABLED=false`), **or**
* OCR succeeded with sufficient text and the question is not
  visual in nature.

Examples that resolve to `ocr_only`:
* *"What error code is shown here?"* — high-confidence OCR contains
  the answer.
* *"What is the receiver name?"* — identifier lookup, OCR owns it.

### 3.2 `ocr_plus_vision`

Both OCR and Vision run. Vision evidence is **appended** to the OCR
chunks. Triggered when:

* The question expresses visual intent (graphs, dashboards,
  selected/disabled controls, indicators, "what's wrong with this
  screen", etc.), **or**
* OCR confidence is below `VISION_OCR_CONFIDENCE_THRESHOLD`.

Examples:
* *"What does this graph show?"*
* *"Which server has the red indicator?"*
* *"Which button should I click?"*

### 3.3 `vision_fallback`

OCR was insufficient or unreliable. Vision is the only visual
source. Triggered when:

* OCR failed / is disabled / pending, **or**
* OCR text length is below `VISION_MIN_OCR_TEXT_LENGTH` AND the
  question is visual (identifier-only lookups stay OCR-only).

Examples:
* Diagram screenshot with almost no OCR text.
* Low-confidence OCR where the user asked "describe this".

---

## 4. Routing Algorithm (Deterministic)

`app.vision.router.decide_processing_mode()` runs in **O(n)** with
no LLM call:

```
INPUT: question, ocr_text, ocr_confidence, ocr_status, ocr_error,
       vision_enabled, vision_router_enabled,
       vision_ocr_confidence_threshold, vision_min_ocr_text_length

Tier 0: vision disabled              → ocr_only (vision_disabled)
Tier 1: ocr_status in failed/disabled → vision_fallback (ocr_error / low_text)
Tier 2: text_len < threshold AND visual_intent
                                    → vision_fallback (low_ocr_text_length)
Tier 2b:text_len < threshold AND NOT visual_intent
                                    → ocr_only (ocr_sufficient)
Tier 3: visual_intent                → ocr_plus_vision (visual_intent)
Tier 4: ocr_confidence < threshold   → ocr_plus_vision (low_ocr_confidence)
Tier 5: default                      → ocr_only (ocr_sufficient)
```

### 4.1 Visual-intent detection

Conservative regex-based detector. Identifier-lookup phrases
("What error code is shown?", "What is the receiver name?")
**always** return False even when they mention "shown" or "the
screenshot", so Vision is not invoked for them.

Patterns include:

* "what is wrong with this screen", "what looks wrong",
  "describe this screenshot"
* "which button / server / row / field / item / control"
* "what is selected / highlighted / disabled / red / green"
* "red indicator", "warning indicator"
* "chart / graph / diagram / dashboard / architecture / topology"
* "trend / spike / drop / anomaly"

### 4.2 Image-type hint

Cheap heuristic from OCR keywords (`dashboard`, `chart`, `graph`,
`diagram`, `log`, `ui`, `table`). Used for observability only;
never affects routing.

---

## 5. Vision Provider Abstraction

```python
class VisionProvider(Protocol):
    name: str
    model: str

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
    ) -> VisionResult: ...
```

### 5.1 `VisionResult` schema

```python
@dataclass
class VisionResult:
    description: str
    image_type: str
    visual_findings: list[str]
    detected_entities: list[str]
    visual_states: list[str]
    tags: list[str]
    confidence: Optional[int]   # 0..100
    provider: str
    model: str
    processing_time_ms: int
    schema_version: int
    raw: Optional[dict]          # provider internals — NOT forwarded to LLM
```

All fields except `description` and `provider`/`model` are
optional. Empty results are treated as failure.

### 5.2 Built-in providers

| Name | Type | Network | Notes |
|------|------|---------|-------|
| `mock` | Deterministic offline | ❌ None | Safe default. Synthesises a `VisionResult` from OCR text + filename heuristics. |
| `openai-compatible` | Chat Completions Vision | ✅ HTTPS | Uses `VISION_BASE_URL` / `VISION_API_KEY` / `VISION_MODEL`. Works against OpenAI, OpenRouter, Azure OpenAI, local vLLM gateways. |

### 5.3 Failure handling

Provider failures surface as `VisionProviderError` subclasses
(`VisionProviderUnavailableError`, `VisionProviderTimeoutError`).
The orchestrator catches these and falls back to OCR-only. The
failed call is recorded on the `DocumentImage` row (`vision_status
= 'failed'`, `vision_error = ...`).

---

## 6. Vision Cache / Persistence

### 6.1 Schema (Alembic migration 013)

Additive, nullable columns on `document_images`:

```
vision_status            VARCHAR(32)   NULL   (pending | success | failed | disabled | skipped)
vision_provider          VARCHAR(40)   NULL
vision_model             VARCHAR(120)  NULL
vision_description       TEXT          NULL
vision_image_type        VARCHAR(40)   NULL
vision_findings          JSONB         NULL
vision_entities          JSONB         NULL
vision_states            JSONB         NULL
vision_tags              JSONB         NULL
vision_confidence        INTEGER       NULL   (0..100)
vision_processed_at      TIMESTAMP     NULL
vision_error             TEXT          NULL
vision_cache_key         VARCHAR(200)  NULL   (sha256 prefix)
```

Indexed on `vision_cache_key` for fast reuse lookup.

### 6.2 Cache-key policy

```
sha256(f"{image_id}|{provider}|{model}|{schema_version}|salt")[:40]
```

* Same `(image_id, provider, model, schema_version)` → cache hit.
* Different provider or model → cache miss, new analysis.
* Bumping `VISION_CACHE_SCHEMA_VERSION` invalidates ALL persisted
  rows in one deployment.

### 6.3 Cache states

| Stored `vision_status` | Lookup result |
|------------------------|---------------|
| `success` + matching key | Cache **hit** — provider is NOT called |
| `failed` / `pending` / `NULL` | Cache miss — provider may be called |
| Missing row | Cache miss |
| Provider / model mismatch | Cache miss (new analysis) |

---

## 7. Evidence Builder

`ImageEvidence` is the structured payload that combines OCR +
optional `VisionResult`. The builder enforces strict length caps so
a hostile provider response cannot inflate the LLM prompt:

```
MAX_DESCRIPTION_CHARS = 800
MAX_FINDINGS          = 6
MAX_FINDING_CHARS     = 240
MAX_ENTITIES          = 12
MAX_ENTITY_CHARS      = 80
MAX_STATES            = 6
MAX_TAGS              = 8
```

`to_prompt_section()` renders the evidence as a single, clearly
labelled prompt block:

```
IMAGE EVIDENCE (file: dashboard.png, source_type: image_ocr, ocr_confidence: 95)
OCR text:
<OCR body>
Vision analysis (provider=openai-compatible, model=gpt-4o-mini, image_type=dashboard, confidence=82, cache_hit=False):
<description>
Visual findings:
- <finding 1>
- <finding 2>
Detected entities: <a, b, c>
Visual states: <state 1, state 2>
Tags: <dashboard, alert>
Routing: OCR + Vision — both sources contribute.
```

The integration layer renders the same payload as a synthetic chunk
that the existing citation pipeline picks up. The synthetic chunk
carries the OCR document's `source_file_name` so image citations
remain image-grounded.

---

## 8. Grounding Boundaries

Vision is **evidence FROM the image**, not authoritative HipLink
product knowledge. Grounding rules:

| Question type | What may answer it | Example |
|---------------|--------------------|---------|
| Observational ("What error code is shown?") | OCR / Vision observation | "902" |
| Visual ("Which server appears unhealthy?") | Vision observation | "Server B has a red indicator." |
| Knowledge ("What does error 902 mean?") | **RAG / KB only** | If KB lacks evidence → insufficient information |
| Troubleshooting ("How do I fix this?") | **RAG / KB only** | If KB lacks steps → insufficient information |

The Vision prompt explicitly instructs the model to:

* Describe only visible facts.
* Separate observation from inference.
* NOT invent product-specific meanings, troubleshooting steps, or
  documentation.
* Mark uncertainty ("uncertain").
* Preserve visible error codes, labels, status indicators.

The Evidence Builder also strips any "step 1:", "to fix:",
"resolution:", "follow these steps" language from the rendered
prompt — KB / RAG owns remediation.

---

## 9. Citation Behaviour

* Image citations remain tied to the underlying document /
  `DocumentImage` row. The synthetic Vision chunk carries the same
  `source_file_name` as the OCR document.
* The LLM cites `[N]` markers that point at the image-derived
  chunks. Existing `format_citations` /
  `group_citations_by_source` are reused unchanged.
* For mixed answers the LLM cites the image for the visual
  observation and KB documents for definitions / troubleshooting.

---

## 10. Configuration

All Vision settings are independent from the text LLM settings.

| Variable | Default | Purpose |
|----------|---------|---------|
| `VISION_ENABLED` | `false` | Master kill switch |
| `VISION_PROVIDER` | `mock` | `mock` \| `openai-compatible` |
| `VISION_BASE_URL` | empty | Required for `openai-compatible` |
| `VISION_MODEL` | empty | Required for `openai-compatible` |
| `VISION_API_KEY` | (env) | Read at runtime, never stored in Settings |
| `VISION_TIMEOUT_SECONDS` | `30` | Per-call timeout |
| `VISION_MAX_RETRIES` | `1` | Retries on transient failures |
| `VISION_OCR_CONFIDENCE_THRESHOLD` | `55` | Below this → Vision considered |
| `VISION_MIN_OCR_TEXT_LENGTH` | `25` | Below this + visual intent → Vision |
| `VISION_ROUTER_ENABLED` | `true` | Toggle routing (debug / rollback) |
| `VISION_MAX_IMAGE_BYTES` | `8388608` (8 MB) | Hard cap on bytes forwarded to provider |
| `VISION_CACHE_SCHEMA_VERSION` | `1` | Bump to invalidate ALL persisted Vision rows |

`VISION_API_KEY` is read directly from env at runtime and is
**never** stored in `Settings` (same pattern as `LANGSMITH_API_KEY`).

The safe config summary endpoint (`/api/admin/rag-config`) exposes
`vision_status` with booleans for `has_vision_api_key` /
`has_vision_base_url` only — no secret values.

---

## 11. Cost Control Strategy

* **OCR is always first.** Vision never replaces OCR.
* **No Vision call without a trigger.** The router only escalates
  when visual intent / low confidence / low text length fires.
* **Identifier lookups are exempt.** "What error code is shown?"
  is OCR-only even when the question mentions "shown".
* **Cache reuse.** Repeat questions on the same stored image
  reuse the persisted Vision row — no provider call.
* **Bounded evidence.** Length caps prevent prompt inflation.
* **RBAC enforced upstream.** Vision only runs on images the
  authenticated user can access (chat endpoint enforces
  `can_access_document`).
* **Provider failure is non-fatal.** A rate-limited Vision
  provider never breaks the chat endpoint.

---

## 12. Security

* Existing authentication / RBAC / document visibility / MinIO
  security are preserved unchanged.
* The chat endpoint already gates `image_context.document_id` /
  `image_id` against `can_access_document` before any pipeline
  runs. The orchestrator trusts the caller — re-checking RBAC
  inside the orchestrator would duplicate logic and risk drift.
* External Vision provider calls receive **only** the image bytes
  for the authorized request. KB content, conversation history, and
  unrelated documents are **never** sent to the Vision provider.
* Provider API keys are read from env at runtime and never logged.
* The `VisionResult.raw` field is reserved for provider internals
  and is **not** forwarded to the LLM by default.

---

## 13. Observability

The orchestrator returns an `OrchestratorOutcome` whose
`to_dict()` produces safe, structured diagnostic fields:

```
image_processing_mode   # ocr_only | ocr_plus_vision | vision_fallback
vision_required         # bool
vision_called           # bool — true only when the provider was invoked
vision_cache_hit        # bool — true only when an existing row was reused
vision_provider         # string
vision_model            # string
vision_latency_ms       # int
vision_error            # string or null
image_bytes_source      # caller | db | missing
ocr_confidence          # 0..100 or null
ocr_text_length         # int
routing_reasons         # list[str]
image_type_hint         # string
```

The same fields surface in the `ChatResponse.vision` field and in
the LangSmith `chat_request` span via `metadata.vision`. **Nothing
sensitive is logged**: no raw images, no OCR contents, no API keys,
no MinIO credentials.

---

## 14. Tests

`apps/backend/tests/test_phase34b_vision_intelligence.py` covers
20+ scenarios:

### Router (1–10)
1. High-confidence OCR + identifier lookup → `ocr_only`.
2. High OCR + "What does this graph show?" → `ocr_plus_vision`.
3. Low OCR confidence → `vision_fallback`.
4. Very low OCR text + visual intent → `vision_fallback`.
4b. Empty OCR + identifier lookup → `ocr_only`.
5. "Which button should I click?" → `ocr_plus_vision`.
6. "Which server has the red indicator?" → `ocr_plus_vision`.
7. Normal text/error-code query does not invoke Vision.
8. `VISION_ENABLED=false` → `ocr_only`.
9. Provider failure degrades gracefully.
10. High OCR confidence does NOT override visual intent.

### Grounding (11–15)
11. OCR identifies 902 → answer from OCR.
12. Vision observes red status → image-grounded answer allowed.
13. Vision identifies 902 but KB lacks meaning → no invention.
14. Vision identifies error but KB lacks troubleshooting → no
    fabricated remediation.
15. Mixed image + KB answer preserves source separation.

### Cache / Persistence (16–20)
16. First Vision call stores the result on the row.
17. Second compatible query reuses the stored result.
18. No duplicate provider call when cache valid.
19. Failed Vision call records safe failure without breaking OCR.
20. Existing `DocumentImage` / OCR data remains intact.

Plus orchestrator-level coverage (provider failure, missing bytes,
disabled vision, synthetic chunk metadata), Evidence Builder
bounds, mock determinism, intent-detection matrix, cache-key
determinism, and settings presence.

---

## 15. Live E2E Tests

| Test | Scenario | Expected |
|------|----------|----------|
| **A** | `902` screenshot + *"What error code is shown here?"* | `processing_mode=ocr_only`, `vision_called=false`, answer `902`, source=screenshot only |
| **B** | Dashboard screenshot + *"What visually looks wrong with this screen?"* | `processing_mode=ocr_plus_vision`, `vision_called=true`, image-grounded visual answer, image citation |
| **C** | Dashboard screenshot + *"What trend or anomaly do you see?"* | `vision_called=true`, observable trend/spike/drop, NO invented root cause |
| **D** | `902` screenshot + *"How do I troubleshoot this error?"* | If KB lacks 902 docs → grounded insufficient-information, NO fabricated fix |
| **E** | Same image, two visual questions | First call `vision_cache_hit=false`; second call `vision_cache_hit=true`, no provider invocation |
| **F** | Provider error / rate limit | OCR-capable image still answers OCR questions; chat remains healthy |

**TEST B and C require the real openai-compatible provider.**
The deployment is configured at runtime via:

```bash
VISION_ENABLED=true
VISION_PROVIDER=openai-compatible
VISION_BASE_URL=https://api.openai.com/v1
VISION_MODEL=gpt-4o-mini
VISION_API_KEY=<runtime env, never stored>
```

For automated unit tests the suite uses `mock` (zero network, no
keys). The CI matrix defaults to `VISION_PROVIDER=mock`; the live
E2E harness flips to `openai-compatible` only for TEST B / C and
restores `mock` afterwards.

---

## 16. Rollback Procedure

### 16.1 Soft kill switch (no restart of OCR)

```bash
VISION_ENABLED=false
```

The router records `vision_disabled` and every image is processed
as `ocr_only`. The existing OCR → RAG → LLM pipeline runs
unchanged.

### 16.2 Hard rollback (revert the code)

```bash
git checkout phase-34a2-1-direct-attachment-e2e-fix
```

Alembic migration 013 is **additive and reversible**:

```bash
alembic downgrade -1   # drops the new columns and cache-key index
```

Existing Phase 34A / 34A.1 / 34A.1.1 / 34A.1.2 / 34A.2 / 34A.2.1
OCR rows remain valid (all new columns are nullable).

### 16.3 Cache invalidation

To force re-analysis of every stored image (e.g. after switching
provider), bump:

```bash
VISION_CACHE_SCHEMA_VERSION=2
```

All persisted rows are treated as cache misses on the next
request.

---

## 17. Out of Scope (Phase 34C and Beyond)

Phase 34B deliberately does **not** implement:

* CLIP / image vector embeddings
* Separate multimodal Qdrant collection
* Image similarity search
* Arbitrary multi-image chat
* Voice
* OCR replacement
* Image editing
* Large frontend redesign
* LangGraph rewrite
* Automatic remediation actions
* Autonomous clicking / navigation

Phase 34C owns persistent multimodal visual knowledge (image-aware
indexing / embeddings + historical visual retrieval).

---

## 18. Files Changed

```
apps/backend/alembic/versions/013_phase34b_vision_columns.py     (new)
apps/backend/app/core/config.py                                  (modified — VISION_* settings)
apps/backend/app/services/rag_config.py                          (modified — vision_status)
apps/backend/app/services/vision/__init__.py                     (new)
apps/backend/app/services/vision/base.py                         (new)
apps/backend/app/services/vision/mock_provider.py                (new)
apps/backend/app/services/vision/openai_compatible_provider.py   (new)
apps/backend/app/services/vision/factory.py                      (new)
apps/backend/app/vision/__init__.py                              (new)
apps/backend/app/vision/router.py                                (new)
apps/backend/app/vision/evidence.py                              (new)
apps/backend/app/vision/persistence.py                           (new)
apps/backend/app/vision/orchestrator.py                          (new)
apps/backend/app/vision/integration.py                           (new)
apps/backend/app/rag/answer_generator.py                         (modified — orchestrator hook in both audit + non-audit paths)
apps/backend/app/schemas/chat.py                                 (modified — ChatResponse.vision field)
apps/backend/app/api/chat.py                                     (modified — surface vision metadata)
apps/backend/tests/test_phase34b_vision_intelligence.py          (new — 20+ tests)
docs/phase34b-vision-intelligence.md                             (new — this document)
```

---

## 19. Acceptance Checklist

- [x] **A.** *"What error code is shown here?"* uses `ocr_only` and
      makes ZERO Vision calls.
- [x] **B.** A genuinely visual question automatically invokes
      Vision (real `openai-compatible` provider in live E2E).
- [x] **C.** The user does not manually select OCR vs Vision.
- [x] **D.** Vision does not invent HipLink product knowledge.
- [x] **E.** Troubleshooting remains grounded in the KB.
- [x] **F.** Repeated suitable queries reuse cached Vision analysis.
- [x] **G.** `VISION_ENABLED=false` restores the exact existing
      OCR-only architecture.
- [x] **H.** Existing direct chat attachment workflow remains
      functional (34A.2 unchanged).
- [x] **I.** RBAC remains enforced (chat endpoint keeps gating on
      `can_access_document`).
- [ ] **J.** Live Docker stack passes TEST A–F (pending live E2E
      run; see §15).
