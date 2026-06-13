# CRAG Readiness Review

**Date:** 2026-06-13
**Phase:** 11.4
**Status:** Architecture Review

---

## Executive Summary

This document reviews the current RAG pipeline and assesses Corrective RAG (CRAG) readiness. The system already implements significant CRAG-like behavior including hybrid retrieval, topic relevance checking, citation enforcement, and fallback blocking. A small gap exists in CRAG decision metadata visibility and optional query expansion for weak retrieval cases.

**Recommendation:** Proceed to Phase 12 (User Authentication). The optional query rewrite/grader step can be added as Phase 11.4B if needed, but is not blocking.

---

## Current RAG Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────────────────┐
│ 1. Query Preprocessing      │
│    - Query rewriter (config)│
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 2. Hybrid Retrieval         │
│    - Vector search (top_k)  │
│    - Keyword search (top_k) │
│    - RRF fusion             │
│    - Optional reranking     │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 3. Permission Filtering     │
│    - Auth context check     │
│    - Document permission    │
│    - Admin bypass           │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 4. Grounding Checks         │
│    - Retrieval guardrail    │
│    - Minimum relevance      │
│    - Topic relevance        │
│    - Answer grounding       │
└─────────────────────────────┘
    │
    ├─→ [BLOCK] → Fallback Message
    │
    ▼
┌─────────────────────────────┐
│ 5. LLM Answer Generation    │
│    - RAG prompt             │
│    - Temperature = 0        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 6. Citation Repair (if miss)│
│    - Retry with strict prompt│
│    - Backend citation attach│
└─────────────────────────────┘
    │
    ├─→ [NO CITATIONS] → Block
    │
    ▼
Answer + Citations + Metadata
```

---

## What Already Qualifies as CRAG-Like Behavior

| Feature | Implementation | CRAG Relevance |
|---------|---------------|----------------|
| **Hybrid retrieval** | Vector + keyword with RRF fusion | Multiple retrieval paths |
| **Permission filtering** | Auth-aware document filtering | Secure retrieval |
| **Minimum relevance threshold** | Configurable score cutoff | Quality gate before LLM |
| **Topic relevance check** | Keyword overlap analysis (Phase 11.2) | Context relevance validation |
| **Fallback for unsupported** | Blocks when evidence insufficient | CRAG-style blocking |
| **Citation enforcement** | Requires [N] citations in answer | Output validation |
| **Citation retry** | Retry with strict prompt if missing | Self-correction |
| **Backend citation repair** | Content overlap citation attachment | Self-correction |
| **Observability logging** | Full audit trail | Traceability |
| **Debug metadata** | Retrieval scores, grounding info | Visibility |

---

## Current Grounding/Fallback Logic

### Block Conditions (apply_grounding_checks)

1. **No chunks retrieved** → `"I don't have enough information in the provided sources..."`

2. **Top score < threshold** (default 0.5) → `"I don't have enough relevant information..."`

3. **Topic not relevant** (keyword overlap < 15%) → Same fallback message

4. **Answer lacks citations** → `"I don't have enough information..."`

### Fallback Message (Same for all cases)
```
"I don't have enough information in the provided sources to answer this question."
```

This is intentional - prevents attackers from learning why a query was blocked.

---

## Current Citation Repair Flow

```
1. Generate answer with RAG prompt
2. Check if answer has citations [N]
   ├─→ YES → Continue
   └─→ NO → Check top_score >= threshold
              ├─→ YES → Retry with strict citation prompt
              │         └─→ Check citations again
              │             ├─→ YES → Continue
              │             └─→ NO → Try backend citation attachment
              └─→ NO → Skip to backend attachment
3. Backend citation attachment (content overlap)
   ├─→ Found matching content → Attach citations
   └─→ No match → Block for lack of citations
4. Final citation count check
```

---

## Current Evaluation Coverage

The evaluation runner tests:
- Docker-related questions with source files
- Topic relevance (unrelated queries should block)
- Citation presence
- Answer quality via keyword matching
- Fallback behavior for unsupported queries

**15/15 tests passing** confirms the current CRAG-like behavior works correctly.

---

## CRAG Gaps and Risks

### Gap 1: Limited CRAG Decision Metadata
**Issue:** Debug mode and observability don't clearly expose CRAG decision flow.

**Current metadata:**
- `blocked`, `block_reason`
- `grounding.chunk_count`, `grounding.top_score`
- `citation_repair` details

**Missing for full CRAG visibility:**
- `retrieval_status`: strong / weak / insufficient
- `correction_attempted`: true / false
- `correction_type`: none / query_rewrite / second_pass / fallback
- `topic_relevance_score`
- `crag_enabled`
- `crag_decision_reason`

**Impact:** Medium - Hard to debug CRAG decisions in production.

### Gap 2: No Query Expansion for Weak Retrieval
**Issue:** When retrieval is weak (low scores) but topic is relevant, system blocks instead of retrying with expanded query.

**Current behavior:**
```
Topic relevant + Weak retrieval = BLOCK
```

**CRAG ideal behavior:**
```
Topic relevant + Weak retrieval = Retry with query expansion
```

**Risk:** False positives on edge cases where relevant content exists but scores are low.

**Mitigation:** The topic relevance check (Phase 11.2) reduces false positives significantly.

### Gap 3: No LLM-Based Context Grading
**Issue:** System uses deterministic heuristics only for grading context quality.

**Current:** Keyword overlap, score thresholds
**CRAG ideal:** Small LLM model grades context relevance

**Risk:** Deterministic checks miss semantic mismatches.

**Mitigation:** Not required for Phase 12. Can be added as Phase 11.4B if needed.

---

## Where Query Rewrite Step Could Be Inserted

```
Current: User Query → retrieve_chunks_with_settings()
                 ↓
New:     User Query → Query Rewriter → retrieve_chunks_with_settings()
```

**Implementation location:** `app/rag/retriever.retrieve_chunks_with_settings()`

**Current support:** Query rewriter is already configured via `RETRIEVAL_QUERY_REWRITER` setting. Default is "none" (no rewrite).

**Future env vars (if enabled):**
```bash
CRAG_USE_QUERY_REWRITE=true
CRAG_REWRITE_PROVIDER=openrouter
CRAG_REWRITE_MODEL=<small-fast-model>
```

**Constraint:** Model name must not be hardcoded. Must be configurable.

---

## Where Context Grader Could Be Inserted

```
Current: retrieve_chunks → apply_grounding_checks → LLM
                           ↓
New:     retrieve_chunks → Context Grader → [pass/fail] → LLM
                                         → [weak] → Query expand → Retry
```

**Implementation location:** `app/rag/answer_generator.generate_answer_with_rag()`

**Constraint:** Must be optional, provider-agnostic, configurable.

**Future env vars:**
```bash
CRAG_USE_LLM_GRADER=true
CRAG_GRADER_PROVIDER=openrouter
CRAG_GRADER_MODEL=<small-fast-model>
```

---

## Recommended Model Strategy for Future CRAG

### Preferred Architecture

1. **Deterministic checks first** (always enabled):
   - Top score threshold
   - Keyword/topic overlap
   - Permission filtering result
   - Citation availability

2. **Optional small/cheap model later** (off by default):
   - Query rewrite: Small fast model (e.g., Qwen 0.5B, Phi-2)
   - Context grading: Small fast model (e.g., Qwen 1.5B)

3. **Main LLM only for final answer**:
   - Answer from selected chunks only
   - Include citations
   - No general-knowledge answers in KB mode

### Env Var Specification

```bash
# CRAG Enabled (default false - deterministic only)
CRAG_ENABLED=false

# Max correction attempts (default 1)
CRAG_MAX_CORRECTION_ATTEMPTS=1

# Query rewrite (default false)
CRAG_USE_QUERY_REWRITE=false
CRAG_REWRITE_PROVIDER=openrouter
CRAG_REWRITE_MODEL=

# Context grader (default false)
CRAG_USE_LLM_GRADER=false
CRAG_GRADER_PROVIDER=openrouter
CRAG_GRADER_MODEL=
```

### Constraints
- Do NOT hardcode model names
- Do NOT require CRAG model for normal operation
- Default must remain deterministic and safe
- Must be provider-agnostic (OpenRouter compatible)

---

## Identified Gaps Summary

| Gap | Severity | Blocking Phase 12? | Recommendation |
|-----|----------|-------------------|----------------|
| Limited CRAG metadata | Medium | No | Add as small improvement |
| No query expansion retry | Low | No | Phase 11.4B optional |
| No LLM context grader | Low | No | Phase 14 or later |

---

## Recommendation

### ✅ Proceed to Phase 12 (User Authentication)

**Rationale:**
1. Current CRAG-like behavior is stable (15/15 tests pass)
2. All blocking conditions work correctly
3. Citation repair flow is robust
4. Topic relevance check (Phase 11.2) significantly reduces false positives
5. Optional improvements (query rewrite, LLM grader) are not blocking

### Optional Phase 11.4B (Non-Blocking)

If resources allow, consider adding:
1. **CRAG decision metadata** - Low risk, high visibility
2. **Query expansion retry** - Medium risk, requires careful testing

**Do NOT implement Phase 11.4B if:**
- It breaks current tests
- It requires major refactoring
- It introduces paid model dependency

---

## Files Reviewed

- `apps/backend/app/api/chat.py` - Chat endpoint
- `apps/backend/app/rag/answer_generator.py` - RAG pipeline
- `apps/backend/app/rag/retriever.py` - Hybrid retrieval
- `apps/backend/app/rag/grounding.py` - Grounding checks
- `apps/backend/app/rag/citations.py` - Citation handling
- `apps/backend/app/rag/hybrid_retriever.py` - Hybrid fusion

---

## Conclusion

The current RAG pipeline already implements substantial CRAG-like behavior through deterministic checks, grounding enforcement, and citation repair. The system is ready for Phase 12 without additional CRAG hardening. Optional improvements can be added later as Phase 11.4B if needed.