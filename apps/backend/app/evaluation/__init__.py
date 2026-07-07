"""
Phase 31C — Evaluation helpers

Provides shared, reusable scoring logic for RAG evaluation cases.
Both `scripts/run_rag_evaluation.py` (classic Knowledge Base) and
`scripts/compare_rag_modes.py` (classic vs agentic comparison) can
import from this package so the pass/fail rules stay identical across
runs.

The scoring rules intentionally mirror the legacy
`_run_evaluation_case_impl` function in `run_rag_evaluation.py`, but
expressed as a pure function over plain dicts so it can be tested
without spinning up the full RAG pipeline.
"""
