"""
Phase 31B — LangGraph Agentic RAG State Module

Defines the TypedDict state object that flows through the agentic RAG
LangGraph, plus safe constructors and merge helpers used by each node.

Design constraints:
- The state is intentionally PLAIN (TypedDict + Optional fields) so it
  can be passed across node boundaries without leaking framework types
  into the surrounding custom RAG pipeline.
- All merge helpers default to "later wins" or "append" semantics that
  match what the LangGraph `StateGraph` expects.
- No imports from `langgraph` are required to use this module — keeping
  the state definition framework-agnostic makes it trivial to unit-test
  in isolation and to plug a different graph framework in later if the
  RAG_AGENTIC_FRAMEWORK setting is extended.
- No secrets or absolute paths ever end up in this state. Filenames are
  stripped to their basename by the redaction helpers in
  `app.services.langsmith_tracing` before they reach the LangSmith
  metadata sink.

The state carries the data needed by every node in `agentic_graph.py`:

  normalize_question      → `normalized_question`
  retrieve_context        → `retrieval_metadata`, `chunks`
  evaluate_evidence       → `evidence_level`, `evidence_meta`
  rewrite_query_if_needed → `rewrite_attempted`, `rewritten_question`,
                            `retry_count`
  retrieve_retry          → `chunks` (overwrite), `retry_count`
  generate_answer         → `answer`, `answer_metadata`,
                            `citation_repair_meta`
  verify_citations        → `citation_verification`
  finalize_response       → `final_answer`, `final_citations`,
                            `final_metadata`, `blocked`, `block_reason`
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


# ---------------------------------------------------------------------------
# State TypedDict
# ---------------------------------------------------------------------------

class AgenticRAGState(TypedDict, total=False):
    """
    State carried across every node of the agentic RAG graph.

    All fields are Optional / have defaults so partial updates from a
    single node do not require the caller to supply the full schema.
    """

    # ----- Inputs (set once, read by every node) -----
    query: str
    """Original user question (raw). Never mutated."""

    auth: Optional[Any]
    """
    Optional AuthContext. Held here so permission-aware retrieval can be
    used by the retrieve_context node. The graph does not introspect it;
    nodes simply pass it through to `retrieve_chunks_with_auth`.
    """

    debug: bool
    """If True, retrieve_context forwards the `debug` flag to retrieval."""

    conversation_context: str
    """Phase 20C: pre-formatted conversation context passed to the LLM."""

    # ----- Pipeline stage outputs -----

    # normalize_question
    normalized_question: str
    """Trimmed, whitespace-collapsed version of `query`."""

    # retrieve_context
    chunks: list[dict]
    """Current set of retrieved chunks. Replaced wholesale on retry."""
    retrieval_metadata: dict
    """Metadata returned by the underlying retriever."""

    # evaluate_evidence
    evidence_level: str
    """'strong' | 'medium' | 'weak' — set by decide_evidence_level."""
    evidence_meta: dict
    """Full metadata dict returned by decide_evidence_level."""

    # rewrite_query_if_needed
    rewrite_attempted: bool
    """Whether the rewrite_query_if_needed node fired in this run."""
    rewritten_question: Optional[str]
    """The rewritten question produced by the rewrite node, if any."""

    # retrieve_retry
    retry_count: int
    """Number of retrieve_retry cycles that have already run."""

    # generate_answer
    answer: str
    """Generated answer text. Empty string until generate_answer runs."""
    answer_metadata: dict
    """Metadata captured during answer generation (LLM latency, etc.)."""
    citation_repair_meta: dict
    """
    Citation repair history (initial/retry/attachment) — mirrors the
    fields already produced by the classic `answer_generator` flow.
    """

    # verify_citations
    citation_verification: dict
    """Output of verify_citations: has_citations, citation_count, etc."""

    # finalize_response
    final_answer: str
    """Answer surfaced to the caller (post-verification / post-fallback)."""
    final_citations: list[dict]
    """Formatted citation list surfaced to the caller."""
    final_metadata: dict
    """Aggregated metadata surfaced to the caller."""
    blocked: bool
    """Whether the agentic graph produced a fallback response."""
    block_reason: Optional[str]
    """Reason the graph fell back to a safe message (if it did)."""


# ---------------------------------------------------------------------------
# Constants — keep small, framework-agnostic
# ---------------------------------------------------------------------------

EVIDENCE_STRONG = "strong"
EVIDENCE_MEDIUM = "medium"
EVIDENCE_WEAK = "weak"

VALID_EVIDENCE_LEVELS = {EVIDENCE_STRONG, EVIDENCE_MEDIUM, EVIDENCE_WEAK}


# ---------------------------------------------------------------------------
# Constructors / merge helpers
# ---------------------------------------------------------------------------

def make_initial_state(
    query: str,
    *,
    auth: Optional[Any] = None,
    debug: bool = False,
    conversation_context: str = "",
) -> AgenticRAGState:
    """
    Build a fresh state for a single agentic RAG run.

    All stage outputs are initialized to safe defaults so nodes can rely
    on `state.get(...)` returning a coherent value.
    """
    return AgenticRAGState(
        # Inputs
        query=query or "",
        auth=auth,
        debug=bool(debug),
        conversation_context=conversation_context or "",
        # Stage defaults
        normalized_question="",
        chunks=[],
        retrieval_metadata={},
        evidence_level=EVIDENCE_WEAK,
        evidence_meta={},
        rewrite_attempted=False,
        rewritten_question=None,
        retry_count=0,
        answer="",
        answer_metadata={},
        citation_repair_meta={},
        citation_verification={},
        final_answer="",
        final_citations=[],
        final_metadata={},
        blocked=False,
        block_reason=None,
    )


def is_weak_evidence(state: AgenticRAGState) -> bool:
    """True when the most recent evidence decision is WEAK."""
    return (state.get("evidence_level") or EVIDENCE_WEAK) == EVIDENCE_WEAK


def is_medium_evidence(state: AgenticRAGState) -> bool:
    """True when the most recent evidence decision is MEDIUM."""
    return (state.get("evidence_level") or EVIDENCE_WEAK) == EVIDENCE_MEDIUM


def is_strong_evidence(state: AgenticRAGState) -> bool:
    """True when the most recent evidence decision is STRONG."""
    return (state.get("evidence_level") or EVIDENCE_WEAK) == EVIDENCE_STRONG


def should_retry(state: AgenticRAGState, max_retries: int) -> bool:
    """
    Whether the graph should attempt another retrieve_retry cycle.

    Retries are bounded by `max_retries` (RAG_AGENTIC_MAX_RETRIES) and
    are only attempted when evidence was weak on the previous pass and
    no successful answer has been generated yet.
    """
    if max_retries <= 0:
        return False
    if state.get("answer"):
        # We already have an answer; further retries are pointless.
        return False
    return int(state.get("retry_count") or 0) < int(max_retries)


def safe_chunk_count(state: AgenticRAGState) -> int:
    """Length of `chunks` if present, else 0. Never raises."""
    try:
        return len(state.get("chunks") or [])
    except Exception:
        return 0


def safe_top_score(state: AgenticRAGState) -> Optional[float]:
    """Score of the top chunk, or None if no chunks."""
    chunks = state.get("chunks") or []
    if not chunks:
        return None
    try:
        return float(chunks[0].get("score") or 0.0)
    except (TypeError, ValueError):
        return None
