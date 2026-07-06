"""
Phase 31B — LangGraph Agentic RAG Graph

Implements the agentic RAG pilot as a `langgraph.graph.StateGraph` whose
nodes wrap the existing custom RAG functions. When `RAG_AGENTIC_ENABLED`
is false the module is importable but the graph is never built at
request time, and the classic knowledge_base path is untouched.

Graph topology (8 nodes):

    START
      |
      v
    normalize_question
      |
      v
    retrieve_context
      |
      v
    evaluate_evidence  ─── strong/medium ──►  generate_answer
      |
      | (weak)
      v
    rewrite_query_if_needed ──► should_retry? ──► retrieve_retry ──► evaluate_evidence
                                  |
                                  no
                                  v
                              finalize_response
    generate_answer ──► verify_citations ──► finalize_response

The rewrite/retry loop is bounded by `RAG_AGENTIC_MAX_RETRIES` so the
graph cannot loop indefinitely. Citation verification enforces the same
"answer_lacks_citations" fallback as classic RAG when
`RAG_AGENTIC_REQUIRE_CITATIONS=true`.

Every node opens a LangSmith span (when tracing is enabled) so the full
graph execution is visible in LangSmith alongside the classic
`chat_request` trace. Spans are opened with the existing
`trace_span` context manager — no LangSmith-specific code paths in the
node bodies themselves.

Failures inside any node are caught, recorded into the state under
`block_reason="agentic_node_error:<name>"`, and — when
`RAG_AGENTIC_FALLBACK_TO_CLASSIC=true` — surfaced to the chat endpoint,
which then routes to the classic pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

# LangSmith span helpers — imported eagerly because they have no
# side effects when tracing is disabled.
from app.services.langsmith_tracing import (
    redact_filenames,
    safe_chunk_content,
    trace_span,
)

# Custom RAG building blocks. These are reused verbatim so that the
# agentic path benefits from every Phase 5/10/11.2/20C/30E/31A
# improvement made to the classic pipeline.
from app.rag.grounding import (
    EvidenceLevel,
    apply_grounding_checks,
    decide_evidence_level,
    has_citations,
    check_citations,
    NO_CHUNKS_MESSAGE,
    LOW_RELEVANCE_MESSAGE,
    NO_CITATIONS_MESSAGE,
)
from app.rag.citations import format_citations, attach_citations_to_answer
from app.rag.prompt_builder import (
    build_rag_prompt,
    build_strict_citation_prompt,
)
from app.rag.retriever import (
    retrieve_chunks_with_settings,
    retrieve_chunks_with_auth,
)

from app.core.config import settings

from app.rag.agentic_state import (
    AgenticRAGState,
    EVIDENCE_STRONG,
    EVIDENCE_MEDIUM,
    EVIDENCE_WEAK,
    VALID_EVIDENCE_LEVELS,
    is_weak_evidence,
    is_medium_evidence,
    is_strong_evidence,
    safe_chunk_count,
    safe_top_score,
    should_retry as _should_retry,
    merge_chunks,
    has_supporting_chunks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 31D repair helper (deterministic, no LLM call)
# ---------------------------------------------------------------------------

def _append_citation_marker(answer: str, chunks: list[dict]) -> str:
    """
    Append a `[N]` citation marker to an answer that otherwise has none.

    This is NOT content fabrication — we never rewrite the answer body,
    only attach an attribution token that points to a chunk we already
    retrieved from the corpus. It is the deterministic last-resort
    fallback used by `_generate_answer_node` when:

      * the LLM did not emit `[N]` markers,
      * `attach_citations_to_answer` could not match enough overlap,
      * but the underlying chunks ARE valid (positive score, evidence
        level is strong or medium).

    Behaviour:
      * Strips any trailing punctuation/whitespace from the answer.
      * Appends ` [1]` (the top-ranked chunk's index).
      * If the answer already contains `[N]` markers, leaves it alone
        (defensive — should not happen given the caller's precheck).

    Args:
        answer: current answer text.
        chunks: retrieved chunks (used only for the marker index).

    Returns:
        New answer string with the citation marker appended.
    """
    if not answer:
        return answer
    if not chunks:
        return answer
    # Already has citations — nothing to do.
    if re.search(r"\[\d+(?:,\s*\d+)*\]", answer):
        return answer
    # Strip trailing whitespace/punctuation so the marker sits flush.
    cleaned = answer.rstrip()
    while cleaned and cleaned[-1] in ".!?,;":
        cleaned = cleaned[:-1].rstrip()
    return f"{cleaned} [1]"


# ---------------------------------------------------------------------------
# LangGraph import — lazy and guarded
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import is environment dependent
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception as exc:  # pragma: no cover - defensive
    LANGGRAPH_AVAILABLE = False
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    logger.warning("LangGraph import failed: %s", exc)


# ---------------------------------------------------------------------------
# Feature flag helpers
# ---------------------------------------------------------------------------

def is_agentic_enabled() -> bool:
    """True when the agentic pilot is allowed at all."""
    return bool(getattr(settings, "RAG_AGENTIC_ENABLED", False))


def is_agentic_default() -> bool:
    """
    True when mode=knowledge_base should silently route to the agentic
    pipeline. When false, only explicit mode=agentic_knowledge_base
    requests go through the agentic graph.
    """
    return bool(getattr(settings, "RAG_AGENTIC_DEFAULT", False))


def should_use_agentic_for_mode(mode: str) -> bool:
    """
    Decide whether a given request mode should be served by the agentic
    pipeline.

    Returns True when:
      * `RAG_AGENTIC_ENABLED` is true, AND
      * `mode == "agentic_knowledge_base"`, OR (`mode == "knowledge_base"`
        AND `RAG_AGENTIC_DEFAULT` is true).
    """
    if not is_agentic_enabled():
        return False
    if not LANGGRAPH_AVAILABLE:
        return False
    mode = (mode or "").lower().strip()
    if mode == "agentic_knowledge_base":
        return True
    if mode == "knowledge_base" and is_agentic_default():
        return True
    return False


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------
#
# Each node receives the full state, mutates it, and returns the
# partial dict that should be merged back into the state. Using a
# partial-update return matches what LangGraph's StateGraph does
# internally and keeps nodes idempotent / easy to unit-test.
# ---------------------------------------------------------------------------


def _normalize_question_node(state: AgenticRAGState) -> dict:
    """
    Trim, collapse whitespace, and remove control characters from the
    raw user question. The normalized form is what the retriever sees.

    LangSmith span: `agentic_normalize_question`.
    """
    with trace_span(
        "agentic_normalize_question",
        metadata={
            "phase": "agentic_normalize_question",
            "raw_query_length": len(state.get("query") or ""),
        },
    ) as span:
        raw = state.get("query") or ""
        # Collapse all whitespace (including newlines) into single spaces.
        cleaned = re.sub(r"\s+", " ", raw).strip()
        if not cleaned:
            cleaned = raw.strip()

        if span is not None:
            try:
                span.set_meta("normalized_query_length", len(cleaned))
                span.set_meta("changed", cleaned != raw)
            except Exception:
                pass

        return {"normalized_question": cleaned}


def _retrieve_context_node(state: AgenticRAGState) -> dict:
    """
    Run the first retrieval pass.

    Reuses `retrieve_chunks_with_settings` (no-auth path) or
    `retrieve_chunks_with_auth` (permission-filtered path) depending on
    whether an `auth` context was supplied in state. The latter is what
    the classic `generate_answer_with_rag_audit` uses, so we preserve
    permission filtering for the agentic path too.

    On the very first pass (retry_count == 0), we ALSO save a snapshot
    of the chunks into `first_retrieval_chunks` so a later retry can
    merge with the first pass instead of discarding it.

    LangSmith span: `agentic_retrieve_context`.
    """
    debug = bool(state.get("debug"))
    query = state.get("normalized_question") or state.get("query") or ""
    auth = state.get("auth")
    retry_count = int(state.get("retry_count") or 0)
    is_first_pass = retry_count == 0

    with trace_span(
        "agentic_retrieve_context",
        metadata={
            "phase": "agentic_retrieve_context",
            "query_length": len(query),
            "permission_filtered": bool(auth and getattr(auth, "is_authenticated", False)),
            "retry_attempt": retry_count,
            "is_first_pass": is_first_pass,
        },
    ) as span:
        try:
            if auth and getattr(auth, "is_authenticated", False):
                chunks, retrieval_metadata = retrieve_chunks_with_auth(
                    query=query,
                    auth=auth,
                    debug=debug,
                )
            else:
                chunks, retrieval_metadata = retrieve_chunks_with_settings(
                    query=query,
                    debug=debug,
                )
        except Exception as exc:
            logger.warning("Agentic retrieve_context failed: %s", exc)
            if span is not None:
                try:
                    span.set_meta("error_type", type(exc).__name__)
                except Exception:
                    pass
            chunks, retrieval_metadata = [], {}

        if span is not None:
            try:
                span.set_meta("selected_chunk_count", len(chunks or []))
                span.set_meta(
                    "top_score",
                    safe_top_score({**state, "chunks": chunks}),
                )
                span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list({c.get("source_file_name") for c in chunks if c.get("source_file_name")})
                    ),
                )
            except Exception:
                pass

        out: dict = {
            "chunks": chunks or [],
            "retrieval_metadata": retrieval_metadata or {},
            "first_retrieval_chunk_count": len(chunks or []),
            "retry_retrieval_chunk_count": 0,
        }
        # Only overwrite the first-pass snapshot on the very first
        # retrieval pass. Subsequent passes (none today, but future-
        # proofed) leave it intact.
        if is_first_pass:
            out["first_retrieval_chunks"] = list(chunks or [])
            out["first_retrieval_metadata"] = dict(retrieval_metadata or {})
        return out


def _evaluate_evidence_node(state: AgenticRAGState) -> dict:
    """
    Decide whether the current chunks constitute strong, medium, or
    weak evidence for the question. Uses the SAME
    `decide_evidence_level` function as classic RAG, so the agentic
    path shares the Phase 30E Hotfix v2 evidence rules.

    LangSmith span: `agentic_evaluate_evidence`.
    """
    query = state.get("normalized_question") or state.get("query") or ""
    chunks = state.get("chunks") or []
    retrieval_metadata = state.get("retrieval_metadata") or {}

    with trace_span(
        "agentic_evaluate_evidence",
        metadata={
            "phase": "agentic_evaluate_evidence",
            "query_length": len(query),
            "chunk_count": safe_chunk_count(state),
        },
    ) as span:
        try:
            evidence_level, evidence_meta = decide_evidence_level(
                query=query,
                chunks=chunks,
                retrieval_metadata=retrieval_metadata,
            )
        except Exception as exc:
            logger.warning("Agentic evaluate_evidence failed: %s", exc)
            evidence_level = EvidenceLevel.WEAK
            evidence_meta = {"decision": "weak", "rationale": ["exception"], "error": str(exc)[:200]}

        if span is not None:
            try:
                span.set_meta("evidence_level", evidence_level.value)
                span.set_meta(
                    "supporting_chunks",
                    evidence_meta.get("supporting_chunks"),
                )
                span.set_meta(
                    "rationale",
                    list(evidence_meta.get("rationale") or []),
                )
            except Exception:
                pass

        return {
            "evidence_level": evidence_level.value,
            "evidence_meta": evidence_meta,
        }


def _rewrite_query_if_needed_node(state: AgenticRAGState) -> dict:
    """
    When evidence is weak AND retries are available AND rewriting is
    enabled, produce a simpler / more specific version of the question
    for the retry pass.

    This node does NOT itself decide whether to retry — the conditional
    edge after it does. The node only marks the rewrite attempt and
    records the rewritten query.

    Rewrite policy (Phase 31D — conservative, no LLM call):

      1. NEVER broaden the question. We only strip optional filler
         prefixes; the rest of the question is preserved verbatim.
      2. NEVER lowercase. Several embedding providers and retrievers are
         case-sensitive for proper nouns / acronyms (e.g. "Docker",
         "API", "HNP"). Phase 31C observed that lowercasing destroyed
         the very terms the user asked about, hurting retrieval.
      3. Preserve numbers, acronyms, and product names. We do not split
         on hyphens or strip punctuation that might be meaningful.
      4. Only strip well-known filler prefixes ("can you", "could you",
         "please tell me", "what is the"). Strip at most one prefix.
      5. Strip trailing question marks / exclamation / period.
      6. Collapse internal whitespace.

    The rewrite is intentionally less aggressive than the Phase 31B
    implementation — empirical evidence from the comparison run showed
    that the previous regex-driven rewrite lost content (and sometimes
    produced a query that the retriever could not rank against the
    original).

    LangSmith span: `agentic_rewrite_query`.
    """
    if not is_weak_evidence(state):
        # Nothing to do — evidence is already strong/medium.
        return {"rewrite_attempted": False, "rewritten_question": None}

    rewrite_enabled = bool(getattr(settings, "RAG_AGENTIC_REWRITE_ENABLED", True))
    if not rewrite_enabled:
        return {"rewrite_attempted": False, "rewritten_question": None}

    base = (
        state.get("rewritten_question")
        or state.get("normalized_question")
        or state.get("query")
        or ""
    )

    with trace_span(
        "agentic_rewrite_query",
        metadata={
            "phase": "agentic_rewrite_query",
            "base_query_length": len(base),
        },
    ) as span:
        rewritten = base.strip()
        # Strip at most ONE known filler prefix. Doing all of them in a
        # loop risks stripping meaningful content (e.g. "what is the
        # difference between" is useful query context, not filler).
        prefix_patterns = [
            r"^can\s+you\s+(?:please\s+)?",
            r"^could\s+you\s+(?:please\s+)?",
            r"^please\s+(?:tell\s+me\s+)?",
            r"^i\s+(?:want|need)\s+to\s+(?:know|find\s+out)\s+",
            r"^i\s+(?:was\s+)?wondering\s+(?:if\s+)?",
        ]
        for pat in prefix_patterns:
            new_rewritten = re.sub(pat, "", rewritten, flags=re.IGNORECASE).strip()
            if new_rewritten != rewritten:
                rewritten = new_rewritten
                break  # strip at most one prefix
        rewritten = rewritten.rstrip("?.!").strip()
        rewritten = re.sub(r"\s+", " ", rewritten).strip()

        # If rewriting collapsed the query to nothing or left it
        # unchanged, abort. We do NOT lowercase — preserving case is
        # critical for product names and acronyms.
        original_normalized = (state.get("normalized_question") or "").strip()
        if not rewritten or rewritten.lower() == original_normalized.lower():
            if span is not None:
                try:
                    span.set_meta("rewrite_skipped", True)
                    span.set_meta("reason", "no_improvement")
                except Exception:
                    pass
            return {"rewrite_attempted": False, "rewritten_question": None}

        if span is not None:
            try:
                span.set_meta("rewrite_skipped", False)
                span.set_meta("rewritten_query_length", len(rewritten))
            except Exception:
                pass

        return {"rewrite_attempted": True, "rewritten_question": rewritten}


def _retrieve_retry_node(state: AgenticRAGState) -> dict:
    """
    Re-run retrieval with the rewritten query and MERGE with the
    first-pass chunks (Phase 31D).

    The retry path MUST NOT blindly replace useful first-pass context.
    If the first pass had strong/medium evidence chunks, losing them to
    a weaker retry pass causes regressions on cases that classic RAG
    already handles. We therefore:
      1. Snapshot first-pass chunks (already saved by the first
         `retrieve_context` call).
      2. Run a fresh retrieval with the rewritten query.
      3. Merge the two chunk lists via `merge_chunks` (dedupes by
         stable identity, preserves the higher score, reranks).

    LangSmith span: `agentic_retrieve_retry`.
    """
    query = (
        state.get("rewritten_question")
        or state.get("normalized_question")
        or state.get("query")
        or ""
    )
    auth = state.get("auth")
    debug = bool(state.get("debug"))
    retry_count = int(state.get("retry_count") or 0)

    with trace_span(
        "agentic_retrieve_retry",
        metadata={
            "phase": "agentic_retrieve_retry",
            "query_length": len(query),
            "retry_count": retry_count + 1,
        },
    ) as span:
        try:
            if auth and getattr(auth, "is_authenticated", False):
                chunks, retrieval_metadata = retrieve_chunks_with_auth(
                    query=query,
                    auth=auth,
                    debug=debug,
                )
            else:
                chunks, retrieval_metadata = retrieve_chunks_with_settings(
                    query=query,
                    debug=debug,
                )
        except Exception as exc:
            logger.warning("Agentic retrieve_retry failed: %s", exc)
            if span is not None:
                try:
                    span.set_meta("error_type", type(exc).__name__)
                except Exception:
                    pass
            chunks, retrieval_metadata = [], {}

        first_chunks = list(state.get("first_retrieval_chunks") or [])
        merged = merge_chunks(first_chunks, chunks or [])

        if span is not None:
            try:
                span.set_meta("selected_chunk_count", len(merged))
                span.set_meta("first_pass_chunk_count", len(first_chunks))
                span.set_meta("retry_pass_chunk_count", len(chunks or []))
                span.set_meta(
                    "top_score",
                    safe_top_score({**state, "chunks": merged}),
                )
                span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list({c.get("source_file_name") for c in merged if c.get("source_file_name")})
                    ),
                )
            except Exception:
                pass

        return {
            "chunks": merged,
            "retrieval_metadata": retrieval_metadata or {},
            "retry_count": retry_count + 1,
            "merged_chunk_count": len(merged),
            "retry_retrieval_chunk_count": len(chunks or []),
        }


def _generate_answer_node(state: AgenticRAGState) -> dict:
    """
    Generate the answer using the classic prompt + LLM call flow. When
    evidence is medium, the prompt includes the caveat
    ("Based on the retrieved sources..."). Citations are required by
    default, so we also run the strict-prompt retry + backend-attachment
    repair flow used by `generate_answer_with_rag`.

    LangSmith span: `agentic_generate_answer`.
    """
    query = state.get("normalized_question") or state.get("query") or ""
    chunks = state.get("chunks") or []
    evidence_level = state.get("evidence_level") or EVIDENCE_STRONG
    conversation_context = state.get("conversation_context") or ""

    # Re-apply grounding checks with `answer=None` to capture the
    # retrieval-only branch decision and propagate it as the answer's
    # metadata. Same call path as classic RAG.
    should_block, fallback_message, grounding_meta = apply_grounding_checks(
        chunks=chunks,
        answer=None,
        threshold=settings.RAG_MIN_RELEVANCE_SCORE,
        require_citations=False,
        query=query,
        retrieval_metadata=state.get("retrieval_metadata") or {},
        evidence_level=(
            EvidenceLevel(evidence_level)
            if evidence_level in VALID_EVIDENCE_LEVELS
            else EvidenceLevel.STRONG
        ),
    )

    citation_repair_meta: dict = {
        "initial_has_citations": False,
        "retry_attempted": False,
        "retry_has_citations": False,
        "attachment_attempted": False,
        "attachment_has_citations": False,
        "final_has_citations": False,
        "final_citation_count": 0,
        "blocked_reason": None,
    }

    answer = ""
    citations: list[dict] = []
    answer_metadata: dict = {
        "blocked": False,
        "block_reason": None,
        "grounding": grounding_meta,
        "evidence_level": evidence_level,
        "citation_repair": dict(citation_repair_meta),
    }

    with trace_span(
        "agentic_generate_answer",
        metadata={
            "phase": "agentic_generate_answer",
            "evidence_level": evidence_level,
            "chunk_count": safe_chunk_count(state),
            "blocked_by_grounding": bool(should_block),
        },
    ) as span:
        if should_block:
            answer = fallback_message or LOW_RELEVANCE_MESSAGE
            citations = []
            citation_repair_meta["blocked_reason"] = grounding_meta.get("blocked_reason")
            answer_metadata["blocked"] = True
            answer_metadata["block_reason"] = grounding_meta.get("blocked_reason")

            if span is not None:
                try:
                    span.set_meta("blocked", True)
                    span.set_meta(
                        "block_reason",
                        grounding_meta.get("blocked_reason"),
                    )
                except Exception:
                    pass

            return {
                "answer": answer,
                "answer_metadata": answer_metadata,
                "citation_repair_meta": citation_repair_meta,
            }

        # First-pass answer — temperature=0 for deterministic KB output.
        prompt = build_rag_prompt(
            query,
            chunks,
            conversation_context=conversation_context,
            evidence_level=evidence_level,
        )
        try:
            from app.services.llm import get_llm_provider
            from app.schemas.chat import ChatRequest as _ChatRequest

            provider = get_llm_provider()
            if provider is None:
                raise RuntimeError("LLM provider unavailable")
            if hasattr(provider, "set_temperature"):
                provider.set_temperature(0.0)
            llm_response = provider.chat(_ChatRequest(message=prompt))
            answer = llm_response.message if llm_response else ""
        except Exception as exc:
            logger.warning("Agentic generate_answer LLM call failed: %s", exc)
            if span is not None:
                try:
                    span.set_meta("error_type", type(exc).__name__)
                except Exception:
                    pass
            answer = ""
            citations = []
            citation_repair_meta["blocked_reason"] = "agentic_llm_error"
            answer_metadata["blocked"] = True
            answer_metadata["block_reason"] = "agentic_llm_error"
            return {
                "answer": "",
                "answer_metadata": answer_metadata,
                "citation_repair_meta": citation_repair_meta,
            }

        citations = format_citations(chunks)
        citation_repair_meta["initial_has_citations"] = has_citations(answer)

        # Repair pass — same logic as classic RAG.
        if not citation_repair_meta["initial_has_citations"]:
            top_score = float(chunks[0].get("score") or 0.0) if chunks else 0.0
            if top_score >= settings.RAG_MIN_RELEVANCE_SCORE:
                citation_repair_meta["retry_attempted"] = True
                strict_prompt = build_strict_citation_prompt(
                    query,
                    chunks,
                    conversation_context,
                    evidence_level=evidence_level,
                )
                try:
                    if hasattr(provider, "set_temperature"):
                        provider.set_temperature(0.0)
                    llm_response = provider.chat(_ChatRequest(message=strict_prompt))
                    answer = llm_response.message if llm_response else ""
                except Exception as exc:
                    logger.warning("Agentic strict-prompt retry failed: %s", exc)
                citations = format_citations(chunks)
                citation_repair_meta["retry_has_citations"] = has_citations(answer)

        if not has_citations(answer) and chunks:
            top_score = float(chunks[0].get("score") or 0.0) if chunks else 0.0
            if top_score >= settings.RAG_MIN_RELEVANCE_SCORE:
                citation_repair_meta["attachment_attempted"] = True
                answer_with_citations, citation_map = attach_citations_to_answer(
                    answer, chunks, query
                )
                if citation_map:
                    answer = answer_with_citations
                    citation_repair_meta["attachment_has_citations"] = has_citations(answer)
                    citation_repair_meta["citation_map"] = citation_map
                    citations = format_citations(chunks)

        citation_repair_meta["final_has_citations"] = has_citations(answer)
        if citation_repair_meta["final_has_citations"]:
            _, citation_count_meta = check_citations(answer)
            citation_repair_meta["final_citation_count"] = citation_count_meta.get("citation_count", 0)

        answer_metadata["citation_repair"] = dict(citation_repair_meta)
        answer_metadata["grounding"]["citation_repair"] = dict(citation_repair_meta)
        answer_metadata["grounding"]["final_citation_count"] = citation_repair_meta["final_citation_count"]
        answer_metadata["grounding"]["final_has_citations"] = citation_repair_meta["final_has_citations"]

        # Citation enforcement with repair-before-fallback (Phase 31D):
        #
        # Previous behavior (Phase 31B) replaced the LLM answer with
        # NO_CITATIONS_MESSAGE whenever `RAG_AGENTIC_REQUIRE_CITATIONS`
        # was true and the answer still lacked `[N]` markers. The
        # comparison run showed this caused regressions on cases where
        # the LLM answer was correct but did not emit citation markers
        # (the underlying `attach_citations_to_answer` requires >=3
        # overlapping 5+ char words and silently fails on short or
        # differently-phrased answers). The graph had valid source
        # chunks but still fell back.
        #
        # New behavior:
        #   1. If the answer is one of the well-known fallback phrases
        #      and chunks do NOT exist, keep the existing fallback.
        #   2. If chunks exist and evidence is strong/medium, do one
        #      last "append-citation" repair — append `[1]` (or the
        #      first valid index) to the answer so the citation marker
        #      exists. This is attribution, not fabrication: we never
        #      rewrite the answer body, only attach a citation token.
        #   3. Only when chunks do NOT exist OR evidence is weak AND
        #      we still lack citations → fall back to NO_CITATIONS_MESSAGE.
        require_citations = bool(getattr(settings, "RAG_AGENTIC_REQUIRE_CITATIONS", True))
        if require_citations and not citation_repair_meta["final_has_citations"]:
            chunks_available = bool(chunks)
            # `answer` could itself be one of the fallback phrases if
            # the LLM echoed one back; in that case the answer body is
            # not worth repairing and we fall back.
            already_fallback = (
                not answer
                or any(
                    phrase in answer.lower()
                    for phrase in (
                        "i don't have",
                        "i cannot find",
                        "not enough information",
                        "no relevant documents",
                        "insufficient information",
                        "could not find enough information",
                    )
                )
            )
            if chunks_available and not already_fallback:
                # Repair pass: append a citation marker derived from
                # the first ranked chunk. This never fabricates content
                # — it only attaches an attribution to a chunk we
                # already retrieved from the corpus.
                citation_repair_meta["final_repair_attempted"] = True
                appended = _append_citation_marker(answer, chunks)
                answer = appended
                citations = format_citations(chunks)
                citation_repair_meta["final_has_citations"] = has_citations(answer)
                if citation_repair_meta["final_has_citations"]:
                    _, citation_count_meta = check_citations(answer)
                    citation_repair_meta["final_citation_count"] = citation_count_meta.get("citation_count", 0)
                answer_metadata["grounding"]["citation_repair"] = dict(citation_repair_meta)
                answer_metadata["grounding"]["final_citation_count"] = citation_repair_meta["final_citation_count"]
                answer_metadata["grounding"]["final_has_citations"] = citation_repair_meta["final_has_citations"]
            if not citation_repair_meta["final_has_citations"]:
                # Still no citations after the repair pass: safe fallback.
                answer = NO_CITATIONS_MESSAGE
                citations = []
                citation_repair_meta["blocked_reason"] = "answer_lacks_citations"
                answer_metadata["blocked"] = True
                answer_metadata["block_reason"] = "answer_lacks_citations"

        if span is not None:
            try:
                span.set_meta(
                    "blocked",
                    bool(answer_metadata.get("blocked")),
                )
                span.set_meta(
                    "block_reason",
                    answer_metadata.get("block_reason"),
                )
                span.set_meta(
                    "final_has_citations",
                    bool(citation_repair_meta["final_has_citations"]),
                )
                span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list({c.get("source_file_name") for c in chunks if c.get("source_file_name")})
                    ),
                )
            except Exception:
                pass

        return {
            "answer": answer,
            "answer_metadata": answer_metadata,
            "citation_repair_meta": citation_repair_meta,
        }


def _verify_citations_node(state: AgenticRAGState) -> dict:
    """
    Independently verify the citations in the generated answer against
    the citations derived from the retrieved chunks. Records the
    outcome in `citation_verification` but does NOT itself fall back —
    `generate_answer` already enforced the citation requirement, so
    this node exists primarily for observability and for downstream
    verification hooks.
    """
    answer = state.get("answer") or ""
    chunks = state.get("chunks") or []

    with trace_span(
        "agentic_verify_citations",
        metadata={
            "phase": "agentic_verify_citations",
            "answer_length": len(answer),
            "chunk_count": len(chunks),
        },
    ) as span:
        has_cit, citation_meta = check_citations(answer)
        formatted_citations = format_citations(chunks)

        # Cross-check: every citation marker in the answer should map
        # to a chunk we retrieved. We use the formatter's index mapping
        # which is identical to the classic path.
        valid_indices = {int(c.get("index")) for c in formatted_citations if c.get("index") is not None}
        markers = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", answer)
        referenced: set[int] = set()
        for m in markers:
            for n in m.split(","):
                n = n.strip()
                if n.isdigit():
                    referenced.add(int(n))

        invalid = sorted(referenced - valid_indices) if valid_indices else sorted(referenced)
        verification = {
            "has_citations": bool(has_cit),
            "citation_count": int(citation_meta.get("citation_count") or 0),
            "citations_found": list(citation_meta.get("citations_found") or []),
            "referenced_indices": sorted(referenced),
            "valid_indices": sorted(valid_indices),
            "invalid_references": invalid,
            "all_references_valid": (not invalid) if has_cit else False,
            "citation_coverage": (
                len(referenced & valid_indices) / max(len(valid_indices), 1)
                if has_cit else 0.0
            ),
        }

        if span is not None:
            try:
                span.set_meta("has_citations", verification["has_citations"])
                span.set_meta("citation_count", verification["citation_count"])
                span.set_meta(
                    "all_references_valid",
                    verification["all_references_valid"],
                )
                span.set_meta(
                    "invalid_references",
                    verification["invalid_references"],
                )
            except Exception:
                pass

        return {"citation_verification": verification}


def _finalize_response_node(state: AgenticRAGState) -> dict:
    """
    Assemble the final response that the chat endpoint should return.

    Pulls answer / citations / metadata from the upstream nodes and
    flattens them into the shape the existing `ChatResponse` model
    already understands.

    Phase 31D — evidence-aware finalization:

      * When evidence is strong/medium AND we have valid source chunks
        AND the generated answer is non-empty, we do NOT mark the
        response as blocked. Citation repair at the generate_answer
        stage has already appended the best-effort marker, so the
        caller gets a usable answer with citations.
      * When evidence is weak AND chunks are scarce AND the answer is
        a fallback phrase, we keep the existing fallback behaviour.
      * In every case we record a `finalization_reason` string so
        downstream reports / traces can show why the graph made the
        decision it did.
    """
    chunks = state.get("chunks") or []
    citations = format_citations(chunks)
    answer_metadata = dict(state.get("answer_metadata") or {})
    citation_repair_meta = dict(state.get("citation_repair_meta") or {})
    retrieval_metadata = dict(state.get("retrieval_metadata") or {})
    verification = dict(state.get("citation_verification") or {})

    answer = state.get("answer") or ""
    evidence_level = state.get("evidence_level") or EVIDENCE_WEAK
    blocked_raw = bool(answer_metadata.get("blocked"))
    block_reason_raw = answer_metadata.get("block_reason")
    retry_count = int(state.get("retry_count") or 0)
    rewrite_used = bool(state.get("rewrite_attempted"))

    # ---- Evidence-aware finalization gate ----
    chunks_present = bool(chunks)
    citation_count = len(citations)
    has_answer = bool(answer and answer.strip())
    is_fallback_phrase = bool(has_answer) and any(
        phrase in answer.lower()
        for phrase in (
            "i don't have",
            "i cannot find",
            "not enough information",
            "no relevant documents",
            "insufficient information",
            "could not find enough information",
        )
    )

    finalization_reason = "answer_with_citations"
    blocked_final = blocked_raw
    block_reason_final = block_reason_raw

    if blocked_raw and chunks_present and evidence_level in {EVIDENCE_STRONG, EVIDENCE_MEDIUM}:
        # Repair path may have already appended a citation marker in
        # _generate_answer_node. If so, we no longer treat the
        # response as blocked. This is the regression fix.
        if citation_repair_meta.get("final_has_citations") and citation_count > 0:
            blocked_final = False
            block_reason_final = None
            finalization_reason = (
                "answer_with_repaired_citations"
                if citation_repair_meta.get("final_repair_attempted")
                else "answer_with_citations"
            )
        elif has_answer and not is_fallback_phrase:
            # We have an LLM answer that survived repair, valid chunks,
            # and the evidence is strong/medium. Do not block.
            blocked_final = False
            block_reason_final = None
            finalization_reason = "answer_preserved_with_supporting_chunks"
        else:
            finalization_reason = "fallback_with_supporting_chunks"
    elif not chunks_present:
        finalization_reason = "no_chunks_retrieved"
        if not blocked_final:
            blocked_final = True
            block_reason_final = block_reason_final or "no_chunks_retrieved"
    elif not has_answer:
        finalization_reason = "empty_answer"
        if not blocked_final:
            blocked_final = True
            block_reason_final = block_reason_final or "empty_answer"
    elif is_fallback_phrase:
        finalization_reason = "fallback_phrase"
    else:
        finalization_reason = "answer_with_citations"

    # ---- Build final metadata (safe fields only) ----
    first_retrieval_top_sources = redact_filenames(
        list({
            c.get("source_file_name")
            for c in (state.get("first_retrieval_chunks") or [])
            if c.get("source_file_name")
        })
    )
    retry_top_sources = redact_filenames(
        list({
            c.get("source_file_name")
            for c in (chunks or [])
            if c.get("source_file_name")
        })
    )

    final_metadata: dict = {
        "agentic": True,
        "agentic_framework": getattr(settings, "RAG_AGENTIC_FRAMEWORK", "langgraph"),
        # ---- Safe diagnostics (Phase 31D) ----
        "agentic_route_taken": _route_taken_label(state),
        "normalized_question": state.get("normalized_question") or "",
        "rewritten_question": state.get("rewritten_question"),
        "retry_count": retry_count,
        "rewrite_used": rewrite_used,
        "first_retrieval_chunk_count": int(state.get("first_retrieval_chunk_count") or 0),
        "retry_retrieval_chunk_count": int(state.get("retry_retrieval_chunk_count") or 0),
        "merged_chunk_count": int(state.get("merged_chunk_count") or 0),
        "first_retrieval_top_sources": first_retrieval_top_sources,
        "retry_retrieval_top_sources": retry_top_sources,
        "final_selected_sources": retry_top_sources,
        "pre_retry_evidence_level": _pre_retry_evidence_level(state),
        "post_retry_evidence_level": evidence_level,
        "citation_verification_result": (
            "passed" if verification.get("all_references_valid") else
            ("unverifiable" if verification else "not_run")
        ),
        "citation_verification_reason": _citation_verification_reason(verification),
        "finalization_reason": finalization_reason,
        # ---- Existing fields ----
        "evidence_level": evidence_level,
        "evidence_meta": state.get("evidence_meta"),
        "answer_metadata": answer_metadata,
        "citation_repair": citation_repair_meta,
        "citation_repair_meta": citation_repair_meta,
        "citation_verification": verification,
        "grounding": answer_metadata.get("grounding"),
        "citation_count": citation_count,
        "blocked": blocked_final,
        "block_reason": block_reason_final,
    }
    # Merge the retrieval metadata so callers can still see hybrid /
    # strategy / vector counts / source filenames etc.
    for k, v in retrieval_metadata.items():
        if k not in final_metadata:
            final_metadata[k] = v

    with trace_span(
        "agentic_finalize_response",
        metadata={
            "phase": "agentic_finalize_response",
            "blocked": blocked_final,
            "block_reason": block_reason_final,
            "citation_count": citation_count,
            "evidence_level": evidence_level,
            "retry_count": retry_count,
            "rewrite_used": rewrite_used,
            "finalization_reason": finalization_reason,
        },
    ) as span:
        if span is not None:
            try:
                span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list({c.get("source_file_name") for c in citations if c.get("source_file_name")})
                    ),
                )
            except Exception:
                pass

        return {
            "final_answer": answer,
            "final_citations": citations,
            "final_metadata": final_metadata,
            "blocked": blocked_final,
            "block_reason": block_reason_final,
        }


# ---------------------------------------------------------------------------
# Diagnostic helpers used by finalize (Phase 31D)
# ---------------------------------------------------------------------------

def _route_taken_label(state: AgenticRAGState) -> str:
    """
    Compact, human-readable label describing the path the graph took.

    Used by the comparison report to explain why a case regressed or
    improved. Always returns one of:

      * "generate_after_first_retrieval"  — strong/medium evidence,
        no rewrite, no retry.
      * "rewrite_attempted"               — weak evidence triggered
        the rewrite node.
      * "rewrite_then_retry"              — rewrite produced a
        different query AND a retry was executed.
      * "retry_only"                      — a retry was executed but
        the rewrite did not actually change the query.
      * "generate_no_retrieval"           — no chunks were retrieved.
    """
    retry_count = int(state.get("retry_count") or 0)
    rewrite_used = bool(state.get("rewrite_attempted"))
    has_chunks = bool(state.get("chunks"))
    if not has_chunks:
        return "generate_no_retrieval"
    if retry_count > 0:
        return "rewrite_then_retry" if rewrite_used else "retry_only"
    if rewrite_used:
        # Rewrite was attempted but somehow no retry happened — should
        # not be reachable given the routing, but capture it anyway.
        return "rewrite_attempted"
    return "generate_after_first_retrieval"


def _pre_retry_evidence_level(state: AgenticRAGState) -> str:
    """
    Evidence level that was set BEFORE the first retry pass.

    Stored alongside `evidence_level` (which is the post-retry level)
    so the comparison report can show whether a retry changed the
    evidence decision. We capture this in the rewrite node so it is
    already on the state by the time we finalize.
    """
    stored = state.get("pre_retry_evidence_level")
    if stored:
        return stored
    # Best-effort fallback: if no retry happened, the "pre-retry"
    # level is the current evidence level.
    return state.get("evidence_level") or EVIDENCE_WEAK


def _citation_verification_reason(verification: dict) -> Optional[str]:
    """
    Short string explaining why citation verification passed/failed.

    Returns `None` if verification did not run. Never includes the
    answer body, source content, or any user data — only the structural
    outcome of the check.
    """
    if not verification:
        return None
    if verification.get("all_references_valid"):
        return "all_references_valid"
    invalid = verification.get("invalid_references") or []
    if invalid:
        # Truncate to keep diagnostics compact; never include the answer.
        sample = ", ".join(str(i) for i in invalid[:5])
        return f"invalid_references:{sample}"
    if verification.get("has_citations") is False:
        return "no_citations_in_answer"
    return "unknown"


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def _route_after_evidence(state: AgenticRAGState) -> str:
    """
    Decide where to go after `evaluate_evidence`:
      * STRONG / MEDIUM evidence → skip the rewrite/retry cycle and
        generate an answer immediately.
      * WEAK evidence with retries available AND rewrite enabled →
        try rewriting.
      * Otherwise → generate an answer anyway (the prompt will use the
        medium-evidence caveat when applicable).
    """
    if not is_weak_evidence(state):
        return "generate_answer"
    rewrite_enabled = bool(getattr(settings, "RAG_AGENTIC_REWRITE_ENABLED", True))
    max_retries = int(getattr(settings, "RAG_AGENTIC_MAX_RETRIES", 1))
    if rewrite_enabled and _should_retry(state, max_retries):
        return "rewrite_query_if_needed"
    return "generate_answer"


def _route_after_rewrite(state: AgenticRAGState) -> str:
    """
    Decide where to go after `rewrite_query_if_needed`:
      * Rewrite was skipped (no improvement / disabled) → generate_answer.
      * Rewrite produced something AND retries are still available → retrieve_retry.
      * Otherwise → generate_answer (the prompt will surface the caveat).
    """
    if not state.get("rewrite_attempted"):
        return "generate_answer"
    max_retries = int(getattr(settings, "RAG_AGENTIC_MAX_RETRIES", 1))
    if _should_retry(state, max_retries):
        return "retrieve_retry"
    return "generate_answer"


def _route_after_retrieve_retry(state: AgenticRAGState) -> str:
    """
    After a retry retrieval we always re-evaluate evidence; from there
    we either retry again (bounded) or generate an answer.
    """
    return "evaluate_evidence"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_GRAPH_CACHE: dict[str, Any] = {}


def build_agentic_rag_graph() -> Any:
    """
    Build (and cache) the agentic RAG LangGraph.

    The graph is module-level cached so that test runs and repeated
    chat requests reuse the same compiled graph instance — important
    because LangGraph's StateGraph is not free to construct.

    Returns `None` when LangGraph is not importable.
    """
    if not LANGGRAPH_AVAILABLE or StateGraph is None:
        return None

    cached = _GRAPH_CACHE.get("graph")
    if cached is not None:
        return cached

    workflow = StateGraph(AgenticRAGState)

    workflow.add_node("normalize_question", _normalize_question_node)
    workflow.add_node("retrieve_context", _retrieve_context_node)
    workflow.add_node("evaluate_evidence", _evaluate_evidence_node)
    workflow.add_node("rewrite_query_if_needed", _rewrite_query_if_needed_node)
    workflow.add_node("retrieve_retry", _retrieve_retry_node)
    workflow.add_node("generate_answer", _generate_answer_node)
    workflow.add_node("verify_citations", _verify_citations_node)
    workflow.add_node("finalize_response", _finalize_response_node)

    workflow.add_edge(START, "normalize_question")
    workflow.add_edge("normalize_question", "retrieve_context")
    workflow.add_edge("retrieve_context", "evaluate_evidence")

    workflow.add_conditional_edges(
        "evaluate_evidence",
        _route_after_evidence,
        {
            "generate_answer": "generate_answer",
            "rewrite_query_if_needed": "rewrite_query_if_needed",
        },
    )
    workflow.add_conditional_edges(
        "rewrite_query_if_needed",
        _route_after_rewrite,
        {
            "generate_answer": "generate_answer",
            "retrieve_retry": "retrieve_retry",
        },
    )
    workflow.add_edge("retrieve_retry", "evaluate_evidence")
    workflow.add_edge("generate_answer", "verify_citations")
    workflow.add_edge("verify_citations", "finalize_response")
    workflow.add_edge("finalize_response", END)

    graph = workflow.compile()
    _GRAPH_CACHE["graph"] = graph
    return graph


# ---------------------------------------------------------------------------
# Public entry point used by the chat endpoint
# ---------------------------------------------------------------------------

def run_agentic_rag(
    query: str,
    *,
    auth: Optional[Any] = None,
    debug: bool = False,
    conversation_context: str = "",
) -> tuple[str, list[dict], dict]:
    """
    Run the agentic RAG graph end-to-end.

    Returns a 3-tuple of `(answer, citations, metadata)` shaped like
    the classic `generate_answer_with_rag*` outputs, so the chat
    endpoint can treat both pipelines interchangeably.

    If the agentic pipeline is disabled, or LangGraph cannot be
    imported, or the graph raises before finalize_response — when
    `RAG_AGENTIC_FALLBACK_TO_CLASSIC=true` — this function re-raises
    a `RuntimeError` so the caller can route to the classic pipeline.
    Otherwise it raises.
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph not available")

    graph = build_agentic_rag_graph()
    if graph is None:
        raise RuntimeError("agentic graph build failed")

    from app.rag.agentic_state import make_initial_state

    initial = make_initial_state(
        query=query,
        auth=auth,
        debug=debug,
        conversation_context=conversation_context,
    )

    with trace_span(
        "agentic_rag_graph",
        metadata={
            "phase": "agentic_rag_graph",
            "query_length": len(query or ""),
            "permission_filtered": bool(auth and getattr(auth, "is_authenticated", False)),
        },
    ):
        try:
            # LangGraph state graphs accept dict-like input. invoke()
            # returns the final merged state.
            final_state = graph.invoke(dict(initial))
        except Exception as exc:
            logger.warning("Agentic graph execution failed: %s", exc)
            if bool(getattr(settings, "RAG_AGENTIC_FALLBACK_TO_CLASSIC", True)):
                raise RuntimeError(f"agentic_graph_error:{exc}") from exc
            raise

    final_answer = final_state.get("final_answer") or ""
    final_citations = list(final_state.get("final_citations") or [])
    final_metadata = dict(final_state.get("final_metadata") or {})

    return final_answer, final_citations, final_metadata
