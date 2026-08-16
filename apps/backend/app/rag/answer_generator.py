"""
Answer Generation for RAG Chat

Provides functions to generate answers with or without RAG.
Supports permission filtering and audit logging (Phase 6).

Phase 10.6: Stability fix for citation-aware answering:
- Automatic retry with stricter citation prompt when LLM omits citations
- Backend citation attachment based on content overlap with retrieved chunks
- Deterministic temperature=0 for Knowledge Base mode

Phase 20C (refined): conversation_context passed separately to prompt,
not prepended to retrieval query.
"""

import re
import time
from typing import Optional
from app.rag.retriever import (
    retrieve_chunks,
    retrieve_chunks_with_settings,
    retrieve_chunks_with_auth,
)
from app.rag.prompt_builder import build_rag_prompt, build_general_chat_prompt, build_strict_citation_prompt
from app.rag.citations import format_citations, attach_citations_to_answer
from app.rag.grounding import (
    apply_grounding_checks,
    get_debug_info,
    NO_CHUNKS_MESSAGE,
    LOW_RELEVANCE_MESSAGE,
    has_citations,
    check_citations,
)
from app.rag.image_routing import select_image_aware_chunks, select_image_content_chunks
from app.rag.query_analysis import analyze_query, QueryAnalysis
from app.rag.image_resolver import (
    ResolvedImage,
    extract_explicit_target,
    resolve_recent_image,
)
from app.services.llm import get_llm_provider
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.core.config import settings

# Phase 31A — LangSmith tracing for LLM call timing / metadata.
try:
    from app.services.langsmith_tracing import (
        trace_span,
        redact_filenames,
        safe_chunk_content,
    )
except Exception:  # pragma: no cover - tracing never required
    from contextlib import contextmanager

    @contextmanager
    def trace_span(*args, **kwargs):
        yield None

    def redact_filenames(value):
        return list(value or [])

    def safe_chunk_content(value):
        return value or ""

# Import security modules for Phase 6
try:
    from app.security.auth import AuthContext
    from app.security.audit import get_audit_logger
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    AuthContext = None


def _call_llm_with_citations(
    query: str,
    chunks: list[dict],
    strict: bool = False,
    temperature: float = 0.0,
    conversation_context: str = "",
) -> str:
    """
    Call LLM to generate answer from retrieved chunks.

    Args:
        query: User's question.
        chunks: Retrieved context chunks.
        strict: If True, use stricter citation prompt (for retry).
        temperature: LLM temperature for deterministic output.
        conversation_context: Formatted conversation context for follow-up handling.

    Returns:
        LLM generated answer text.
    """
    if strict:
        prompt = build_strict_citation_prompt(query, chunks, conversation_context)
    else:
        prompt = build_rag_prompt(query, chunks, conversation_context=conversation_context)

    return _invoke_llm(
        prompt=prompt,
        chunks=chunks,
        strict=strict,
        temperature=temperature,
        conversation_context=conversation_context,
    )


def _invoke_llm(
    prompt: str,
    chunks: list[dict],
    strict: bool,
    temperature: float,
    conversation_context: str,
) -> str:
    """
    Internal: invoke the LLM provider inside a `llm_call` span.

    The span captures provider, model, latency_ms, retry_count, error_type,
    and (when LANGSMITH_LOG_LLM_OUTPUT is on) the output length and the
    number of citation markers in the output.
    """
    provider = get_llm_provider()
    provider_name = getattr(provider, 'provider_name', 'unknown') if provider else 'unknown'
    model_name = getattr(provider, 'model', 'unknown') if provider else 'unknown'

    with trace_span(
        "llm_call",
        metadata={
            "phase": "llm_call",
            "provider": provider_name,
            "model": model_name,
            "strict_prompt": strict,
            "temperature": temperature,
            "context_chunk_count": len(chunks or []),
            "has_conversation_context": bool(conversation_context),
        },
    ) as llm_span:
        start = time.time() * 1000.0
        error_type: Optional[str] = None
        error_message: Optional[str] = None
        try:
            if hasattr(provider, 'set_temperature'):
                provider.set_temperature(temperature)
            llm_response = provider.chat(ChatRequest(message=prompt))
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)[:200]
            if llm_span is not None:
                try:
                    llm_span.set_meta("error_type", error_type)
                    llm_span.set_meta("error_message", error_message)
                    llm_span.set_meta("latency_ms", int(time.time() * 1000.0 - start))
                except Exception:
                    pass
            raise
        latency_ms = int(time.time() * 1000.0 - start)
        output = llm_response.message if llm_response else ""

        if llm_span is not None:
            try:
                llm_span.set_meta("latency_ms", latency_ms)
                llm_span.set_meta("output_length", len(output or ""))
                # Count citation markers like [1], [2] etc. — useful summary
                # even when full output logging is disabled.
                import re as _re
                markers = _re.findall(r"\[(\d+(?:,\s*\d+)*)\]", output or "")
                unique_markers = set()
                for m in markers:
                    for n in m.split(','):
                        n = n.strip()
                        if n:
                            unique_markers.add(n)
                llm_span.set_meta("citation_markers_found", len(unique_markers))
                if settings.LANGSMITH_LOG_LLM_OUTPUT:
                    # Operator has explicitly opted into logging LLM output
                    llm_span.set_meta("llm_output_preview", safe_chunk_content(output))
                llm_span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list({c.get("source_file_name") for c in chunks if c.get("source_file_name")})
                    ),
                )
            except Exception:
                pass

        return output


def _call_llm_with_citations_and_evidence(
    query: str,
    chunks: list[dict],
    strict: bool = False,
    temperature: float = 0.0,
    conversation_context: str = "",
    evidence_level: str = "strong",
) -> str:
    """
    Call LLM to generate answer with evidence-level-aware prompt.

    Phase 30E Hotfix v2: When evidence_level is "medium", the prompt includes
    a "Based on the retrieved sources..." caveat that instructs the model to
    answer cautiously and not overstate confidence.

    Args:
        query: User's question.
        chunks: Retrieved context chunks.
        strict: If True, use stricter citation prompt (for retry).
        temperature: LLM temperature for deterministic output.
        conversation_context: Formatted conversation context for follow-up handling.
        evidence_level: "strong" (default) or "medium".

    Returns:
        LLM generated answer text.
    """
    if strict:
        prompt = build_strict_citation_prompt(
            query, chunks, conversation_context, evidence_level=evidence_level
        )
    else:
        prompt = build_rag_prompt(
            query, chunks, conversation_context=conversation_context, evidence_level=evidence_level
        )

    return _invoke_llm(
        prompt=prompt,
        chunks=chunks,
        strict=strict,
        temperature=temperature,
        conversation_context=conversation_context,
    )


def _should_retry_for_citations(
    answer: str,
    chunks: list[dict],
    min_relevance_score: float,
) -> bool:
    """
    Determine if we should retry LLM with stricter citation prompt.

    Returns True if:
    - Answer lacks citations
    - Chunks exist and are strong (top_score >= threshold)
    - Answer has some content overlap with retrieved chunks (not hallucinated)

    Args:
        answer: Generated answer text.
        chunks: Retrieved chunks.
        min_relevance_score: Minimum relevance threshold.

    Returns:
        True if retry is warranted.
    """
    if has_citations(answer):
        return False

    if not chunks:
        return False

    # Check if top chunk score meets threshold
    top_score = chunks[0].get("score", 0) if chunks else 0
    if top_score < min_relevance_score:
        return False

    # Check if answer has some content overlap with chunks (not pure hallucination)
    # Use simple keyword overlap as heuristic
    answer_lower = answer.lower()
    chunk_texts = " ".join(c.get("content", "").lower() for c in chunks)

    # Extract key terms from answer (words >= 5 chars)
    answer_terms = set(re.findall(r'\b[a-z]{5,}\b', answer_lower))
    chunk_terms = set(re.findall(r'\b[a-z]{5,}\b', chunk_texts))

    # If answer has significant overlap with chunk content, it's likely grounded
    if answer_terms and chunk_terms:
        overlap = len(answer_terms & chunk_terms) / len(answer_terms)
        if overlap >= 0.3:  # At least 30% of answer terms appear in chunks
            return True

    return False


def generate_answer_with_rag(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    use_hybrid: bool = True,
    debug: bool = False,
    min_relevance_score: Optional[float] = None,
    conversation_context: str = "",
    image_context: Optional[dict] = None,
    db: Optional["object"] = None,
    auth: Optional["object"] = None,
) -> tuple[str, list[dict], dict]:
    """
    Full RAG pipeline: retrieve chunks, build prompt, call LLM, return answer + citations.

    Includes grounding checks (Phase 10):
    - Retrieval guardrail: Don't call LLM if no chunks retrieved
    - Minimum relevance threshold: Reject chunks below threshold
    - Citation enforcement: Verify answer includes citations
    - Automatic retry: Retry once with strict citations if LLM omitted them
    - Backend citation attachment: Attach citations based on content overlap

    Phase 20C: conversation_context is passed to the prompt SEPARATELY from retrieval.
    The retrieval query remains clean (user message only).

    Phase 34A.1.2 — image-content retrieval fix:
        When the query analysis classifies the question as
        ``image_content`` (e.g. "What error code is shown in the image
        I uploaded?"), the resolver attempts to:
          1. Use the explicit image_context if supplied
          2. Resolve the most recently uploaded accessible image via
             ``image_resolver.resolve_recent_image`` if ``db`` and
             ``auth`` are provided
        and then fetches the OCR chunks DIRECTLY from Qdrant by
        ``document_id`` / ``image_id`` payload filters, bypassing the
        semantic similarity floor — the source relationship itself is
        the relevance signal.

    Args:
        query: User's question.
        top_k: Number of chunks to retrieve (used when use_hybrid=False).
        score_threshold: Minimum score threshold (used when use_hybrid=False).
        use_hybrid: If True, use hybrid retrieval with all Phase 5 features.
                    If False, use original vector-only retrieval.
        debug: If True, return additional debug metadata about retrieval.
        min_relevance_score: Minimum relevance score. Defaults to RAG_MIN_RELEVANCE_SCORE.
        conversation_context: Formatted conversation context for follow-up handling.
        image_context: Optional frontend-supplied image context dict.
        db: Optional SQLAlchemy session (used by the recent-image resolver).
        auth: Optional AuthContext (used by the recent-image resolver).

    Returns:
        Tuple of (answer, citations, metadata).
        Metadata is empty when debug=False.
    """
    if min_relevance_score is None:
        min_relevance_score = settings.RAG_MIN_RELEVANCE_SCORE

    # Phase 34A.1.2 — fast path for image_content questions. We attempt
    # to resolve the image FIRST so the OCR is used as primary evidence
    # without depending on semantic similarity against the entire KB.
    #
    # Phase 34A.2.1 — also attempt pre-resolution when `image_context`
    # is supplied even if the query text does NOT match image-intent
    # patterns. "What error code is shown here?" with explicit
    # `image_context` should still use the in-scope image.
    pre_resolved_meta: dict = {}
    pre_resolved_chunks: list[dict] = []
    try:
        pre_analysis = analyze_query(query)
        
        # Phase 34A.2.1 — check for image_context even when the
        # query analysis did not classify as "image_content".
        # The frontend supplies `image_context` with real IDs when
        # the user has an attached image in scope. This is a
        # stronger signal than the query text.
        has_context_for_image = bool(
            image_context and isinstance(image_context, dict)
            and (image_context.get("document_id") or image_context.get("image_id"))
        )
        
        if pre_analysis.query_type == "image_content" or has_context_for_image:
            # Step 1: explicit image_context
            explicit = extract_explicit_target(image_context)
            resolved_doc_id = explicit.document_id
            resolved_image_id = explicit.image_id
            resolution_mode = "explicit" if explicit.resolved else None

            # Step 2: backend resolver for the recent accessible image
            if (resolved_doc_id is None and resolved_image_id is None
                    and db is not None and auth is not None):
                try:
                    recent = resolve_recent_image(db, auth)
                    pre_resolved_meta["image_resolver"] = recent.diagnostics or {}
                    if recent.resolved:
                        resolved_doc_id = recent.document_id
                        resolved_image_id = recent.image_id
                        resolution_mode = "recent"
                        pre_resolved_meta["image_resolver_resolved"] = True
                        pre_resolved_meta["resolved_document_id"] = recent.document_id
                        pre_resolved_meta["resolved_image_id"] = recent.image_id
                        pre_resolved_meta["resolved_filename"] = recent.document_filename
                except Exception as exc:
                    pre_resolved_meta["image_resolver_error"] = str(exc)[:200]

            if resolved_doc_id is not None or resolved_image_id is not None:
                resolved_chunks, resolved_meta = select_image_content_chunks(
                    pre_analysis,
                    image_context,
                    resolved_document_id=resolved_doc_id,
                    resolved_image_id=resolved_image_id,
                    resolved_filename=pre_resolved_meta.get("resolved_filename"),
                )
                pre_resolved_meta.update(resolved_meta)
                pre_resolved_meta["image_resolution_mode"] = resolution_mode or "explicit"
                if resolved_chunks:
                    pre_resolved_chunks = resolved_chunks
    except Exception as exc:
        # Never let pre-resolution crash the request — fall through to
        # standard hybrid retrieval.
        pre_resolved_meta["image_resolver_error"] = str(exc)[:200]

    if pre_resolved_chunks:
        # Image-content questions: use the resolved OCR chunks directly.
        # The relevance floor DOES NOT apply here — the source
        # relationship is the relevance signal.
        retrieval_metadata = {
            "image_routing": pre_resolved_meta,
            "image_context_used": bool(image_context),
            "retrieval_mode": pre_resolved_meta.get("routing_mode") or "image_content",
            "query_type": "image_content",
            "candidate_count": len(pre_resolved_chunks),
            "filtered_count": 0,
            "final_source_count": len(pre_resolved_chunks),
        }
        return _run_rag_with_chunks(
            query=query,
            chunks=pre_resolved_chunks,
            retrieval_metadata=retrieval_metadata,
            min_relevance_score=min_relevance_score,
            conversation_context=conversation_context,
            debug=debug,
            relevance_threshold_bypass=True,
            image_routing_meta=pre_resolved_meta,
        )

    if use_hybrid:
        chunks, retrieval_metadata = retrieve_chunks_with_settings(query, debug=debug)
    else:
        chunks = retrieve_chunks(query=query, limit=top_k, score_threshold=score_threshold)
        retrieval_metadata = {}

    # Phase 34A.1 — image-aware routing. When the query references an
    # uploaded image, scope retrieval to OCR-derived chunks so unrelated
    # KB sources (e.g. admin_test.txt) do not pollute the answer.
    analysis_for_routing = QueryAnalysis(query=query)
    try:
        qa = retrieval_metadata.get("query_analysis") or {}
        analysis_for_routing = QueryAnalysis(
            query=query,
            query_type=qa.get("query_type", "general"),
            error_codes=list(qa.get("error_codes", [])),
            technical_terms=list(qa.get("technical_terms", [])),
            references_uploaded_image=bool(retrieval_metadata.get("image_context_used"))
            or (
                retrieval_metadata.get("query_type") == "image_content"
            ),
        )
    except Exception:
        pass

    if getattr(settings, "RAG_IMAGE_AWARE_ROUTING_ENABLED", True):
        chunks, routing_meta = select_image_aware_chunks(
            chunks, analysis_for_routing, image_context=image_context
        )
        retrieval_metadata["image_routing"] = routing_meta
        retrieval_metadata["image_context_used"] = bool(
            routing_meta.get("image_context_used")
        )
    else:
        retrieval_metadata["image_routing"] = {
            "routing_mode": "disabled",
            "image_context_used": False,
            "kept_count": len(chunks),
            "dropped_count": 0,
        }
        retrieval_metadata["image_context_used"] = False

    # Surface the resolver diagnostics (e.g. "no_ocr_candidates",
    # "rbac_denied") even when the pre-resolved path didn't fire so
    # observability can see why.
    if pre_resolved_meta:
        retrieval_metadata.setdefault("image_resolver", {})
        retrieval_metadata["image_resolver"].update(pre_resolved_meta)

    return _run_rag_with_chunks(
        query=query,
        chunks=chunks,
        retrieval_metadata=retrieval_metadata,
        min_relevance_score=min_relevance_score,
        conversation_context=conversation_context,
        debug=debug,
        relevance_threshold_bypass=False,
        image_routing_meta=retrieval_metadata.get("image_routing", {}),
    )


def _run_rag_with_chunks(
    *,
    query: str,
    chunks: list[dict],
    retrieval_metadata: dict,
    min_relevance_score: float,
    conversation_context: str,
    debug: bool,
    relevance_threshold_bypass: bool,
    image_routing_meta: dict,
) -> tuple[str, list[dict], dict]:
    """Phase 34A.1.2 — shared LLM/grounding pipeline used by both the
    pre-resolved image-content path and the standard hybrid path.

    When ``relevance_threshold_bypass`` is True (image_content path)
    we skip the ``RAG_MIN_RELEVANCE_FLOOR`` because the source
    relationship IS the relevance signal. All other grounding checks
    (citation enforcement, retries, attachment) still run.
    """

    # Phase 34A.1.2 — for image_content paths, by-id chunks carry
    # ``score=None`` (Qdrant payload filter lookup, not similarity).
    # The relevance threshold check must NOT block them, so we
    # override the threshold to 0 for this branch and record the
    # bypass explicitly in metadata.
    grounding_threshold = min_relevance_score
    if relevance_threshold_bypass:
        grounding_threshold = 0.0
        retrieval_metadata["relevance_threshold_bypass"] = True
        retrieval_metadata["relevance_threshold_bypass_reason"] = (
            "image_content_source_relationship_is_relevance_signal"
        )

    # Phase 10: Apply grounding checks BEFORE calling LLM.
    # Phase 34A.1.2 — SKIP pre-LLM grounding entirely when the
    # source was resolved by image_id / document_id. The source
    # relationship IS the relevance signal. Post-LLM citation
    # enforcement still runs.
    if relevance_threshold_bypass:
        should_block = False
        fallback_message = None
        grounding_meta = {
            "evidence_level": "strong",
            "evidence_meta": {"decision": "strong", "rationale": ["resolved_image_content_source"]},
            "pre_llm_skipped": True,
            "pre_llm_skip_reason": "image_content_resolved_source",
        }
    else:
        should_block, fallback_message, grounding_meta = apply_grounding_checks(
            chunks=chunks,
            answer=None,  # No answer yet, only check retrieval
            threshold=grounding_threshold,
            require_citations=False,  # Can't require citations without an answer
            query=query,  # For topic relevance check
        )

    # CRAG decision metadata - tracks corrective RAG flow
    crag_decision = {
        "retrieval_status": "insufficient",
        "correction_attempted": False,
        "correction_type": "none",
        "fallback_reason": grounding_meta.get("blocked_reason"),
        "topic_relevance_score": grounding_meta.get("keyword_overlap_ratio"),
        "top_score": chunks[0].get("score") if chunks else None,
        "citation_count": 0,
        "blocked": False,
        "crag_enabled": False,  # Deterministic mode only for now
        "crag_decision_reason": None,
    }
    retrieval_metadata["grounding"] = grounding_meta
    retrieval_metadata["crag_decision"] = crag_decision

    if should_block:
        retrieval_metadata["blocked"] = True
        retrieval_metadata["block_reason"] = grounding_meta.get("blocked_reason", "unknown")
        if debug:
            retrieval_metadata["debug_info"] = get_debug_info(
                chunks, min_relevance_score, (should_block, fallback_message, grounding_meta)
            )
        return fallback_message, [], retrieval_metadata

    retrieval_metadata["blocked"] = False

    # Phase 10.6: Track citation repair flow explicitly
    citation_repair_meta = {
        "initial_has_citations": False,
        "retry_attempted": False,
        "retry_has_citations": False,
        "attachment_attempted": False,
        "attachment_has_citations": False,
        "final_has_citations": False,
        "final_citation_count": 0,
        "blocked_reason": None,
    }

    # Proceed with LLM call - use temperature=0 for deterministic KB output
    # Phase 20C: Pass conversation_context separately to prompt
    # Phase 30E Hotfix v2: pass evidence_level so the prompt uses the
    # medium-evidence caveat when appropriate.
    evidence_level = grounding_meta.get("evidence_level", "strong") or "strong"
    answer = _call_llm_with_citations_and_evidence(
        query,
        chunks,
        strict=False,
        temperature=0.0,
        conversation_context=conversation_context,
        evidence_level=evidence_level,
    )
    citations = format_citations(chunks)

    # Phase 10.6: Check initial citations
    citation_repair_meta["initial_has_citations"] = has_citations(answer)

    # Phase 10.6: Check answer grounding AFTER LLM generates response
    # Phase 34A.1.2 — pass retrieval_metadata so the post-LLM evidence
    # decision recognises resolved image_content sources and grants
    # STRONG evidence rather than rejecting with weak_evidence.
    should_block, fallback_message, answer_grounding_meta = apply_grounding_checks(
        chunks=chunks,
        answer=answer,
        threshold=grounding_threshold,
        require_citations=True,  # Require citations in the answer
        retrieval_metadata=retrieval_metadata,
        bypass_topic_relevance=relevance_threshold_bypass,
    )

    retrieval_metadata["grounding"].update(answer_grounding_meta)

    # Phase 10.6: Retry logic for missing citations with strong retrieval
    if not citation_repair_meta["initial_has_citations"]:
        top_score = chunks[0].get("score") or 0 if chunks else 0
        if top_score >= min_relevance_score or relevance_threshold_bypass:
            # Retry once with strict citation prompt
            citation_repair_meta["retry_attempted"] = True
            crag_decision["correction_attempted"] = True
            crag_decision["correction_type"] = "retry"
            answer = _call_llm_with_citations_and_evidence(
                query,
                chunks,
                strict=True,
                temperature=0.0,
                conversation_context=conversation_context,
                evidence_level=evidence_level,
            )
            citations = format_citations(chunks)
            citation_repair_meta["retry_has_citations"] = has_citations(answer)

    # Phase 10.6: Backend citation attachment if citations still missing but chunks are strong
    if not has_citations(answer) and chunks:
        top_score = chunks[0].get("score") or 0 if chunks else 0
        if top_score >= min_relevance_score or relevance_threshold_bypass:
            # Try to attach citations based on content overlap
            citation_repair_meta["attachment_attempted"] = True
            crag_decision["correction_attempted"] = True
            crag_decision["correction_type"] = "attachment"
            answer_with_citations, citation_map = attach_citations_to_answer(
                answer, chunks, query
            )
            if citation_map:  # If we found matching content
                answer = answer_with_citations
                citation_repair_meta["attachment_has_citations"] = has_citations(answer)
                citation_repair_meta["citation_map"] = citation_map
                retrieval_metadata["grounding"]["backend_citations_attached"] = True

    # Phase 10.6: Final check
    final_has_citations = has_citations(answer)
    citation_repair_meta["final_has_citations"] = final_has_citations
    if final_has_citations:
        _, citation_count_meta = check_citations(answer)
        citation_repair_meta["final_citation_count"] = citation_count_meta.get("citation_count", 0)
        crag_decision["citation_count"] = citation_count_meta.get("citation_count", 0)

    # Final block check only if citations still missing after ALL repair attempts.
    # Phase 34A.1.2 — for resolved image_content sources, the source
    # relationship IS the relevance signal. We have an answer and the
    # citation retry + attachment paths have already attempted to
    # add citations. If citations are still missing, do NOT invoke
    # the full evidence-aware grounding decision (which would reject
    # because the OCR body shares no vocabulary with the user's
    # meta-question). Skip the block and ship the answer; record
    # the bypass in metadata for observability.
    should_block = False
    if not final_has_citations:
        if relevance_threshold_bypass:
            retrieval_metadata["grounding"]["citation_block_bypassed"] = True
            retrieval_metadata["grounding"]["citation_block_bypass_reason"] = (
                "image_content_resolved_source_relationship_is_relevance_signal"
            )
            citation_repair_meta["blocked_reason"] = None
            # Don't block: ship the answer as-is. The LLM may have
            # produced a usable response without formal [n] markers.
        else:
            should_block, fallback_message, answer_grounding_meta = apply_grounding_checks(
                chunks=chunks,
                answer=answer,
                threshold=grounding_threshold,
                require_citations=True,
                retrieval_metadata=retrieval_metadata,
                bypass_topic_relevance=relevance_threshold_bypass,
            )
            retrieval_metadata["grounding"].update(answer_grounding_meta)
            citation_repair_meta["blocked_reason"] = "answer_lacks_citations"

    # Merge citation repair metadata into grounding metadata
    retrieval_metadata["grounding"]["citation_repair"] = citation_repair_meta
    retrieval_metadata["grounding"]["final_citation_count"] = citation_repair_meta["final_citation_count"]
    retrieval_metadata["grounding"]["final_has_citations"] = final_has_citations

    if should_block:
        retrieval_metadata["blocked"] = True
        retrieval_metadata["block_reason"] = citation_repair_meta["blocked_reason"] or "answer_lacks_citations"
        if debug:
            retrieval_metadata["debug_info"] = get_debug_info(
                chunks, min_relevance_score, (should_block, fallback_message, answer_grounding_meta)
            )
        return fallback_message, [], retrieval_metadata

    # Add debug info if requested
    if debug:
        retrieval_metadata["debug_info"] = get_debug_info(
            chunks, min_relevance_score, (False, None, retrieval_metadata["grounding"])
        )

    # Phase 31A — emit a `final_response` span summarising the outcome.
    with trace_span(
        "final_response",
        metadata={
            "phase": "final_response",
            "blocked": retrieval_metadata.get("blocked", False),
            "block_reason": retrieval_metadata.get("block_reason"),
            "citation_count": len(citations or []),
            "evidence_level": (retrieval_metadata.get("grounding") or {}).get("evidence_level"),
            "fallback_reason": (retrieval_metadata.get("grounding") or {}).get("blocked_reason"),
        },
    ) as final_span:
        if final_span is not None:
            try:
                final_span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list({c.get("source_file_name") for c in citations if c.get("source_file_name")})
                    ),
                )
            except Exception:
                pass

    return answer, citations, retrieval_metadata


def generate_answer_with_rag_audit(
    query: str,
    auth: AuthContext,
    debug: bool = False,
    use_hybrid: bool = True,
    request_ip: Optional[str] = None,
    request_user_agent: Optional[str] = None,
    min_relevance_score: Optional[float] = None,
    conversation_context: str = "",
    image_context: Optional[dict] = None,
    db: Optional["object"] = None,
) -> tuple[str, list[dict], dict]:
    """
    Full RAG pipeline with permission filtering and audit logging (Phase 6).

    Includes grounding checks (Phase 10):
    - Retrieval guardrail: Don't call LLM if no chunks retrieved
    - Minimum relevance threshold: Reject chunks below threshold
    - Citation enforcement: Verify answer includes citations

    Phase 20C: conversation_context is passed to the prompt SEPARATELY from retrieval.
    The retrieval query remains clean (user message only).

    Phase 34A.1.2 — image-content retrieval fast path. Mirrors the
    non-audit ``generate_answer_with_rag``: when the query analysis
    classifies the question as ``image_content`` we resolve the
    relevant image (explicit context first, then the most recent
    accessible image via ``image_resolver.resolve_recent_image``) and
    fetch its OCR chunks DIRECTLY from Qdrant by payload filter,
    bypassing semantic similarity. The relevance floor is skipped
    because the source relationship is the relevance signal.

    This function:
    1. Performs permission-aware retrieval (filters unauthorized documents)
    2. Applies grounding checks (Phase 10)
    3. Generates answer using LLM
    4. Logs the interaction to audit log

    Args:
        query: User's question.
        auth: Authentication context with user info and role.
        debug: If True, return additional debug metadata.
        use_hybrid: Use hybrid retrieval (default True).
        request_ip: Client IP for audit logging.
        request_user_agent: Client user agent for audit logging.
        min_relevance_score: Minimum relevance score. Defaults to RAG_MIN_RELEVANCE_SCORE.
        conversation_context: Formatted conversation context for follow-up handling.
        image_context: Optional frontend-supplied image context dict.
        db: Optional SQLAlchemy session (used by the recent-image resolver).

    Returns:
        Tuple of (answer, citations, metadata).
    """
    if min_relevance_score is None:
        min_relevance_score = settings.RAG_MIN_RELEVANCE_SCORE

    # Get LLM provider info
    provider = get_llm_provider()
    model_provider = getattr(provider, 'provider_name', 'unknown') if provider else 'unknown'
    model_name = getattr(provider, 'model', 'unknown') if provider else 'unknown'

    # Phase 34A.1.2 — image-content pre-resolution. When the user is
    # asking about the content of an uploaded image, attempt to
    # resolve the relevant image BEFORE doing hybrid retrieval so the
    # OCR becomes the primary evidence.
    pre_resolved_meta: dict = {}
    pre_resolved_chunks: list[dict] = []
    try:
        pre_analysis = analyze_query(query)
        if pre_analysis.query_type == "image_content":
            explicit = extract_explicit_target(image_context)
            resolved_doc_id = explicit.document_id
            resolved_image_id = explicit.image_id
            resolution_mode = "explicit" if explicit.resolved else None

            if (resolved_doc_id is None and resolved_image_id is None
                    and db is not None and auth is not None):
                try:
                    recent = resolve_recent_image(db, auth)
                    pre_resolved_meta["image_resolver"] = recent.diagnostics or {}
                    if recent.resolved:
                        resolved_doc_id = recent.document_id
                        resolved_image_id = recent.image_id
                        resolution_mode = "recent"
                        pre_resolved_meta["image_resolver_resolved"] = True
                        pre_resolved_meta["resolved_document_id"] = recent.document_id
                        pre_resolved_meta["resolved_image_id"] = recent.image_id
                        pre_resolved_meta["resolved_filename"] = recent.document_filename
                except Exception as exc:
                    pre_resolved_meta["image_resolver_error"] = str(exc)[:200]

            if resolved_doc_id is not None or resolved_image_id is not None:
                resolved_chunks, resolved_meta = select_image_content_chunks(
                    pre_analysis,
                    image_context,
                    resolved_document_id=resolved_doc_id,
                    resolved_image_id=resolved_image_id,
                    resolved_filename=pre_resolved_meta.get("resolved_filename"),
                )
                pre_resolved_meta.update(resolved_meta)
                pre_resolved_meta["image_resolution_mode"] = resolution_mode or "explicit"
                if resolved_chunks:
                    pre_resolved_chunks = resolved_chunks
    except Exception as exc:
        pre_resolved_meta["image_resolver_error"] = str(exc)[:200]

    if pre_resolved_chunks:
        # Permission filtering is enforced INSIDE the by-id fetch path:
        # we still apply the existing RBAC filter to the resolved
        # document_id so a user cannot pull another tenant's OCR via
        # the resolver.
        from app.security.permissions import can_access_document
        if HAS_SECURITY and auth is not None and auth.is_authenticated:
            try:
                accessible = can_access_document(
                    auth, int(pre_resolved_chunks[0].get("document_id") or 0)
                )
            except Exception:
                accessible = False
            if not accessible and not getattr(auth, "is_admin", lambda: False)():
                # Re-check with the explicit admin/sysadmin helpers used
                # elsewhere in the codebase; non-admins get the
                # grounded fallback.
                from app.security.permissions import _is_admin, _is_sysadmin, _dev_bypass_active
                if not (_is_admin(auth) or _is_sysadmin(auth) or _dev_bypass_active(auth)):
                    return (
                        "I don't have enough information in the provided sources on that topic.",
                        [],
                        {
                            "blocked": True,
                            "block_reason": "rbac_denied",
                            "image_routing": pre_resolved_meta,
                            "image_resolver": pre_resolved_meta,
                            "query_type": "image_content",
                            "retrieval_mode": pre_resolved_meta.get("routing_mode") or "image_content",
                            "candidate_count": 0,
                            "filtered_count": 0,
                            "final_source_count": 0,
                        },
                    )

        retrieval_metadata = {
            "image_routing": pre_resolved_meta,
            "image_context_used": bool(image_context),
            "retrieval_mode": pre_resolved_meta.get("routing_mode") or "image_content",
            "query_type": "image_content",
            "candidate_count": len(pre_resolved_chunks),
            "filtered_count": 0,
            "final_source_count": len(pre_resolved_chunks),
        }
        answer, citations, retrieval_metadata = _run_rag_with_chunks_audit(
            query=query,
            chunks=pre_resolved_chunks,
            retrieval_metadata=retrieval_metadata,
            min_relevance_score=min_relevance_score,
            conversation_context=conversation_context,
            debug=debug,
            relevance_threshold_bypass=True,
            image_routing_meta=pre_resolved_meta,
            auth=auth,
            model_provider=model_provider,
            model_name=model_name,
            request_ip=request_ip,
            request_user_agent=request_user_agent,
        )
        return answer, citations, retrieval_metadata

    # Perform permission-aware retrieval
    if HAS_SECURITY and auth.is_authenticated:
        chunks, retrieval_metadata = retrieve_chunks_with_auth(
            query=query,
            auth=auth,
            debug=debug,
        )
    else:
        # No auth, fall back to regular retrieval
        chunks, retrieval_metadata = retrieve_chunks_with_settings(query, debug=debug)

    # Phase 34A.1 — image-aware routing. Restrict results when the user
    # is asking about content in an uploaded image; otherwise pass
    # through.
    analysis_for_routing = QueryAnalysis(query=query)
    try:
        qa = retrieval_metadata.get("query_analysis") or {}
        analysis_for_routing = QueryAnalysis(
            query=query,
            query_type=qa.get("query_type", "general"),
            error_codes=list(qa.get("error_codes", [])),
            technical_terms=list(qa.get("technical_terms", [])),
            references_uploaded_image=bool(retrieval_metadata.get("image_context_used"))
            or (retrieval_metadata.get("query_type") == "image_content"),
        )
    except Exception:
        pass

    if getattr(settings, "RAG_IMAGE_AWARE_ROUTING_ENABLED", True):
        chunks, routing_meta = select_image_aware_chunks(
            chunks, analysis_for_routing, image_context=image_context
        )
        retrieval_metadata["image_routing"] = routing_meta

    # Surface the resolver diagnostics (e.g. "no_ocr_candidates",
    # "rbac_denied") even when the pre-resolved path didn't fire so
    # observability can see why.
    if pre_resolved_meta:
        retrieval_metadata.setdefault("image_resolver", {})
        retrieval_metadata["image_resolver"].update(pre_resolved_meta)

    return _run_rag_with_chunks_audit(
        query=query,
        chunks=chunks,
        retrieval_metadata=retrieval_metadata,
        min_relevance_score=min_relevance_score,
        conversation_context=conversation_context,
        debug=debug,
        relevance_threshold_bypass=False,
        image_routing_meta=retrieval_metadata.get("image_routing", {}),
        auth=auth,
        model_provider=model_provider,
        model_name=model_name,
        request_ip=request_ip,
        request_user_agent=request_user_agent,
    )


def _run_rag_with_chunks_audit(
    *,
    query: str,
    chunks: list[dict],
    retrieval_metadata: dict,
    min_relevance_score: float,
    conversation_context: str,
    debug: bool,
    relevance_threshold_bypass: bool,
    image_routing_meta: dict,
    auth,
    model_provider: str,
    model_name: str,
    request_ip: Optional[str],
    request_user_agent: Optional[str],
) -> tuple[str, list[dict], dict]:
    """Phase 34A.1.2 — audit-aware counterpart of ``_run_rag_with_chunks``.

    Same bypass semantics for the relevance threshold on
    ``image_content`` paths. Adds the existing audit logging + permission
    filter pass-through that the original function relied on.
    """
    # Phase 34A.1.2 — bypass the relevance floor for image_content
    # sources (by-id chunks have score=None). Source relationship is
    # the relevance signal.
    grounding_threshold = min_relevance_score
    if relevance_threshold_bypass:
        grounding_threshold = 0.0
        retrieval_metadata["relevance_threshold_bypass"] = True
        retrieval_metadata["relevance_threshold_bypass_reason"] = (
            "image_content_source_relationship_is_relevance_signal"
        )

    # Phase 10: Apply grounding checks BEFORE calling LLM.
    # Phase 34A.1.2 — SKIP pre-LLM grounding entirely when the source
    # was resolved by image_id / document_id. The source relationship
    # IS the relevance signal; OCR body rarely shares vocabulary with
    # the user's meta-question so the topic-relevance and evidence-level
    # checks would otherwise reject every well-resolved image-content
    # question. Post-LLM citation enforcement still runs.
    if relevance_threshold_bypass:
        should_block = False
        fallback_message = None
        grounding_meta = {
            "evidence_level": "strong",
            "evidence_meta": {"decision": "strong", "rationale": ["resolved_image_content_source"]},
            "pre_llm_skipped": True,
            "pre_llm_skip_reason": "image_content_resolved_source",
        }
    else:
        should_block, fallback_message, grounding_meta = apply_grounding_checks(
            chunks=chunks,
            answer=None,
            threshold=grounding_threshold,
            require_citations=False,
            query=query,  # For topic relevance check
        )

    # CRAG decision metadata - tracks corrective RAG flow
    crag_decision = {
        "retrieval_status": "insufficient",
        "correction_attempted": False,
        "correction_type": "none",
        "fallback_reason": grounding_meta.get("blocked_reason"),
        "topic_relevance_score": grounding_meta.get("keyword_overlap_ratio"),
        "top_score": chunks[0].get("score") if chunks else None,
        "citation_count": 0,
        "blocked": False,
        "crag_enabled": False,  # Deterministic mode only for now
        "crag_decision_reason": None,
    }
    retrieval_metadata["grounding"] = grounding_meta
    retrieval_metadata["crag_decision"] = crag_decision

    if should_block:
        retrieval_metadata["blocked"] = True
        retrieval_metadata["block_reason"] = grounding_meta.get("blocked_reason", "unknown")
        crag_decision["blocked"] = True
        crag_decision["crag_decision_reason"] = "blocked_" + (grounding_meta.get("blocked_reason") or "unknown")

        # Log blocked retrieval
        if HAS_SECURITY and auth.is_authenticated:
            try:
                audit_logger = get_audit_logger()
                audit_logger.log_chat_rag(
                    auth=auth,
                    question=query,
                    retrieved_document_ids=[],
                    retrieved_chunk_ids=[],
                    model_provider=model_provider,
                    model_name=model_name,
                    status="blocked",  # Blocked due to low grounding
                    request_ip=request_ip,
                    request_user_agent=request_user_agent,
                )
            except Exception:
                pass

        retrieval_metadata["audit_logged"] = False

        if debug:
            retrieval_metadata["debug_info"] = get_debug_info(
                chunks, min_relevance_score, (should_block, fallback_message, grounding_meta)
            )

        return fallback_message, [], retrieval_metadata

    retrieval_metadata["blocked"] = False

    # Extract document and chunk IDs for audit
    retrieved_doc_ids = list(set(c.get("document_id") for c in chunks if c.get("document_id")))
    retrieved_chunk_ids = list(set(c.get("chunk_id") for c in chunks if c.get("chunk_id")))

    # Phase 30E Hotfix v2: extract evidence_level from grounding meta so the
    # prompt uses the medium-evidence caveat when appropriate.
    evidence_level = grounding_meta.get("evidence_level", "strong") or "strong"

    # Build prompt and generate answer - use temperature=0 for deterministic KB output
    # Phase 20C: Pass conversation_context separately to prompt
    # Phase 34A.1.2 — resolve the provider locally (this helper is also
    # called from the pre-resolved image_content path which does NOT
    # have provider in scope).
    _audit_provider = get_llm_provider()
    prompt = build_rag_prompt(
        query,
        chunks,
        conversation_context=conversation_context,
        evidence_level=evidence_level,
    )
    llm_request = ChatRequest(message=prompt)
    if _audit_provider is not None and hasattr(_audit_provider, 'set_temperature'):
        _audit_provider.set_temperature(0.0)
    if _audit_provider is None:
        raise RuntimeError("LLM provider is not configured")
    llm_response = _audit_provider.chat(llm_request)
    answer = llm_response.message
    citations = format_citations(chunks)

    # Phase 10.6: Track citation repair flow explicitly
    citation_repair_meta = {
        "initial_has_citations": has_citations(answer),
        "retry_attempted": False,
        "retry_has_citations": False,
        "attachment_attempted": False,
        "attachment_has_citations": False,
        "final_has_citations": False,
        "final_citation_count": 0,
        "blocked_reason": None,
    }

    # Phase 10.6: Retry logic for missing citations with strong retrieval
    if not citation_repair_meta["initial_has_citations"]:
        top_score = chunks[0].get("score") or 0 if chunks else 0
        if top_score >= min_relevance_score or relevance_threshold_bypass:
            # Retry once with strict citation prompt
            citation_repair_meta["retry_attempted"] = True
            crag_decision["correction_attempted"] = True
            crag_decision["correction_type"] = "retry"
            strict_prompt = build_strict_citation_prompt(
                query,
                chunks,
                conversation_context,
                evidence_level=evidence_level,
            )
            llm_request = ChatRequest(message=strict_prompt)
            if _audit_provider is not None and hasattr(_audit_provider, 'set_temperature'):
                _audit_provider.set_temperature(0.0)
            llm_response = _audit_provider.chat(llm_request)
            answer = llm_response.message
            citations = format_citations(chunks)
            citation_repair_meta["retry_has_citations"] = has_citations(answer)

    # Phase 10.6: Backend citation attachment if citations still missing but chunks are strong
    if not has_citations(answer) and chunks:
        top_score = chunks[0].get("score") or 0 if chunks else 0
        if top_score >= min_relevance_score or relevance_threshold_bypass:
            # Try to attach citations based on content overlap
            citation_repair_meta["attachment_attempted"] = True
            crag_decision["correction_attempted"] = True
            crag_decision["correction_type"] = "attachment"
            answer_with_citations, citation_map = attach_citations_to_answer(
                answer, chunks, query
            )
            if citation_map:  # If we found matching content
                answer = answer_with_citations
                citation_repair_meta["attachment_has_citations"] = has_citations(answer)
                citation_repair_meta["citation_map"] = citation_map
                retrieval_metadata["grounding"]["backend_citations_attached"] = True

    # Phase 10.6: Final check
    final_has_citations = has_citations(answer)
    citation_repair_meta["final_has_citations"] = final_has_citations
    if final_has_citations:
        _, citation_count_meta = check_citations(answer)
        citation_repair_meta["final_citation_count"] = citation_count_meta.get("citation_count", 0)

    retrieval_metadata["grounding"]["citation_repair"] = citation_repair_meta
    retrieval_metadata["grounding"]["final_citation_count"] = citation_repair_meta["final_citation_count"]
    retrieval_metadata["grounding"]["final_has_citations"] = final_has_citations
    crag_decision["citation_count"] = citation_repair_meta.get("final_citation_count", 0)

    # Block only if citations still missing after ALL repair attempts.
    # Phase 34A.1.2 — resolved image_content sources short-circuit the
    # final block check; see _run_rag_with_chunks for the rationale.
    if not final_has_citations:
        if relevance_threshold_bypass:
            retrieval_metadata["grounding"]["citation_block_bypassed"] = True
            retrieval_metadata["grounding"]["citation_block_bypass_reason"] = (
                "image_content_resolved_source_relationship_is_relevance_signal"
            )
            citation_repair_meta["blocked_reason"] = None
        else:
            should_block, fallback_message, answer_grounding_meta = apply_grounding_checks(
                chunks=chunks,
                answer=answer,
                threshold=grounding_threshold,
                require_citations=True,
                retrieval_metadata=retrieval_metadata,
                bypass_topic_relevance=relevance_threshold_bypass,
            )
            retrieval_metadata["grounding"].update(answer_grounding_meta)
            citation_repair_meta["blocked_reason"] = "answer_lacks_citations"
            retrieval_metadata["blocked"] = True
            retrieval_metadata["block_reason"] = "answer_lacks_citations"

        # Log blocked due to no citations
        if HAS_SECURITY and auth.is_authenticated:
            try:
                audit_logger = get_audit_logger()
                audit_logger.log_chat_rag(
                    auth=auth,
                    question=query,
                    retrieved_document_ids=retrieved_doc_ids,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    model_provider=model_provider,
                    model_name=model_name,
                    status="blocked_no_citations",
                    request_ip=request_ip,
                    request_user_agent=request_user_agent,
                )
            except Exception:
                pass

        retrieval_metadata["audit_logged"] = False

        if debug:
            retrieval_metadata["debug_info"] = get_debug_info(
                chunks, min_relevance_score, (should_block, fallback_message, answer_grounding_meta)
            )

        return fallback_message, [], retrieval_metadata

    # Log successful RAG interaction
    if HAS_SECURITY and auth.is_authenticated:
        try:
            audit_logger = get_audit_logger()
            audit_logger.log_chat_rag(
                auth=auth,
                question=query,
                retrieved_document_ids=retrieved_doc_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                model_provider=model_provider,
                model_name=model_name,
                status="success",
                request_ip=request_ip,
                request_user_agent=request_user_agent,
            )
            retrieval_metadata["audit_logged"] = True
        except Exception:
            retrieval_metadata["audit_logged"] = False

    # Add debug info if requested
    if debug:
        retrieval_metadata["debug_info"] = get_debug_info(
            chunks, min_relevance_score, (False, None, retrieval_metadata["grounding"])
        )

    # Phase 31A — final_response span summarising the audited outcome.
    with trace_span(
        "final_response",
        metadata={
            "phase": "final_response",
            "blocked": retrieval_metadata.get("blocked", False),
            "block_reason": retrieval_metadata.get("block_reason"),
            "citation_count": len(citations or []),
            "evidence_level": (retrieval_metadata.get("grounding") or {}).get("evidence_level"),
            "fallback_reason": (retrieval_metadata.get("grounding") or {}).get("blocked_reason"),
            "audit_logged": retrieval_metadata.get("audit_logged", False),
        },
    ) as final_span:
        if final_span is not None:
            try:
                final_span.set_meta(
                    "source_file_names",
                    redact_filenames(
                        list({c.get("source_file_name") for c in citations if c.get("source_file_name")})
                    ),
                )
            except Exception:
                pass

    return answer, citations, retrieval_metadata


def generate_answer_without_rag(query: str) -> str:
    """General chat without RAG - uses business-friendly prompt."""
    provider = get_llm_provider()
    prompt = build_general_chat_prompt(query)
    llm_request = ChatRequest(message=prompt)
    llm_response = provider.chat(llm_request)
    return llm_response.message


def generate_answer_without_rag_audit(
    query: str,
    auth: AuthContext,
    request_ip: Optional[str] = None,
    request_user_agent: Optional[str] = None,
) -> str:
    """
    Normal chat without RAG, with audit logging (Phase 6).
    """
    # Log the chat message
    if HAS_SECURITY and auth.is_authenticated:
        try:
            audit_logger = get_audit_logger()
            audit_logger.log_chat_message(
                auth=auth,
                status="success",
                request_ip=request_ip,
                request_user_agent=request_user_agent,
            )
        except Exception:
            pass  # Don't fail the request if audit logging fails

    return generate_answer_without_rag(query)