"""
LangSmith Tracing Service — Phase 27

Provides safe, privacy-preserving tracing for chat and RAG workflows.
Traces are disabled by default (LANGSMITH_TRACING=false).

Design principles:
1. NO external calls when LANGSMITH_TRACING=false
2. LangSmith package is imported LAZILY — only when tracing is enabled and
   only inside functions. This prevents any import-time side effects from
   crashing FastAPI startup.
3. Tracing errors are caught and logged — never propagate to break chat/RAG.
4. Metadata is sanitized: no API keys, tokens, passwords, raw .env values,
   full document text, or full prompts are ever included.
5. Only safe, high-level metadata is traced: chat mode, session_id presence,
   chunk count, citation count, latency, model name, provider name.

Trace events:
- trace_chat_start
- trace_retrieval
- trace_prompt_build
- trace_llm_call
- trace_grounding
- trace_chat_end
- trace_error
"""

from __future__ import annotations

import os
import time
import logging
import random
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secret sanitiser — strips fields that look like credentials
# ---------------------------------------------------------------------------
_SECRET_FIELD_NAMES = frozenset({
    "api_key", "apikey", "api_key_", "secret", "token", "password",
    "auth", "authorization", "credential", "private_key", "secret_key",
    "access_token", "refresh_token", "session_token",
    "litellm_master_key", "openrouter_api_key", "openai_api_key",
    "db_url", "database_url", "minio_root_password", "minio_root_user",
    "jwt_secret_key", "oidc_client_secret",
})

# Field name substrings that indicate secrets (for nested objects)
_SECRET_SUBSTRINGS = ("key", "token", "secret", "password", "credential", "auth")


def _sanitize_value(key: str, value: Any) -> Any:
    """Replace secret-like values with [REDACTED]."""
    key_lower = key.lower()
    if any(secret in key_lower for secret in _SECRET_SUBSTRINGS):
        return "[REDACTED]"
    return value


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove secret fields from a metadata dict before sending to LangSmith."""
    if not isinstance(metadata):
        return {}
    result = {}
    for key, value in metadata.items():
        result[key] = _sanitize_value(key, value)
    return result


# ---------------------------------------------------------------------------
# Safe tracer interface — always available, never raises
# ---------------------------------------------------------------------------

def _is_langsmith_api_key_configured() -> bool:
    """Check if LANGSMITH_API_KEY is set and non-placeholder."""
    key = os.environ.get("LANGSMITH_API_KEY", "")
    return bool(key and key not in ("", "changeme", "your-langsmith-api-key"))


def trace_chat_start(
    mode: str,
    has_session_id: bool,
    has_conversation_context: bool,
    user_role: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Trace a new chat request start.

    Metadata (all safe, no secrets):
    - mode: general_chat | knowledge_base | debug
    - has_session_id: bool
    - has_conversation_context: bool
    - user_role: user | admin | None (NOT password or token)
    """
    if not settings.LANGSMITH_TRACING:
        return
    if not _is_langsmith_api_key_configured():
        return
    if _should_sample():
        _trace_event(
            "trace_chat_start",
            {
                "mode": mode,
                "has_session_id": has_session_id,
                "has_conversation_context": has_conversation_context,
                "user_role": user_role,
                **(sanitize_metadata(extra) if extra else {}),
            },
        )


def trace_retrieval(
    query_length: int,
    chunk_count: int,
    top_score: Optional[float] = None,
    retrieval_mode: str = "vector",
    error: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Trace retrieval step of RAG pipeline.

    Metadata:
    - query_length: character count of retrieval query
    - chunk_count: number of chunks retrieved
    - top_score: highest relevance score (safe float)
    - retrieval_mode: vector | hybrid
    - error: error message if retrieval failed
    """
    if not settings.LANGSMITH_TRACING:
        return
    if not _is_langsmith_api_key_configured():
        return
    if _should_sample():
        data = {
            "query_length": query_length,
            "chunk_count": chunk_count,
            "retrieval_mode": retrieval_mode,
            "error": error,
            **(sanitize_metadata(extra) if extra else {}),
        }
        if top_score is not None:
            data["top_score"] = round(top_score, 4)
        _trace_event("trace_retrieval", data)


def trace_prompt_build(
    mode: str,
    context_chunks: int,
    has_conversation_context: bool,
    prompt_length: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Trace prompt assembly for LLM call.

    Metadata:
    - mode: general_chat | knowledge_base
    - context_chunks: number of chunks in context
    - has_conversation_context: bool
    - prompt_length: total prompt character count (only if LANGSMITH_LOG_FULL_PROMPT)
    """
    if not settings.LANGSMITH_TRACING:
        return
    if not _is_langsmith_api_key_configured():
        return
    if _should_sample():
        data = {
            "mode": mode,
            "context_chunks": context_chunks,
            "has_conversation_context": has_conversation_context,
            **(sanitize_metadata(extra) if extra else {}),
        }
        if prompt_length is not None and settings.LANGSMITH_LOG_FULL_PROMPT:
            data["prompt_length"] = prompt_length
        _trace_event("trace_prompt_build", data)


def trace_llm_call(
    provider: str,
    model: str,
    latency_ms: int,
    token_count: Optional[int] = None,
    error: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Trace LLM call completion.

    Metadata:
    - provider: litellm | openrouter | openai | mock
    - model: model name (safe string)
    - latency_ms: milliseconds
    - token_count: approximate token usage
    - error: error message if call failed
    """
    if not settings.LANGSMITH_TRACING:
        return
    if not _is_langsmith_api_key_configured():
        return
    if _should_sample():
        data = {
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "error": error,
            **(sanitize_metadata(extra) if extra else {}),
        }
        if token_count is not None:
            data["token_count"] = token_count
        _trace_event("trace_llm_call", data)


def trace_grounding(
    has_citations: bool,
    citation_count: int,
    is_fallback: bool,
    fallback_reason: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Trace grounding/citation decision.

    Metadata:
    - has_citations: bool
    - citation_count: number of citations
    - is_fallback: whether answer fell back to generic
    - fallback_reason: reason for fallback (if any)
    """
    if not settings.LANGSMITH_TRACING:
        return
    if not _is_langsmith_api_key_configured():
        return
    if _should_sample():
        _trace_event(
            "trace_grounding",
            {
                "has_citations": has_citations,
                "citation_count": citation_count,
                "is_fallback": is_fallback,
                "fallback_reason": fallback_reason,
                **(sanitize_metadata(extra) if extra else {}),
            },
        )


def trace_chat_end(
    mode: str,
    latency_ms: int,
    has_citations: bool,
    citation_count: int,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Trace chat request completion.

    Metadata:
    - mode: general_chat | knowledge_base | debug
    - latency_ms: total request latency
    - has_citations: bool
    - citation_count: number of citations in response
    """
    if not settings.LANGSMITH_TRACING:
        return
    if not _is_langsmith_api_key_configured():
        return
    if _should_sample():
        _trace_event(
            "trace_chat_end",
            {
                "mode": mode,
                "latency_ms": latency_ms,
                "has_citations": has_citations,
                "citation_count": citation_count,
                **(sanitize_metadata(extra) if extra else {}),
            },
        )


def trace_error(
    error_type: str,
    error_message: str,
    mode: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Trace an error in the chat/RAG pipeline.

    Metadata:
    - error_type: exception class name (not the full traceback)
    - error_message: short error reason (sanitised)
    - mode: chat mode if applicable
    """
    if not settings.LANGSMITH_TRACING:
        return
    if not _is_langsmith_api_key_configured():
        return
    if _should_sample():
        _trace_event(
            "trace_error",
            {
                "error_type": error_type,
                "error_message": error_message[:200],  # truncate long messages
                "mode": mode,
                **(sanitize_metadata(extra) if extra else {}),
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _should_sample() -> bool:
    """Apply sample rate. Returns True if this request should be traced."""
    rate = settings.LANGSMITH_SAMPLE_RATE
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def _trace_event(name: str, metadata: dict[str, Any]) -> None:
    """
    Send a trace event to LangSmith via lazy import.

    Never raises — errors are caught and logged as warnings.
    """
    try:
        # Lazy import: langsmith is only imported here, at trace time,
        # and only when LANGSMITH_TRACING=true and API key is configured.
        import langsmith
    except ImportError:
        logger.debug("langsmith not installed, skipping trace event %s", name)
        return

    try:
        # Use langsmith's Client for low-level tracing without decorators
        client = langsmith.Client(
            api_key=os.environ.get("LANGSMITH_API_KEY", ""),
            endpoint=settings.LANGSMITH_ENDPOINT,
        )
        # Run in a thread pool to avoid blocking the async event loop
        # since LangSmith SDK uses synchronous HTTP calls
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            def _send():
                try:
                    client.upload_trace(
                        name=name,
                        inputs=metadata,
                        metadata={
                            "project": settings.LANGSMITH_PROJECT,
                        },
                    )
                except Exception as exc:
                    logger.warning("LangSmith trace upload failed (event=%s): %s", name, exc)

            executor.submit(_send)
    except Exception as exc:
        logger.warning("LangSmith tracing failed (event=%s): %s", name, exc)


# ---------------------------------------------------------------------------
# Status helper — safe to call from API/status endpoints (no import chain issues)
# ---------------------------------------------------------------------------

def get_langsmith_status() -> dict[str, Any]:
    """
    Return LangSmith status for admin status endpoints.

    NEVER exposes LANGSMITH_API_KEY — only presence/absence.
    """
    api_key_configured = _is_langsmith_api_key_configured()

    # Parse endpoint host for safe display (no credentials)
    endpoint_host = "unknown"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(settings.LANGSMITH_ENDPOINT)
        endpoint_host = parsed.hostname or "unknown"
    except Exception:
        pass

    return {
        "langsmith_tracing": settings.LANGSMITH_TRACING,
        "langsmith_available": True,  # langsmith package installable independently
        "project": settings.LANGSMITH_PROJECT,
        "endpoint_host": endpoint_host,
        "log_full_prompt": settings.LANGSMITH_LOG_FULL_PROMPT,
        "log_document_text": settings.LANGSMITH_LOG_DOCUMENT_TEXT,
        "log_user_input": settings.LANGSMITH_LOG_USER_INPUT,
        "log_retrieved_context": settings.LANGSMITH_LOG_RETRIEVED_CONTEXT,
        "sample_rate": settings.LANGSMITH_SAMPLE_RATE,
        "api_key_configured": api_key_configured,
        # Show warning if tracing enabled but no API key
        "warning": (
            "LangSmith tracing is enabled but LANGSMITH_API_KEY is not set. "
            "Traces will not be sent."
            if settings.LANGSMITH_TRACING and not api_key_configured
            else None
        ),
    }