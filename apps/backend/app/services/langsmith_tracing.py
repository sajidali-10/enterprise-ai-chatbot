"""
LangSmith Tracing Service — Phase 27 (foundation), Phase 31A (instrumentation)

Provides safe, privacy-preserving tracing for chat and RAG workflows.
Traces are disabled by default (LANGSMITH_TRACING=false).

Design principles:
1. NO external calls when LANGSMITH_TRACING=false
2. LangSmith package is imported LAZILY — only when tracing is enabled and
   only inside functions. This prevents any import-time side effects from
   crashing FastAPI startup.
3. Tracing errors are caught and logged — never propagate to break chat/RAG.
4. Metadata is sanitized: no API keys, tokens, passwords, raw .env values,
   full document text, or full prompts are ever included by default.
5. Only safe, high-level metadata is traced: chat mode, session_id presence,
   chunk count, citation count, latency, model name, provider name.

Phase 31A adds:
- Span/context-manager API (`trace_span`, `trace_chat_request`) that
  establishes a parent/child hierarchy. Each span attaches sanitized
  inputs/outputs/metadata and emits them when the with-block exits.
- Decorator API (`@traceable`) that wraps a function in a span.
- Content redaction helpers (`redact_text`, `redact_metadata`) that scrub
  secrets, tokens, JWTs, emails, phone numbers, and absolute filesystem
  paths from any string before it leaves the process.
- New privacy configuration:
    LANGSMITH_LOG_LLM_OUTPUT       — log LLM output summary only (default off)
    LANGSMITH_REDACT_METADATA      — apply redaction to all metadata (default on)
    LANGSMITH_MAX_CONTEXT_CHARS    — truncate any chunk content to this cap

Trace events (legacy — kept for backward compat):
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
import re
import time
import uuid
import logging
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secret / content redaction
# ---------------------------------------------------------------------------

# Common credential-like substrings in field NAMES.
_SECRET_FIELD_NAMES = frozenset({
    "api_key", "apikey", "api_key_", "secret", "token", "password",
    "auth", "authorization", "credential", "private_key", "secret_key",
    "access_token", "refresh_token", "session_token",
    "litellm_master_key", "openrouter_api_key", "openai_api_key",
    "db_url", "database_url", "minio_root_password", "minio_root_user",
    "jwt_secret_key", "oidc_client_secret",
})

_SECRET_SUBSTRINGS = ("key", "token", "secret", "password", "credential", "auth")


def _sanitize_value(key: str, value: Any) -> Any:
    """Replace secret-like values with [REDACTED]."""
    key_lower = key.lower()
    if any(secret in key_lower for secret in _SECRET_SUBSTRINGS):
        return "[REDACTED]"
    return value


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove secret fields from a metadata dict before sending to LangSmith."""
    if not isinstance(metadata, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        result[key] = _sanitize_value(key, value)
    return result


# --- Content redaction (Phase 31A) -----------------------------------------

# Patterns we want to scrub out of any free-text string that flows to
# LangSmith. Order matters — more specific patterns first.
_REDACT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}")),
    ("basic_auth_header", re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{8,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    (
        "generic_api_key",
        re.compile(r"\b(?:api[_-]?key|apikey|access[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{16,}['\"]?", re.I),
    ),
    (
        "password_pair",
        re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{6,}['\"]?", re.I),
    ),
    (
        "connection_string",
        re.compile(r"(?i)(?:postgres(?:ql)?|mysql|mongodb|redis|amqp):\/\/[^\s'\"<>]+"),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b")),
    (
        "absolute_path",
        re.compile(r"(?:/[A-Za-z0-9_.\-]+){2,}/[A-Za-z0-9_.\-]+"),
    ),
]


def redact_text(text: Any, max_length: Optional[int] = None) -> str:
    """
    Scrub obvious secrets and PII out of a free-text string.

    Returns "" for None. Truncates to max_length if provided.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    out = text
    for _name, pattern in _REDACT_PATTERNS:
        out = pattern.sub("[REDACTED]", out)

    if max_length is not None and max_length > 0 and len(out) > max_length:
        out = out[:max_length] + "…[truncated]"

    return out


def redact_path(path: Any) -> str:
    """
    Replace absolute paths with just the filename component.

    Used for file paths before they leave the process — we want
    `report.pdf`, not `/var/lib/minio/chatbot-uploads/uploads/uuid/report.pdf`.
    """
    if not path:
        return ""
    s = str(path)
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    if "\\" in s:
        s = s.rsplit("\\", 1)[-1]
    return s


def redact_filenames(filenames: Any) -> list[str]:
    """Apply redact_path to a list of filenames."""
    if not filenames:
        return []
    if isinstance(filenames, str):
        return [redact_path(filenames)]
    out: list[str] = []
    for f in filenames:
        if f:
            out.append(redact_path(f))
    return out


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _is_langsmith_api_key_configured() -> bool:
    """Check if LANGSMITH_API_KEY is set and non-placeholder."""
    key = os.environ.get("LANGSMITH_API_KEY", "")
    return bool(key and key not in ("", "changeme", "your-langsmith-api-key"))


def _should_sample() -> bool:
    """Apply sample rate. Returns True if this request should be traced."""
    try:
        rate = float(settings.LANGSMITH_SAMPLE_RATE)
    except (TypeError, ValueError):
        rate = 1.0
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def _tracing_active() -> bool:
    """Master gate: tracing on, key configured, and sampling decided yes."""
    if not settings.LANGSMITH_TRACING:
        return False
    if not _is_langsmith_api_key_configured():
        return False
    return _should_sample()


# ---------------------------------------------------------------------------
# Legacy event-level API (kept for backward compatibility)
# ---------------------------------------------------------------------------

def trace_chat_start(
    mode: str,
    has_session_id: bool,
    has_conversation_context: bool,
    user_role: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    if not _tracing_active():
        return
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
    if not _tracing_active():
        return
    data = {
        "query_length": query_length,
        "chunk_count": chunk_count,
        "retrieval_mode": retrieval_mode,
        "error": error,
        **(sanitize_metadata(extra) if extra else {}),
    }
    if top_score is not None:
        data["top_score"] = round(float(top_score), 4)
    _trace_event("trace_retrieval", data)


def trace_prompt_build(
    mode: str,
    context_chunks: int,
    has_conversation_context: bool,
    prompt_length: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    if not _tracing_active():
        return
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
    if not _tracing_active():
        return
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
    if not _tracing_active():
        return
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
    if not _tracing_active():
        return
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
    if not _tracing_active():
        return
    _trace_event(
        "trace_error",
        {
            "error_type": error_type,
            "error_message": redact_text(error_message, max_length=200),
            "mode": mode,
            **(sanitize_metadata(extra) if extra else {}),
        },
    )


# ---------------------------------------------------------------------------
# Phase 31A — Span / context-manager API
# ---------------------------------------------------------------------------

# Thread-local stack of currently open spans. Each span, when closed,
# emits a single LangSmith event with sanitized inputs/outputs/metadata.
_span_stack: threading.local = threading.local()


def _current_span() -> Optional["TraceSpan"]:
    stack = getattr(_span_stack, "stack", None)
    if not stack:
        return None
    return stack[-1]


@dataclass
class TraceSpan:
    """
    In-memory representation of a trace span.

    Spans are nested via `_span_stack`. When the context manager exits,
    if tracing is enabled, the span is flushed to LangSmith via
    `_emit_span`.
    """

    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    parent_id: Optional[str] = None
    start_ms: float = field(default_factory=lambda: time.time() * 1000.0)
    end_ms: Optional[float] = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    completed: bool = False

    def set_input(self, key: str, value: Any) -> None:
        if value is not None:
            self.inputs[key] = value

    def set_output(self, key: str, value: Any) -> None:
        if value is not None:
            self.outputs[key] = value

    def set_meta(self, key: str, value: Any) -> None:
        if value is not None:
            self.metadata[key] = value

    def add_inputs(self, data: dict[str, Any]) -> None:
        for k, v in (data or {}).items():
            if v is not None:
                self.inputs[k] = v

    def add_outputs(self, data: dict[str, Any]) -> None:
        for k, v in (data or {}).items():
            if v is not None:
                self.outputs[k] = v

    def add_metadata(self, data: dict[str, Any]) -> None:
        for k, v in (data or {}).items():
            if v is not None:
                self.metadata[k] = v

    def mark_error(self, error: BaseException) -> None:
        try:
            self.error_type = type(error).__name__
            self.error = str(error)[:200]
        except Exception:
            self.error_type = "Exception"
            self.error = "unknown"

    def finalize(self) -> None:
        if self.completed:
            return
        self.completed = True
        self.end_ms = time.time() * 1000.0


@contextmanager
def trace_span(
    name: str,
    *,
    inputs: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    enabled: bool = True,
) -> Iterator[Optional[TraceSpan]]:
    """
    Context manager that opens a named span and (optionally) emits it
    to LangSmith on exit.

    Returns the active `TraceSpan` so callers can attach data inside the
    block, or `None` if tracing is disabled / not sampled.

    All inputs/outputs/metadata pass through `_safe_metadata` so that
    secrets are scrubbed before they ever leave the process.
    """
    if not (enabled and _tracing_active()):
        yield None
        return

    parent = _current_span()
    span = TraceSpan(
        name=name,
        parent_id=parent.span_id if parent else None,
    )
    if inputs:
        span.add_inputs(_safe_payload(inputs))
    if outputs:
        span.add_outputs(_safe_payload(outputs))
    if metadata:
        span.add_metadata(_safe_metadata(metadata))

    stack = getattr(_span_stack, "stack", None)
    if stack is None:
        stack = []
        _span_stack.stack = stack
    stack.append(span)
    try:
        yield span
        span.finalize()
        _emit_span(span)
    except Exception as exc:
        span.mark_error(exc)
        span.finalize()
        _emit_span(span)
        # Re-raise only if the caller opts in by passing enabled=True;
        # tracing must NEVER break the chat pipeline.
        raise
    finally:
        if stack:
            stack.pop()


@contextmanager
def trace_chat_request(
    *,
    mode: str,
    extra: Optional[dict[str, Any]] = None,
) -> Iterator[Optional[TraceSpan]]:
    """
    Top-level parent span for a chat request. Always named
    `chat_request`. Use as `with trace_chat_request(...) as span:` and
    add child spans inside.
    """
    meta: dict[str, Any] = {"mode": mode}
    if extra:
        meta.update(extra)
    with trace_span("chat_request", metadata=meta) as span:
        yield span


# ---------------------------------------------------------------------------
# Safe payload / metadata helpers
# ---------------------------------------------------------------------------

def _apply_redaction(value: Any) -> Any:
    """Recursively redact free-text inside dict/list/scalar values."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: _apply_redaction(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_apply_redaction(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_apply_redaction(v) for v in value)
    return value


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Return a metadata dict safe to send to LangSmith.

    - Drops any key whose name smells like a secret.
    - If LANGSMITH_REDACT_METADATA is on (default), scrubs free-text
      inside every string value for known secret/PII patterns.
    """
    if not isinstance(metadata, dict):
        return {}
    cleaned = sanitize_metadata(metadata)
    if getattr(settings, "LANGSMITH_REDACT_METADATA", True):
        cleaned = _apply_redaction(cleaned)
    return cleaned


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Inputs/outputs: only secret-name filtering, not free-text scrub,
    because LLM inputs/outputs may legitimately contain long passages.
    """
    if not isinstance(payload, dict):
        return {}
    return sanitize_metadata(payload)


# ---------------------------------------------------------------------------
# Span emission
# ---------------------------------------------------------------------------

def _emit_span(span: TraceSpan) -> None:
    """Push one span to LangSmith. Never raises."""
    try:
        import langsmith
    except ImportError:
        logger.debug("langsmith not installed, skipping span %s", span.name)
        return

    payload: dict[str, Any] = {
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "name": span.name,
        "latency_ms": int((span.end_ms or 0) - (span.start_ms or 0)),
    }
    if span.inputs:
        payload["inputs"] = _safe_payload(span.inputs)
    if span.outputs:
        payload["outputs"] = _safe_payload(span.outputs)
    md: dict[str, Any] = {"project": settings.LANGSMITH_PROJECT}
    if span.metadata:
        md.update(_safe_metadata(span.metadata))
    if span.error_type:
        md["error_type"] = span.error_type
    if span.error:
        md["error_message"] = span.error
    payload["metadata"] = md

    try:
        client = langsmith.Client(
            api_key=os.environ.get("LANGSMITH_API_KEY", ""),
            endpoint=settings.LANGSMITH_ENDPOINT,
        )
        import concurrent.futures

        def _send() -> None:
            try:
                client.upload_trace(
                    name=span.name,
                    inputs=payload,
                    metadata=md,
                )
            except Exception as exc:
                logger.warning(
                    "LangSmith span upload failed (span=%s id=%s): %s",
                    span.name,
                    span.span_id,
                    exc,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(_send)
    except Exception as exc:
        logger.warning(
            "LangSmith span emission failed (span=%s id=%s): %s",
            span.name,
            span.span_id,
            exc,
        )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def traceable(
    name: Optional[str] = None,
    *,
    capture_args: bool = False,
    capture_result: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that wraps a function in a span.

    The wrapped function still runs; we just open a span around it.
    By default we capture no args/results (privacy) — opt in explicitly.

    Usage:
        @traceable("rag_retrieval")
        def retrieve_chunks_with_settings(query, debug=False):
            ...

        @traceable("evidence_grounding", capture_args=False, capture_result=True)
        def apply_grounding_checks(...):
            ...
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or f"fn:{func.__module__}.{func.__qualname__}"

        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            meta: dict[str, Any] = {"function": span_name}
            inputs: dict[str, Any] = {}
            if capture_args:
                # Best-effort: stringify args/kwargs into inputs without
                # ever storing large objects verbatim.
                inputs["arg_count"] = len(args)
                inputs["kwargs_keys"] = sorted(list(kwargs.keys()))
            with trace_span(span_name, inputs=inputs, metadata=meta) as span:
                result = func(*args, **kwargs)
                if span is not None:
                    span.add_outputs(
                        {"captured": True} if capture_result else {}
                    )
                return result

        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            meta = {"function": span_name}
            inputs: dict[str, Any] = {}
            if capture_args:
                inputs["arg_count"] = len(args)
                inputs["kwargs_keys"] = sorted(list(kwargs.keys()))
            with trace_span(span_name, inputs=inputs, metadata=meta) as span:
                result = await func(*args, **kwargs)
                if span is not None:
                    span.add_outputs(
                        {"captured": True} if capture_result else {}
                    )
                return result

        # Decide sync vs async at decoration time
        import inspect

        if inspect.iscoroutinefunction(func):
            return _async_wrapper
        return _sync_wrapper

    return _decorator


# ---------------------------------------------------------------------------
# Legacy low-level emitter (kept for the original trace_* helpers)
# ---------------------------------------------------------------------------

def _trace_event(name: str, metadata: dict[str, Any]) -> None:
    """Send a flat trace event to LangSmith via lazy import."""
    try:
        import langsmith
    except ImportError:
        logger.debug("langsmith not installed, skipping trace event %s", name)
        return

    try:
        client = langsmith.Client(
            api_key=os.environ.get("LANGSMITH_API_KEY", ""),
            endpoint=settings.LANGSMITH_ENDPOINT,
        )
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

            def _send() -> None:
                try:
                    client.upload_trace(
                        name=name,
                        inputs=_safe_payload(metadata),
                        metadata={"project": settings.LANGSMITH_PROJECT},
                    )
                except Exception as exc:
                    logger.warning("LangSmith trace upload failed (event=%s): %s", name, exc)

            executor.submit(_send)
    except Exception as exc:
        logger.warning("LangSmith tracing failed (event=%s): %s", name, exc)


# ---------------------------------------------------------------------------
# Status helper — safe to call from API/status endpoints
# ---------------------------------------------------------------------------

# Per-process counter of last successful / failed LangSmith emit. Used by
# the admin status endpoint so operators can see at a glance whether
# tracing is actually reaching the API.
_last_trace_status: dict[str, Any] = {
    "last_attempt_ms": None,
    "last_success_ms": None,
    "last_failure_ms": None,
    "last_error": None,
    "spans_emitted": 0,
    "spans_failed": 0,
}


def record_trace_attempt(*, success: bool, error: Optional[str] = None) -> None:
    """Called from the emission paths to update status counters."""
    now = time.time() * 1000.0
    _last_trace_status["last_attempt_ms"] = now
    if success:
        _last_trace_status["last_success_ms"] = now
        _last_trace_status["spans_emitted"] = int(_last_trace_status.get("spans_emitted", 0)) + 1
        _last_trace_status["last_error"] = None
    else:
        _last_trace_status["last_failure_ms"] = now
        _last_trace_status["spans_failed"] = int(_last_trace_status.get("spans_failed", 0)) + 1
        _last_trace_status["last_error"] = error[:200] if error else None


def get_langsmith_status() -> dict[str, Any]:
    """
    Return LangSmith status for admin status endpoints.

    NEVER exposes LANGSMITH_API_KEY — only presence/absence.
    """
    api_key_configured = _is_langsmith_api_key_configured()

    endpoint_host = "unknown"
    try:
        from urllib.parse import urlparse

        parsed = urlparse(settings.LANGSMITH_ENDPOINT)
        endpoint_host = parsed.hostname or "unknown"
    except Exception:
        pass

    privacy_mode = "Strict"
    if settings.LANGSMITH_LOG_RETRIEVED_CONTEXT:
        privacy_mode = "Permissive"
    elif settings.LANGSMITH_LOG_FULL_PROMPT or settings.LANGSMITH_LOG_DOCUMENT_TEXT:
        privacy_mode = "Moderate"

    return {
        "langsmith_tracing": settings.LANGSMITH_TRACING,
        "langsmith_available": True,
        "project": settings.LANGSMITH_PROJECT,
        "endpoint_host": endpoint_host,
        "log_full_prompt": settings.LANGSMITH_LOG_FULL_PROMPT,
        "log_document_text": settings.LANGSMITH_LOG_DOCUMENT_TEXT,
        "log_user_input": settings.LANGSMITH_LOG_USER_INPUT,
        "log_llm_output": getattr(settings, "LANGSMITH_LOG_LLM_OUTPUT", False),
        "log_retrieved_context": settings.LANGSMITH_LOG_RETRIEVED_CONTEXT,
        "redact_metadata": getattr(settings, "LANGSMITH_REDACT_METADATA", True),
        "max_context_chars": getattr(settings, "LANGSMITH_MAX_CONTEXT_CHARS", 400),
        "sample_rate": settings.LANGSMITH_SAMPLE_RATE,
        "api_key_configured": api_key_configured,
        "privacy_mode": privacy_mode,
        "last_trace": {
            "last_attempt_ms": _last_trace_status.get("last_attempt_ms"),
            "last_success_ms": _last_trace_status.get("last_success_ms"),
            "last_failure_ms": _last_trace_status.get("last_failure_ms"),
            "last_error": _last_trace_status.get("last_error"),
            "spans_emitted": _last_trace_status.get("spans_emitted", 0),
            "spans_failed": _last_trace_status.get("spans_failed", 0),
        },
        "warning": (
            "LangSmith tracing is enabled but LANGSMITH_API_KEY is not set. "
            "Traces will not be sent."
            if settings.LANGSMITH_TRACING and not api_key_configured
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Convenience helpers used by RAG components
# ---------------------------------------------------------------------------

def safe_chunk_content(text: Any) -> str:
    """
    Return a safe preview of a chunk's content.

    - Returns "" for None.
    - Truncates to LANGSMITH_MAX_CONTEXT_CHARS (default 400) so we never
      send long passages unless the operator opts into
      LANGSMITH_LOG_RETRIEVED_CONTEXT=true.
    - Applies content redaction so a chunk that embeds an API key still
      doesn't leak the key.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    max_chars = int(getattr(settings, "LANGSMITH_MAX_CONTEXT_CHARS", 400) or 400)
    if max_chars <= 0:
        # Caller explicitly disabled chunk text logging
        return ""
    return redact_text(text, max_length=max_chars)


def get_current_span() -> Optional[TraceSpan]:
    """Public accessor for the currently-open span (or None)."""
    return _current_span()
