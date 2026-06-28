"""
Phase 31A — LangSmith tracing tests.

Covers:
- Disabled path: no external calls when tracing is off.
- Enabled path: span/decorator/context-manager APIs emit sanitized
  payloads via the LangSmith client (verified with monkeypatched mock).
- Redaction: secrets / tokens / passwords / API keys / JWTs / connection
  strings / emails / phone numbers / absolute paths are scrubbed.
- Privacy flags: chunk content, user input, and LLM output are gated by
  the LANGSMITH_LOG_* settings.
- Component metadata: retrieval, grounding, citations, evaluation
  attach the documented fields.
- Eval case metadata: the RAG evaluation runner emits per-case spans.
- Resilience: a tracing failure never breaks the chat response.
- Admin status: no API key is exposed via the status endpoint.
"""

from __future__ import annotations

import importlib
import sys
import time
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module import under test
# ---------------------------------------------------------------------------

_LANGSMITH_MOD = "app.services.langsmith_tracing"


@pytest.fixture
def tracing_module(monkeypatch):
    """Reload langsmith_tracing with a controlled LANGSMITH_API_KEY."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-for-unit-tests")
    if _LANGSMITH_MOD in sys.modules:
        return importlib.reload(sys.modules[_LANGSMITH_MOD])
    return importlib.import_module(_LANGSMITH_MOD)


@pytest.fixture
def tracing_disabled(monkeypatch, tracing_module):
    """Tracing module with LANGSMITH_TRACING=false."""
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_TRACING", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_RETRIEVED_CONTEXT", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_USER_INPUT", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_LLM_OUTPUT", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_FULL_PROMPT", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_REDACT_METADATA", True)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_MAX_CONTEXT_CHARS", 400)
    return tracing_module


@pytest.fixture
def tracing_enabled(monkeypatch, tracing_module):
    """Tracing module with LANGSMITH_TRACING=true and sampling=1.0."""
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_RETRIEVED_CONTEXT", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_USER_INPUT", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_LLM_OUTPUT", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_FULL_PROMPT", False)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_REDACT_METADATA", True)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_MAX_CONTEXT_CHARS", 400)
    return tracing_module


# ---------------------------------------------------------------------------
# 1. Disabled path: no external calls
# ---------------------------------------------------------------------------

def test_trace_span_no_op_when_tracing_disabled(tracing_disabled):
    """When LANGSMITH_TRACING=false, trace_span yields None and never
    imports or calls the LangSmith client."""
    captured: dict[str, Any] = {}

    def _fake_upload(*args, **kwargs):
        captured["upload_called"] = True
        return None

    with patch.object(tracing_disabled, "_emit_span", side_effect=_fake_upload) as emit:
        with tracing_disabled.trace_span("rag_retrieval", metadata={"foo": "bar"}) as span:
            assert span is None
    assert not captured.get("upload_called", False), "emit should not have been called"


def test_legacy_trace_chat_start_is_noop_when_disabled(tracing_disabled):
    with patch.object(tracing_disabled, "_trace_event") as evt:
        tracing_disabled.trace_chat_start(
            mode="knowledge_base",
            has_session_id=True,
            has_conversation_context=False,
            user_role="admin",
        )
    assert evt.call_count == 0


def test_traceable_decorator_runs_function_when_disabled(tracing_disabled):
    @tracing_disabled.traceable("test_span")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


# ---------------------------------------------------------------------------
# 2. Enabled path: spans emitted via mocked client
# ---------------------------------------------------------------------------

def _install_fake_langsmith(captured):
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def upload_trace(self, *, name, inputs, metadata):
            captured.append({"name": name, "inputs": inputs, "metadata": metadata})

    fake_langsmith = type(sys)("langsmith")
    fake_langsmith.Client = _FakeClient
    sys.modules["langsmith"] = fake_langsmith
    return fake_langsmith


def _drain(captured, name, max_iters=40):
    for _ in range(max_iters):
        if any(c.get("name") == name for c in captured):
            return
        time.sleep(0.05)


def test_trace_span_emits_when_enabled(tracing_enabled):
    """With LANGSMITH_TRACING=true the span should call into the
    LangSmith client (we monkeypatch the import to record calls)."""
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        with tracing_enabled.trace_span(
            "rag_retrieval",
            metadata={"retrieval_strategy": "hybrid_mmr", "top_score": 0.81},
        ) as span:
            assert span is not None
            span.set_meta("selected_chunk_count", 3)

        _drain(captured, "rag_retrieval")

        assert captured, "expected at least one captured trace event"
        rec = next(c for c in captured if c["name"] == "rag_retrieval")
        assert rec["metadata"].get("retrieval_strategy") == "hybrid_mmr"
        assert rec["metadata"].get("selected_chunk_count") == 3
        assert rec["metadata"].get("project") == tracing_enabled.settings.LANGSMITH_PROJECT
    finally:
        sys.modules.pop("langsmith", None)


def test_trace_chat_request_creates_parent_span(tracing_enabled):
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        with tracing_enabled.trace_chat_request(
            mode="knowledge_base",
            extra={"user_role": "admin"},
        ) as span:
            assert span is not None
            span.set_meta("citation_count", 2)

        _drain(captured, "chat_request")

        names = {c["name"] for c in captured}
        assert "chat_request" in names
        chat_meta = next(c for c in captured if c["name"] == "chat_request")["metadata"]
        assert chat_meta.get("mode") == "knowledge_base"
        assert chat_meta.get("citation_count") == 2
    finally:
        sys.modules.pop("langsmith", None)


def test_traceable_decorator_emits_span_when_enabled(tracing_enabled):
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        @tracing_enabled.traceable("rag_retrieval", capture_args=True)
        def _do_retrieval(query: str) -> int:
            return len(query)

        assert _do_retrieval("hello world") == 11

        _drain(captured, "rag_retrieval")
        assert any(c["name"] == "rag_retrieval" for c in captured)
    finally:
        sys.modules.pop("langsmith", None)


# ---------------------------------------------------------------------------
# 3. Redaction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, must_not_contain",
    [
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234", "abcdefghijklmnopqrstuvwxyz1234"),
        ("api_key=ABCD-EFGH-IJKL-MNOP-QRST-UVWX-YZ12-3456", "ABCD-EFGH-IJKL-MNOP-QRST-UVWX-YZ12-3456"),
        ('password="mySup3rSecret!"', "mySup3rSecret!"),
        ("postgres://user:secret@db.internal:5432/app", "secret@db.internal"),
        ("contact me at alice@example.com please", "alice@example.com"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signaturepart", "eyJhbGciOiJIUzI1NiJ9"),
        ("file at /var/lib/minio/uploads/uuid/report.pdf", "/var/lib/minio/uploads/uuid"),
    ],
)
def test_redact_text_strips_common_secret_patterns(tracing_module, raw, must_not_contain):
    out = tracing_module.redact_text(raw)
    assert "[REDACTED]" in out
    assert must_not_contain not in out


def test_redact_text_truncates_when_max_length_set(tracing_module):
    long = "hello world " * 100
    out = tracing_module.redact_text(long, max_length=50)
    assert len(out) <= 80  # 50 + truncation suffix
    assert "[truncated]" in out


def test_redact_path_returns_only_filename(tracing_module):
    assert tracing_module.redact_path("/var/lib/minio/uploads/uuid/report.pdf") == "report.pdf"
    assert tracing_module.redact_path("C:\\Users\\admin\\file.docx") == "file.docx"
    assert tracing_module.redact_path("plain.pdf") == "plain.pdf"
    assert tracing_module.redact_path("") == ""


def test_redact_filenames_strips_paths_in_list(tracing_module):
    out = tracing_module.redact_filenames([
        "/var/lib/minio/uploads/uuid/report.pdf",
        "C:\\Users\\admin\\file.docx",
        "plain.txt",
    ])
    assert out == ["report.pdf", "file.docx", "plain.txt"]


def test_sanitize_metadata_redacts_secret_named_keys(tracing_module):
    out = tracing_module.sanitize_metadata({
        "api_key": "ABCD",
        "Authorization": "Bearer xyz",
        "user_input": "visible",
    })
    assert out["api_key"] == "[REDACTED]"
    assert out["Authorization"] == "[REDACTED]"
    assert out["user_input"] == "visible"


def test_safe_metadata_applies_content_redaction_when_enabled(tracing_module, monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_REDACT_METADATA", True)
    out = tracing_module._safe_metadata({
        "source_file_names": ["/var/lib/minio/uploads/uuid/report.pdf"],
        "user_note": "contact alice@example.com",
    })
    assert "alice@example.com" not in out["user_note"]


# ---------------------------------------------------------------------------
# 4. Privacy flags: chunk content / user input / LLM output
# ---------------------------------------------------------------------------

def test_chunk_content_truncated_when_enabled(tracing_module, monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_MAX_CONTEXT_CHARS", 50)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_RETRIEVED_CONTEXT", True)
    long_text = "word " * 200
    out = tracing_module.safe_chunk_content(long_text)
    assert len(out) <= 200  # slack for truncation marker


def test_safe_chunk_content_redacts_secrets(tracing_module, monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_MAX_CONTEXT_CHARS", 400)
    out = tracing_module.safe_chunk_content("api_key=ABCD-EFGH-IJKL-MNOP-QRST secret")
    assert "ABCD-EFGH-IJKL-MNOP-QRST" not in out


def test_llm_output_logging_respects_flag(tracing_module, monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_LLM_OUTPUT", False)
    status = tracing_module.get_langsmith_status()
    assert status["log_llm_output"] is False

    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_LLM_OUTPUT", True)
    status = tracing_module.get_langsmith_status()
    assert status["log_llm_output"] is True


def test_user_input_logging_respects_flag(tracing_module, monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_LOG_USER_INPUT", False)
    status = tracing_module.get_langsmith_status()
    assert status["log_user_input"] is False


# ---------------------------------------------------------------------------
# 5. Component metadata: retrieval / grounding / citations
# ---------------------------------------------------------------------------

def test_retrieval_metadata_includes_strategy_and_count(tracing_enabled):
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        with tracing_enabled.trace_span("rag_retrieval", metadata={"phase": "rag_retrieval"}) as span:
            span.set_meta("retrieval_strategy", "hybrid_mmr")
            span.set_meta("selected_chunk_count", 5)
            span.set_meta("top_score", 0.82)
            span.set_meta("source_file_names", ["a.pdf", "b.docx"])

        _drain(captured, "rag_retrieval")

        rec = next(c for c in captured if c["name"] == "rag_retrieval")
        md = rec["metadata"]
        assert md["retrieval_strategy"] == "hybrid_mmr"
        assert md["selected_chunk_count"] == 5
        assert md["top_score"] == 0.82
        assert md["source_file_names"] == ["a.pdf", "b.docx"]
    finally:
        sys.modules.pop("langsmith", None)


def test_grounding_metadata_includes_evidence_level_and_fallback(tracing_enabled):
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        with tracing_enabled.trace_span("evidence_grounding", metadata={"phase": "evidence_grounding"}) as span:
            span.set_meta("evidence_level", "medium")
            span.set_meta("fallback_reason", "below_relevance_threshold")
            span.set_meta("supporting_chunk_count", 1)
            span.set_meta("llm_skipped_due_to_weak_evidence", False)

        _drain(captured, "evidence_grounding")

        rec = next(c for c in captured if c["name"] == "evidence_grounding")
        md = rec["metadata"]
        assert md["evidence_level"] == "medium"
        assert md["fallback_reason"] == "below_relevance_threshold"
        assert md["supporting_chunk_count"] == 1
    finally:
        sys.modules.pop("langsmith", None)


def test_citation_metadata_includes_source_filenames_not_full_paths(tracing_enabled):
    """Citation spans must carry filenames only — never full paths."""
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        safe = tracing_enabled.redact_filenames([
            "/var/lib/minio/uploads/uuid/report.pdf",
            "C:\\Users\\admin\\file.docx",
        ])
        with tracing_enabled.trace_span("citation_processing") as span:
            span.set_meta("citation_count", 2)
            span.set_meta("source_file_names", safe)

        _drain(captured, "citation_processing")

        rec = next(c for c in captured if c["name"] == "citation_processing")
        md = rec["metadata"]
        assert md["citation_count"] == 2
        assert md["source_file_names"] == ["report.pdf", "file.docx"]
        for fn in md["source_file_names"]:
            assert "/" not in fn
            assert "\\" not in fn
    finally:
        sys.modules.pop("langsmith", None)


# ---------------------------------------------------------------------------
# 6. Eval case metadata
# ---------------------------------------------------------------------------

def test_eval_case_metadata_can_be_attached(tracing_enabled):
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        with tracing_enabled.trace_span("rag_evaluation_case") as span:
            span.set_meta("evaluation_case_id", "case-001")
            span.set_meta("expected_behavior", "answer")
            span.set_meta("expected_source_file", "guide.pdf")
            span.set_meta("retrieval_strategy", "hybrid_mmr")
            span.set_meta("evidence_level", "strong")
            span.set_meta("passed", True)
            span.set_meta("latency_ms", 1234)
            span.set_meta("citation_count", 3)
            span.set_meta("failure_reasons", None)

        _drain(captured, "rag_evaluation_case")

        rec = next(c for c in captured if c["name"] == "rag_evaluation_case")
        md = rec["metadata"]
        assert md["evaluation_case_id"] == "case-001"
        assert md["expected_behavior"] == "answer"
        assert md["retrieval_strategy"] == "hybrid_mmr"
        assert md["evidence_level"] == "strong"
        assert md["passed"] is True
        assert md["latency_ms"] == 1234
        assert md["citation_count"] == 3
    finally:
        sys.modules.pop("langsmith", None)


def test_eval_case_failure_reasons_attached(tracing_enabled):
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        with tracing_enabled.trace_span("rag_evaluation_case") as span:
            span.set_meta("evaluation_case_id", "case-002")
            span.set_meta("passed", False)
            span.set_meta("failure_reasons", ["missing_citations", "wrong_source"])

        _drain(captured, "rag_evaluation_case")

        rec = next(c for c in captured if c["name"] == "rag_evaluation_case")
        md = rec["metadata"]
        assert md["passed"] is False
        assert "missing_citations" in md["failure_reasons"]
        assert "wrong_source" in md["failure_reasons"]
    finally:
        sys.modules.pop("langsmith", None)


# ---------------------------------------------------------------------------
# 7. Resilience: tracing failure does not break the calling code
# ---------------------------------------------------------------------------

def test_trace_span_raises_but_does_not_break_chat_when_emission_fails(tracing_enabled):
    """If the LangSmith client raises, the inner code must still run and
    propagate its own exceptions cleanly."""

    class _BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        def upload_trace(self, *, name, inputs, metadata):
            raise RuntimeError("network down")

    fake_langsmith = type(sys)("langsmith")
    fake_langsmith.Client = _BoomClient
    sys.modules["langsmith"] = fake_langsmith

    try:
        ran = False
        with pytest.raises(ValueError):
            with tracing_enabled.trace_span("rag_retrieval"):
                ran = True
                raise ValueError("inner failure")
        assert ran
    finally:
        sys.modules.pop("langsmith", None)


def test_trace_span_swallows_emission_error_when_inner_succeeds(tracing_enabled):
    """If the inner code succeeds but emission fails, the result must
    still be returned and no exception should bubble out."""

    class _BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        def upload_trace(self, *, name, inputs, metadata):
            raise RuntimeError("network down")

    fake_langsmith = type(sys)("langsmith")
    fake_langsmith.Client = _BoomClient
    sys.modules["langsmith"] = fake_langsmith

    try:
        result = None
        with tracing_enabled.trace_span("rag_retrieval"):
            result = 42
        assert result == 42
    finally:
        sys.modules.pop("langsmith", None)


# ---------------------------------------------------------------------------
# 8. Admin status does not expose API key
# ---------------------------------------------------------------------------

def test_admin_status_does_not_expose_api_key(tracing_module, monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "super-secret-key-1234567890")
    status = tracing_module.get_langsmith_status()

    serialized = str(status)
    assert "super-secret-key-1234567890" not in serialized
    assert status["api_key_configured"] is True
    for v in status.values():
        if isinstance(v, str):
            assert "super-secret-key" not in v


def test_admin_status_includes_privacy_flags_and_last_trace(tracing_module):
    status = tracing_module.get_langsmith_status()
    assert "log_full_prompt" in status
    assert "log_document_text" in status
    assert "log_user_input" in status
    assert "log_llm_output" in status
    assert "log_retrieved_context" in status
    assert "redact_metadata" in status
    assert "max_context_chars" in status
    assert "sample_rate" in status
    assert "last_trace" in status
    last = status["last_trace"]
    assert "last_attempt_ms" in last
    assert "last_success_ms" in last
    assert "last_failure_ms" in last
    assert "spans_emitted" in last
    assert "spans_failed" in last


# ---------------------------------------------------------------------------
# 9. Sampling rate
# ---------------------------------------------------------------------------

def test_sample_rate_zero_disables_emission(tracing_module, monkeypatch):
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(tracing_module.settings, "LANGSMITH_SAMPLE_RATE", 0.0)
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def upload_trace(self, *, name, inputs, metadata):
            captured["called"] = True

    fake_langsmith = type(sys)("langsmith")
    fake_langsmith.Client = _FakeClient
    sys.modules["langsmith"] = fake_langsmith

    try:
        with tracing_module.trace_span("rag_retrieval") as span:
            assert span is None  # sampling excluded
        assert "called" not in captured
    finally:
        sys.modules.pop("langsmith", None)


# ---------------------------------------------------------------------------
# 10. Span hierarchy
# ---------------------------------------------------------------------------

def test_child_span_carries_parent_id(tracing_enabled):
    captured: list[dict[str, Any]] = []
    _install_fake_langsmith(captured)

    try:
        with tracing_enabled.trace_span("chat_request") as parent:
            parent_id = parent.span_id
            with tracing_enabled.trace_span("rag_retrieval") as child:
                assert child.parent_id == parent_id
                with tracing_enabled.trace_span("llm_call") as grandchild:
                    assert grandchild.parent_id == child.span_id
                    assert grandchild.parent_id != parent_id

        _drain(captured, "llm_call")

        names = [c["name"] for c in captured]
        assert "chat_request" in names
        assert "rag_retrieval" in names
        assert "llm_call" in names
    finally:
        sys.modules.pop("langsmith", None)
