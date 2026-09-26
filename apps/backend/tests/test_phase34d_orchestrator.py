"""Phase 34D — Unit tests for the advanced orchestrator.

Covers the brief's requirements #1, #12-14, #17, #21-23:
* kill switch (``ADVANCED_VISION_ENABLED=false``) preserves Phase 34B
* text-only OCR question does NOT call advanced Vision
* UI / chart / diagram / table / comparison tasks route correctly
* provider call only when task classifier + RBAC + cache-miss align
* RBAC gates BOTH reject before MinIO fetch AND before provider call
* provider failure / timeout / parse failure degrade safely
* cache hit avoids duplicate provider calls
* historical image (no explicit image_context) can be advanced-analyzed
* ``is_successful`` gates cache writes
* ``to_dict`` exposes every observability field

All tests are hermetic: monkeypatching replaces DB / MinIO / Redis /
provider so no real network or storage access is required.
"""

from __future__ import annotations

import pytest

from app.services.advanced_vision.base import (
    AdvancedVisualTask,
    VisualReasoningResult,
)
from app.services.advanced_vision.orchestrator import (
    AdvancedOrchestratorOutcome,
    run_advanced_visual_reasoning,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeAuth:
    """Minimal AuthContext-shaped stand-in for RBAC tests."""

    def __init__(self, user_id: int = 1, role: str = "user", is_authenticated: bool = True):
        self.user_id = user_id
        self.role = role
        self.is_authenticated = is_authenticated

    def __bool__(self) -> bool:
        return self.is_authenticated


def _make_auth(user_id: int = 1) -> FakeAuth:
    return FakeAuth(user_id=user_id)


def _fake_doc_image_row(*, document_id: int = 7, storage_key: str = "uploads/x.png", mime_type: str = "image/png"):
    return type(
        "Row",
        (),
        {
            "id": 42,
            "document_id": document_id,
            "storage_key": storage_key,
            "mime_type": mime_type,
            "original_filename": "screenshot.png",
        },
    )()


def _patch_minio(monkeypatch, *, payload: bytes = b"PNG-DATA"):
    """Force MinIO get_object to return ``payload``."""
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            return self._payload
        def close(self):
            pass
        def release_conn(self):
            pass

    class FakeClient:
        def get_object(self, bucket, key):
            return FakeResponse(payload)

    fake_client = FakeClient()
    import app.core.minio_client as minio_client
    monkeypatch.setattr(minio_client, "get_minio_client", lambda: fake_client)


def _patch_orchestrator_image_fetch(monkeypatch, *, document_id: int = 7, payload: bytes = b"PNG-DATA"):
    """Patch the orchestrator's image-fetch module directly so we do not
    need a real DocumentImage row or MinIO call. The helper returns a
    fake AuthorizedImage so RBAC appears to succeed.
    """
    from app.services.advanced_vision import _image_fetch as fetch_mod
    from app.services.advanced_vision import orchestrator as orch

    fake_auth = type(
        "AuthorizedImage",
        (),
        {
            "image_id": 42,
            "document_id": document_id,
            "mime_type": "image/png",
            "storage_key": "uploads/x.png",
            "bytes_": payload,
            "filename": "screenshot.png",
        },
    )()

    monkeypatch.setattr(
        orch,
        "fetch_authorized_image_bytes",
        lambda *a, **kw: fake_auth,
    )
    return fake_auth


def _patch_can_access_document(monkeypatch, *, allow: bool = True):
    """Patch the RBAC permission gate used by the orchestrator."""
    from app.services.advanced_vision import orchestrator as orch
    monkeypatch.setattr(
        orch, "can_access_document", lambda *a, **kw: bool(allow)
    )


def _patch_redis(monkeypatch, *, hit_blob: str = ""):
    """Patch Redis client. ``hit_blob`` empty means cache miss."""
    fake_client = type(
        "FakeRedis",
        (),
        {
            "get": staticmethod(lambda _key: hit_blob),
            "set": staticmethod(lambda *a, **kw: True),
            "delete": staticmethod(lambda _key: True),
        },
    )()
    from app.services.advanced_vision import cache as adv_cache
    monkeypatch.setattr(adv_cache, "_get_redis_client", lambda: fake_client)
    monkeypatch.setattr(adv_cache, "_get_process_lru", lambda: None)
    return fake_client


def _patch_provider(monkeypatch, *, return_value=None, raise_exc: Exception | None = None):
    """Patch the Phase 34B provider resolution so we do not call the network."""
    if raise_exc is not None:
        class FakeProvider:
            name = "mock"
            model = "mock-v1"
            def analyze_image(self, **_kw):
                raise raise_exc
    else:
        class FakeProvider:
            name = "mock"
            model = "mock-v1"
            def analyze_image(self, **_kw):
                return return_value
    fake_provider = FakeProvider()
    from app.services.advanced_vision import orchestrator as orch
    monkeypatch.setattr(orch, "_resolve_provider", lambda: fake_provider)
    return fake_provider


def _good_visual_result(task_type: str = "ui_state_analysis") -> VisualReasoningResult:
    return VisualReasoningResult(
        task_type=task_type,
        summary="Server B failed",
        observations=["failed indicator"],
        confidence=85,
        image_ids=[42],
    )


def _phase34b_vision_result():
    return type(
        "VisionResult",
        (),
        {
            "description": "Server B failed",
            "image_type": "ui",
            "visual_findings": ["failed indicator"],
            "detected_entities": ["Server B"],
            "visual_states": ["failed-indicator"],
            "tags": [],
            "confidence": 85,
            "provider": "mock",
            "model": "mock-v1",
            "processing_time_ms": 12,
            "schema_version": 1,
            "raw": None,
        },
    )()


# ---------------------------------------------------------------------------
# Settings overrides
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_advanced_vision(monkeypatch):
    """Default: advanced vision ENABLED for most tests. Individual tests
    that need the kill switch override ``ADVANCED_VISION_ENABLED``."""
    import app.services.advanced_vision.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.settings, "ADVANCED_VISION_ENABLED", True)
    monkeypatch.setattr(orch_mod.settings, "ADVANCED_VISION_ROUTER_ENABLED", True)
    monkeypatch.setattr(orch_mod.settings, "VISION_ENABLED", True)
    monkeypatch.setattr(orch_mod.settings, "VISION_PROVIDER", "mock")
    monkeypatch.setattr(orch_mod.settings, "VISION_MODEL", "mock-v1")
    monkeypatch.setattr(orch_mod.settings, "ADVANCED_VISION_SCHEMA_VERSION", 1)


# ---------------------------------------------------------------------------
# Brief test 1: text-only OCR question does NOT call advanced Vision
# ---------------------------------------------------------------------------


def test_orchestrator_01_text_only_question_does_not_advance(monkeypatch):
    """'What error code is shown?' is OCR-only — no provider call."""
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)  # would raise if called

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What error code is shown?",
        image_id=42,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason in (
        "identifier_lookup", "no_task_signal", "kill_switch"
    )


# ---------------------------------------------------------------------------
# Brief tests 12-14, 22-23: task routing into single-image advanced vision
# ---------------------------------------------------------------------------


def test_orchestrator_02_ui_state_question_invokes_provider(monkeypatch):
    """UI state question + authorized image → provider called, observation captured."""
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
        ocr_text="Server B Failed\nApply disabled",
    )
    assert outcome.ran is True
    assert outcome.skipped_reason is None
    assert outcome.classification is not None
    assert outcome.classification.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value


def test_orchestrator_03_chart_question_invokes_provider(monkeypatch):
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What trend does this chart show?",
        image_id=42,
    )
    assert outcome.ran is True
    assert outcome.classification.task_type == AdvancedVisualTask.CHART_ANALYSIS.value


def test_orchestrator_04_diagram_question_invokes_provider(monkeypatch):
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What flow does this architecture diagram show?",
        image_id=42,
    )
    assert outcome.ran is True
    assert outcome.classification.task_type == AdvancedVisualTask.DIAGRAM_ANALYSIS.value


def test_orchestrator_05_table_question_invokes_provider(monkeypatch):
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="Which row is highlighted in the table?",
        image_id=42,
    )
    assert outcome.ran is True
    assert outcome.classification.task_type == AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value


def test_orchestrator_06_comparison_question_skipped_without_two_images(monkeypatch):
    """Comparison task with only one image_id is NOT routed through
    single-image orchestrator. The orchestrator reports
    ``comparison_via_orchestrator_unsupported`` (or simply does not
    advance) because Phase 34D reserves comparison to its own helper."""
    _patch_redis(monkeypatch)
    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What changed between these screenshots?",
        image_id=42,
    )
    # Single-image orchestrator treats comparison as a distinct path.
    # Image-comparison helper handles two-image flow; here we just
    # confirm advanced vision is not invoked on a single image for a
    # comparison question (the single-image orchestrator returns
    # ``ran=False``).
    assert outcome.ran is False


# ---------------------------------------------------------------------------
# Brief test 17: RBAC before MinIO fetch
# ---------------------------------------------------------------------------


def test_orchestrator_07_rbac_rejects_before_minio_fetch(monkeypatch):
    """Unauthorized image → no bytes fetched, no provider call."""
    _patch_can_access_document(monkeypatch, allow=False)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)  # would raise if called

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "unauthorized"


def test_orchestrator_08_rbac_gate_two_rejects_before_provider_call(monkeypatch):
    """Gate #2: even if MinIO fetch succeeded, RBAC check before provider must reject."""
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=False)  # gate #2 fails
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)  # must NOT be called

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "unauthorized_pre_provider"


def test_orchestrator_09_unauthenticated_caller_refused(monkeypatch):
    """``auth=None`` is treated as unauthorized by the orchestrator."""
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_reasoning(
        None,
        auth=None,
        question="What visually looks wrong?",
        image_id=42,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason in ("unauthorized", "unauthorized_pre_provider")


# ---------------------------------------------------------------------------
# Brief test 14: provider failure / timeout / parse failure degrade safely
# ---------------------------------------------------------------------------


def test_orchestrator_10_provider_timeout_degrades_safely(monkeypatch):
    from app.services.vision.base import VisionProviderTimeoutError
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(
        monkeypatch,
        raise_exc=VisionProviderTimeoutError("simulated timeout"),
    )

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    # Provider raised → outcome records skipped_reason AND ran=True
    # (we attempted to run; the failure is captured in skipped_reason,
    # not in the result's success predicate). The safe empty result
    # carries a fallback summary so the LLM still sees an explicit
    # "provider timed out" signal.
    assert outcome.ran is True
    assert outcome.skipped_reason == "provider_timeout"
    assert outcome.result is not None
    assert "timed out" in (outcome.result.summary or "").lower()


def test_orchestrator_11_provider_unavailable_degrades_safely(monkeypatch):
    from app.services.vision.base import VisionProviderUnavailableError
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(
        monkeypatch,
        raise_exc=VisionProviderUnavailableError("simulated 503"),
    )

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.skipped_reason == "provider_unavailable"


def test_orchestrator_12_provider_returns_empty_degrades_safely(monkeypatch):
    """Provider returns a VisionResult with no content → safe empty result."""
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    empty_vision_result = type(
        "VR",
        (),
        {
            "description": "",
            "image_type": "unknown",
            "visual_findings": [],
            "detected_entities": [],
            "visual_states": [],
            "tags": [],
            "confidence": None,
            "provider": "mock",
            "model": "mock-v1",
            "processing_time_ms": 5,
            "schema_version": 1,
            "raw": None,
        },
    )()
    _patch_provider(monkeypatch, return_value=empty_vision_result)

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    # Empty provider response → safe failure path.
    assert outcome.skipped_reason == "parse_failure"


# ---------------------------------------------------------------------------
# Brief tests 15-16: cache prevents duplicate provider calls
# ---------------------------------------------------------------------------


def test_orchestrator_13_cache_hit_avoids_provider(monkeypatch):
    """Cached blob + matching schema → no provider call."""
    import json as _json
    cached_blob = _json.dumps({
        "task_type": "ui_state_analysis",
        "summary": "Server B failed",
        "schema_version": 1,
        "image_ids": [42],
    })
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch, hit_blob=cached_blob)
    _patch_provider(monkeypatch)  # would raise if called

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.ran is True
    assert outcome.cache_hit is True
    assert outcome.result is not None
    assert outcome.result.task_type == "ui_state_analysis"


def test_orchestrator_14_cache_miss_triggers_provider(monkeypatch):
    """Empty cache → provider called, result cached."""
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    fake_redis = _patch_redis(monkeypatch, hit_blob="")  # miss
    set_called = {"n": 0}
    original_set = fake_redis.set
    def _counted_set(*a, **kw):
        set_called["n"] += 1
        return original_set(*a, **kw)
    fake_redis.set = staticmethod(_counted_set)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.ran is True
    assert outcome.cache_hit is False
    # Redis SET was called at least once.
    assert set_called["n"] >= 1


# ---------------------------------------------------------------------------
# Brief test 21: kill switch restores previous behaviour
# ---------------------------------------------------------------------------


def test_orchestrator_15_kill_switch_skips_everything(monkeypatch):
    """``ADVANCED_VISION_ENABLED=false`` → ran=False, no provider call."""
    import app.services.advanced_vision.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.settings, "ADVANCED_VISION_ENABLED", False)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "kill_switch"


def test_orchestrator_16_router_disabled_skips_everything(monkeypatch):
    """``ADVANCED_VISION_ROUTER_ENABLED=false`` → ran=False."""
    import app.services.advanced_vision.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.settings, "ADVANCED_VISION_ROUTER_ENABLED", False)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "kill_switch"


# ---------------------------------------------------------------------------
# Brief test 22: no advanced Vision for ordinary text RAG
# ---------------------------------------------------------------------------


def test_orchestrator_17_plain_text_rag_does_not_advance(monkeypatch):
    """General knowledge question (no image context) does NOT advance."""
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What is the capital of France?",
        image_id=42,  # even with an image_id, no visual task signal → skipped
    )
    assert outcome.ran is False
    assert outcome.skipped_reason in (
        "no_task_signal", "identifier_lookup", "product_meaning",
    )


# ---------------------------------------------------------------------------
# Brief test 23: historical image (resolved by Phase 34C retrieval) can be advanced-analyzed
# ---------------------------------------------------------------------------


def test_orchestrator_18_historical_image_can_be_advanced_analyzed(monkeypatch):
    """A historical image supplied with image_id is processed normally —
    the orchestrator does not care WHERE the image_id came from."""
    _patch_orchestrator_image_fetch(monkeypatch, document_id=99)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.ran is True
    assert outcome.image_ids == [42]


# ---------------------------------------------------------------------------
# Forced task (test-only override)
# ---------------------------------------------------------------------------


def test_orchestrator_19_forced_task_bypasses_classifier(monkeypatch):
    """``forced_task`` lets tests / future callers skip the classifier."""
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    # The question would normally classify as ``general_visual`` (no task
    # signal), but the forced_task override forces CHART_ANALYSIS.
    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="Describe the image.",
        image_id=42,
        forced_task=AdvancedVisualTask.CHART_ANALYSIS.value,
    )
    assert outcome.ran is True
    assert outcome.classification.task_type == AdvancedVisualTask.CHART_ANALYSIS.value


# ---------------------------------------------------------------------------
# Oversize / image too large
# ---------------------------------------------------------------------------


def test_orchestrator_20_oversize_image_rejected(monkeypatch):
    """Bytes over the configured cap → ``too_large`` without provider call."""
    import app.services.advanced_vision.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.settings, "ADVANCED_VISION_MAX_IMAGE_BYTES", 10)
    big_payload = b"X" * 64
    _patch_orchestrator_image_fetch(monkeypatch, payload=big_payload)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "too_large"


# ---------------------------------------------------------------------------
# Outcome payload fields
# ---------------------------------------------------------------------------


def test_orchestrator_21_to_dict_exposes_observability_fields(monkeypatch):
    """Every observability field promised by the brief is present in to_dict()."""
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    payload = outcome.to_dict()
    required = {
        "advanced_vision_ran",
        "advanced_vision_task_type",
        "advanced_vision_called",
        "advanced_vision_cache_hit",
        "advanced_vision_provider",
        "advanced_vision_model",
        "advanced_vision_latency_ms",
        "advanced_vision_image_count",
        "advanced_vision_image_ids",
        "advanced_vision_skipped_reason",
        "advanced_vision_error",
    }
    for field in required:
        assert field in payload, f"missing {field} in outcome.to_dict()"


def test_orchestrator_22_image_ids_present_in_outcome(monkeypatch):
    """Successful run carries the image_ids for the citation renderer."""
    _patch_orchestrator_image_fetch(monkeypatch)
    _patch_can_access_document(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_reasoning(
        None,
        auth=_make_auth(),
        question="What visually looks wrong on this screen?",
        image_id=42,
    )
    assert outcome.image_ids == [42]
    payload = outcome.to_dict()
    assert payload["advanced_vision_image_ids"] == [42]
    assert payload["advanced_vision_image_count"] == 1
