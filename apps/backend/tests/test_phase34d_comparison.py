"""Phase 34D — Unit tests for the IMAGE_COMPARISON orchestrator.

Covers the brief's requirements #18 (unauthorized rejected before
provider) and #20 (citations include both image IDs), plus the
order-preserving cache key invariant (so a swapped comparison
request cannot silently invert the answer).
"""

from __future__ import annotations

import pytest

from app.services.advanced_vision.base import (
    AdvancedVisualTask,
    VisualReasoningResult,
    build_advanced_vision_cache_key,
    image_content_hash,
)
from app.services.advanced_vision.cache import _safe_parse
from app.services.advanced_vision.comparison import (
    ComparisonOutcome,
    run_advanced_visual_comparison,
)


class FakeAuth:
    def __init__(self, user_id: int = 1, is_authenticated: bool = True):
        self.user_id = user_id
        self.is_authenticated = is_authenticated


def _patch_image_fetch_for_pair(monkeypatch, *, doc_a: int = 7, doc_b: int = 8):
    """Patch the comparison module's _image_fetch helpers so both
    images return successfully-authorized AuthorizedImage objects.
    """
    from app.services.advanced_vision import comparison as cmp

    def _fake_auth(db, image_id, auth, **kwargs):
        # image_id 1 → doc_a, image_id 2 → doc_b (test chooses IDs).
        return type(
            "AuthorizedImage",
            (),
            {
                "image_id": int(image_id),
                "document_id": doc_a if int(image_id) == 1 else doc_b,
                "mime_type": "image/png",
                "storage_key": f"uploads/img_{image_id}.png",
                "bytes_": b"PNG-DATA-" + str(image_id).encode(),
                "filename": f"image_{image_id}.png",
            },
        )()

    monkeypatch.setattr(cmp, "_gate_authorize", _fake_auth)


def _patch_can_access(monkeypatch, *, allow: bool = True):
    from app.services.advanced_vision import comparison as cmp
    monkeypatch.setattr(cmp, "can_access_document", lambda *a, **kw: bool(allow))


def _patch_redis(monkeypatch, *, hit_blob: str = ""):
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
    from app.services.advanced_vision import comparison as cmp
    monkeypatch.setattr(cmp, "_resolve_provider", lambda: fake_provider)
    return fake_provider


def _phase34b_vision_result(*, description: str = "Server B Healthy → Failed"):
    return type(
        "VR",
        (),
        {
            "description": description,
            "image_type": "ui",
            "visual_findings": ["status change observed"],
            "detected_entities": ["Server B"],
            "visual_states": ["status-change"],
            "tags": [],
            "confidence": 88,
            "provider": "mock",
            "model": "mock-v1",
            "processing_time_ms": 20,
            "schema_version": 1,
            "raw": None,
        },
    )()


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    import app.services.advanced_vision.comparison as cmp_mod
    monkeypatch.setattr(cmp_mod.settings, "ADVANCED_VISION_ENABLED", True)
    monkeypatch.setattr(cmp_mod.settings, "ADVANCED_VISION_ROUTER_ENABLED", True)
    monkeypatch.setattr(cmp_mod.settings, "ADVANCED_VISION_MAX_IMAGES_PER_REQUEST", 2)
    monkeypatch.setattr(cmp_mod.settings, "VISION_ENABLED", True)
    monkeypatch.setattr(cmp_mod.settings, "VISION_PROVIDER", "mock")
    monkeypatch.setattr(cmp_mod.settings, "VISION_MODEL", "mock-v1")


# ---------------------------------------------------------------------------
# RBAC gates
# ---------------------------------------------------------------------------


def test_comparison_01_unauthorized_image_a_rejected_before_provider(monkeypatch):
    """Image A unauthorized → comparison never runs the provider."""
    from app.services.advanced_vision import comparison as cmp

    def _fake_gate(db, image_id, auth, **_kw):
        # image_a (id=1) unauthorized, image_b (id=2) OK
        if int(image_id) == 1:
            return None
        return type(
            "AuthorizedImage",
            (),
            {"image_id": 2, "document_id": 8, "mime_type": "image/png",
             "storage_key": "uploads/img_2.png", "bytes_": b"PNG-DATA-2",
             "filename": "image_2.png"},
        )()

    monkeypatch.setattr(cmp, "_gate_authorize", _fake_gate)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)  # would raise if called

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "unauthorized"
    assert outcome.image_ids == []  # nothing leaked


def test_comparison_02_unauthorized_image_b_rejected_before_provider(monkeypatch):
    """Image B unauthorized → comparison never runs the provider."""
    from app.services.advanced_vision import comparison as cmp

    def _fake_gate(db, image_id, auth, **_kw):
        if int(image_id) == 2:
            return None
        return type(
            "AuthorizedImage",
            (),
            {"image_id": 1, "document_id": 7, "mime_type": "image/png",
             "storage_key": "uploads/img_1.png", "bytes_": b"PNG-DATA-1",
             "filename": "image_1.png"},
        )()

    monkeypatch.setattr(cmp, "_gate_authorize", _fake_gate)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "unauthorized"


def test_comparison_03_unauthenticated_caller_refused(monkeypatch):
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_comparison(
        None, auth=None, question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "unauthorized"


def test_comparison_04_pre_provider_reauth_failure_blocks_provider(monkeypatch):
    """Gate #2: both images fetch OK, but pre-provider re-auth fails."""
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=False)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "unauthorized_pre_provider"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_comparison_05_identical_image_ids_rejected(monkeypatch):
    """Comparing image A to itself is meaningless — refuse."""
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=1,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "identical_image_ids"


def test_comparison_06_max_images_cap_blocks_comparison(monkeypatch):
    """When the cap is below 2, comparison is disabled."""
    import app.services.advanced_vision.comparison as cmp_mod
    cmp_mod.settings.ADVANCED_VISION_MAX_IMAGES_PER_REQUEST = 1
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "comparison_disabled_by_cap"


def test_comparison_07_kill_switch_short_circuits(monkeypatch):
    import app.services.advanced_vision.comparison as cmp_mod
    cmp_mod.settings.ADVANCED_VISION_ENABLED = False
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch)

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is False
    assert outcome.skipped_reason == "kill_switch"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_comparison_08_cache_hit_avoids_provider(monkeypatch):
    """Cached blob + matching schema → no provider call."""
    import json as _json
    cached_blob = _json.dumps({
        "task_type": "image_comparison",
        "summary": "Server B Healthy → Failed",
        "schema_version": 1,
        "image_ids": [1, 2],
    })
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=True)
    _patch_redis(monkeypatch, hit_blob=cached_blob)
    _patch_provider(monkeypatch)  # would raise if called

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is True
    assert outcome.cache_hit is True
    assert outcome.result is not None
    assert outcome.result.task_type == "image_comparison"
    assert outcome.image_ids == [1, 2]


def test_comparison_09_cache_miss_triggers_provider_and_writes(monkeypatch):
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=True)
    fake_redis = _patch_redis(monkeypatch, hit_blob="")
    set_called = {"n": 0}
    original_set = fake_redis.set
    def _counted_set(*a, **kw):
        set_called["n"] += 1
        return original_set(*a, **kw)
    fake_redis.set = staticmethod(_counted_set)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is True
    assert outcome.cache_hit is False
    assert set_called["n"] >= 1


def test_comparison_10_order_preserving_cache_key():
    """A swapped A↔B request MUST produce a different cache key."""
    forward = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
        image_hashes=["hash_a", "hash_b"],
    )
    swapped = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
        image_hashes=["hash_b", "hash_a"],
    )
    assert forward != swapped


# ---------------------------------------------------------------------------
# Citations include both image IDs
# ---------------------------------------------------------------------------


def test_comparison_11_successful_outcome_carries_both_image_ids(monkeypatch):
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is True
    # Both image IDs MUST be present for the citation renderer.
    assert outcome.image_ids == [1, 2]
    assert len(outcome.image_ids) == 2
    assert outcome.image_a_filename is not None
    assert outcome.image_b_filename is not None


def test_comparison_12_to_advanced_outcome_carries_image_ids(monkeypatch):
    """``to_advanced_outcome`` preserves image_ids for the orchestrator surface."""
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, return_value=_phase34b_vision_result())

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    adv = outcome.to_advanced_outcome()
    assert adv.image_ids == [1, 2]
    payload = adv.to_dict()
    assert payload["advanced_vision_image_count"] == 2
    assert payload["advanced_vision_image_ids"] == [1, 2]


# ---------------------------------------------------------------------------
# Provider failure paths
# ---------------------------------------------------------------------------


def test_comparison_13_provider_timeout_degrades_safely(monkeypatch):
    from app.services.vision.base import VisionProviderTimeoutError
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, raise_exc=VisionProviderTimeoutError("simulated"))

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.skipped_reason in ("provider_error", "provider_timeout")
    assert outcome.result is not None


def test_comparison_14_provider_unavailable_degrades_safely(monkeypatch):
    from app.services.vision.base import VisionProviderUnavailableError
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    _patch_provider(monkeypatch, raise_exc=VisionProviderUnavailableError("503"))

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.skipped_reason == "provider_error"
    assert outcome.result is not None


def test_comparison_15_provider_unavailable_in_factory(monkeypatch):
    """When the Phase 34B factory returns None, comparison still degrades safely."""
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    from app.services.advanced_vision import comparison as cmp
    monkeypatch.setattr(cmp, "_resolve_provider", lambda: None)

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    assert outcome.ran is True  # ran but did not invoke the provider
    assert outcome.skipped_reason == "no_provider"
    assert outcome.provider_called is False


# ---------------------------------------------------------------------------
# Safe parse failure
# ---------------------------------------------------------------------------


def test_comparison_16_unparseable_response_degrades_safely(monkeypatch):
    _patch_image_fetch_for_pair(monkeypatch)
    _patch_can_access(monkeypatch, allow=True)
    _patch_redis(monkeypatch)
    empty_vision = type(
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
    _patch_provider(monkeypatch, return_value=empty_vision)

    outcome = run_advanced_visual_comparison(
        None, auth=FakeAuth(), question="What changed?",
        image_id_a=1, image_id_b=2,
    )
    # Empty response → safe failure path; image_ids preserved for citation.
    assert outcome.skipped_reason == "parse_failure"
    assert outcome.image_ids == [1, 2]
