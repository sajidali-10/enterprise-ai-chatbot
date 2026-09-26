"""Phase 34D — Unit tests for the Redis-backed advanced Vision cache.

Covers:
* cache key determinism and order-preservation for comparison
* cache key changes when task_type changes
* cache key changes when schema_version changes
* safe (de)serialisation of VisualReasoningResult through JSON
* corrupt cached JSON degrades safely
* cache miss/failure never raises (Redis outage → provider path)
* set_cached_result is no-op when result is empty/failed
* to_dict/from_dict round-trip preserves every length-bounded field
* image_content_hash is deterministic and short
"""

from __future__ import annotations

import json

from app.services.advanced_vision.base import (
    AdvancedVisualTask,
    VisualReasoningResult,
    build_advanced_vision_cache_key,
    empty_result,
    image_content_hash,
)
from app.services.advanced_vision.cache import (
    CacheReadResult,
    _safe_parse,
    cache_enabled,
    get_cached_result,
    set_cached_result,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_cache_key_01_deterministic_for_same_inputs():
    a = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type=AdvancedVisualTask.UI_STATE_ANALYSIS.value,
        image_hashes=["abc123", "def456"],
    )
    b = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type=AdvancedVisualTask.UI_STATE_ANALYSIS.value,
        image_hashes=["abc123", "def456"],
    )
    assert a == b


def test_cache_key_02_changes_when_task_type_changes():
    base = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type=AdvancedVisualTask.UI_STATE_ANALYSIS.value,
        image_hashes=["h"],
    )
    other = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type=AdvancedVisualTask.CHART_ANALYSIS.value,
        image_hashes=["h"],
    )
    assert base != other


def test_cache_key_03_changes_when_schema_version_changes():
    a = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type="ui_state_analysis", image_hashes=["h"],
    )
    b = build_advanced_vision_cache_key(
        schema_version=2, provider="mock", model="mock-v1",
        task_type="ui_state_analysis", image_hashes=["h"],
    )
    assert a != b


def test_cache_key_04_changes_when_provider_or_model_changes():
    base = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type="ui_state_analysis", image_hashes=["h"],
    )
    other_provider = build_advanced_vision_cache_key(
        schema_version=1, provider="openai-compatible", model="mock-v1",
        task_type="ui_state_analysis", image_hashes=["h"],
    )
    other_model = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="gpt-4o-mini",
        task_type="ui_state_analysis", image_hashes=["h"],
    )
    assert base != other_provider
    assert base != other_model


def test_cache_key_05_comparison_preserves_a_to_b_order():
    """A swapped comparison key must produce a different key."""
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


def test_cache_key_06_includes_all_hashes():
    """Single-image tasks include the single hash; comparison includes both."""
    single = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type="ui_state_analysis", image_hashes=["hash_only"],
    )
    comparison = build_advanced_vision_cache_key(
        schema_version=1, provider="mock", model="mock-v1",
        task_type="image_comparison", image_hashes=["hash_a", "hash_b"],
    )
    assert "hash_only" in single
    assert "hash_a" in comparison
    assert "hash_b" in comparison


# ---------------------------------------------------------------------------
# image_content_hash
# ---------------------------------------------------------------------------


def test_content_hash_01_deterministic():
    a = image_content_hash(b"the quick brown fox")
    b = image_content_hash(b"the quick brown fox")
    assert a == b


def test_content_hash_02_distinct_for_different_bytes():
    assert image_content_hash(b"alpha") != image_content_hash(b"beta")


def test_content_hash_03_short_hex_string():
    h = image_content_hash(b"x" * 1_000_000)
    # SHA-256 prefix is 32 hex chars (16 bytes).
    assert len(h) == 32
    int(h, 16)  # raises if not valid hex


def test_content_hash_04_empty_bytes_returns_sentinel():
    assert image_content_hash(None) == "empty"
    assert image_content_hash(b"") == "empty"


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_serialise_01_round_trip_preserves_task_type_and_summary():
    original = VisualReasoningResult(
        task_type=AdvancedVisualTask.CHART_ANALYSIS.value,
        summary="Traffic rises sharply around 14:00.",
        observations=["Spike around 14:00", "Decline afterward"],
        confidence=82,
        image_ids=[42],
    )
    payload = original.to_dict()
    blob = json.dumps(payload)
    round_tripped = VisualReasoningResult.from_dict(json.loads(blob))
    assert round_tripped.task_type == AdvancedVisualTask.CHART_ANALYSIS.value
    assert round_tripped.summary == "Traffic rises sharply around 14:00."
    assert round_tripped.confidence == 82
    assert round_tripped.image_ids == [42]
    assert round_tripped.observations == ["Spike around 14:00", "Decline afterward"]


def test_serialise_02_round_trip_preserves_chart_findings():
    original = VisualReasoningResult(
        task_type=AdvancedVisualTask.CHART_ANALYSIS.value,
        summary="Chart findings",
        chart_findings=[
            {"kind": "spike", "location": "around 14:00", "evidence": "tall bar"},
            {"kind": "downward-trend", "location": "right side", "evidence": "slope down"},
        ],
    )
    blob = json.dumps(original.to_dict())
    parsed = VisualReasoningResult.from_dict(json.loads(blob))
    assert parsed.chart_findings[0]["kind"] == "spike"
    assert parsed.chart_findings[1]["kind"] == "downward-trend"


def test_serialise_03_round_trip_preserves_comparison_changes():
    original = VisualReasoningResult(
        task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
        summary="Server B changed.",
        comparison_changes=[
            {"subject": "Server B", "kind": "status-change",
             "before": "Healthy", "after": "Failed", "evidence": "label changed"},
        ],
        image_ids=[10, 20],
    )
    blob = json.dumps(original.to_dict())
    parsed = VisualReasoningResult.from_dict(json.loads(blob))
    assert parsed.comparison_changes[0]["before"] == "Healthy"
    assert parsed.comparison_changes[0]["after"] == "Failed"
    assert parsed.image_ids == [10, 20]


def test_serialise_04_truncates_summary():
    """to_dict enforces MAX_SUMMARY_CHARS."""
    original = VisualReasoningResult(
        task_type=AdvancedVisualTask.GENERAL_VISUAL.value,
        summary="x" * 1_000,
    )
    payload = original.to_dict()
    assert len(payload["summary"]) <= 280


def test_serialise_05_truncates_observations():
    """to_dict enforces MAX_OBSERVATIONS and MAX_OBSERVATION_CHARS."""
    original = VisualReasoningResult(
        task_type=AdvancedVisualTask.GENERAL_VISUAL.value,
        observations=["x" * 500] * 50,
    )
    payload = original.to_dict()
    assert len(payload["observations"]) <= 8
    for obs in payload["observations"]:
        assert len(obs) <= 200


def test_serialise_06_invalid_task_type_falls_back_to_general():
    """from_dict tolerates corrupted task_type without crashing."""
    payload = {"task_type": "hacker-injected", "summary": "x"}
    parsed = VisualReasoningResult.from_dict(payload)
    assert parsed.task_type == AdvancedVisualTask.GENERAL_VISUAL.value


def test_serialise_07_non_dict_payload_yields_empty_result():
    parsed = VisualReasoningResult.from_dict("not a dict")  # type: ignore[arg-type]
    assert parsed.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert parsed.is_successful() is False


# ---------------------------------------------------------------------------
# Cache read / write safety
# ---------------------------------------------------------------------------


def test_cache_safe_parse_01_returns_none_for_garbage():
    assert _safe_parse("not json", expect_schema_version=1) is None
    assert _safe_parse("{}", expect_schema_version=1) is None
    assert _safe_parse(123, expect_schema_version=1) is None  # type: ignore[arg-type]
    assert _safe_parse(None, expect_schema_version=1) is None  # type: ignore[arg-type]


def test_cache_safe_parse_02_returns_none_for_schema_mismatch():
    blob = json.dumps({"task_type": "ui_state_analysis", "summary": "x"})
    # schema mismatch → cache miss (the orchestrator will re-run)
    assert _safe_parse(blob, expect_schema_version=99) is None


def test_cache_safe_parse_03_returns_result_for_matching_schema():
    blob = json.dumps({
        "task_type": "ui_state_analysis",
        "summary": "Server B failed",
        "schema_version": 1,
    })
    parsed = _safe_parse(blob, expect_schema_version=1)
    assert parsed is not None
    assert parsed.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value


def test_cache_write_01_empty_result_is_not_written(monkeypatch):
    """empty_result / unsuccessful results are NEVER cached."""
    # Mock the in-process LRU to fail the write so we can prove the
    # write is rejected upstream (when cache_enabled is False).
    monkeypatch.setattr("app.services.advanced_vision.cache._get_process_lru", lambda: None)
    monkeypatch.setattr("app.services.advanced_vision.cache._get_redis_client", lambda: None)

    # ``empty_result`` has a fallback summary by design so the LLM
    # still sees "advanced vision unavailable". A TRULY-empty result
    # (all fields blank) is what the cache must reject.
    failed = VisualReasoningResult(
        task_type=AdvancedVisualTask.UI_STATE_ANALYSIS.value,
        summary="",
    )
    assert failed.is_successful() is False
    # set_cached_result refuses to write unsuccessful results.
    assert set_cached_result("any:key", failed) is False


def test_cache_write_02_disabled_cache_returns_false(monkeypatch):
    monkeypatch.setattr("app.services.advanced_vision.cache.cache_enabled", lambda: False)
    good = VisualReasoningResult(
        task_type=AdvancedVisualTask.UI_STATE_ANALYSIS.value,
        summary="Server B failed",
    )
    assert set_cached_result("any:key", good) is False


def test_cache_read_01_redis_outage_returns_miss(monkeypatch):
    """Redis unreachable → cache miss (provider path takes over)."""
    monkeypatch.setattr("app.services.advanced_vision.cache.cache_enabled", lambda: True)
    monkeypatch.setattr("app.services.advanced_vision.cache._get_process_lru", lambda: None)
    # Force the Redis client to return None (simulating outage).
    monkeypatch.setattr("app.services.advanced_vision.cache._get_redis_client", lambda: None)
    result = get_cached_result("any:key", expect_schema_version=1)
    assert result.hit is False
    assert result.source == "miss"


def test_cache_read_02_disabled_cache_returns_miss(monkeypatch):
    monkeypatch.setattr("app.services.advanced_vision.cache.cache_enabled", lambda: False)
    result = get_cached_result("any:key", expect_schema_version=1)
    assert result.hit is False
    assert result.source == "disabled"


def test_cache_read_03_corrupt_blob_returns_miss(monkeypatch):
    """Corrupt cached JSON → cache miss (provider re-runs)."""
    monkeypatch.setattr("app.services.advanced_vision.cache.cache_enabled", lambda: True)
    monkeypatch.setattr("app.services.advanced_vision.cache._get_process_lru", lambda: None)
    fake_client = type(
        "FakeRedis",
        (),
        {"get": staticmethod(lambda _key: "{not valid json")},
    )()
    monkeypatch.setattr("app.services.advanced_vision.cache._get_redis_client", lambda: fake_client)
    result = get_cached_result("any:key", expect_schema_version=1)
    assert result.hit is False
    assert result.source == "miss"


def test_cache_read_04_schema_mismatch_in_blob_returns_miss(monkeypatch):
    """Schema-version mismatch → cache miss (orchestrator re-runs)."""
    monkeypatch.setattr("app.services.advanced_vision.cache.cache_enabled", lambda: True)
    monkeypatch.setattr("app.services.advanced_vision.cache._get_process_lru", lambda: None)
    blob = json.dumps({
        "task_type": "ui_state_analysis",
        "summary": "x",
        "schema_version": 1,
    })
    fake_client = type(
        "FakeRedis",
        (),
        {"get": staticmethod(lambda _key: blob)},
    )()
    monkeypatch.setattr("app.services.advanced_vision.cache._get_redis_client", lambda: fake_client)
    # Expecting a different schema version → miss
    result = get_cached_result("any:key", expect_schema_version=99)
    assert result.hit is False
    assert result.source == "miss"


def test_cache_read_05_valid_blob_returns_hit(monkeypatch):
    """Valid blob + matching schema → hit."""
    monkeypatch.setattr("app.services.advanced_vision.cache.cache_enabled", lambda: True)
    monkeypatch.setattr("app.services.advanced_vision.cache._get_process_lru", lambda: None)
    blob = json.dumps({
        "task_type": "ui_state_analysis",
        "summary": "Server B failed",
        "schema_version": 1,
    })
    fake_client = type(
        "FakeRedis",
        (),
        {"get": staticmethod(lambda _key: blob)},
    )()
    monkeypatch.setattr("app.services.advanced_vision.cache._get_redis_client", lambda: fake_client)
    result = get_cached_result("any:key", expect_schema_version=1)
    assert result.hit is True
    assert result.source == "redis"
    assert result.result is not None
    assert result.result.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value
