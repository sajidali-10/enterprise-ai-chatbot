"""Phase 34D — Advanced Visual Understanding: base types.

Defines the structured types Phase 34D adds on top of the Phase 34B
Vision pipeline:

* ``AdvancedVisualTask`` — the task taxonomy (``general_visual``,
  ``ui_state_analysis``, ``chart_analysis``, ``diagram_analysis``,
  ``table_visual_analysis``, ``image_comparison``).

* ``VisualReasoningResult`` — a structured dataclass that captures
  every observation an advanced Vision pass can produce. Every field
  is length-bounded so a hostile provider response cannot inflate
  the LLM prompt or the Redis cache value.

* Cache key builders — deterministic and order-preserving for the
  comparison path (image A is the *before* state, image B is the
  *after* state — swapping them would silently invert the answer).

The module is pure: no DB, no network, no provider calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bump when the VisualReasoningResult schema changes in a
# backward-incompatible way. The Redis cache key embeds this value, so
# a bump invalidates every cached entry in one deployment.
ADVANCED_VISION_SCHEMA_VERSION = 1

# Hard length caps applied both by ``to_dict`` (sanity) and by
# ``from_dict`` (defensive parsing of cached JSON). The values mirror
# the spec's design intent — every list / string is bounded so a
# pathological provider response cannot grow without limit.
MAX_SUMMARY_CHARS = 280
MAX_OBSERVATIONS = 8
MAX_OBSERVATION_CHARS = 200
MAX_ANOMALIES = 6
MAX_ANOMALY_CHARS = 200
MAX_RELATIONSHIPS = 24
MAX_UI_STATES = 12
MAX_CHART_FINDINGS = 12
MAX_DIAGRAM_FINDINGS = 12
MAX_TABLE_FINDINGS = 12
MAX_COMPARISON_CHANGES = 24
MAX_ENTITIES = 12
MAX_ENTITY_CHARS = 80
MAX_EVIDENCE_REFS = 8
MAX_DICT_KEY_CHARS = 64
MAX_DICT_VALUE_CHARS = 240

# Cap for the optional in-process LRU.
DEFAULT_PROCESS_CACHE_MAXSIZE = 64


class AdvancedVisualTask(str, Enum):
    """Task taxonomy for advanced visual reasoning.

    Values are lowercase strings so they survive JSON serialisation
    into Redis, LangSmith traces, and the chat response without any
    mapping layer.
    """

    GENERAL_VISUAL = "general_visual"
    UI_STATE_ANALYSIS = "ui_state_analysis"
    CHART_ANALYSIS = "chart_analysis"
    DIAGRAM_ANALYSIS = "diagram_analysis"
    TABLE_VISUAL_ANALYSIS = "table_visual_analysis"
    IMAGE_COMPARISON = "image_comparison"


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------


@dataclass
class VisualReasoningResult:
    """Structured output of one advanced visual reasoning pass.

    The provider is expected to return JSON matching this shape; the
    dataclass ``from_dict`` parser is lenient about missing optional
    fields but strict about types and length caps so a hostile
    payload cannot bypass the safety bounds.
    """

    task_type: str = AdvancedVisualTask.GENERAL_VISUAL.value
    summary: str = ""
    observations: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    ui_states: List[Dict[str, Any]] = field(default_factory=list)
    chart_findings: List[Dict[str, Any]] = field(default_factory=list)
    diagram_findings: List[Dict[str, Any]] = field(default_factory=list)
    table_findings: List[Dict[str, Any]] = field(default_factory=list)
    comparison_changes: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    confidence: Optional[int] = None
    provider: str = ""
    model: str = ""
    processing_time_ms: int = 0
    schema_version: int = ADVANCED_VISION_SCHEMA_VERSION
    image_ids: List[int] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    # --- (de)serialisation -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict with every field length-capped.

        The serialised payload is what we store in Redis — by
        construction it MUST NOT contain image bytes, raw provider
        payload, OCR bodies, or API keys. None of the dataclass
        fields can carry those values, so the contract is enforced
        by the schema, not by the serializer.
        """
        return {
            "task_type": str(self.task_type or AdvancedVisualTask.GENERAL_VISUAL.value),
            "summary": _truncate(self.summary, MAX_SUMMARY_CHARS),
            "observations": _list_capped(self.observations, MAX_OBSERVATIONS, MAX_OBSERVATION_CHARS),
            "anomalies": _list_capped(self.anomalies, MAX_ANOMALIES, MAX_ANOMALY_CHARS),
            "relationships": _list_of_dict_capped(
                self.relationships, MAX_RELATIONSHIPS,
                max_keys=4, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=True,
            ),
            "ui_states": _list_of_dict_capped(
                self.ui_states, MAX_UI_STATES,
                max_keys=8, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            "chart_findings": _list_of_dict_capped(
                self.chart_findings, MAX_CHART_FINDINGS,
                max_keys=8, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            "diagram_findings": _list_of_dict_capped(
                self.diagram_findings, MAX_DIAGRAM_FINDINGS,
                max_keys=12, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            "table_findings": _list_of_dict_capped(
                self.table_findings, MAX_TABLE_FINDINGS,
                max_keys=8, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            "comparison_changes": _list_of_dict_capped(
                self.comparison_changes, MAX_COMPARISON_CHANGES,
                max_keys=8, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            "entities": _list_capped(self.entities, MAX_ENTITIES, MAX_ENTITY_CHARS),
            "confidence": _safe_int_or_none(self.confidence),
            "provider": _truncate(self.provider, MAX_DICT_VALUE_CHARS),
            "model": _truncate(self.model, MAX_DICT_VALUE_CHARS),
            "processing_time_ms": max(0, int(self.processing_time_ms or 0)),
            "schema_version": int(self.schema_version or ADVANCED_VISION_SCHEMA_VERSION),
            "image_ids": _list_capped(
                [str(x) for x in (self.image_ids or [])], MAX_RELATIONSHIPS, 32
            ),
            "evidence_refs": _list_capped(
                self.evidence_refs, MAX_EVIDENCE_REFS, MAX_DICT_VALUE_CHARS
            ),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisualReasoningResult":
        """Lenient parser with strict bounds enforcement.

        Missing optional fields default to empty. Unknown fields are
        ignored. Field types that do not match the schema collapse to
        their defaults (no exception). Length caps are applied so a
        maliciously-grown cached payload cannot survive a round-trip.
        """
        if not isinstance(payload, dict):
            return cls()

        task_type = str(
            payload.get("task_type") or AdvancedVisualTask.GENERAL_VISUAL.value
        ).strip().lower() or AdvancedVisualTask.GENERAL_VISUAL.value
        # Unknown task_type falls back to general_visual — defensive.
        valid = {t.value for t in AdvancedVisualTask}
        if task_type not in valid:
            task_type = AdvancedVisualTask.GENERAL_VISUAL.value

        return cls(
            task_type=task_type,
            summary=_truncate(str(payload.get("summary") or ""), MAX_SUMMARY_CHARS),
            observations=_list_capped(payload.get("observations") or [], MAX_OBSERVATIONS, MAX_OBSERVATION_CHARS),
            anomalies=_list_capped(payload.get("anomalies") or [], MAX_ANOMALIES, MAX_ANOMALY_CHARS),
            relationships=_list_of_dict_capped(
                payload.get("relationships") or [], MAX_RELATIONSHIPS,
                max_keys=4, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=True,
            ),
            ui_states=_list_of_dict_capped(
                payload.get("ui_states") or [], MAX_UI_STATES,
                max_keys=8, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            chart_findings=_list_of_dict_capped(
                payload.get("chart_findings") or [], MAX_CHART_FINDINGS,
                max_keys=8, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            diagram_findings=_list_of_dict_capped(
                payload.get("diagram_findings") or [], MAX_DIAGRAM_FINDINGS,
                max_keys=12, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            table_findings=_list_of_dict_capped(
                payload.get("table_findings") or [], MAX_TABLE_FINDINGS,
                max_keys=8, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            comparison_changes=_list_of_dict_capped(
                payload.get("comparison_changes") or [], MAX_COMPARISON_CHANGES,
                max_keys=8, max_key_chars=MAX_DICT_KEY_CHARS,
                max_value_chars=MAX_DICT_VALUE_CHARS,
                string_values_only=False,
            ),
            entities=_list_capped(payload.get("entities") or [], MAX_ENTITIES, MAX_ENTITY_CHARS),
            confidence=_safe_int_or_none(payload.get("confidence")),
            provider=_truncate(str(payload.get("provider") or ""), MAX_DICT_VALUE_CHARS),
            model=_truncate(str(payload.get("model") or ""), MAX_DICT_VALUE_CHARS),
            processing_time_ms=max(0, int(payload.get("processing_time_ms") or 0)),
            schema_version=int(
                payload.get("schema_version") or ADVANCED_VISION_SCHEMA_VERSION
            ),
            image_ids=[
                int(x) for x in (payload.get("image_ids") or []) if _safe_int_or_none(x) is not None
            ][:MAX_RELATIONSHIPS],
            evidence_refs=_list_capped(
                payload.get("evidence_refs") or [], MAX_EVIDENCE_REFS, MAX_DICT_VALUE_CHARS
            ),
        )

    def is_successful(self) -> bool:
        """True when the result has any usable content.

        A result with only a fallback summary IS successful: the LLM
        still gets an explicit "advanced vision unavailable" signal
        and the cache treats it as a real, cacheable observation. The
        orchestrator's failure-path detection relies on
        ``skipped_reason`` rather than this predicate.
        """
        return bool(
            self.summary
            or self.observations
            or self.anomalies
            or self.relationships
            or self.ui_states
            or self.chart_findings
            or self.diagram_findings
            or self.table_findings
            or self.comparison_changes
        )

    def image_count(self) -> int:
        """Number of images this result was produced for.

        Always >= 1 when the result is meaningful. ``IMAGE_COMPARISON``
        results are produced for exactly 2 images.
        """
        return max(1, len(self.image_ids or []))


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def image_content_hash(image_bytes: Optional[bytes]) -> str:
    """Stable, short, content-derived hash for an image.

    First 16 bytes of SHA-256 hex-encoded (32 chars). Enough to
    distinguish re-uploads without paying for a full hash on every
    request. Empty input produces a sentinel so cache keys remain
    deterministic when bytes are unavailable.
    """
    if not image_bytes:
        return "empty"
    return hashlib.sha256(image_bytes).hexdigest()[:32]


def build_advanced_vision_cache_key(
    *,
    schema_version: int,
    provider: str,
    model: str,
    task_type: str,
    image_hashes: List[str],
    extra_salt: str = "",
) -> str:
    """Deterministic Redis key for an advanced-vision analysis.

    Order of ``image_hashes`` is preserved (NOT lexicographically
    sorted) because the ``IMAGE_COMPARISON`` task treats image A as
    *before* and image B as *after*. Sorting would silently invert
    the answer.

    The key is short enough to fit Redis key-length limits
    comfortably and human-readable for ops debugging.
    """
    hashes = list(image_hashes or [])
    # Only the comparison task explicitly supports multiple images.
    # Single-image tasks ignore everything after the first hash; we
    # still include all hashes to keep the key shape uniform.
    salt = _truncate(str(extra_salt or ""), 64)
    task = str(task_type or AdvancedVisualTask.GENERAL_VISUAL.value)
    return (
        f"av:{schema_version}:{provider}:{model}:{task}:"
        + ":".join(hashes)
        + (f":{salt}" if salt else "")
    )


# ---------------------------------------------------------------------------
# Failure-safe defaults
# ---------------------------------------------------------------------------


def empty_result(
    *,
    task_type: str = AdvancedVisualTask.GENERAL_VISUAL.value,
    provider: str = "",
    model: str = "",
    processing_time_ms: int = 0,
    summary: str = "",
    image_ids: Optional[List[int]] = None,
) -> VisualReasoningResult:
    """Build a safe empty result for provider-failure / parse-failure paths.

    ``is_successful()`` returns False so the orchestrator routes back
    to OCR-only / Phase 34B evidence. ``summary`` is intentionally
    non-empty so the LLM still receives an explicit "no advanced
    observation available" signal when rendering the answer.
    """
    return VisualReasoningResult(
        task_type=str(task_type or AdvancedVisualTask.GENERAL_VISUAL.value),
        summary=_truncate(summary or "advanced vision unavailable", MAX_SUMMARY_CHARS),
        provider=_truncate(provider, MAX_DICT_VALUE_CHARS),
        model=_truncate(model, MAX_DICT_VALUE_CHARS),
        processing_time_ms=max(0, int(processing_time_ms or 0)),
        image_ids=list(image_ids or []),
    )


def safe_parse_provider_payload(
    parsed: Dict[str, Any],
    *,
    provider: str,
    model: str,
    processing_time_ms: int,
    task_type: str,
    image_ids: Optional[List[int]] = None,
) -> VisualReasoningResult:
    """Convert a parsed provider payload into a bounded result.

    The provider may return either the full ``VisualReasoningResult``
    schema OR a partial payload (e.g. only ``summary``). The parser
    is defensive: missing optional fields collapse to empty defaults
    rather than raising.
    """
    result = VisualReasoningResult.from_dict(parsed)
    result.provider = _truncate(provider, MAX_DICT_VALUE_CHARS)
    result.model = _truncate(model, MAX_DICT_VALUE_CHARS)
    result.processing_time_ms = max(0, int(processing_time_ms or 0))
    # Preserve the requested task_type if the provider omitted it.
    if not result.task_type or result.task_type == AdvancedVisualTask.GENERAL_VISUAL.value:
        result.task_type = str(task_type or AdvancedVisualTask.GENERAL_VISUAL.value)
    if image_ids:
        result.image_ids = list(image_ids)
    return result


# ---------------------------------------------------------------------------
# Pure helpers (tested in isolation)
# ---------------------------------------------------------------------------


def _truncate(value: Any, max_chars: int) -> str:
    s = "" if value is None else str(value)
    if len(s) <= max_chars:
        return s
    return s[: max(1, max_chars - 1)].rsplit(" ", 1)[0] + "…"


def _safe_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, min(100, n))


def _list_capped(value: Any, max_items: int, max_chars: int) -> List[str]:
    """Coerce ``value`` to a list of bounded strings."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [value]
    else:
        items = [str(value)]
    out: List[str] = []
    seen: set = set()
    for raw in items:
        if raw is None:
            continue
        s = _truncate(raw, max_chars).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _list_of_dict_capped(
    value: Any,
    max_items: int,
    *,
    max_keys: int,
    max_key_chars: int,
    max_value_chars: int,
    string_values_only: bool,
) -> List[Dict[str, Any]]:
    """Coerce ``value`` to a list of bounded dicts."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    out: List[Dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            # Skip non-dict entries — a hostile provider might emit a
            # list of strings here. They do not survive schema
            # validation; dropping them is the safest response.
            continue
        entry: Dict[str, Any] = {}
        for i, (k, v) in enumerate(raw.items()):
            if i >= max_keys:
                break
            key = _truncate(k, max_key_chars).strip()
            if not key:
                continue
            if string_values_only and not isinstance(v, str):
                v = str(v)
            entry[key] = _truncate(v, max_value_chars)
        if entry:
            out.append(entry)
        if len(out) >= max_items:
            break
    return out


__all__ = [
    "ADVANCED_VISION_SCHEMA_VERSION",
    "AdvancedVisualTask",
    "VisualReasoningResult",
    "image_content_hash",
    "build_advanced_vision_cache_key",
    "empty_result",
    "safe_parse_provider_payload",
    "MAX_SUMMARY_CHARS",
    "MAX_OBSERVATIONS",
    "MAX_OBSERVATION_CHARS",
    "MAX_ANOMALIES",
    "MAX_ANOMALY_CHARS",
    "MAX_RELATIONSHIPS",
    "MAX_UI_STATES",
    "MAX_CHART_FINDINGS",
    "MAX_DIAGRAM_FINDINGS",
    "MAX_TABLE_FINDINGS",
    "MAX_COMPARISON_CHANGES",
    "MAX_ENTITIES",
    "MAX_ENTITY_CHARS",
    "MAX_EVIDENCE_REFS",
    "MAX_DICT_KEY_CHARS",
    "MAX_DICT_VALUE_CHARS",
    "DEFAULT_PROCESS_CACHE_MAXSIZE",
]
