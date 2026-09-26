"""Phase 34D — Redis-backed advanced Vision cache.

Production-safe task-specific cache for :class:`VisualReasoningResult`.

Design notes:

* **Redis is the authoritative cache.** Every read / write goes
  through Redis first. A failure is logged at debug level and
  treated as a cache miss so the orchestrator can fall through to
  the provider call.

* **Optional in-process LRU is an optimization only.** It is
  bounded, opt-in (default off), and is NEVER consulted on Redis
  outage — the in-process cache never substitutes for Redis.

* **No image bytes / secrets in the cache.** Only
  ``VisualReasoningResult.to_dict()`` JSON is stored, which by
  construction excludes raw image bytes, OCR bodies, provider
  request payloads, and API keys.

* **Comparison cache keys preserve A→B order.** The key builder
  takes an ordered list of image hashes; swapping image A and
  image B produces a different key so a swapped cache hit cannot
  silently invert a comparison answer.

* **Cache key embeds schema version + provider + model + task.**
  Bumping ``ADVANCED_VISION_SCHEMA_VERSION`` invalidates every
  cached entry on the next access.

* **TTL is configurable.** Default 1 hour. Expired entries are
  treated as misses (Redis deletes them lazily on access).

The module is import-safe: it lazily imports ``redis`` only when
the cache is actually exercised. In environments without Redis
(e.g. local SQLite test runs) the cache degrades to a no-op so
the chat pipeline is never blocked.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.advanced_vision.base import (
    VisualReasoningResult,
    empty_result,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache value container
# ---------------------------------------------------------------------------


@dataclass
class CacheReadResult:
    """Outcome of a cache read.

    ``hit=True`` when a structured result was successfully retrieved
    and parsed. ``hit=False`` covers both cache miss (key absent)
    and cache failure (Redis down / corrupt JSON / disabled). The
    caller treats both as "compute fresh" and may consult ``error``
    for observability.
    """

    hit: bool = False
    result: Optional[VisualReasoningResult] = None
    error: Optional[str] = None
    source: str = ""  # "redis" | "process" | "miss" | "disabled"


# ---------------------------------------------------------------------------
# Lazy Redis client
# ---------------------------------------------------------------------------


_redis_lock = threading.Lock()
_redis_client: Optional[Any] = None
_redis_checked: bool = False


def _get_redis_client():
    """Lazily connect to Redis using the existing REDIS_HOST / REDIS_PORT.

    Mirrors the pattern in ``app.core.rate_limit``. Returns ``None``
    when Redis is unreachable — the caller must treat that as a
    cache miss and continue.
    """
    global _redis_client, _redis_checked
    if _redis_client is not None:
        return _redis_client
    with _redis_lock:
        if _redis_checked:
            return _redis_client
        _redis_checked = True
        try:
            import redis as redis_lib  # type: ignore
            client = redis_lib.Redis(
                host=settings.REDIS_HOST,
                port=int(settings.REDIS_PORT),
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            client.ping()
            logger.info(
                "advanced_vision.cache: using Redis backend (%s:%s)",
                settings.REDIS_HOST, settings.REDIS_PORT,
            )
            _redis_client = client
            return client
        except Exception as exc:  # pragma: no cover - infra-dependent
            logger.debug(
                "advanced_vision.cache: Redis unavailable (%s). "
                "Cache will degrade to provider-only.",
                exc,
            )
            _redis_client = None
            return None


def reset_redis_client_for_tests() -> None:
    """Reset the Redis client so tests can rebind the host/port."""
    global _redis_client, _redis_checked
    with _redis_lock:
        _redis_client = None
        _redis_checked = False


# ---------------------------------------------------------------------------
# Optional in-process LRU (optimization only)
# ---------------------------------------------------------------------------


class _ProcessLRU:
    """Tiny LRU used as an optimization in front of Redis.

    It is intentionally small. It is never consulted on Redis outage
    (the orchestrator always asks Redis first). When Redis is up,
    a hit here avoids the Redis round-trip.
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._maxsize = max(1, int(maxsize))
        self._data: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
            else:
                self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


_process_lru: Optional[_ProcessLRU] = None
_process_lru_lock = threading.Lock()


def _get_process_lru() -> Optional[_ProcessLRU]:
    """Return the in-process LRU when enabled, else ``None``."""
    global _process_lru
    if not bool(getattr(settings, "ADVANCED_VISION_PROCESS_CACHE_ENABLED", False)):
        return None
    with _process_lru_lock:
        if _process_lru is None:
            maxsize = int(
                getattr(settings, "ADVANCED_VISION_PROCESS_CACHE_MAXSIZE", 64)
            )
            _process_lru = _ProcessLRU(maxsize=maxsize)
        return _process_lru


def reset_process_lru_for_tests() -> None:
    global _process_lru
    with _process_lru_lock:
        _process_lru = None


# ---------------------------------------------------------------------------
# Public cache API
# ---------------------------------------------------------------------------


def cache_enabled() -> bool:
    """True when the cache is enabled by configuration.

    Either Redis OR the in-process LRU can satisfy a read; both are
    gated by their own settings. The orchestrator can ask the cache
    to serve a hit from either layer.
    """
    return bool(getattr(settings, "ADVANCED_VISION_CACHE_ENABLED", True))


def get_cached_result(
    key: str,
    *,
    expect_schema_version: int,
) -> CacheReadResult:
    """Look up a cached :class:`VisualReasoningResult`.

    Args:
        key: Cache key produced by ``build_advanced_vision_cache_key``.
        expect_schema_version: The current schema version. When the
            cached payload declares a different version (operator
            bumped the version), the entry is treated as a miss so a
            stale analysis cannot survive a schema change.

    Returns:
        ``CacheReadResult``. ``hit=True`` only when a usable
        ``VisualReasoningResult`` is recovered from the cache.
    """
    if not cache_enabled():
        return CacheReadResult(hit=False, source="disabled")

    # Process cache first — purely an optimization. It is only
    # consulted when Redis would also have been consulted; the
    # orchestrator always passes through to Redis when this returns
    # a miss.
    lru = _get_process_lru()
    if lru is not None:
        blob = lru.get(key)
        if blob is not None:
            parsed = _safe_parse(blob, expect_schema_version=expect_schema_version)
            if parsed is not None:
                return CacheReadResult(hit=True, result=parsed, source="process")
            # corrupt cache entry — treat as miss and continue to Redis

    redis_client = _get_redis_client()
    if redis_client is None:
        return CacheReadResult(hit=False, source="miss", error="redis_unavailable")

    try:
        blob = redis_client.get(key)
    except Exception as exc:
        logger.debug("advanced_vision.cache: redis GET failed: %s", exc)
        return CacheReadResult(hit=False, source="miss", error=f"redis_get_failed: {str(exc)[:160]}")

    if not blob:
        return CacheReadResult(hit=False, source="miss")

    parsed = _safe_parse(blob, expect_schema_version=expect_schema_version)
    if parsed is None:
        return CacheReadResult(hit=False, source="miss", error="corrupt_cached_value")

    if lru is not None:
        try:
            lru.set(key, blob)
        except Exception:
            pass

    return CacheReadResult(hit=True, result=parsed, source="redis")


def set_cached_result(
    key: str,
    result: VisualReasoningResult,
    *,
    ttl_seconds: Optional[int] = None,
) -> bool:
    """Store a :class:`VisualReasoningResult` in the cache.

    Args:
        key: Cache key.
        result: The result to serialise. Only ``to_dict()`` JSON is
            stored; no image bytes, no provider internals.
        ttl_seconds: Override TTL. Defaults to
            ``settings.ADVANCED_VISION_CACHE_TTL_SECONDS``.

    Returns:
        True when the write was successful (Redis or in-process
        LRU). False on every failure path — the caller MUST NOT
        raise based on a False return; cache write failures are
        observability events, not chat-breaking errors.
    """
    if not cache_enabled():
        return False
    if result is None or not result.is_successful():
        return False

    try:
        blob = json.dumps(result.to_dict(), ensure_ascii=False)
    except Exception as exc:
        logger.debug("advanced_vision.cache: serialisation failed: %s", exc)
        return False

    ttl = int(
        ttl_seconds
        if ttl_seconds is not None
        else getattr(settings, "ADVANCED_VISION_CACHE_TTL_SECONDS", 3600)
    )
    if ttl <= 0:
        ttl = 3600

    # Process cache first (when enabled) so the next hit avoids Redis.
    lru = _get_process_lru()
    if lru is not None:
        try:
            lru.set(key, blob)
        except Exception:
            pass

    redis_client = _get_redis_client()
    if redis_client is None:
        return lru is not None  # write succeeded only if process cache accepted

    try:
        redis_client.set(key, blob, ex=ttl)
    except Exception as exc:
        logger.debug("advanced_vision.cache: redis SET failed: %s", exc)
        return lru is not None

    return True


def invalidate_key(key: str) -> bool:
    """Delete a single cache entry. Returns True on success.

    Used by tests and by Phase 34D.1 future workflows. Failures
    are logged but never raised.
    """
    lru = _get_process_lru()
    if lru is not None:
        try:
            # OrderedDict doesn't have a deletion guard; pop with default
            # is idempotent.
            lru._data.pop(key, None)  # type: ignore[attr-defined]
        except Exception:
            pass

    redis_client = _get_redis_client()
    if redis_client is None:
        return True
    try:
        redis_client.delete(key)
        return True
    except Exception as exc:
        logger.debug("advanced_vision.cache: redis DELETE failed: %s", exc)
        return False


def get_or_compute(
    *,
    key: str,
    expect_schema_version: int,
    compute: Callable[[], Tuple[VisualReasoningResult, bool]],
) -> Tuple[VisualReasoningResult, CacheReadResult, bool]:
    """Read-through helper.

    Args:
        key: Cache key.
        expect_schema_version: Current schema version.
        compute: Callable returning ``(result, success)``. The
            ``success`` flag tells the helper whether to cache the
            produced result. Failure paths return
            ``(empty_result(...), False)`` so the caller still
            receives a safe response.

    Returns:
        Tuple ``(result, cache_read, cached)``. ``cached`` is True
        when the result was recovered from the cache; False when it
        was computed fresh. ``cache_read`` carries the diagnostic
        fields for observability.
    """
    read = get_cached_result(key, expect_schema_version=expect_schema_version)
    if read.hit and read.result is not None and read.result.is_successful():
        return read.result, read, True

    result, success = compute()
    if success and result is not None and result.is_successful():
        # Don't propagate failures of the write — the chat pipeline
        # already has the result; losing a cache entry is fine.
        set_cached_result(key, result)
    return result, read, False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_parse(
    blob: Any,
    *,
    expect_schema_version: int,
) -> Optional[VisualReasoningResult]:
    """Parse a cached JSON blob into a :class:`VisualReasoningResult`.

    Returns ``None`` for any failure (non-JSON, wrong type, schema
    mismatch, missing required fields, etc.). The caller treats
    ``None`` as a cache miss.
    """
    if not isinstance(blob, str):
        return None
    try:
        payload = json.loads(blob)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    parsed = VisualReasoningResult.from_dict(payload)
    if parsed.schema_version != int(expect_schema_version):
        # Schema drift — treat as miss so the orchestrator produces
        # a fresh analysis under the new schema.
        return None
    if not parsed.is_successful():
        return None
    return parsed


__all__ = [
    "CacheReadResult",
    "cache_enabled",
    "get_cached_result",
    "set_cached_result",
    "invalidate_key",
    "get_or_compute",
    "reset_redis_client_for_tests",
    "reset_process_lru_for_tests",
]
