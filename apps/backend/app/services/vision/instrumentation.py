"""Phase 34C.1 -- Vision provider call counter (in-process, env-gated).

Used by Test F to prove the Phase 34C.1 backfill issues zero Vision
provider invocations. The counter is in-memory only (per worker
process) and reset by ``reset_vision_call_counter``. It is wrapped
around the Vision provider factory so every concrete provider
construction is counted exactly once per call, including cache hits
within a process.

**Disabled by default.** Every ``record_*`` call short-circuits to a
no-op unless the environment variable
``VISION_INSTRUMENTATION_ENABLED=1`` is set. Production deployments
have zero overhead from this module; tests set the env var to opt
in.
"""

from __future__ import annotations

import os
import threading
from typing import Dict


_ENABLED = os.environ.get("VISION_INSTRUMENTATION_ENABLED") == "1"

_lock = threading.Lock()
_counts: Dict[str, int] = {
    "factory_calls": 0,           # every successful get_vision_provider() returning a provider
    "factory_returned_none": 0,   # VISION_ENABLED is false or no provider configured
    "analyze_image_calls": 0,     # every analyze_image() entry (across all providers)
}


def _is_enabled() -> bool:
    """Re-read the env var on every call so tests can flip it at runtime."""
    return os.environ.get("VISION_INSTRUMENTATION_ENABLED") == "1"


def record_factory_call(returned_provider: bool) -> None:
    if not _is_enabled():
        return
    with _lock:
        if returned_provider:
            _counts["factory_calls"] += 1
        else:
            _counts["factory_returned_none"] += 1


def record_analyze_image_call(provider_name: str = "") -> None:
    if not _is_enabled():
        return
    with _lock:
        _counts["analyze_image_calls"] += 1


def get_vision_call_counts() -> Dict[str, int]:
    if not _is_enabled():
        # Return zeros (an empty counter) when disabled so callers see
        # the same shape regardless of state.
        return {"factory_calls": 0, "factory_returned_none": 0, "analyze_image_calls": 0}
    with _lock:
        return dict(_counts)


def reset_vision_call_counter() -> None:
    if not _is_enabled():
        return
    with _lock:
        for k in list(_counts.keys()):
            _counts[k] = 0
