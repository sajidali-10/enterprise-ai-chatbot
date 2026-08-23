"""Phase 34B — Vision provider base interface.

Defines the narrow contract every Vision provider must satisfy.
The contract is intentionally minimal so adding a new provider
does not require changing application code.

Lifecycle:
    1. The router decides Vision is justified (see ``app.vision.router``).
    2. The persistence layer checks for a cached ``VisionResult`` on
       the ``DocumentImage`` row (same provider + model + schema_version).
    3. If no cache hit, the factory returns a configured provider.
    4. The provider's ``analyze_image`` is called with the raw
       image bytes the caller is authorized to send.
    5. The structured ``VisionResult`` is persisted and forwarded
       into the Evidence Builder.

The contract NEVER returns a free-form paragraph. ``VisionResult``
is always structured so it can be cached, diffed, and used as
grounded evidence by the LLM.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


# Bump when the VisionResult schema changes in a backward-incompatible
# way. The cache layer uses this to invalidate stale rows.
VISION_SCHEMA_VERSION = 1


class VisionProviderError(Exception):
    """Base class for Vision provider failures."""


class VisionProviderUnavailableError(VisionProviderError):
    """Provider is misconfigured, offline, or rate-limited.

    The router falls back to OCR evidence when this is raised.
    """


class VisionProviderTimeoutError(VisionProviderError):
    """Provider call exceeded VISION_TIMEOUT_SECONDS."""


@dataclass
class VisionResult:
    """Structured visual understanding result.

    The router and Evidence Builder treat this as EVIDENCE derived
    from the image, NOT as authoritative HipLink product knowledge.
    Any claim about product behavior, troubleshooting, or KB
    semantics must come from RAG — Vision only describes what is
    visually present.
    """

    description: str = ""
    image_type: str = "unknown"
    visual_findings: List[str] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)
    visual_states: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    confidence: Optional[int] = None
    provider: str = ""
    model: str = ""
    processing_time_ms: int = 0
    schema_version: int = VISION_SCHEMA_VERSION
    # Provider-specific raw payload for debugging/observability. NOT
    # forwarded to the LLM by default — the router only renders the
    # structured fields above into evidence.
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "image_type": self.image_type,
            "visual_findings": list(self.visual_findings or []),
            "detected_entities": list(self.detected_entities or []),
            "visual_states": list(self.visual_states or []),
            "tags": list(self.tags or []),
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
            "processing_time_ms": self.processing_time_ms,
            "schema_version": self.schema_version,
        }

    def is_successful(self) -> bool:
        return bool(self.description or self.visual_findings or self.detected_entities)


class VisionProvider(Protocol):
    """Minimal Vision provider contract.

    Implementations MUST:

    * never log or echo the API key,
    * never store raw bytes past the call,
    * surface transient failures as ``VisionProviderUnavailableError``
      or ``VisionProviderTimeoutError`` so the router can degrade
      gracefully to OCR.
    """

    name: str
    model: str

    def analyze_image(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        ocr_text: str = "",
        context_hint: str = "",
        max_tokens: int = 600,
        timeout_s: Optional[float] = None,
    ) -> VisionResult:
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_cache_key(
    *,
    document_image_id: int,
    provider: str,
    model: str,
    schema_version: int = VISION_SCHEMA_VERSION,
    extra_salt: str = "",
) -> str:
    """Deterministic cache key for Vision result reuse.

    Two requests with the same ``document_image_id``,
    ``provider``, ``model``, ``schema_version`` and ``extra_salt``
    MUST hit the same cached row.
    """
    payload = f"{document_image_id}|{provider}|{model}|{schema_version}|{extra_salt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]


def timing_wrapper(fn):
    """Decorator that times ``analyze_image`` calls in ms and stores
    the latency on the returned ``VisionResult``."""

    def _wrapped(*args, **kwargs) -> VisionResult:
        start = time.time() * 1000.0
        result = fn(*args, **kwargs)
        elapsed = int(time.time() * 1000.0 - start)
        if isinstance(result, VisionResult) and result.processing_time_ms == 0:
            result.processing_time_ms = elapsed
        return result

    return _wrapped
