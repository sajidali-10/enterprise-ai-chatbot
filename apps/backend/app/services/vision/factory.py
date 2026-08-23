"""Phase 34B — Vision provider factory.

Resolves ``VISION_PROVIDER`` to a concrete ``VisionProvider``
instance, mirroring the LLM provider factory pattern. The factory
NEVER raises at import time — provider construction errors are
surfaced per-call so the router can fall back to OCR.

Cache note: factory results are memoised in-process for the
lifetime of the worker. The Vision cache lives in PostgreSQL on
``document_images`` so reloads pick up where they left off.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from app.core.config import settings
from app.services.vision.base import (
    VisionProvider,
    VisionProviderError,
    VisionProviderUnavailableError,
)

logger = logging.getLogger(__name__)


AVAILABLE_VISION_PROVIDERS: List[str] = ["mock", "openai-compatible"]

_cached_provider: Dict[str, VisionProvider] = {}
_provider_lock = threading.Lock()


def list_vision_providers() -> List[str]:
    """Return the supported provider names."""
    return list(AVAILABLE_VISION_PROVIDERS)


def is_vision_provider_available(name: str) -> bool:
    """Return True if ``name`` is a recognised provider."""
    return (name or "").strip().lower() in AVAILABLE_VISION_PROVIDERS


def get_vision_provider(name: Optional[str] = None) -> Optional[VisionProvider]:
    """Resolve a ``VisionProvider`` instance.

    Args:
        name: Optional override. Defaults to ``settings.VISION_PROVIDER``.

    Returns:
        A configured ``VisionProvider`` instance, or ``None`` when
        ``VISION_ENABLED`` is false. The router treats ``None`` as
        OCR-only and never calls the provider.

    Raises:
        VisionProviderUnavailableError: when the provider name is
            unknown or the provider cannot be constructed (missing
            key, missing base URL, missing httpx, etc.). The router
            catches this and falls back to OCR.
    """
    if not getattr(settings, "VISION_ENABLED", False):
        return None

    provider_name = (name or settings.VISION_PROVIDER or "mock").strip().lower()

    with _provider_lock:
        cached = _cached_provider.get(provider_name)
        if cached is not None:
            return cached

        if provider_name == "mock":
            from app.services.vision.mock_provider import MockVisionProvider

            instance: VisionProvider = MockVisionProvider()
        elif provider_name == "openai-compatible":
            from app.services.vision.openai_compatible_provider import (
                OpenAICompatibleVisionProvider,
            )

            instance = OpenAICompatibleVisionProvider()
        else:
            raise VisionProviderUnavailableError(
                f"Unknown VISION_PROVIDER '{provider_name}'. "
                f"Available: {AVAILABLE_VISION_PROVIDERS}."
            )

        _cached_provider[provider_name] = instance
        return instance


def reset_vision_provider_cache() -> None:
    """Clear the in-process provider cache.

    Used by tests that swap the provider between runs. Production
    code never needs to call this — provider instances are immutable
    in shape (constructor reads env at process start).
    """
    with _provider_lock:
        _cached_provider.clear()
