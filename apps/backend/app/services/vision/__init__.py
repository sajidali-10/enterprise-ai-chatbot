"""Phase 34B — Vision provider package.

Exposes the public Vision provider interface and a safe
``get_vision_provider()`` factory.

Architectural rules (Phase 34B):
    * OCR remains the default and cheapest path. Vision is ADDITIONAL
      visual understanding capability on top of OCR.
    * The provider abstraction is intentionally narrow:
      ``analyze_image(bytes, mime_type, prompt, max_tokens) -> VisionResult``.
    * The factory returns a ``VisionProvider`` instance configured by
      ``VISION_PROVIDER``. Mock is the safe default — deterministic
      and offline. openai-compatible speaks the OpenAI Chat
      Completions Vision protocol and is selected at deploy time.
    * The provider layer NEVER touches secrets, MinIO, RBAC, or
      Qdrant. It receives only the image bytes the caller is
      authorized to send and returns a structured ``VisionResult``.
"""

from app.services.vision.base import (
    VisionProvider,
    VisionResult,
    VisionProviderError,
    VisionProviderUnavailableError,
    VisionProviderTimeoutError,
    VISION_SCHEMA_VERSION,
)
from app.services.vision.factory import (
    get_vision_provider,
    list_vision_providers,
    is_vision_provider_available,
)

__all__ = [
    "VisionProvider",
    "VisionResult",
    "VisionProviderError",
    "VisionProviderUnavailableError",
    "VisionProviderTimeoutError",
    "VISION_SCHEMA_VERSION",
    "get_vision_provider",
    "list_vision_providers",
    "is_vision_provider_available",
]
