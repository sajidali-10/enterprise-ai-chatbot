"""Phase 34D — Advanced Visual Understanding package.

Adds a task-aware, structured reasoning layer on top of the Phase 34B
Vision pipeline. The package reuses the Phase 34B
``VisionProvider`` protocol (extended with optional ``task`` and
``images`` parameters) and the existing Redis client.

Public surface (re-exported for callers):

* :class:`AdvancedVisualTask` — the task taxonomy.
* :class:`VisualReasoningResult` — the structured result schema.
* :func:`classify_visual_task` — deterministic task classifier.
* :func:`run_advanced_visual_reasoning` — single-image orchestrator.
* :func:`run_advanced_visual_comparison` — two-image comparison
  orchestrator.

When ``ADVANCED_VISION_ENABLED=false`` every entry point short-
circuits to a no-op and the Phase 34A / 34B / 34C / 34C.1
behaviour is unchanged.

Importing this package is always safe; it does not touch the
network or any DB at import time.
"""

from app.services.advanced_vision.base import (
    ADVANCED_VISION_SCHEMA_VERSION,
    AdvancedVisualTask,
    VisualReasoningResult,
    build_advanced_vision_cache_key,
    image_content_hash,
)
from app.services.advanced_vision.comparison import (
    ComparisonOutcome,
    run_advanced_visual_comparison,
)
from app.services.advanced_vision.orchestrator import (
    AdvancedOrchestratorOutcome,
    run_advanced_visual_reasoning,
)
from app.services.advanced_vision.router import (
    TaskClassification,
    classify_visual_task,
)


__all__ = [
    "ADVANCED_VISION_SCHEMA_VERSION",
    "AdvancedVisualTask",
    "AdvancedOrchestratorOutcome",
    "ComparisonOutcome",
    "TaskClassification",
    "VisualReasoningResult",
    "build_advanced_vision_cache_key",
    "classify_visual_task",
    "image_content_hash",
    "run_advanced_visual_comparison",
    "run_advanced_visual_reasoning",
]
