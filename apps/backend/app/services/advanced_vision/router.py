"""Phase 34D — Advanced Visual Task Classifier.

Decides — deterministically and without any LLM call — which
``AdvancedVisualTask`` a user question implies.

The classifier is intentionally narrow:

* Identical input → identical output.
* Conservative: when a question carries no clear task signal, the
  classifier returns ``general_visual`` and the orchestrator treats
  the question as "no advanced reasoning needed" (the Phase 34B
  generic ``VisionResult`` already covers ``general_visual``).
* Identifier-only and product-meaning questions always return
  ``general_visual`` so the orchestrator skips advanced Vision and
  defers to OCR / KB respectively.
* IMAGE_COMPARISON is decided purely by the question text. The
  orchestrator separately validates that the request actually has
  two authorized images; a comparison question with one image (or
  zero images) is skipped, not silently downgraded.

The classifier is pure — no DB, no network, no provider call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


# Re-export for callers that already import from this module.
from app.services.advanced_vision.base import AdvancedVisualTask  # noqa: F401


# ---------------------------------------------------------------------------
# Identifier-lookup exclusion patterns
# ---------------------------------------------------------------------------
#
# Questions whose intent is to look up an exact visible value
# (error code, status code, name, number). These are answered by
# OCR alone. Advanced Vision would add latency without adding
# signal — they are excluded before any task-specific pattern runs.
_IDENTIFIER_LOOKUP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bwhat\s+(?:error\s+code|error\s+number|code|number|status\s+code|status\s+number)\s+is\s+(?:shown|displayed|visible)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:text|message|words?)\s+(?:is\s+)?(?:shown|displayed|visible|contained)\b[^.]{0,80}?\b(?:screenshot|image|page|attachment|photo)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:text|message|words?)\s+(?:does|do|is)\s+(?:this|the)\s+(?:screenshot|image|page)\s+(?:contain|show|display|say)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+the\s+(?:receiver|sender|recipient|customer|user|name|address|phone|email|amount|value)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+number\s+is\s+displayed\b", re.IGNORECASE),
    re.compile(r"\bread\s+(?:out|back)\s+the\s+(?:text|error|message)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Product-meaning exclusion patterns
# ---------------------------------------------------------------------------
#
# Questions whose answer requires authoritative product knowledge
# (meanings, causes, fixes). KB owns these; advanced Vision
# observations cannot help — they would either be ignored by the
# grounding check or risk being interpreted as product authority.
_PRODUCT_MEANING_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bhow\s+do\s+i\s+(?:fix|resolve|configure|setup|set\s+up|install|use|troubleshoot|clear|reset|recover|restore|debug|address|handle|work\s+around)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+to\s+(?:fix|resolve|configure|setup|set\s+up|install|use|troubleshoot|clear|reset|recover|restore|debug|address|handle|work\s+around)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+.+\s+mean\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+(?:the\s+)?(?:meaning|definition|cause)\s+of\b", re.IGNORECASE),
    re.compile(r"\broot\s+cause\b", re.IGNORECASE),
    re.compile(r"\bwhy\s+(?:is|does|did|am|are|do|don't|has|have|should|would|will|won't|can't|cannot)\b", re.IGNORECASE),
    re.compile(r"\btroubleshoot(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\bhow\s+can\s+i\b", re.IGNORECASE),
    re.compile(r"\bhow\s+should\s+i\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+should\s+i\s+do\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Task-specific patterns (in priority order — first match wins)
# ---------------------------------------------------------------------------

_UI_STATE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bwhat\s+(?:visually\s+)?(?:is|does)\s+(?:wrong|happening|going\s+on)\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s|\s+is)\s+wrong\s+with\s+(?:this|that|the)\s+(?:screen|page|view|window|dashboard|screenshot|image|ui)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+looks?\s+wrong\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:visually\s+)?looks?\s+(?:wrong|off|broken|unhealthy|failing|degraded)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:do\s+you\s+see|are\s+you\s+seeing)\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(?:button|icon|server|row|field|item|control|menu|tab|column|node|box|line)\s+(?:is\s+)?(?:disabled|enabled|selected|highlighted|active|inactive|focused|clicked|pressed)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+(?:selected|highlighted|disabled|active|inactive|focused|clicked|enabled|red|green|yellow|amber)\b", re.IGNORECASE),
    re.compile(r"\b(?:red|green|yellow|amber|orange|blue)\s+(?:indicator|status|dot|light|marker|icon)\b", re.IGNORECASE),
    re.compile(r"\bwarning\s+(?:indicator|state|message|icon|banner|sign)\b", re.IGNORECASE),
    re.compile(r"\b(?:error|warning)\s+banner\b", re.IGNORECASE),
    re.compile(r"\bmodal\s+(?:open|visible|present)\b", re.IGNORECASE),
    re.compile(r"\b(?:which|what)\s+server\s+(?:appears?|looks?)\s+(?:failed|failing|down|unhealthy|broken|offline|degraded)\b", re.IGNORECASE),
    re.compile(r"\bserver\s+(?:status|health)\b", re.IGNORECASE),
    re.compile(r"\bform\s+validation\b", re.IGNORECASE),
    re.compile(r"\bselected\s+(?:navigation|nav|tab|row)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+the\s+(?:current\s+)?status\s+(?:of|shown|displayed)\b", re.IGNORECASE),
]


_CHART_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:chart|graph|dashboard)\s+(?:shows?|displays?|depicts?|indicates?|reveals?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:does\s+(?:this|that|the)\s+)?(?:chart|graph)\s+(?:show|display|depict|reveal|indicate)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:trend|pattern)\s+(?:does|do|is|are)\s+(?:this|the|these)\s+(?:chart|graph|dashboard)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:is|are)\s+the\s+(?:trend|pattern|anomal(?:y|ies))\b", re.IGNORECASE),
    re.compile(r"\b(?:trend|spike|dip|drop|plateau|outlier|peak|valley|sudden\s+change|anomal(?:y|ies))\b", re.IGNORECASE),
    re.compile(r"\b(?:highest|lowest|largest|biggest|smallest)\s+(?:spike|peak|drop|dip|value|point)\b", re.IGNORECASE),
    re.compile(r"\b(?:rising|falling|increasing|decreasing|climbing|dropping)\s+(?:trend|line|curve)\b", re.IGNORECASE),
    re.compile(r"\b(?:traffic|usage|load|throughput|latency|requests?)\s+(?:spike|drop|surge|decline|trend|pattern)\b", re.IGNORECASE),
    re.compile(r"\b(?:line|bar|pie|area|scatter)\s+chart\b", re.IGNORECASE),
    re.compile(r"\b(?:time\s+series|timeseries)\b", re.IGNORECASE),
    re.compile(r"\b(?:y-axis|x-axis|axis\s+label|legend|series)\b", re.IGNORECASE),
]


_DIAGRAM_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bwhat\s+does\s+(?:this|that|the)\s+(?:diagram|architecture|topology|flow|workflow)\s+(?:show|display|depict|reveal)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:flow|architecture|topology|workflow)\s+(?:does|is|shows?)\s+(?:this|that|the)?\s*(?:diagram|architecture|topology|flow|workflow)?\s*(?:show|display|depict|reveal|indicate)\b", re.IGNORECASE),
    re.compile(r"\b(?:architecture|topology|workflow|flow\s+diagram|system\s+flow)\s+(?:diagram|shown|displayed)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:is\s+)?connected\s+to\s+(?:the\s+)?[A-Za-z][\w\- ]{0,40}\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:connects?|flows?)\s+(?:to|from|into|between|through)\b", re.IGNORECASE),
    re.compile(r"\b(?:nodes?|edges?|arrows?|connections?|boundaries?|groupings?)\s+(?:in|of|on)\s+(?:this|the|that)\s+(?:diagram|architecture|topology|flow)\b", re.IGNORECASE),
    re.compile(r"\b(?:sequence|uml|entity-relationship|e-r|er)\s+diagram\b", re.IGNORECASE),
    re.compile(r"\b(?:network\s+)?topology\b", re.IGNORECASE),
    re.compile(r"\b(?:data\s+)?flow\s+(?:diagram|shown)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+(?:connected|linked|attached|downstream|upstream)\s+(?:to|from)\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+does\s+.+\s+(?:connect|flow|go)\s+(?:to|from|into)\b", re.IGNORECASE),
]


_TABLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:which|what)\s+row\s+(?:is\s+)?(?:highlighted|selected|abnormal|flagged|marked)\b", re.IGNORECASE),
    re.compile(r"\b(?:which|what)\s+row\s+(?:looks?|appears?|seems?)\s+(?:abnormal|highlighted|selected|flagged|marked|outlier)\b", re.IGNORECASE),
    re.compile(r"\bhighlighted\s+row\b", re.IGNORECASE),
    re.compile(r"\bselected\s+(?:row|column|cell)\b", re.IGNORECASE),
    re.compile(r"\b(?:abnormal|anomalous|unusual|outlier)\s+row\b", re.IGNORECASE),
    re.compile(r"\b(?:spreadsheet|grid|table)\s+(?:shown|displayed|in\s+the\s+image)\b", re.IGNORECASE),
    re.compile(r"\b(?:column|row)\s+headers?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:does|is)\s+(?:this|the|that)\s+(?:table|grid|spreadsheet)\s+show\b", re.IGNORECASE),
    re.compile(r"\bvisible\s+(?:rows?|columns?)\b", re.IGNORECASE),
]


_COMPARISON_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bcompare\s+(?:these|the|these\s+two|the\s+two)\s+(?:screenshots?|images?|diagrams?|pictures?|photos?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:is\s+)?(?:the\s+)?(?:difference|change[s]?)\s+between\s+(?:these|the|these\s+two|the\s+two|this|that)?\s*(?:screenshots?|images?|diagrams?|pictures?|photos?)\b", re.IGNORECASE),
    re.compile(r"\bbefore\s+(?:and|&)\s+after\b", re.IGNORECASE),
    re.compile(r"\b(?:compare|versus|vs\.?)\b\s+(?:the|two|both)?\s*(?:screenshots?|images?|pictures?)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:did|has|have)\s+.+\s+(?:change[d]?|differ[ed]?)\s+(?:between|from|since)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+changed\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(?:status|control|value|indicator)\s+(?:changed|changed\s+from|switched)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Decision object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskClassification:
    """Result of running ``classify_visual_task`` on a question.

    Attributes are read-only so callers cannot mutate the decision.
    """

    task_type: str = AdvancedVisualTask.GENERAL_VISUAL.value
    is_identifier_lookup: bool = False
    is_product_meaning: bool = False
    requires_image: bool = False
    requires_comparison: bool = False
    matched_patterns: List[str] = field(default_factory=list)

    @property
    def should_run_advanced_vision(self) -> bool:
        """True only when the orchestrator should call advanced Vision.

        Conditions:
        * the question is not an identifier-lookup (OCR suffices)
        * the question is not a product-meaning / troubleshooting
          question (KB authority required)
        * the question carries a task signal stronger than
          ``general_visual`` (Phase 34B already covers general)
        * either requires a comparison flow (caller must have two
          authorized images) or a single-image task
        """
        if self.is_identifier_lookup or self.is_product_meaning:
            return False
        if self.task_type == AdvancedVisualTask.GENERAL_VISUAL.value:
            return False
        return True


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_visual_task(
    question: str,
    *,
    image_count: int = 1,
) -> TaskClassification:
    """Decide which advanced visual task (if any) the question implies.

    Args:
        question: The user question text (already cleaned).
        image_count: Number of images the request supplies. Defaults
            to 1. The classifier checks the question text only;
            ``image_count`` is exposed as a field on the decision so
            the orchestrator can refuse to run a comparison flow
            against an image_count that doesn't match.

    Returns:
        ``TaskClassification``. The function never raises; bad input
        collapses to ``general_visual`` so the chat pipeline is
        always answerable.

    Priority order:
        1. Identifier-lookup exclusion (OCR suffices).
        2. Product-meaning exclusion (KB authority required).
        3. IMAGE_COMPARISON if comparison patterns match.
        4. UI_STATE_ANALYSIS, CHART_ANALYSIS, DIAGRAM_ANALYSIS,
           TABLE_VISUAL_ANALYSIS — first match wins.
        5. GENERAL_VISUAL fallback.
    """
    text = (question or "").strip()

    matched: List[str] = []
    is_identifier = False
    is_product = False

    if text:
        for pat in _IDENTIFIER_LOOKUP_PATTERNS:
            if pat.search(text):
                is_identifier = True
                matched.append(f"identifier:{pat.pattern[:48]}")
                break

        if not is_identifier:
            for pat in _PRODUCT_MEANING_PATTERNS:
                if pat.search(text):
                    is_product = True
                    matched.append(f"product_meaning:{pat.pattern[:48]}")
                    break

        if not is_identifier and not is_product:
            task_type, pattern_label = _first_match(text, _COMPARISON_PATTERNS, "comparison")
            if task_type is not None:
                matched.append(pattern_label or "comparison")
                return TaskClassification(
                    task_type=AdvancedVisualTask.IMAGE_COMPARISON.value,
                    is_identifier_lookup=False,
                    is_product_meaning=False,
                    requires_image=True,
                    requires_comparison=True,
                    matched_patterns=list(matched),
                )

            task_type, pattern_label = _first_match(text, _CHART_PATTERNS, "chart")
            if task_type is not None:
                matched.append(pattern_label or "chart")
                return TaskClassification(
                    task_type=AdvancedVisualTask.CHART_ANALYSIS.value,
                    is_identifier_lookup=False,
                    is_product_meaning=False,
                    requires_image=True,
                    requires_comparison=False,
                    matched_patterns=list(matched),
                )

            task_type, pattern_label = _first_match(text, _DIAGRAM_PATTERNS, "diagram")
            if task_type is not None:
                matched.append(pattern_label or "diagram")
                return TaskClassification(
                    task_type=AdvancedVisualTask.DIAGRAM_ANALYSIS.value,
                    is_identifier_lookup=False,
                    is_product_meaning=False,
                    requires_image=True,
                    requires_comparison=False,
                    matched_patterns=list(matched),
                )

            task_type, pattern_label = _first_match(text, _TABLE_PATTERNS, "table")
            if task_type is not None:
                matched.append(pattern_label or "table")
                return TaskClassification(
                    task_type=AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value,
                    is_identifier_lookup=False,
                    is_product_meaning=False,
                    requires_image=True,
                    requires_comparison=False,
                    matched_patterns=list(matched),
                )

            task_type, pattern_label = _first_match(text, _UI_STATE_PATTERNS, "ui_state")
            if task_type is not None:
                matched.append(pattern_label or "ui_state")
                return TaskClassification(
                    task_type=AdvancedVisualTask.UI_STATE_ANALYSIS.value,
                    is_identifier_lookup=False,
                    is_product_meaning=False,
                    requires_image=True,
                    requires_comparison=False,
                    matched_patterns=list(matched),
                )

    return TaskClassification(
        task_type=AdvancedVisualTask.GENERAL_VISUAL.value,
        is_identifier_lookup=is_identifier,
        is_product_meaning=is_product,
        requires_image=False,
        requires_comparison=False,
        matched_patterns=list(matched),
    )


def _first_match(
    text: str,
    patterns: List[re.Pattern],
    label: str,
) -> tuple[Optional[str], Optional[str]]:
    """Return (task_type, pattern_label) for the first matching pattern.

    Returns ``(None, None)`` when no pattern matches. The pattern
    label is a short, debug-friendly tag (e.g. ``comparison:...``)
    that the orchestrator can surface in observability.
    """
    for pat in patterns:
        if pat.search(text):
            return label, f"{label}:{pat.pattern[:48]}"
    return None, None


__all__ = [
    "AdvancedVisualTask",
    "TaskClassification",
    "classify_visual_task",
]
