"""Phase 34D — Task-specific advanced Vision prompts.

Each task has its own system + user prompt template. The templates
share a non-negotiable safety prefix that forbids the model from:

* inventing product-specific behaviour, troubleshooting steps, or
  root-cause analysis;
* inferring hidden state (e.g. "the service crashed");
* quoting content that is not visible in the image;
* mapping identifiers to authoritative meanings (KB owns that).

The JSON contract matches :class:`VisualReasoningResult`. The
provider is asked to produce strict JSON; the runtime parser is
lenient (reuses the Phase 34B ``_parse_json_lenient`` pattern) and
collapses malformed responses to a safe ``general_visual`` result
with a non-empty summary so the LLM still receives an explicit
"advanced vision unavailable" signal.

The module is pure: no DB, no network, no provider call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.services.advanced_vision.base import AdvancedVisualTask


# ---------------------------------------------------------------------------
# Safety prefix (shared across every task)
# ---------------------------------------------------------------------------

_SAFETY_PREFIX = (
    "You are analyzing an enterprise technical image. You are NOT "
    "an authoritative source on the product, its documentation, or "
    "its troubleshooting. You are a vision observer. Follow these "
    "rules strictly:\n"
    "1. Report ONLY what is visually observable in the image. Do NOT "
    "invent product-specific behaviour, troubleshooting steps, root "
    "cause, error resolutions, configuration advice, or support steps.\n"
    "2. Do NOT infer hidden state. If a value is not visible in the "
    "image, say so. Do not guess why a control is disabled, what "
    "happened next, or what caused the observed state.\n"
    "3. Preserve visible error codes, labels, status indicators, "
    "relationships, graph trends, selected controls, and warnings "
    "exactly as they appear. Use the OCR text below as ground truth "
    "for any text/numbers/labels — do not re-spell them.\n"
    "4. Use uncertainty language ('appears', 'approximately', "
    "'visually', 'uncertain') when precision is limited. Avoid "
    "fabricated exact numeric values.\n"
    "5. Do NOT map identifiers to meanings (e.g. 'error 902 means X'). "
    "Authority over meanings and fixes lives outside this image.\n"
    "6. Respond with STRICT JSON. No prose outside the JSON object. "
    "Do not wrap in markdown code fences.\n"
)


# ---------------------------------------------------------------------------
# Task-specific system prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    AdvancedVisualTask.GENERAL_VISUAL.value: (
        _SAFETY_PREFIX
        + "\nTask: GENERAL_VISUAL. Summarise what the image shows "
        "(layout, components, visual style). List the most notable "
        "observations. Do not enumerate every visible item."
    ),
    AdvancedVisualTask.UI_STATE_ANALYSIS.value: (
        _SAFETY_PREFIX
        + "\nTask: UI_STATE_ANALYSIS. Identify observable UI state "
        "elements such as: warning/error banners, colour-coded "
        "status indicators (red/green/yellow/amber), selected "
        "navigation/tabs, disabled controls, enabled controls, "
        "visible status labels, highlighted rows, modal/dialog "
        "presence, success/error states, form-validation indicators, "
        "and visible server/service health states. Each finding must "
        "describe what is observable and the exact visible label or "
        "text associated with it."
    ),
    AdvancedVisualTask.CHART_ANALYSIS.value: (
        _SAFETY_PREFIX
        + "\nTask: CHART_ANALYSIS. Identify observable chart "
        "characteristics: trend (upward/downward/plateau), spike, "
        "dip, outlier, peak, trough, sudden change, approximate "
        "time/location of change, legend/series distinction (when "
        "visible), relative comparison between series. Do NOT "
        "fabricate exact numeric values. Use words like "
        "'approximately', 'around', 'appears'. Do not invent "
        "causes; observation only."
    ),
    AdvancedVisualTask.DIAGRAM_ANALYSIS.value: (
        _SAFETY_PREFIX
        + "\nTask: DIAGRAM_ANALYSIS. Extract observable structural "
        "elements: nodes (with visible labels), edges/connections "
        "(with visible direction arrows), grouping boundaries, "
        "annotations. Describe the visible flow/relationships as "
        "shown. Do NOT invent protocols, ports, security controls, "
        "or behaviour that is not visible in the diagram or "
        "supported by the OCR text. Do not label a node as "
        "'database' unless the word 'database' is visible in the "
        "diagram or OCR text."
    ),
    AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value: (
        _SAFETY_PREFIX
        + "\nTask: TABLE_VISUAL_ANALYSIS. Identify observable table "
        "characteristics: visible column headers, row grouping, "
        "visually abnormal/highlighted rows, status columns, "
        "relative magnitude (largest/smallest when visible), "
        "selected row. Use OCR for exact cell values; vision "
        "supplements layout and state."
    ),
    AdvancedVisualTask.IMAGE_COMPARISON.value: (
        _SAFETY_PREFIX
        + "\nTask: IMAGE_COMPARISON. Image 1 (A) is the BEFORE "
        "state; image 2 (B) is the AFTER state. Identify observable "
        "differences between them: status changes, new/removed "
        "banners, enabled/disabled control changes, changed values, "
        "changed chart shape, added/removed components, visual state "
        "changes. For each change, state what changed, in which "
        "image, and from what to what. Do NOT invent root causes or "
        "explain WHY something changed — only what changed."
    ),
}


def get_system_prompt(task_type: str) -> str:
    """Return the safety-bounded system prompt for a given task.

    Falls back to the ``general_visual`` prompt for unknown task
    types so the provider always receives a coherent instruction.
    """
    return _SYSTEM_PROMPTS.get(
        task_type or AdvancedVisualTask.GENERAL_VISUAL.value,
        _SYSTEM_PROMPTS[AdvancedVisualTask.GENERAL_VISUAL.value],
    )


# ---------------------------------------------------------------------------
# Task-specific user prompts (filled with image + question context)
# ---------------------------------------------------------------------------

_USER_PROMPT_GENERAL = (
    "Analyse the image and produce STRICT JSON matching the schema "
    "below. Use the OCR text as ground truth for any text/numbers; "
    "do not re-spell or paraphrase OCR content.\n\n"
    "Question: {question}\n\n"
    "OCR text (already extracted from the image):\n{ocr_text}\n\n"
    "Schema (return ONLY this JSON, no other text):\n"
    "{{\n"
    '  "task_type": "general_visual",\n'
    '  "summary": string (<= 280 chars),\n'
    '  "observations": [string],\n'
    '  "anomalies": [string],\n'
    '  "relationships": [{{"from": string, "to": string, "label": string}}],\n'
    '  "entities": [string],\n'
    '  "confidence": int (0..100)\n'
    "}}\n"
)

_USER_PROMPT_UI_STATE = (
    "Identify observable UI state elements in the image. Produce "
    "STRICT JSON. Each finding must describe what is observable "
    "and the exact visible label/text.\n\n"
    "Question: {question}\n\n"
    "OCR text:\n{ocr_text}\n\n"
    "Schema:\n"
    "{{\n"
    '  "task_type": "ui_state_analysis",\n'
    '  "summary": string (<= 280 chars),\n'
    '  "observations": [string],\n'
    '  "ui_states": [\n'
    '    {{\n'
    '      "kind": string (one of: warning-banner, error-banner, '
    'status-indicator, selected-nav, disabled-control, enabled-control, '
    'visible-status-label, highlighted-row, modal-dialog, success-state, '
    'error-state, validation-indicator, server-health),\n'
    '      "label": string (the visible label / text, or ""), \n'
    '      "value": string (the visible value if any, or ""), \n'
    '      "evidence": string (the visible cue that justifies the finding)\n'
    "    }}\n"
    "  ],\n"
    '  "anomalies": [string],\n'
    '  "entities": [string],\n'
    '  "confidence": int (0..100)\n'
    "}}\n"
)

_USER_PROMPT_CHART = (
    "Identify observable chart characteristics. Do NOT fabricate "
    "exact numeric values. Use uncertainty language where precision "
    "is limited.\n\n"
    "Question: {question}\n\n"
    "OCR text:\n{ocr_text}\n\n"
    "Schema:\n"
    "{{\n"
    '  "task_type": "chart_analysis",\n'
    '  "summary": string (<= 280 chars),\n'
    '  "observations": [string],\n'
    '  "chart_findings": [\n'
    '    {{\n'
    '      "kind": string (one of: upward-trend, downward-trend, '
    'plateau, spike, dip, outlier, peak, trough, sudden-change, '
    'series-distinction, relative-comparison),\n'
    '      "location": string (approximate time/axis location, '
    'e.g. "around 14:00", "right end", or ""),\n'
    '      "evidence": string (the visible cue that justifies the finding)\n'
    "    }}\n"
    "  ],\n"
    '  "anomalies": [string],\n'
    '  "entities": [string],\n'
    '  "confidence": int (0..100)\n'
    "}}\n"
)

_USER_PROMPT_DIAGRAM = (
    "Extract observable structural elements from the diagram. Use "
    "ONLY labels visible in the image or in the OCR text. Do not "
    "invent protocols, ports, security controls, or behaviour.\n\n"
    "Question: {question}\n\n"
    "OCR text:\n{ocr_text}\n\n"
    "Schema:\n"
    "{{\n"
    '  "task_type": "diagram_analysis",\n'
    '  "summary": string (<= 280 chars),\n'
    '  "observations": [string],\n'
    '  "relationships": [\n'
    '    {{\n'
    '      "from": string (label of source node),\n'
    '      "to": string (label of target node),\n'
    '      "label": string (visible arrow/edge label, or ""),\n'
    '      "direction": string (visible direction, e.g. "down", "right", or "")\n'
    "    }}\n"
    "  ],\n"
    '  "diagram_findings": [\n'
    '    {{\n'
    '      "kind": string (one of: nodes, edges, grouping, boundary, annotation, flow),\n'
    '      "subject": string,\n'
    '      "evidence": string\n'
    "    }}\n"
    "  ],\n"
    '  "entities": [string],\n'
    '  "confidence": int (0..100)\n'
    "}}\n"
)

_USER_PROMPT_TABLE = (
    "Identify observable table characteristics. Use OCR for exact "
    "cell values; vision supplements layout and state.\n\n"
    "Question: {question}\n\n"
    "OCR text:\n{ocr_text}\n\n"
    "Schema:\n"
    "{{\n"
    '  "task_type": "table_visual_analysis",\n'
    '  "summary": string (<= 280 chars),\n'
    '  "observations": [string],\n'
    '  "table_findings": [\n'
    '    {{\n'
    '      "kind": string (one of: header, row, highlighted-row, '
    'selected-row, abnormal-row, status-column, relative-magnitude),\n'
    '      "subject": string (column/row identifier, e.g. "Server B" '
    'or "column: Status"),\n'
    '      "value": string (visible cell value, or ""),\n'
    '      "evidence": string (the visible cue that justifies the finding)\n'
    "    }}\n"
    "  ],\n"
    '  "anomalies": [string],\n'
    '  "entities": [string],\n'
    '  "confidence": int (0..100)\n'
    "}}\n"
)

_USER_PROMPT_COMPARISON = (
    "Image 1 (A) is the BEFORE state. Image 2 (B) is the AFTER "
    "state. Identify observable differences. Do NOT explain WHY "
    "anything changed.\n\n"
    "Question: {question}\n\n"
    "OCR text for image A (before):\n{ocr_text_a}\n\n"
    "OCR text for image B (after):\n{ocr_text_b}\n\n"
    "Schema:\n"
    "{{\n"
    '  "task_type": "image_comparison",\n'
    '  "summary": string (<= 280 chars),\n'
    '  "observations": [string],\n'
    '  "comparison_changes": [\n'
    '    {{\n'
    '      "subject": string (the entity that changed, e.g. "Server B"),\n'
    '      "kind": string (one of: status-change, banner-added, '
    'banner-removed, control-enabled, control-disabled, value-changed, '
    'chart-shape-changed, component-added, component-removed, '
    'visual-state-changed),\n'
    '      "before": string (visible state in image A, or ""),\n'
    '      "after": string (visible state in image B, or ""),\n'
    '      "evidence": string (the visible cue that justifies the change)\n'
    "    }}\n"
    "  ],\n"
    '  "anomalies": [string],\n'
    '  "entities": [string],\n'
    '  "confidence": int (0..100)\n'
    "}}\n"
)


_USER_PROMPTS = {
    AdvancedVisualTask.GENERAL_VISUAL.value: _USER_PROMPT_GENERAL,
    AdvancedVisualTask.UI_STATE_ANALYSIS.value: _USER_PROMPT_UI_STATE,
    AdvancedVisualTask.CHART_ANALYSIS.value: _USER_PROMPT_CHART,
    AdvancedVisualTask.DIAGRAM_ANALYSIS.value: _USER_PROMPT_DIAGRAM,
    AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value: _USER_PROMPT_TABLE,
}


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltPrompt:
    """Result of building a task-aware Vision prompt.

    Both ``system`` and ``user`` are returned so the provider can
    choose how to combine them. ``max_tokens`` is a task-aware hint
    — chart/diagram tasks need more room than UI tasks because they
    carry richer structured findings.
    """

    system: str
    user: str
    task_type: str
    max_tokens: int
    prompt_version: int


_TASK_MAX_TOKENS = {
    AdvancedVisualTask.GENERAL_VISUAL.value: 600,
    AdvancedVisualTask.UI_STATE_ANALYSIS.value: 800,
    AdvancedVisualTask.CHART_ANALYSIS.value: 800,
    AdvancedVisualTask.DIAGRAM_ANALYSIS.value: 1000,
    AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value: 800,
    AdvancedVisualTask.IMAGE_COMPARISON.value: 1200,
}


def build_advanced_vision_prompt(
    *,
    task_type: str,
    question: str,
    ocr_text: str = "",
    comparison_ocr_text: str = "",
    prompt_version: int = 1,
    ocr_max_chars: int = 2000,
) -> BuiltPrompt:
    """Build a complete system + user prompt for a single-image task.

    Args:
        task_type: ``AdvancedVisualTask`` value. Unknown values fall
            back to ``general_visual``.
        question: The user question (cleaned).
        ocr_text: Pre-extracted OCR body. Truncated to
            ``ocr_max_chars`` before being embedded in the prompt.
        comparison_ocr_text: Required only for the comparison flow.
            Ignored for single-image tasks.
        prompt_version: Bump to refresh task prompts without
            invalidating the Redis cache. Embedded in the result
            for observability.
        ocr_max_chars: Per-OCR length cap to keep the prompt bounded.

    Returns:
        ``BuiltPrompt``. The provider receives ``system`` and
        ``user`` and may combine them in whatever format the
        protocol requires (the openai-compatible provider prepends
        ``system`` to the messages list).
    """
    raw_task = task_type or AdvancedVisualTask.GENERAL_VISUAL.value
    valid_tasks = {t.value for t in AdvancedVisualTask}
    task = raw_task if raw_task in valid_tasks else AdvancedVisualTask.GENERAL_VISUAL.value
    template = _USER_PROMPTS.get(task, _USER_PROMPT_GENERAL)
    ocr_clip = _clip(ocr_text or "", ocr_max_chars)
    user_prompt = template.format(question=(question or "").strip(), ocr_text=ocr_clip)
    return BuiltPrompt(
        system=get_system_prompt(task),
        user=user_prompt,
        task_type=task,
        max_tokens=_TASK_MAX_TOKENS.get(task, 600),
        prompt_version=int(prompt_version or 1),
    )


def build_comparison_prompt(
    *,
    question: str,
    ocr_text_a: str,
    ocr_text_b: str,
    prompt_version: int = 1,
    ocr_max_chars: int = 1500,
) -> BuiltPrompt:
    """Build a prompt for the IMAGE_COMPARISON flow.

    Both OCR bodies are truncated to ``ocr_max_chars``. Order is
    preserved: A is the *before* state, B is the *after* state.
    """
    task = AdvancedVisualTask.IMAGE_COMPARISON.value
    ocr_a = _clip(ocr_text_a or "", ocr_max_chars)
    ocr_b = _clip(ocr_text_b or "", ocr_max_chars)
    user_prompt = _USER_PROMPT_COMPARISON.format(
        question=(question or "").strip(),
        ocr_text_a=ocr_a,
        ocr_text_b=ocr_b,
    )
    return BuiltPrompt(
        system=get_system_prompt(task),
        user=user_prompt,
        task_type=task,
        max_tokens=_TASK_MAX_TOKENS[task],
        prompt_version=int(prompt_version or 1),
    )


def _clip(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)].rsplit(" ", 1)[0] + "…"


__all__ = [
    "BuiltPrompt",
    "build_advanced_vision_prompt",
    "build_comparison_prompt",
    "get_system_prompt",
]
