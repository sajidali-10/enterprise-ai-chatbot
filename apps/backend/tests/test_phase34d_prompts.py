"""Phase 34D — Unit tests for the task-specific prompt templates.

Covers:
* every AdvancedVisualTask value has a system + user prompt
* every prompt contains the safety prefix forbidding product
  invention / root cause / hidden state
* max_tokens is task-aware (chart/diagram/comparison need more room)
* comparison prompt carries OCR text for both images in order
* general_visual fallback works for unknown task_type values
"""

from __future__ import annotations

from app.services.advanced_vision.base import (
    ADVANCED_VISION_SCHEMA_VERSION,
    AdvancedVisualTask,
)
from app.services.advanced_vision.prompts import (
    build_advanced_vision_prompt,
    build_comparison_prompt,
    get_system_prompt,
)


SAFETY_KEYWORDS = [
    "visually observable",
    "do not infer",
    "do not invent",
    "uncertainty language",
    "ground truth",
]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


def test_prompt_01_general_visual_has_safety_prefix():
    sys = get_system_prompt(AdvancedVisualTask.GENERAL_VISUAL.value)
    for kw in SAFETY_KEYWORDS:
        assert kw in sys.lower(), f"GENERAL_VISUAL missing safety keyword: {kw}"


def test_prompt_02_ui_state_has_safety_prefix():
    sys = get_system_prompt(AdvancedVisualTask.UI_STATE_ANALYSIS.value)
    for kw in SAFETY_KEYWORDS:
        assert kw in sys.lower(), f"UI_STATE_ANALYSIS missing safety keyword: {kw}"
    # UI_STATE prompts MUST mention UI state elements explicitly.
    assert "ui state" in sys.lower()


def test_prompt_03_chart_has_safety_prefix():
    sys = get_system_prompt(AdvancedVisualTask.CHART_ANALYSIS.value)
    for kw in SAFETY_KEYWORDS:
        assert kw in sys.lower(), f"CHART_ANALYSIS missing safety keyword: {kw}"
    assert "chart" in sys.lower()
    assert "do not fabricate" in sys.lower()


def test_prompt_04_diagram_has_safety_prefix():
    sys = get_system_prompt(AdvancedVisualTask.DIAGRAM_ANALYSIS.value)
    for kw in SAFETY_KEYWORDS:
        assert kw in sys.lower(), f"DIAGRAM_ANALYSIS missing safety keyword: {kw}"
    assert "diagram" in sys.lower()
    assert "protocol" in sys.lower()  # explicitly forbids inventing protocols


def test_prompt_05_table_has_safety_prefix():
    sys = get_system_prompt(AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value)
    for kw in SAFETY_KEYWORDS:
        assert kw in sys.lower(), f"TABLE_VISUAL_ANALYSIS missing safety keyword: {kw}"


def test_prompt_06_comparison_has_safety_prefix():
    sys = get_system_prompt(AdvancedVisualTask.IMAGE_COMPARISON.value)
    for kw in SAFETY_KEYWORDS:
        assert kw in sys.lower(), f"IMAGE_COMPARISON missing safety keyword: {kw}"
    # Comparison prompt MUST distinguish A (before) and B (after).
    assert "before" in sys.lower()
    assert "after" in sys.lower()


def test_prompt_07_unknown_task_falls_back_to_general_visual():
    sys = get_system_prompt("not_a_real_task")
    general_sys = get_system_prompt(AdvancedVisualTask.GENERAL_VISUAL.value)
    # Unknown task type MUST fall back to the general_visual prompt
    # so the provider always receives a coherent instruction.
    assert sys == general_sys


# ---------------------------------------------------------------------------
# Built prompt structure
# ---------------------------------------------------------------------------


def test_prompt_10_build_general_visual_carries_question_and_ocr():
    built = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.GENERAL_VISUAL.value,
        question="Describe the image.",
        ocr_text="Server B Failed\nApply button disabled",
    )
    assert built.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert "Describe the image." in built.user
    assert "Server B Failed" in built.user
    assert built.max_tokens >= 64


def test_prompt_11_build_ui_state_includes_kind_enum():
    built = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.UI_STATE_ANALYSIS.value,
        question="Which button is disabled?",
        ocr_text="Apply",
    )
    assert "disabled-control" in built.user
    assert "warning-banner" in built.user
    assert built.max_tokens >= 600


def test_prompt_12_build_chart_includes_kind_enum():
    built = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.CHART_ANALYSIS.value,
        question="What trend do you see?",
        ocr_text="14:00 spike",
    )
    assert "spike" in built.user
    assert "upward-trend" in built.user


def test_prompt_13_build_diagram_forbids_inventing_protocols():
    built = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.DIAGRAM_ANALYSIS.value,
        question="What does this architecture diagram show?",
        ocr_text="Internet -> Nginx -> Backend -> PostgreSQL",
    )
    assert "protocols" in built.user or "protocol" in built.user


def test_prompt_14_build_table_includes_kind_enum():
    built = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value,
        question="Which row is highlighted?",
        ocr_text="row 3",
    )
    assert "highlighted-row" in built.user


def test_prompt_15_build_truncates_oversized_ocr():
    long_ocr = "x" * 10_000
    built = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.GENERAL_VISUAL.value,
        question="Q",
        ocr_text=long_ocr,
    )
    # OCR is clipped inside the prompt — never raw 10k chars.
    assert len(built.user) < 5_000


def test_prompt_16_unknown_task_type_falls_back_to_general_visual():
    built = build_advanced_vision_prompt(
        task_type="definitely_not_real",
        question="Describe the image.",
        ocr_text="",
    )
    assert built.task_type == AdvancedVisualTask.GENERAL_VISUAL.value


# ---------------------------------------------------------------------------
# Comparison prompt
# ---------------------------------------------------------------------------


def test_prompt_20_comparison_prompt_has_both_ocr_bodies():
    built = build_comparison_prompt(
        question="What changed between these screenshots?",
        ocr_text_a="Server B Healthy",
        ocr_text_b="Server B Failed",
    )
    assert built.task_type == AdvancedVisualTask.IMAGE_COMPARISON.value
    assert "Server B Healthy" in built.user
    assert "Server B Failed" in built.user
    # Before / after language must appear in the comparison prompt.
    assert "before" in built.user.lower()
    assert "after" in built.user.lower()


def test_prompt_21_comparison_max_tokens_higher_than_single_image():
    single = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.UI_STATE_ANALYSIS.value,
        question="Q",
        ocr_text="",
    )
    compare = build_comparison_prompt(
        question="Q",
        ocr_text_a="",
        ocr_text_b="",
    )
    assert compare.max_tokens > single.max_tokens


def test_prompt_22_comparison_prompt_truncates_long_ocr():
    long_a = "a" * 5_000
    long_b = "b" * 5_000
    built = build_comparison_prompt(
        question="Q",
        ocr_text_a=long_a,
        ocr_text_b=long_b,
    )
    assert len(built.user) < 5_000


def test_prompt_23_prompt_version_propagates():
    built = build_advanced_vision_prompt(
        task_type=AdvancedVisualTask.GENERAL_VISUAL.value,
        question="Q",
        prompt_version=7,
    )
    assert built.prompt_version == 7


def test_prompt_24_advanced_vision_schema_version_default():
    """Phase 34D carries its own schema version constant."""
    assert ADVANCED_VISION_SCHEMA_VERSION >= 1
