"""Phase 34D — Unit tests for the deterministic task classifier.

Covers:
* identifier-lookup exclusion (OCR suffices)
* product-meaning exclusion (KB authority required)
* UI_STATE_ANALYSIS detection
* CHART_ANALYSIS detection
* DIAGRAM_ANALYSIS detection
* TABLE_VISUAL_ANALYSIS detection
* IMAGE_COMPARISON detection (text-only — image_count is orthogonal)
* GENERAL_VISUAL fallback
* first-match-wins priority
* should_run_advanced_vision property
"""

from __future__ import annotations

import pytest

from app.services.advanced_vision.router import (
    AdvancedVisualTask,
    TaskClassification,
    classify_visual_task,
)


# ---------------------------------------------------------------------------
# Identifier-lookup exclusion (OCR suffices; advanced Vision is NOT needed)
# ---------------------------------------------------------------------------


def test_classifier_01_identifier_lookup_error_code_does_not_advance():
    """What error code is shown? → OCR suffices → general_visual."""
    decision = classify_visual_task("What error code is shown in the image?")
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.is_identifier_lookup is True
    assert decision.should_run_advanced_vision is False


def test_classifier_02_identifier_lookup_text_question_does_not_advance():
    """What text is shown? → OCR suffices."""
    decision = classify_visual_task(
        "What text is shown in the screenshot you just uploaded?"
    )
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.is_identifier_lookup is True
    assert decision.should_run_advanced_vision is False


def test_classifier_03_identifier_lookup_receiver_name_does_not_advance():
    """Identifier-only lookups never trigger advanced Vision."""
    decision = classify_visual_task("What is the receiver name?")
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.is_identifier_lookup is True


# ---------------------------------------------------------------------------
# Product-meaning exclusion (KB owns these)
# ---------------------------------------------------------------------------


def test_classifier_04_product_meaning_how_do_i_fix_does_not_advance():
    """How do I fix error 902? → KB authority → general_visual."""
    decision = classify_visual_task("How do I fix error 902?")
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.is_product_meaning is True
    assert decision.should_run_advanced_vision is False


def test_classifier_05_product_meaning_what_does_mean_does_not_advance():
    """What does error 902 mean? → KB authority."""
    decision = classify_visual_task("What does error 902 mean?")
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.is_product_meaning is True


def test_classifier_06_product_meaning_root_cause_does_not_advance():
    """Root cause questions → KB authority."""
    decision = classify_visual_task("What is the root cause of the failed delivery?")
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.is_product_meaning is True


def test_classifier_07_product_meaning_troubleshooting_does_not_advance():
    """Troubleshooting → KB authority."""
    decision = classify_visual_task(
        "Help me troubleshoot the failed login in the screenshot."
    )
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.is_product_meaning is True


# ---------------------------------------------------------------------------
# UI_STATE_ANALYSIS
# ---------------------------------------------------------------------------


def test_classifier_10_ui_state_what_looks_wrong_advances():
    decision = classify_visual_task("What visually looks wrong on this screen?")
    assert decision.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value
    assert decision.requires_image is True
    assert decision.should_run_advanced_vision is True


def test_classifier_11_ui_state_which_button_disabled_advances():
    decision = classify_visual_task("Which button is disabled on this screen?")
    assert decision.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value


def test_classifier_12_ui_state_red_indicator_advances():
    decision = classify_visual_task("Which server has the red indicator?")
    assert decision.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value


def test_classifier_13_ui_state_server_health_advances():
    decision = classify_visual_task("Which server appears failed?")
    assert decision.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value


def test_classifier_14_ui_state_warning_banner_advances():
    decision = classify_visual_task("Is there a warning banner visible?")
    assert decision.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value


# ---------------------------------------------------------------------------
# CHART_ANALYSIS
# ---------------------------------------------------------------------------


def test_classifier_20_chart_what_trend_advances():
    decision = classify_visual_task("What trend does this chart show?")
    assert decision.task_type == AdvancedVisualTask.CHART_ANALYSIS.value


def test_classifier_21_chart_highest_spike_advances():
    decision = classify_visual_task("What was the highest spike in this graph?")
    assert decision.task_type == AdvancedVisualTask.CHART_ANALYSIS.value


def test_classifier_22_chart_spike_drop_advances():
    decision = classify_visual_task("Was there a spike or drop in traffic?")
    assert decision.task_type == AdvancedVisualTask.CHART_ANALYSIS.value


def test_classifier_23_chart_dashboard_keyword_advances():
    decision = classify_visual_task("What does the dashboard indicate?")
    assert decision.task_type == AdvancedVisualTask.CHART_ANALYSIS.value


# ---------------------------------------------------------------------------
# DIAGRAM_ANALYSIS
# ---------------------------------------------------------------------------


def test_classifier_30_diagram_what_flow_advances():
    decision = classify_visual_task("What flow does this diagram show?")
    assert decision.task_type == AdvancedVisualTask.DIAGRAM_ANALYSIS.value


def test_classifier_31_diagram_architecture_advances():
    decision = classify_visual_task("What does the architecture diagram show?")
    assert decision.task_type == AdvancedVisualTask.DIAGRAM_ANALYSIS.value


def test_classifier_32_diagram_what_connects_to_advances():
    decision = classify_visual_task("What is connected to the database in this diagram?")
    assert decision.task_type == AdvancedVisualTask.DIAGRAM_ANALYSIS.value


def test_classifier_33_diagram_topology_advances():
    decision = classify_visual_task("Describe the network topology shown in the diagram.")
    assert decision.task_type == AdvancedVisualTask.DIAGRAM_ANALYSIS.value


# ---------------------------------------------------------------------------
# TABLE_VISUAL_ANALYSIS
# ---------------------------------------------------------------------------


def test_classifier_40_table_highlighted_row_advances():
    decision = classify_visual_task("Which row is highlighted in the table?")
    assert decision.task_type == AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value


def test_classifier_41_table_abnormal_row_advances():
    decision = classify_visual_task("Which row looks abnormal in the spreadsheet?")
    assert decision.task_type == AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value


def test_classifier_42_table_headers_advances():
    decision = classify_visual_task("What are the column headers in this table?")
    assert decision.task_type == AdvancedVisualTask.TABLE_VISUAL_ANALYSIS.value


# ---------------------------------------------------------------------------
# IMAGE_COMPARISON (text-only signal — image_count handled by orchestrator)
# ---------------------------------------------------------------------------


def test_classifier_50_comparison_what_changed_advances():
    decision = classify_visual_task("What changed between these two screenshots?")
    assert decision.task_type == AdvancedVisualTask.IMAGE_COMPARISON.value
    assert decision.requires_comparison is True


def test_classifier_51_comparison_compare_keyword_advances():
    decision = classify_visual_task("Compare these two screenshots.")
    assert decision.task_type == AdvancedVisualTask.IMAGE_COMPARISON.value


def test_classifier_52_comparison_before_after_advances():
    decision = classify_visual_task("Show me the before and after.")
    assert decision.task_type == AdvancedVisualTask.IMAGE_COMPARISON.value


# ---------------------------------------------------------------------------
# Priority / fallback
# ---------------------------------------------------------------------------


def test_classifier_60_fallback_to_general_visual():
    """Unrelated question → general_visual fallback."""
    decision = classify_visual_task("Tell me about the history of the company.")
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.should_run_advanced_vision is False


def test_classifier_61_empty_question():
    decision = classify_visual_task("")
    assert decision.task_type == AdvancedVisualTask.GENERAL_VISUAL.value
    assert decision.should_run_advanced_vision is False


def test_classifier_62_image_count_one_default():
    decision = classify_visual_task("Which server appears failed?")
    # image_count defaults to 1; orchestrator handles the comparison case
    # separately by inspecting image_context.comparison.
    assert decision.task_type == AdvancedVisualTask.UI_STATE_ANALYSIS.value
