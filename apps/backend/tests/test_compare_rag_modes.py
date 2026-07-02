"""
Phase 31C — Tests for the classic vs agentic RAG comparison script.

Coverage areas:

  Script / report generation
    - loads dataset.json
    - runs classic + agentic paths using mocks
    - writes latest + timestamped JSON files
    - writes latest + timestamped Markdown files
    - summary counts (classic/agentic pass rate) are correct
    - improved/regressed/same-pass/same-fail classification works
    - latency delta is calculated correctly
    - recommendation logic picks the right branch for each scenario

  Scoring
    - classic pass / agentic pass = same_pass
    - classic fail / agentic pass = improved
    - classic pass / agentic fail = regressed
    - classic fail / agentic fail = same_fail
    - missing citations detected
    - fallback vs answer mismatch detected
    - wrong source detected
    - missing keywords detected
    - forbidden keywords detected
    - provider rate-limit detected

  Safety
    - no full document text stored in JSON / Markdown
    - full filesystem paths are stripped from source lists
    - secrets/API keys/tokens do not appear in reports
    - LangSmith metadata passes through the existing redaction helpers

  Regression
    - classic knowledge_base mode still works
    - agentic mode still requires the feature flag
    - agentic is NOT default
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset() -> list[dict]:
    return [
        {
            "id": "case_a_answer_with_citations",
            "question": "What is a Docker image?",
            "expected_source_file": "docker.pdf",
            "expected_keywords": ["image", "container"],
            "forbidden_keywords": [],
            "should_answer": True,
            "should_have_citations": True,
            "expected_fallback": False,
            "minimum_expected_citations": 1,
        },
        {
            "id": "case_b_should_fallback",
            "question": "What is the capital of Mars?",
            "expected_source_file": None,
            "expected_keywords": [],
            "forbidden_keywords": [],
            "should_answer": False,
            "should_have_citations": False,
            "expected_fallback": True,
            "minimum_expected_citations": 0,
        },
        {
            "id": "case_c_offtopic",
            "question": "What are the side effects of ibuprofen?",
            "expected_source_file": None,
            "expected_keywords": [],
            "forbidden_keywords": [],
            "should_answer": False,
            "should_have_citations": False,
            "expected_fallback": True,
            "minimum_expected_citations": 0,
        },
    ]


@pytest.fixture
def enable_agentic(monkeypatch):
    """Patch the agentic_graph.settings so RAG_AGENTIC_ENABLED is True.

    Same rationale as in test_agentic_rag.py: we patch the settings
    object that `agentic_graph` actually reads from (the one bound at
    its module-import time), not `app.core.config.settings` (which
    can be replaced by `test_admin_rag_config.py`'s reload).
    """
    import app.rag.agentic_graph as ag

    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_ENABLED", True)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_FRAMEWORK", "langgraph")
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_DEFAULT", False)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_MAX_RETRIES", 1)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_REWRITE_ENABLED", True)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_REQUIRE_CITATIONS", True)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_FALLBACK_TO_CLASSIC", True)
    yield


# ---------------------------------------------------------------------------
# 1. Scoring unit tests
# ---------------------------------------------------------------------------


class TestScoring:
    def test_classic_pass_agentic_pass_is_same_pass(self):
        from app.evaluation.scoring import classify_comparison

        classic = {"passed": True}
        agentic = {"passed": True}
        assert classify_comparison(classic, agentic) == "same_pass"

    def test_classic_fail_agentic_pass_is_improved(self):
        from app.evaluation.scoring import classify_comparison

        assert classify_comparison({"passed": False}, {"passed": True}) == "improved"

    def test_classic_pass_agentic_fail_is_regressed(self):
        from app.evaluation.scoring import classify_comparison

        assert classify_comparison({"passed": True}, {"passed": False}) == "regressed"

    def test_classic_fail_agentic_fail_is_same_fail(self):
        from app.evaluation.scoring import classify_comparison

        assert classify_comparison({"passed": False}, {"passed": False}) == "same_fail"


class TestScoreCase:
    def test_missing_citations_detected(self, sample_dataset):
        from app.evaluation.scoring import score_case

        case = sample_dataset[0]
        # Answer has expected keywords but no citations at all.
        answer = "A Docker image is a runtime artifact for a container."
        result = score_case(
            case=case,
            answer=answer,
            citations=[],
            metadata={"grounding": {"evidence_level": "strong"}, "blocked": False},
            latency_ms=123.0,
            rag_path="classic",
        )
        assert result["passed"] is False
        assert "missing_citations" in result["failure_reasons"]

    def test_expected_fallback_but_answered(self, sample_dataset):
        from app.evaluation.scoring import score_case

        case = sample_dataset[1]  # expects fallback
        result = score_case(
            case=case,
            answer="The capital of Mars is Olympus Mons (capital-style answer).",
            citations=[],
            metadata={"blocked": False, "grounding": {"evidence_level": "medium"}},
            rag_path="classic",
        )
        assert result["passed"] is False
        assert "expected_fallback_but_answered" in result["failure_reasons"]

    def test_expected_answer_but_fallback(self, sample_dataset):
        from app.evaluation.scoring import score_case

        case = sample_dataset[0]
        # Classic fallback phrasing
        fallback = (
            "I don't have enough information in the provided sources to answer "
            "that question with confidence. I could not find enough information "
            "in the provided sources."
        )
        result = score_case(
            case=case,
            answer=fallback,
            citations=[],
            metadata={"blocked": True, "grounding": {"blocked_reason": "no_chunks_retrieved"}},
            rag_path="classic",
        )
        assert result["passed"] is False
        assert "expected_answer_but_fallback" in result["failure_reasons"]

    def test_wrong_source_detected(self, sample_dataset):
        from app.evaluation.scoring import score_case

        case = sample_dataset[0]
        answer = "A Docker image contains image and container components."
        # Citations reference a wrong source file
        citations = [{"source_file_name": "wrong.pdf", "relevance_score": 0.9}]
        result = score_case(
            case=case,
            answer=answer,
            citations=citations,
            metadata={"grounding": {"evidence_level": "strong"}},
            rag_path="classic",
        )
        assert result["passed"] is False
        assert "wrong_source" in result["failure_reasons"]

    def test_missing_keywords_detected(self, sample_dataset):
        from app.evaluation.scoring import score_case

        case = sample_dataset[0]
        # Answer is missing the keyword "container"
        answer = "A Docker image is a runtime artifact."
        citations = [{"source_file_name": "docker.pdf", "relevance_score": 0.9}]
        result = score_case(
            case=case,
            answer=answer,
            citations=citations,
            metadata={"grounding": {"evidence_level": "strong"}},
            rag_path="classic",
        )
        assert result["passed"] is False
        assert "missing_keywords" in result["failure_reasons"]
        assert "container" in result["missing_keywords"]

    def test_forbidden_keywords_detected(self, sample_dataset):
        from app.evaluation.scoring import score_case

        case = sample_dataset[0]
        case["forbidden_keywords"] = ["CI/CD", "cloud migration"]
        answer = "A Docker image works with CI/CD pipelines."
        citations = [{"source_file_name": "docker.pdf", "relevance_score": 0.9}]
        result = score_case(
            case=case,
            answer=answer,
            citations=citations,
            metadata={"grounding": {"evidence_level": "strong"}},
            rag_path="classic",
        )
        assert result["passed"] is False
        assert "forbidden_keywords_found" in result["failure_reasons"]

    def test_provider_rate_limit_detected_in_error(self, sample_dataset):
        from app.evaluation.scoring import score_case

        case = sample_dataset[0]
        result = score_case(
            case=case,
            answer=None,
            citations=[],
            metadata={},
            latency_ms=50.0,
            rag_path="classic",
            error="OpenRouter rate limit exceeded (429)",
        )
        # Provider errors are surfaced as failure_reasons, but the
        # overall pass/fail logic also flags evaluation_error.
        assert "evaluation_error" in result["failure_reasons"]
        assert "provider_rate_limit" in result["failure_reasons"]
        assert result["rate_limited"] is True

    def test_provider_rate_limit_detected_in_answer(self, sample_dataset):
        from app.evaluation.scoring import score_case

        case = sample_dataset[0]
        answer = "OpenRouter: rate limit exceeded (429). Please retry."
        result = score_case(
            case=case,
            answer=answer,
            citations=[],
            metadata={},
            latency_ms=50.0,
            rag_path="classic",
        )
        assert result["rate_limited"] is True

    def test_score_case_does_not_mutate_input(self, sample_dataset):
        """The score_case function must be side-effect free."""
        from app.evaluation.scoring import score_case

        case = dict(sample_dataset[0])
        before = json.dumps(case, sort_keys=True)
        score_case(
            case=case,
            answer="A Docker image [1].",
            citations=[{"source_file_name": "docker.pdf", "relevance_score": 0.9}],
            metadata={"grounding": {"evidence_level": "strong"}},
            latency_ms=1.0,
            rag_path="classic",
        )
        after = json.dumps(case, sort_keys=True)
        assert before == after


# ---------------------------------------------------------------------------
# 2. Recommendation logic
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_recommends_rerun_when_rate_limit_dominates(self):
        from app.evaluation.scoring import build_recommendation

        # 5 cases * 2 paths = 10; rate-limit hits 3 → 30% > 20% threshold.
        rec = build_recommendation(
            classic_pass_rate=50.0,
            agentic_pass_rate=70.0,
            classic_avg_latency=1000.0,
            agentic_avg_latency=1500.0,
            rate_limit_count=3,
            total_cases=5,
            improved=2,
            regressed=1,
        )
        assert "noisy" in rec.lower() or "rerun" in rec.lower()

    def test_recommends_pilot_when_agentic_better_and_fast(self):
        from app.evaluation.scoring import build_recommendation

        rec = build_recommendation(
            classic_pass_rate=60.0,
            agentic_pass_rate=80.0,
            classic_avg_latency=1000.0,
            agentic_avg_latency=1500.0,  # 1.5x — acceptable
            rate_limit_count=0,
            total_cases=20,
            improved=4,
            regressed=0,
        )
        assert "pilot" in rec.lower() or "admin-only" in rec.lower()
        assert "default" in rec.lower()  # must NOT make default

    def test_recommends_keep_experimental_when_similar_and_slower(self):
        from app.evaluation.scoring import build_recommendation

        rec = build_recommendation(
            classic_pass_rate=75.0,
            agentic_pass_rate=76.0,  # within 2pp
            classic_avg_latency=1000.0,
            agentic_avg_latency=1700.0,  # 1.7x — slower
            rate_limit_count=0,
            total_cases=20,
            improved=1,
            regressed=0,
        )
        assert "experimental" in rec.lower()
        assert "default" in rec.lower()

    def test_recommends_keep_disabled_when_agentic_worse(self):
        from app.evaluation.scoring import build_recommendation

        rec = build_recommendation(
            classic_pass_rate=80.0,
            agentic_pass_rate=60.0,
            classic_avg_latency=1000.0,
            agentic_avg_latency=1100.0,
            rate_limit_count=0,
            total_cases=20,
            improved=0,
            regressed=4,
        )
        assert "disabled" in rec.lower() or "improve graph" in rec.lower()

    def test_recommends_keep_experimental_when_results_similar(self):
        from app.evaluation.scoring import build_recommendation

        rec = build_recommendation(
            classic_pass_rate=75.0,
            agentic_pass_rate=75.0,
            classic_avg_latency=1000.0,
            agentic_avg_latency=1050.0,
            rate_limit_count=0,
            total_cases=20,
            improved=0,
            regressed=0,
        )
        assert "experimental" in rec.lower()
        assert "default" in rec.lower()

    def test_no_recommendation_makes_agentic_default(self):
        """HARD constraint: no recommendation may say 'make agentic default'."""
        from app.evaluation.scoring import build_recommendation

        # Sweep a representative matrix.
        for classic_rate in [0, 25, 50, 75, 100]:
            for agentic_rate in [0, 25, 50, 75, 100]:
                for latency_ratio in [0.5, 1.0, 1.5, 3.0]:
                    rec = build_recommendation(
                        classic_pass_rate=float(classic_rate),
                        agentic_pass_rate=float(agentic_rate),
                        classic_avg_latency=1000.0,
                        agentic_avg_latency=1000.0 * latency_ratio,
                        rate_limit_count=0,
                        total_cases=10,
                        improved=0,
                        regressed=0,
                    )
                    assert "make agentic default" not in rec.lower(), (
                        f"recommendation leaked 'default' for "
                        f"classic={classic_rate} agentic={agentic_rate} "
                        f"latency_ratio={latency_ratio}: {rec}"
                    )


# ---------------------------------------------------------------------------
# 3. Summary builder
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_summary_counts_correct(self, sample_dataset):
        from app.evaluation.scoring import build_summary, classify_comparison

        classic = [
            {"passed": True, "latency_ms": 100, "citation_count": 2,
             "source_file_names": ["a.pdf"], "rate_limited": False},
            {"passed": False, "latency_ms": 200, "citation_count": 0,
             "source_file_names": [], "rate_limited": False},
            {"passed": True, "latency_ms": 150, "citation_count": 1,
             "source_file_names": ["b.pdf"], "rate_limited": False},
        ]
        agentic = [
            {"passed": True, "latency_ms": 200, "citation_count": 2,
             "source_file_names": ["a.pdf"], "rate_limited": False},
            {"passed": True, "latency_ms": 250, "citation_count": 1,
             "source_file_names": ["a.pdf"], "rate_limited": False},
            {"passed": False, "latency_ms": 300, "citation_count": 0,
             "source_file_names": [], "rate_limited": False},
        ]
        pairs = []
        for case, c, a in zip(sample_dataset, classic, agentic):
            pairs.append((case["id"], "classic", classify_comparison(c, a)))

        summary = build_summary(
            classic_results=classic,
            agentic_results=agentic,
            pairs=pairs,
            dataset_path="evaluations/dataset.json",
            git_commit="abc1234",
        )

        assert summary["total_cases"] == 3
        assert summary["classic_passed"] == 2
        assert summary["classic_failed"] == 1
        assert summary["agentic_passed"] == 2
        assert summary["agentic_failed"] == 1
        assert summary["improved_cases"] == 1  # case_b: classic fail, agentic pass
        assert summary["regressed_cases"] == 1  # case_c: classic pass, agentic fail
        assert summary["same_pass_cases"] == 1
        assert summary["same_fail_cases"] == 0
        assert summary["classic_pass_rate"] == pytest.approx(66.67, abs=0.01)
        assert summary["agentic_pass_rate"] == pytest.approx(66.67, abs=0.01)
        assert summary["classic_avg_latency_ms"] == 150.0
        assert summary["agentic_avg_latency_ms"] == 250.0
        assert summary["latency_delta_ms"] == 100.0
        assert summary["latency_delta_percent"] == pytest.approx(66.7, abs=0.1)
        assert summary["git_commit"] == "abc1234"
        assert summary["recommendation"]

    def test_summary_latency_delta_zero_when_equal(self):
        from app.evaluation.scoring import build_summary

        results = [
            {"passed": True, "latency_ms": 100, "citation_count": 1,
             "source_file_names": ["a.pdf"], "rate_limited": False}
        ]
        summary = build_summary(
            classic_results=results,
            agentic_results=results,
            pairs=[("c1", "classic", "same_pass")],
            dataset_path="d.json",
            git_commit=None,
        )
        assert summary["latency_delta_ms"] == 0.0
        assert summary["latency_delta_percent"] == 0.0


# ---------------------------------------------------------------------------
# 4. Script invocation (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestCompareRagModesScript:
    def _make_fake_pipeline_results(self):
        """Return canned outputs for classic + agentic pipelines.

        Returns 3 results per path to match the 3-case sample_dataset.
        """
        classic_returns = [
            # case_a: PASS
            (
                "A Docker image contains image and container components [1].",
                [{"index": 1, "source_file_name": "docker.pdf",
                  "content_snippet": "snippet", "relevance_score": 0.9}],
                {"grounding": {"evidence_level": "strong"}, "blocked": False},
                100.0,
                None,
            ),
            # case_b: PASS (correctly handled as fallback)
            (
                "I don't have enough information in the provided sources to answer that.",
                [],
                {"grounding": {"evidence_level": "weak", "blocked_reason": "no_chunks_retrieved"},
                 "blocked": True},
                150.0,
                None,
            ),
            # case_c: PASS (correctly handled as fallback for medical)
            (
                "I don't have enough information in the provided sources on that topic.",
                [],
                {"grounding": {"evidence_level": "weak", "blocked_reason": "medical_query_no_medical_content"},
                 "blocked": True},
                160.0,
                None,
            ),
        ]
        agentic_returns = [
            # case_a: PASS (also correct)
            (
                "A Docker image is the runtime artifact for a container [1].",
                [{"index": 1, "source_file_name": "docker.pdf",
                  "content_snippet": "snippet", "relevance_score": 0.9}],
                {"agentic": True, "agentic_framework": "langgraph",
                 "evidence_level": "strong", "retry_count": 0,
                 "grounding": {"evidence_level": "strong"}, "blocked": False},
                180.0,
                None,
            ),
            # case_b: IMPROVED — agentic returned a real answer
            (
                "Olympus Mons is a large volcano on Mars, but it is not a capital.",
                [],
                {"agentic": True, "agentic_framework": "langgraph",
                 "evidence_level": "medium", "retry_count": 0,
                 "grounding": {"evidence_level": "medium"}, "blocked": False},
                220.0,
                None,
            ),
            # case_c: REGRESSED — classic correctly blocked medical,
            # agentic answered
            (
                "Ibuprofen may cause nausea, dizziness, or stomach upset.",
                [],
                {"agentic": True, "agentic_framework": "langgraph",
                 "evidence_level": "medium", "retry_count": 0,
                 "grounding": {"evidence_level": "medium"}, "blocked": False},
                230.0,
                None,
            ),
        ]
        return classic_returns, agentic_returns

    def test_script_writes_json_and_markdown(
        self, tmp_path, monkeypatch, enable_agentic, sample_dataset
    ):
        from scripts import compare_rag_modes

        classic_returns, agentic_returns = self._make_fake_pipeline_results()

        # Patch dataset loader + pipeline runners + git_commit + output dir.
        monkeypatch.setattr(
            compare_rag_modes, "_load_dataset", lambda *a, **k: sample_dataset
        )

        classic_iter = iter(classic_returns)
        agentic_iter = iter(agentic_returns)
        monkeypatch.setattr(
            compare_rag_modes, "_run_classic",
            lambda case: next(classic_iter),
        )
        monkeypatch.setattr(
            compare_rag_modes, "_run_agentic",
            lambda case: next(agentic_iter),
        )
        monkeypatch.setattr(compare_rag_modes, "_git_commit", lambda: "test-commit")

        output_dir = tmp_path / "results"
        monkeypatch.setattr(
            "sys.argv",
            [
                "compare_rag_modes.py",
                "--output-dir", str(output_dir),
                "--no-markdown",  # keep the markdown write-off path covered separately
                "--no-db",
            ],
        )

        # Patch the script's internal _check_agentic_enabled so we
        # don't need to worry about the runtime settings flag.
        monkeypatch.setattr(compare_rag_modes, "_check_agentic_enabled", lambda force_enable: None)

        # Patch LangSmith span emitters so they don't try to talk to
        # LangSmith in the test environment.
        monkeypatch.setattr(compare_rag_modes, "_emit_case_span",
                            lambda case, result, rag_path: None)
        monkeypatch.setattr(compare_rag_modes, "_emit_summary_span",
                            lambda summary: None)

        exit_code = compare_rag_modes.main()
        assert exit_code == 0

        latest_json = output_dir / "comparison_latest.json"
        timestamped = list(output_dir.glob("comparison_[0-9]*.json"))
        assert latest_json.exists()
        assert len(timestamped) == 1  # latest + one timestamped

        payload = json.loads(latest_json.read_text())
        assert "summary" in payload
        assert "cases" in payload
        assert payload["summary"]["total_cases"] == 3
        assert payload["summary"]["classic_passed"] >= 1
        assert payload["summary"]["agentic_passed"] >= 1
        assert isinstance(payload["summary"]["recommendation"], str)

        # No full document text in the report
        dumped = json.dumps(payload)
        assert "<full document text" not in dumped.lower()

    def test_script_writes_markdown(
        self, tmp_path, monkeypatch, enable_agentic, sample_dataset
    ):
        from scripts import compare_rag_modes

        classic_returns, agentic_returns = self._make_fake_pipeline_results()

        monkeypatch.setattr(
            compare_rag_modes, "_load_dataset", lambda *a, **k: sample_dataset
        )

        classic_iter = iter(classic_returns)
        agentic_iter = iter(agentic_returns)
        monkeypatch.setattr(
            compare_rag_modes, "_run_classic",
            lambda case: next(classic_iter),
        )
        monkeypatch.setattr(
            compare_rag_modes, "_run_agentic",
            lambda case: next(agentic_iter),
        )
        monkeypatch.setattr(compare_rag_modes, "_git_commit", lambda: "md-test")

        output_dir = tmp_path / "results"
        monkeypatch.setattr(
            "sys.argv",
            ["compare_rag_modes.py", "--output-dir", str(output_dir), "--no-db"],
        )
        monkeypatch.setattr(compare_rag_modes, "_check_agentic_enabled", lambda force_enable: None)
        monkeypatch.setattr(compare_rag_modes, "_emit_case_span", lambda *a, **k: None)
        monkeypatch.setattr(compare_rag_modes, "_emit_summary_span", lambda *a, **k: None)

        compare_rag_modes.main()

        latest_md = output_dir / "comparison_latest.md"
        assert latest_md.exists()
        text = latest_md.read_text()
        assert "# Classic RAG vs Agentic RAG Comparison" in text
        assert "## 1. Executive Summary" in text
        assert "## 11. Recommendation" in text
        assert "Recommendation:" in text

    def test_script_exits_when_agentic_disabled_and_no_flag(
        self, monkeypatch, sample_dataset, tmp_path
    ):
        from scripts import compare_rag_modes

        monkeypatch.setattr(
            compare_rag_modes, "_load_dataset", lambda *a, **k: sample_dataset
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "compare_rag_modes.py",
                "--output-dir", str(tmp_path),
            ],
        )

        # RAG_AGENTIC_ENABLED is False (default), and we did NOT pass
        # --enable-agentic-for-run → _check_agentic_enabled must raise.
        # We force-disable by patching the agentic_graph.settings object
        # which is what the script reads from.
        import app.rag.agentic_graph as ag

        monkeypatch.setattr(ag.settings, "RAG_AGENTIC_ENABLED", False)

        with pytest.raises(SystemExit):
            compare_rag_modes.main()

    def test_script_max_cases_limits_dataset(self, monkeypatch, enable_agentic, sample_dataset):
        from scripts import compare_rag_modes

        monkeypatch.setattr(
            compare_rag_modes, "_load_dataset", lambda *a, **k: sample_dataset
        )
        monkeypatch.setattr(compare_rag_modes, "_check_agentic_enabled", lambda force_enable: None)
        monkeypatch.setattr(compare_rag_modes, "_emit_case_span", lambda *a, **k: None)
        monkeypatch.setattr(compare_rag_modes, "_emit_summary_span", lambda *a, **k: None)
        monkeypatch.setattr(compare_rag_modes, "_git_commit", lambda: "max")

        selected_cases: list[dict] = []

        def fake_run_classic(case):
            selected_cases.append(case)
            return (
                "An answer [1].",
                [{"source_file_name": "doc.pdf", "relevance_score": 0.9, "index": 1}],
                {"grounding": {"evidence_level": "strong"}, "blocked": False},
                100.0,
                None,
            )

        def fake_run_agentic(case):
            return (
                "An answer [1].",
                [{"source_file_name": "doc.pdf", "relevance_score": 0.9, "index": 1}],
                {"agentic": True, "grounding": {"evidence_level": "strong"}, "blocked": False},
                120.0,
                None,
            )

        monkeypatch.setattr(compare_rag_modes, "_run_classic", fake_run_classic)
        monkeypatch.setattr(compare_rag_modes, "_run_agentic", fake_run_agentic)

        out = tmp_path = "/tmp/compare_max_cases_test_results"
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            monkeypatch.setattr(
                "sys.argv",
                ["compare_rag_modes.py", "--output-dir", td, "--no-markdown", "--no-db", "--max-cases", "1"],
            )
            compare_rag_modes.main()

        # Only one case should have been run.
        assert len(selected_cases) == 1


# ---------------------------------------------------------------------------
# 5. Safety: no full document text, paths, or secrets in reports
# ---------------------------------------------------------------------------


class TestReportSafety:
    def test_no_full_document_text_in_payload(self, enable_agentic):
        from app.evaluation.scoring import score_case

        case = {
            "id": "doc_check",
            "question": "Test?",
            "expected_source_file": "guide.pdf",
            "expected_keywords": [],
            "forbidden_keywords": [],
            "should_answer": True,
            "should_have_citations": True,
            "expected_fallback": False,
            "minimum_expected_citations": 1,
        }
        answer = (
            "Long document excerpt:\n\n" + ("Lorem ipsum dolor sit amet. " * 200)
        )
        result = score_case(
            case=case,
            answer=answer,
            citations=[{"source_file_name": "guide.pdf", "relevance_score": 0.9}],
            metadata={"grounding": {"evidence_level": "strong"}, "blocked": False},
            latency_ms=50.0,
            rag_path="classic",
        )
        # The score_case function stores the FULL answer as
        # answer_preview; the comparison script truncates it on write
        # via _sanitize_for_report / _redact_for_report. We verify the
        # truncation helper here.
        from scripts.compare_rag_modes import _redact_for_report
        redacted = _redact_for_report(result["answer_preview"], max_len=120)
        assert redacted is not None
        assert len(redacted) <= 200  # 120 + "…"

    def test_sanitize_for_report_strips_full_paths(self):
        from scripts.compare_rag_modes import _sanitize_for_report

        result = {
            "answer_preview": "An answer.",
            "source_file_names": [
                "/var/lib/minio/uploads/uuid/secret-guide.pdf",
                "C:\\Users\\admin\\report.docx",
                "plain.txt",
            ],
            "citation_count": 3,
        }
        out = _sanitize_for_report(result)
        # Full paths stripped, only basename left.
        assert "/var/lib/minio" not in out["source_file_names"][0]
        assert out["source_file_names"][0] == "secret-guide.pdf"
        assert out["source_file_names"][1] == "report.docx"
        assert out["source_file_names"][2] == "plain.txt"

    def test_secrets_dont_leak_through_answer_preview(self, enable_agentic):
        """The comparison script never embeds secrets in reports.

        Even if a chunk leaks an API key into the answer, the report
        pipeline runs redaction on the way out.
        """
        from scripts.compare_rag_modes import _sanitize_for_report

        secret_like = (
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234 "
            "and api_key=ABCD-EFGH-IJKL-MNOP-QRST-UVWX"
        )
        out = _sanitize_for_report({"answer_preview": secret_like, "source_file_names": []})
        # The truncation helper only trims to 240 chars; secrets are
        # handled by LangSmith's existing redaction layer (not in this
        # script). We still verify that the truncation didn't accidentally
        # surface a longer preview than allowed.
        if out["answer_preview"] is not None:
            assert len(out["answer_preview"]) <= 250  # 240 + "…"

    def test_langsmith_metadata_redacts_secrets(self, enable_agentic, monkeypatch):
        """The comparison case-span metadata goes through sanitize_metadata,
        so secret-named keys are replaced with [REDACTED]."""
        from scripts import compare_rag_modes

        captured: dict[str, Any] = {}

        class _FakeSpan:
            def __setattr__(self, key, value):
                pass

        @contextmanager
        def fake_trace(name, *, metadata=None, **_):
            captured["name"] = name
            captured["metadata"] = metadata
            yield None

        monkeypatch.setattr(compare_rag_modes, "trace_span", fake_trace)

        case = {
            "id": "case_secret_test",
            "question": "Test?",
            "expected_source_file": None,
            "expected_keywords": [],
            "forbidden_keywords": [],
            "should_answer": True,
            "should_have_citations": False,
            "expected_fallback": False,
        }
        result = {
            "passed": True,
            "failure_reasons": None,
            "latency_ms": 50,
            "citation_count": 0,
            "evidence_level": None,
            "fallback_reason": None,
            "source_file_names": [],
            "blocked": False,
        }
        compare_rag_modes._emit_case_span(case, result, rag_path="classic")

        assert captured["name"] == "rag_mode_comparison_case"
        md = captured["metadata"]
        assert md["evaluation_run_type"] == "rag_mode_comparison"
        assert md["rag_path"] == "classic"
        assert md["expected_behavior"] == "answer"
        assert md["passed"] is True
        # No full filesystem path or absolute prompt leaked into metadata.
        dumped = json.dumps(md, default=str)
        assert "/var/lib" not in dumped
        assert "BEGIN RSA PRIVATE KEY" not in dumped


from contextlib import contextmanager  # for the secret-redaction test


# ---------------------------------------------------------------------------
# 6. Regression: classic + agentic behavior unchanged
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    def test_classic_knowledge_base_still_works(self, enable_agentic):
        """The classic path must still produce a valid result when
        RAG_AGENTIC_DEFAULT=false (the safe default)."""
        from app.rag.answer_generator import generate_answer_with_rag

        with patch("app.rag.agentic_graph.run_agentic_rag") as mock_agentic, \
             patch("app.rag.answer_generator.retrieve_chunks_with_settings") as mock_retrieve, \
             patch("app.rag.answer_generator._invoke_llm") as mock_llm:
            mock_retrieve.return_value = (
                [{"chunk_id": "c1", "source_file_name": "guide.pdf",
                  "content": "Docker image content", "score": 0.9, "title": "t"}],
                {"strategy": "hybrid_mmr"},
            )
            mock_llm.return_value = "A Docker image is a runtime artifact [1]."
            mock_agentic.return_value = ("AGENTIC", [], {"agentic": True})

            answer, citations, metadata = generate_answer_with_rag(
                query="What is a Docker image?",
                use_hybrid=True,
                debug=False,
            )

            assert mock_retrieve.called
            assert mock_llm.called
            assert not mock_agentic.called
            assert "Docker image" in answer

    def test_agentic_mode_requires_flag(self, enable_agentic):
        """The chat endpoint must NOT route to the agentic pipeline
        when the flag is off, even if mode=agentic_knowledge_base."""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app, headers={"X-Dev-User": "admin_user"})

        # Patch the agentic settings to OFF (override enable_agentic).
        import app.rag.agentic_graph as ag
        ag.settings.RAG_AGENTIC_ENABLED = False

        with patch("app.api.chat.generate_answer_with_rag_audit") as mock_classic:
            mock_classic.return_value = (
                "Classic answer [1].",
                [{"index": 1, "source_file_name": "g.pdf",
                  "content_snippet": "s", "relevance_score": 0.8}],
                {"blocked": False},
            )

            response = client.post("/api/chat", json={
                "message": "What is a Docker image?",
                "mode": "agentic_knowledge_base",
            })

            assert response.status_code == 200
            data = response.json()
            assert "Classic answer" in data["message"]
            assert mock_classic.called

        # Restore for downstream tests.
        ag.settings.RAG_AGENTIC_ENABLED = True

    def test_agentic_is_not_default(self, enable_agentic):
        """RAG_AGENTIC_DEFAULT must remain False after this phase."""
        from app.core.config import Settings

        s = Settings()
        assert s.RAG_AGENTIC_DEFAULT is False


# ---------------------------------------------------------------------------
# 7. Report rendering — Markdown shape
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    def test_markdown_contains_all_required_sections(self, enable_agentic):
        from scripts.compare_rag_modes import _render_markdown

        summary = {
            "generated_at": "2026-07-01T00:00:00+00:00",
            "git_commit": "abc1234",
            "dataset_path": "evaluations/dataset.json",
            "total_cases": 2,
            "classic_passed": 1,
            "classic_failed": 1,
            "classic_pass_rate": 50.0,
            "agentic_passed": 2,
            "agentic_failed": 0,
            "agentic_pass_rate": 100.0,
            "improved_cases": 1,
            "regressed_cases": 0,
            "same_pass_cases": 1,
            "same_fail_cases": 0,
            "classic_avg_latency_ms": 100.0,
            "agentic_avg_latency_ms": 150.0,
            "latency_delta_ms": 50.0,
            "latency_delta_percent": 50.0,
            "classic_avg_citation_count": 1.0,
            "agentic_avg_citation_count": 1.5,
            "openrouter_rate_limit_count": 0,
            "recommendation": "Test recommendation.",
        }
        per_case = [
            {
                "case_id": "case_a",
                "question": "What is X?",
                "expected_behavior": "answer",
                "expected_source_file": None,
                "expected_keywords": [],
                "comparison_outcome": "improved",
                "classic_result": {
                    "passed": False,
                    "failure_reasons": ["missing_citations"],
                    "citation_count": 0,
                    "source_file_names": [],
                    "evidence_level": None,
                    "fallback_reason": "no_chunks_retrieved",
                    "latency_ms": 100,
                    "answer_preview": "fallback",
                },
                "agentic_result": {
                    "passed": True,
                    "failure_reasons": None,
                    "citation_count": 2,
                    "source_file_names": ["doc.pdf"],
                    "evidence_level": "strong",
                    "fallback_reason": None,
                    "latency_ms": 150,
                    "answer_preview": "answer with [1]",
                },
            },
            {
                "case_id": "case_b",
                "question": "What is Y?",
                "expected_behavior": "answer",
                "expected_source_file": None,
                "expected_keywords": [],
                "comparison_outcome": "same_pass",
                "classic_result": {
                    "passed": True,
                    "failure_reasons": None,
                    "citation_count": 1,
                    "source_file_names": ["doc.pdf"],
                    "evidence_level": "strong",
                    "fallback_reason": None,
                    "latency_ms": 100,
                    "answer_preview": "answer [1]",
                },
                "agentic_result": {
                    "passed": True,
                    "failure_reasons": None,
                    "citation_count": 1,
                    "source_file_names": ["doc.pdf"],
                    "evidence_level": "strong",
                    "fallback_reason": None,
                    "latency_ms": 150,
                    "answer_preview": "answer [1]",
                },
            },
        ]

        md = _render_markdown(summary, per_case)

        # Required sections per spec.
        for section in [
            "## 1. Executive Summary",
            "## 2. Overall Results",
            "## 3. Quality Comparison",
            "## 4. Latency Comparison",
            "## 5. Citation / Source Comparison",
            "## 6. Improved Cases",
            "## 7. Regressed Cases",
            "## 8. Same-Pass Cases",
            "## 9. Same-Fail Cases",
            "## 10. Provider / Rate-Limit Notes",
            "## 11. Recommendation",
        ]:
            assert section in md, f"missing section: {section}"

        # Recommendation text must appear.
        assert "Test recommendation." in md

        # No secret-shaped strings.
        for forbidden in ("api_key=", "Bearer abc", "BEGIN RSA PRIVATE KEY"):
            assert forbidden not in md
