"""
Tests for Phase 30 — Evaluation Report History.

GET /api/admin/evaluations/history returns read-only history of
custom evaluation runs (from DB) and RAGAS reports (from RAGAS_REPORT_DIR).

Scope: read-only, admin-only, no secrets, no full paths.
"""

import pytest
import json
from pathlib import Path


class TestEvaluationHistoryEndpoint:
    """Phase 30: GET /api/admin/evaluations/history returns safe history data."""

    def test_requires_admin(self, client):
        """Non-admin requests are forbidden."""
        response = client.get("/api/admin/evaluations/history")
        assert response.status_code in (401, 403)

    def test_admin_access_returns_200_with_empty_lists(self, jwt_admin_client):
        """Admin access returns HTTP 200 even when no history exists yet."""
        response = jwt_admin_client.get("/api/admin/evaluations/history")
        assert response.status_code == 200
        d = response.json()
        assert "custom_eval_runs" in d
        assert "custom_eval_reports" in d
        assert "ragas_reports" in d
        assert "warnings" in d

    def test_custom_eval_runs_from_db(self, jwt_admin_client, db_session):
        """When evaluation runs exist in DB, they appear in history."""
        from app.models.evaluation import EvaluationRun

        run = EvaluationRun(
            total_tests=20,
            passed_tests=18,
            failed_tests=2,
            pass_percentage=90.0,
            average_latency_ms=1200.5,
            average_top_score=0.85,
            status="completed",
        )
        db_session.add(run)
        db_session.commit()

        response = jwt_admin_client.get("/api/admin/evaluations/history")
        assert response.status_code == 200
        d = response.json()
        runs = d["custom_eval_runs"]
        assert len(runs) >= 1
        # Most recent first
        latest = runs[0]
        assert latest["total_tests"] == 20
        assert latest["passed"] == 18
        assert latest["failed"] == 2
        assert latest["pass_rate"] == 90.0
        assert latest["status"] == "completed"

    def test_ragas_reports_from_dir(self, jwt_admin_client, monkeypatch, tmp_path):
        """When RAGAS JSON reports exist, they appear in history."""
        import app.core.config

        original = app.core.config.settings.RAGAS_REPORT_DIR
        app.core.config.settings.RAGAS_REPORT_DIR = str(tmp_path)

        report = {
            "timestamp": "2026-06-22T12:00:00Z",
            "case_count": 20,
            "evaluator_provider": "litellm",
            "evaluator_model": "openrouter-gpt-oss",
            "avg_scores": {
                "faithfulness": {"avg": 0.85, "min": 0.7, "max": 0.95, "count": 20},
                "answer_relevancy": {"avg": 0.78, "min": 0.6, "max": 0.9, "count": 20},
                "context_precision": {"avg": 0.91, "min": 0.8, "max": 1.0, "count": 20},
            },
            "skipped_metrics": ["context_recall", "answer_correctness"],
        }
        report_file = tmp_path / "ragas_report_20260622_120000.json"
        report_file.write_text(json.dumps(report))

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            d = response.json()
            reports = d["ragas_reports"]
            assert len(reports) >= 1
            latest = reports[0]
            assert latest["report_name"] == "ragas_report_20260622_120000.json"
            assert latest["metrics"] is not None
            assert latest["metrics"]["faithfulness"] == 0.85
            assert latest["total_cases"] == 20
            # Only filename exposed, not full path
            assert "/" not in latest["report_name"]
            assert "\\" not in latest["report_name"]
        finally:
            app.core.config.settings.RAGAS_REPORT_DIR = original

    def test_malformed_ragas_json_returns_warning_not_500(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """Malformed RAGAS JSON skips the file gracefully with a warning."""
        import app.core.config

        original = app.core.config.settings.RAGAS_REPORT_DIR
        app.core.config.settings.RAGAS_REPORT_DIR = str(tmp_path)

        # Write one good report and one bad report
        good_report = {
            "timestamp": "2026-06-22T12:00:00Z",
            "case_count": 20,
            "evaluator_provider": "litellm",
            "evaluator_model": "openrouter-gpt-oss",
            "avg_scores": {
                "faithfulness": {"avg": 0.85},
                "answer_relevancy": {"avg": 0.78},
                "context_precision": {"avg": 0.91},
            },
            "skipped_metrics": ["context_recall", "answer_correctness"],
        }
        good_file = tmp_path / "ragas_report_20260622_120000.json"
        good_file.write_text(json.dumps(good_report))

        bad_file = tmp_path / "ragas_report_20260623_120000.json"
        bad_file.write_text("this is not json{{{")

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            d = response.json()
            # Good report is returned
            assert len(d["ragas_reports"]) == 1
            # Warning explains the bad file was skipped
            assert any("skip" in w.lower() or "could not read" in w.lower() for w in d["warnings"])
        finally:
            app.core.config.settings.RAGAS_REPORT_DIR = original

    def test_no_secrets_in_response(self, jwt_admin_client, monkeypatch, tmp_path):
        """History response must not expose any secrets."""
        import app.core.config

        original = app.core.config.settings.RAGAS_REPORT_DIR
        app.core.config.settings.RAGAS_REPORT_DIR = str(tmp_path)

        report = {
            "timestamp": "2026-06-22T12:00:00Z",
            "case_count": 20,
            "evaluator_provider": "litellm",
            "evaluator_model": "openrouter-gpt-oss",
            "avg_scores": {
                "faithfulness": {"avg": 0.85},
                "answer_relevancy": {"avg": 0.78},
                "context_precision": {"avg": 0.91},
            },
            "skipped_metrics": ["context_recall", "answer_correctness"],
            # Simulate a report that accidentally contains secrets
            "api_key_used": "sk-1234567890abcdef",
            "litellm_master_key": "sk-super-secret",
        }
        report_file = tmp_path / "ragas_report_20260622_120000.json"
        report_file.write_text(json.dumps(report))

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            raw = str(response.json()).lower()
            secret_keywords = [
                "secret", "password", "token", "api_key", "credential",
                "key", "master_key", "litellm_master", "sk-12345", "sk-super",
            ]
            found = [kw for kw in secret_keywords if kw in raw]
            assert not found, f"Secret keyword(s) found in history response: {found}"
        finally:
            app.core.config.settings.RAGAS_REPORT_DIR = original

    def test_no_full_filesystem_paths_in_response(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """RAGAS report names are filenames only; no /app/ or C:\\ paths."""
        import app.core.config

        original = app.core.config.settings.RAGAS_REPORT_DIR
        app.core.config.settings.RAGAS_REPORT_DIR = str(tmp_path)

        report = {
            "timestamp": "2026-06-22T12:00:00Z",
            "case_count": 20,
            "evaluator_provider": "litellm",
            "evaluator_model": "openrouter-gpt-oss",
            "avg_scores": {
                "faithfulness": {"avg": 0.85},
                "answer_relevancy": {"avg": 0.78},
                "context_precision": {"avg": 0.91},
            },
            "skipped_metrics": ["context_recall", "answer_correctness"],
        }
        report_file = tmp_path / "ragas_report_20260622_120000.json"
        report_file.write_text(json.dumps(report))

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            raw = str(response.json())
            # Should not contain the tmp_path prefix
            assert str(tmp_path) not in raw, "Full filesystem path leaked into history response"
            # Should not contain /app/evals/ragas either
            assert "/app/evals/ragas" not in raw
            assert "/app/evaluations/results" not in raw
        finally:
            app.core.config.settings.RAGAS_REPORT_DIR = original

    def test_limit_parameter(self, jwt_admin_client, db_session):
        """The limit parameter caps the number of returned runs."""
        from app.models.evaluation import EvaluationRun

        # Create 5 runs
        for i in range(5):
            run = EvaluationRun(
                total_tests=20,
                passed_tests=20 - i,
                failed_tests=i,
                pass_percentage=(20 - i) / 20 * 100,
                average_latency_ms=1000.0 + i * 100,
                average_top_score=0.85,
                status="completed",
            )
            db_session.add(run)
        db_session.commit()

        response = jwt_admin_client.get("/api/admin/evaluations/history?limit=3")
        assert response.status_code == 200
        d = response.json()
        # At most 3 custom eval runs
        assert len(d["custom_eval_runs"]) <= 3

    def test_latest_first_ordering(self, jwt_admin_client, db_session):
        """Custom eval runs are sorted newest first."""
        from app.models.evaluation import EvaluationRun

        for i in range(3):
            run = EvaluationRun(
                total_tests=20,
                passed_tests=15 + i,
                failed_tests=5 - i,
                pass_percentage=(15 + i) / 20 * 100,
                average_latency_ms=1000.0,
                average_top_score=0.85,
                status="completed",
            )
            db_session.add(run)
        db_session.commit()

        response = jwt_admin_client.get("/api/admin/evaluations/history")
        assert response.status_code == 200
        d = response.json()
        runs = d["custom_eval_runs"]
        if len(runs) >= 2:
            # Verify descending order by timestamp
            timestamps = [r["timestamp"] for r in runs]
            assert timestamps == sorted(timestamps, reverse=True)

    def test_custom_eval_reports_from_results_dir(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """Timestamped evaluation JSON files appear as custom_eval_reports."""
        import app.api.evaluation as ev_api

        # Override the module-level constant to point to tmp_path
        original_const = ev_api._EVALUATION_RESULTS_DIR
        ev_api._EVALUATION_RESULTS_DIR = tmp_path

        # Write a timestamped evaluation JSON
        eval_data = {
            "run_id": 42,
            "timestamp": "2026-06-22T12:00:00Z",
            "results": [
                {"passed": True, "latency_ms": 1000, "top_score": 0.9},
                {"passed": True, "latency_ms": 1100, "top_score": 0.85},
                {"passed": False, "latency_ms": 900, "top_score": 0.6, "failure_reasons": ["missing_keywords"]},
            ],
        }
        eval_file = tmp_path / "evaluation_20260622_120000.json"
        eval_file.write_text(json.dumps(eval_data))

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            d = response.json()
            assert "custom_eval_reports" in d
        finally:
            ev_api._EVALUATION_RESULTS_DIR = original_const


    def test_custom_eval_reports_parsing(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """Custom eval report JSON is parsed for summary stats."""
        import app.api.evaluation as ev_api

        # Override the module-level constant to point to tmp_path
        original_const = ev_api._EVALUATION_RESULTS_DIR
        ev_api._EVALUATION_RESULTS_DIR = tmp_path

        # Write a timestamped evaluation JSON
        eval_data = {
            "run_id": 5,
            "timestamp": "2026-06-22T12:00:00Z",
            "results": [
                {"passed": True, "latency_ms": 1000.0, "top_score": 0.9},
                {"passed": True, "latency_ms": 2000.0, "top_score": 0.8},
                {"passed": False, "latency_ms": 1500.0, "top_score": 0.5, "failure_reasons": ["missing_keywords"]},
            ],
        }
        eval_file = tmp_path / "evaluation_20260622_120000.json"
        eval_file.write_text(json.dumps(eval_data))

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            d = response.json()
            assert "custom_eval_reports" in d
            assert len(d["custom_eval_reports"]) >= 1
            rep = d["custom_eval_reports"][0]
            assert rep["report_name"] == "evaluation_20260622_120000.json"
            assert rep["total_tests"] == 3
            assert rep["passed"] == 2
            assert rep["failed"] == 1
            # No full paths in response
            assert "/app/" not in str(d)
        finally:
            ev_api._EVALUATION_RESULTS_DIR = original_const

    def test_malformed_custom_eval_json_does_not_crash(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """Malformed custom eval JSON skips the file gracefully."""
        import app.api.evaluation as ev_api

        # Override the module-level constant to point to tmp_path
        original_const = ev_api._EVALUATION_RESULTS_DIR
        ev_api._EVALUATION_RESULTS_DIR = tmp_path

        # Create one good and one bad file
        good_data = {
            "run_id": 1,
            "timestamp": "2026-06-22T12:00:00Z",
            "results": [{"passed": True, "latency_ms": 500, "top_score": 0.9}],
        }
        good_file = tmp_path / "evaluation_20260622_120000.json"
        good_file.write_text(json.dumps(good_data))

        bad_file = tmp_path / "evaluation_20260623_120000.json"
        bad_file.write_text("{ this is not json ")

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            d = response.json()
            # Good file should be returned despite bad file existing
            names = [r["report_name"] for r in d["custom_eval_reports"]]
            assert "evaluation_20260622_120000.json" in names
            # Warning should mention skipping
            assert any("skip" in w.lower() or "could not read" in w.lower() for w in d["warnings"])
        finally:
            ev_api._EVALUATION_RESULTS_DIR = original_const

    def test_latest_results_json_marked_as_latest_alias(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """latest_results.json is included with report_type='latest_results_json'."""
        import app.api.evaluation as ev_api

        # Override the module-level constant to point to tmp_path
        original_const = ev_api._EVALUATION_RESULTS_DIR
        ev_api._EVALUATION_RESULTS_DIR = tmp_path

        latest_data = {
            "run_id": 99,
            "timestamp": "2026-06-24T10:00:00Z",
            "results": [{"passed": True, "latency_ms": 700, "top_score": 0.92}],
        }
        latest_file = tmp_path / "latest_results.json"
        latest_file.write_text(json.dumps(latest_data))

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            d = response.json()
            # latest_results.json should appear with correct type
            latest_reports = [r for r in d["custom_eval_reports"] if r["report_name"] == "latest_results.json"]
            assert len(latest_reports) == 1
            assert latest_reports[0]["report_type"] == "latest_results_json"
            # Should be first in the list (latest-first)
            assert d["custom_eval_reports"][0]["report_name"] == "latest_results.json"
        finally:
            ev_api._EVALUATION_RESULTS_DIR = original_const

    def test_no_app_filesystem_paths_in_custom_eval_reports(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """custom_eval_reports contain no /app/ or full filesystem paths."""
        import app.api.evaluation as ev_api

        # Override the module-level constant to point to tmp_path
        original_const = ev_api._EVALUATION_RESULTS_DIR
        ev_api._EVALUATION_RESULTS_DIR = tmp_path

        eval_data = {
            "run_id": 1,
            "timestamp": "2026-06-22T12:00:00Z",
            "results": [{"passed": True, "latency_ms": 500, "top_score": 0.9}],
        }
        eval_file = tmp_path / "evaluation_20260622_120000.json"
        eval_file.write_text(json.dumps(eval_data))

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            raw = str(response.json())
            assert "/app/" not in raw
            assert "/home/ubuntu/" not in raw
            # tmp_path str may appear in tracebacks but not as a report field
            assert str(tmp_path) not in raw
        finally:
            ev_api._EVALUATION_RESULTS_DIR = original_const

    def test_custom_eval_reports_limit_behavior(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """Limit parameter caps custom_eval_reports returned."""
        import app.api.evaluation as ev_api

        # Override the module-level constant to point to tmp_path
        original_const = ev_api._EVALUATION_RESULTS_DIR
        ev_api._EVALUATION_RESULTS_DIR = tmp_path

        # Create 5 timestamped files
        for i in range(5):
            data = {
                "run_id": i,
                "timestamp": f"2026-06-{22-i:02d}T12:00:00Z",
                "results": [{"passed": True, "latency_ms": 500, "top_score": 0.9}],
            }
            (tmp_path / f"evaluation_202606{22-i:02d}_120000.json").write_text(json.dumps(data))

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history?limit=3")
            assert response.status_code == 200
            d = response.json()
            assert len(d["custom_eval_reports"]) <= 3
        finally:
            ev_api._EVALUATION_RESULTS_DIR = original_const

    def test_ragas_reports_empty_shows_friendly_warning(
        self, jwt_admin_client, monkeypatch, tmp_path
    ):
        """When RAGAS dir exists but has no reports, a friendly warning is returned."""
        import app.core.config

        original_ragas_dir = app.core.config.settings.RAGAS_REPORT_DIR
        app.core.config.settings.RAGAS_REPORT_DIR = str(tmp_path)

        # No ragas_report_*.json files, just the .gitkeep
        (tmp_path / ".gitkeep").touch()

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/history")
            assert response.status_code == 200
            d = response.json()
            assert len(d["ragas_reports"]) == 0
            assert any("no ragas reports" in w.lower() for w in d["warnings"])
        finally:
            app.core.config.settings.RAGAS_REPORT_DIR = original_ragas_dir


class TestExistingEvaluationsUnchanged:
    """Phase 30: existing evaluation endpoint behavior must not change."""

    def test_latest_evaluation_still_works(self, jwt_admin_client):
        """GET /api/admin/evaluations/latest still returns custom eval results."""
        response = jwt_admin_client.get("/api/admin/evaluations/latest")
        assert response.status_code == 200
        d = response.json()
        assert "latest_run" in d or d.get("latest_run") is None

    def test_runs_list_still_works(self, jwt_admin_client):
        """GET /api/admin/evaluations/runs still works."""
        response = jwt_admin_client.get("/api/admin/evaluations/runs")
        assert response.status_code == 200

    def test_ragas_summary_still_works(self, jwt_admin_client):
        """GET /api/admin/evaluations/ragas-summary still works."""
        response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
        assert response.status_code == 200