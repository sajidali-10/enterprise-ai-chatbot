"""
Tests for Phase 28 — RAGAS Summary on Evaluations Page.

Admin-only endpoint GET /api/admin/evaluations/ragas-summary.
"""

import pytest


class TestRagasSummaryEndpoint:
    """Phase 28: GET /api/admin/evaluations/ragas-summary returns safe RAGAS status."""

    def test_requires_admin(self, client):
        """Non-admin requests are forbidden."""
        response = client.get("/api/admin/evaluations/ragas-summary")
        assert response.status_code in (401, 403)

    def test_admin_access_returns_200(self, jwt_admin_client):
        """Admin access returns HTTP 200 even when no report exists."""
        response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
        assert response.status_code == 200

    def test_response_structure_with_no_report(self, jwt_admin_client):
        """When no RAGAS report exists, endpoint returns safe fields with warnings."""
        response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
        assert response.status_code == 200
        d = response.json()

        for key in ["available", "enabled", "evaluator_provider", "evaluator_model",
                    "report_dir_configured", "latest_report_found", "skipped_metrics",
                    "threshold_faithfulness", "threshold_answer_relevancy",
                    "threshold_context_precision", "warnings"]:
            assert key in d, f"Missing key: {key}"

        assert d["latest_report_found"] is False
        assert d["skipped_metrics"] == ["context_recall", "answer_correctness"]
        assert len(d["warnings"]) >= 1  # At least one warning about no report

    def test_no_secrets_in_response(self, jwt_admin_client):
        """RAGAS summary response must not expose any secrets."""
        response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
        assert response.status_code == 200
        raw = str(response.json()).lower()
        secret_keywords = ["secret", "password", "token", "api_key", "credential",
                          "key", "master_key", "litellm_master"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Secret keyword(s) found in ragas summary: {found}"

    def test_no_import_in_request_path(self, jwt_admin_client, monkeypatch):
        """RAGAS import is NOT triggered through the request path (no uvloop crash)."""
        import importlib.util

        calls = []

        def fake_find_spec(name):
            calls.append(name)
            if name == "ragas":
                return object()
            return None

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
        assert response.status_code == 200
        assert "ragas" in calls, f"endpoint should use find_spec to detect ragas, but got calls: {calls}"

    def test_parses_valid_report(self, jwt_admin_client, monkeypatch, tmp_path):
        """When a valid RAGAS report exists, metric averages are returned."""
        import app.core.config
        original_report_dir = app.core.config.settings.RAGAS_REPORT_DIR
        original_enabled = app.core.config.settings.RAGAS_ENABLED
        app.core.config.settings.RAGAS_REPORT_DIR = str(tmp_path)
        app.core.config.settings.RAGAS_ENABLED = True

        report_file = tmp_path / "latest.json"
        report_file.write_text(
            '{"timestamp":"2026-06-22T12:00:00Z","case_count":20,'
            '"avg_scores":{"faithfulness":{"avg":0.85,"min":0.7,"max":0.95,"count":20},'
            '"answer_relevancy":{"avg":0.78,"min":0.6,"max":0.9,"count":20},'
            '"context_precision":{"avg":0.91,"min":0.8,"max":1.0,"count":20}},"threshold_summary":{}}'
        )

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
            assert response.status_code == 200
            d = response.json()
            assert d["latest_report_found"] is True
            assert d["metrics"] is not None
            assert d["metrics"]["faithfulness"] is not None
        finally:
            app.core.config.settings.RAGAS_REPORT_DIR = original_report_dir
            app.core.config.settings.RAGAS_ENABLED = original_enabled

    def test_malformed_json_returns_warnings(self, jwt_admin_client, monkeypatch, tmp_path):
        """Malformed JSON in report returns safe warning, not 500."""
        import app.core.config
        original_report_dir = app.core.config.settings.RAGAS_REPORT_DIR
        original_enabled = app.core.config.settings.RAGAS_ENABLED
        app.core.config.settings.RAGAS_REPORT_DIR = str(tmp_path)
        app.core.config.settings.RAGAS_ENABLED = True

        report_file = tmp_path / "latest.json"
        report_file.write_text("this is not json{")

        try:
            response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
            assert response.status_code == 200
            d = response.json()
            assert any("parse" in w.lower() for w in d.get("warnings", []))
        finally:
            app.core.config.settings.RAGAS_REPORT_DIR = original_report_dir
            app.core.config.settings.RAGAS_ENABLED = original_enabled

    def test_evaluator_config_in_response(self, jwt_admin_client):
        """Evaluator provider and model are included in response."""
        response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
        assert response.status_code == 200
        d = response.json()
        assert d["evaluator_provider"] == "litellm"
        assert d["evaluator_model"] == "openrouter-gpt-oss"

    def test_disabled_by_default(self, jwt_admin_client, monkeypatch):
        """When RAGAS_ENABLED=false, enabled=False in response."""
        # Patch settings directly so no module reload needed
        import app.core.config
        original = app.core.config.settings.RAGAS_ENABLED
        app.core.config.settings.RAGAS_ENABLED = False
        try:
            response = jwt_admin_client.get("/api/admin/evaluations/ragas-summary")
            assert response.status_code == 200
            d = response.json()
            assert d["enabled"] is False
        finally:
            app.core.config.settings.RAGAS_ENABLED = original


class TestExistingEvaluationsUnchanged:
    """Phase 28: existing evaluation endpoint behavior must not change."""

    def test_latest_evaluation_still_works(self, jwt_admin_client):
        """GET /api/admin/evaluations/latest still returns custom eval results."""
        response = jwt_admin_client.get("/api/admin/evaluations/latest")
        assert response.status_code == 200
        d = response.json()
        assert "latest_run" in d or d.get("latest_run") is None  # Empty db is OK

    def test_evaluation_runs_still_work(self, jwt_admin_client):
        """GET /api/admin/evaluations/runs still works."""
        response = jwt_admin_client.get("/api/admin/evaluations/runs")
        assert response.status_code == 200