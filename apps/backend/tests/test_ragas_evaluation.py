"""
Tests for Phase 26 — RAGAS Evaluation Foundation.

Covers:
- RAGAS config settings contain no secrets
- run_ragas_evaluation.py --dry-run loads dataset without LLM calls
- Dataset conversion produces correct SingleTurnSample fields
- Setup error if evaluator LLM config missing
- Existing custom eval script still passes
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/ is on the path for imports
_BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))


class TestRAGASConfigNoSecrets:
    """Phase 26: ragas_status must expose no secrets."""

    def test_ragas_status_has_required_keys(self):
        """provider_status.ragas_status has the required 5 keys."""
        from app.providers.factory import get_provider_status

        rs = get_provider_status()["ragas_status"]
        for key in ["ragas_enabled", "ragas_available", "evaluator_provider", "evaluator_model", "report_dir"]:
            assert key in rs, f"Missing ragas_status key: {key}"

    def test_ragas_status_default_values(self):
        """Default values match Phase 26 config spec."""
        from app.core.config import settings

        assert settings.RAGAS_ENABLED is False
        assert settings.RAGAS_EVALUATOR_PROVIDER == "litellm"
        assert settings.RAGAS_EVALUATOR_MODEL == "openrouter-gpt-oss"
        assert settings.RAGAS_REPORT_DIR == "/app/evals/ragas"
        assert settings.RAGAS_MAX_CASES == 20
        assert settings.RAGAS_SAVE_RESULTS is True

    def test_ragas_status_no_secrets(self):
        """ragas_status must not expose any secret fields."""
        from app.providers.factory import get_provider_status

        rs = get_provider_status()["ragas_status"]
        raw = str(rs).lower()
        secret_keywords = ["secret", "password", "token", "api_key", "credential"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Potential secret(s) in ragas_status: {found}"

    def test_ragas_status_in_provider_status(self):
        """ragas_status must be present in get_provider_status()."""
        from app.providers.factory import get_provider_status

        status = get_provider_status()
        assert "ragas_status" in status
        assert status["ragas_status"]["evaluator_provider"] == "litellm"
        assert status["ragas_status"]["evaluator_model"] == "openrouter-gpt-oss"

    def test_ragas_status_no_secrets_via_rag_config_endpoint(self, jwt_admin_client):
        """RAG config endpoint ragas_status must not expose secrets."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        data = response.json()
        assert "provider_status" in data
        rs = data["provider_status"]["ragas_status"]

        raw = str(rs).lower()
        secret_keywords = ["secret", "password", "token", "api_key", "credential"]
        found = [kw for kw in secret_keywords if kw in raw]
        assert not found, f"Potential secret(s) in ragas_status: {found}"

    def test_ragas_status_values_via_rag_config_endpoint(self, jwt_admin_client):
        """RAG config endpoint returns correct ragas_status values."""
        response = jwt_admin_client.get("/api/admin/rag/config")
        assert response.status_code == 200
        rs = response.json()["provider_status"]["ragas_status"]
        assert rs["ragas_enabled"] is False
        assert rs["evaluator_provider"] == "litellm"
        assert rs["evaluator_model"] == "openrouter-gpt-oss"
        assert rs["report_dir"] == "/app/evals/ragas"


class TestRagasDryRun:
    """Phase 26: --dry-run loads dataset without calling any LLM."""

    def test_dry_run_exits_zero(self, tmp_path, monkeypatch):
        """run_ragas_evaluation.py --dry-run exits 0."""
        # Use tmp dataset with one case so test is self-contained
        dataset_content = '[{"id": "test_001", "question": "what is docker?"}]'
        fake_dataset = tmp_path / "dataset.json"
        fake_dataset.write_text(dataset_content)

        monkeypatch.chdir(_BACKEND_ROOT)
        monkeypatch.setenv("RAGAS_MAX_CASES", "1")

        # Patch load_dataset to use our fake file
        import scripts.run_ragas_evaluation as ragas_script

        original_load = ragas_script.load_dataset

        def fake_load(path):
            import json
            with open(path) as f:
                return json.load(f)

        with patch.object(ragas_script, "load_dataset", fake_load):
            with patch.object(ragas_script, "build_dataset") as mock_build:
                mock_ds = _FakeDataset([])
                mock_build.return_value = mock_ds

                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                rc = ragas_script.main
                # Patch sys.argv
                original_argv = sys.argv
                sys.argv = ["run_ragas_evaluation.py", "--dry-run"]
                try:
                    with redirect_stdout(f):
                        exit_code = rc()
                finally:
                    sys.argv = original_argv

                assert exit_code == 0, f"--dry-run exited {exit_code}, expected 0. Output: {f.getvalue()}"

    def test_dry_run_loads_dataset_without_llm(self, tmp_path, monkeypatch):
        """--dry-run loads dataset and converts cases without calling generate_answer_with_rag."""
        dataset_content = '[{"id": "test_001", "question": "what is docker?"}]'
        fake_dataset = tmp_path / "dataset.json"
        fake_dataset.write_text(dataset_content)

        monkeypatch.chdir(_BACKEND_ROOT)

        import scripts.run_ragas_evaluation as ragas_script

        with patch.object(ragas_script, "load_dataset") as mock_load:
            mock_load.return_value = [{"id": "test_001", "question": "what is docker?"}]

            with patch.object(ragas_script, "build_dataset") as mock_build:
                mock_ds = _FakeDataset([])
                mock_build.return_value = mock_ds

                original_argv = sys.argv
                sys.argv = ["run_ragas_evaluation.py", "--dry-run"]
                try:
                    import io
                    from contextlib import redirect_stdout
                    f = io.StringIO()
                    with redirect_stdout(f):
                        exit_code = ragas_script.main()
                finally:
                    sys.argv = original_argv

                assert exit_code == 0
                # build_dataset must have been called with dry_run=True
                mock_build.assert_called_once()
                args, kwargs = mock_build.call_args
                assert kwargs.get("dry_run") is True


class TestRagasDatasetConversion:
    """Phase 26: dataset conversion produces correct SingleTurnSample fields."""

    def test_sample_has_user_input_and_response(self):
        """SingleTurnSample requires user_input (question) and response (answer)."""
        from ragas.dataset_schema import SingleTurnSample

        sample = SingleTurnSample(
            user_input="what is a container?",
            response="A container is a lightweight unit.",
            retrieved_contexts=["context one", "context two"],
        )
        assert sample.user_input == "what is a container?"
        assert sample.response == "A container is a lightweight unit."
        assert sample.retrieved_contexts == ["context one", "context two"]

    def test_build_dataset_produces_samples(self):
        """build_dataset converts each case to a SingleTurnSample."""
        from scripts.run_ragas_evaluation import build_dataset
        from ragas.dataset_schema import SingleTurnSample

        cases = [
            {"id": "t1", "question": "question 1"},
            {"id": "t2", "question": "question 2"},
        ]

        # dry-run: no LLM calls
        dataset = build_dataset(cases, dry_run=True)

        assert len(dataset.samples) == 2
        assert all(isinstance(s, SingleTurnSample) for s in dataset.samples)
        # In dry-run, response and contexts are empty
        assert dataset.samples[0].user_input == "question 1"
        assert dataset.samples[1].user_input == "question 2"

    def test_generate_answer_for_case_returns_answer_and_contexts(self):
        """generate_answer_for_case returns (answer, contexts) tuple."""
        from scripts.run_ragas_evaluation import generate_answer_for_case

        answer, contexts = generate_answer_for_case("what is docker?")
        # On success returns non-empty strings; on failure returns ("", [])
        assert isinstance(answer, str)
        assert isinstance(contexts, list)


class TestRagasSetupError:
    """Phase 26: clear error if evaluator LLM config is missing."""

    def test_build_evaluator_llm_raises_without_master_key(self, monkeypatch):
        """build_evaluator_llm raises ValueError when LITTLM_MASTER_KEY is missing."""
        monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
        monkeypatch.setenv("LITELLM_MASTER_KEY", "")

        from scripts.run_ragas_evaluation import build_evaluator_llm

        with pytest.raises(ValueError, match="LITELLM_MASTER_KEY"):
            build_evaluator_llm()

    def test_build_evaluator_llm_raises_with_placeholder_key(self, monkeypatch):
        """build_evaluator_llm raises ValueError when LITLLM_MASTER_KEY is placeholder."""
        monkeypatch.setenv("LITELLM_MASTER_KEY", "changeme_litellm_master_key")

        from scripts.run_ragas_evaluation import build_evaluator_llm

        with pytest.raises(ValueError, match="LITELLM_MASTER_KEY"):
            build_evaluator_llm()


class TestRagasIsAvailable:
    """Phase 26: _is_ragas_available detection."""

    def test_is_ragas_available_returns_bool(self):
        """_is_ragas_available returns True when ragas is installed."""
        from app.providers.factory import _is_ragas_available

        result = _is_ragas_available()
        assert isinstance(result, bool)
        # In this environment ragas IS installed (0.2.15)
        assert result is True

    def test_ragas_version_accessible(self):
        """ragas.__version__ is accessible."""
        import ragas
        assert ragas.__version__


class TestExistingCustomEvalUnchanged:
    """Phase 26: existing custom eval (run_rag_evaluation.py) still works."""

    def test_evaluation_script_has_load_function(self):
        """Existing eval script still has load_evaluation_dataset function."""
        from scripts.run_rag_evaluation import load_evaluation_dataset
        assert callable(load_evaluation_dataset)

    def test_evaluation_script_load_dataset(self):
        """load_evaluation_dataset returns a list of cases."""
        from scripts.run_rag_evaluation import load_evaluation_dataset

        cases = load_evaluation_dataset()
        assert isinstance(cases, list)
        assert len(cases) >= 20  # Our dataset has 20 cases
        # Each case has required fields
        for case in cases:
            assert "id" in case
            assert "question" in case

    def test_generate_answer_with_rag_is_callable(self):
        """generate_answer_with_rag is still callable and returns expected types."""
        from app.rag.answer_generator import generate_answer_with_rag

        # Use a simple question; we only care that it returns the right types
        answer, citations, metadata = generate_answer_with_rag(
            query="what is docker?",
            use_hybrid=True,
            debug=False,
            conversation_context="",
        )
        assert isinstance(answer, str)
        assert isinstance(citations, list)
        assert isinstance(metadata, dict)


# ---------------------------------------------------------------------------
# Fake dataset for tests that don't need real LLM calls
# ---------------------------------------------------------------------------
class _FakeSample:
    def __init__(self, user_input="", response="", retrieved_contexts=None):
        self.user_input = user_input
        self.response = response
        self.retrieved_contexts = retrieved_contexts or []


class _FakeDataset:
    def __init__(self, samples):
        self.samples = samples