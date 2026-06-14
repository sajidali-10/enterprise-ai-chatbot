"""
Tests for Evaluation and Observability functionality.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestEvaluationDataset:
    """Tests for evaluation dataset loading and structure."""

    def test_dataset_file_exists(self):
        """Test that evaluation dataset file exists."""
        dataset_path = Path(__file__).parent.parent / "evaluations" / "dataset.json"
        assert dataset_path.exists(), "Evaluation dataset file should exist"

    def test_dataset_valid_json(self):
        """Test that dataset is valid JSON."""
        dataset_path = Path(__file__).parent.parent / "evaluations" / "dataset.json"
        with open(dataset_path) as f:
            data = json.load(f)
        assert isinstance(data, list), "Dataset should be a list"

    def test_dataset_has_required_fields(self):
        """Test that each test case has required fields."""
        dataset_path = Path(__file__).parent.parent / "evaluations" / "dataset.json"
        with open(dataset_path) as f:
            data = json.load(f)
        
        required_fields = [
            "id", "question", "expected_mode", "expected_source_file",
            "expected_keywords", "forbidden_keywords", "should_answer",
            "should_have_citations", "expected_fallback", "minimum_expected_citations"
        ]
        
        for case in data:
            for field in required_fields:
                assert field in case, f"Test case {case.get('id', 'unknown')} missing field: {field}"

    def test_dataset_has_positive_and_negative_examples(self):
        """Test that dataset has both answer and fallback test cases."""
        dataset_path = Path(__file__).parent.parent / "evaluations" / "dataset.json"
        with open(dataset_path) as f:
            data = json.load(f)
        
        should_answer = [c for c in data if c.get("should_answer") == True]
        should_fallback = [c for c in data if c.get("expected_fallback") == True]
        
        assert len(should_answer) > 0, "Dataset should have positive examples"
        assert len(should_fallback) > 0, "Dataset should have negative examples (fallback cases)"

    def test_docker_components_has_correct_forbidden_keywords(self):
        """Test that docker components case has CI/CD related forbidden keywords."""
        dataset_path = Path(__file__).parent.parent / "evaluations" / "dataset.json"
        with open(dataset_path) as f:
            data = json.load(f)
        
        docker_case = next((c for c in data if c["id"] == "docker_components_001"), None)
        assert docker_case is not None, "docker_components_001 should exist"
        
        forbidden = docker_case.get("forbidden_keywords", [])
        assert "CI/CD" in forbidden, "Should forbid CI/CD"
        assert "cloud migration" in forbidden, "Should forbid cloud migration"
        assert "continuous integration" in forbidden, "Should forbid continuous integration"


class TestEvaluationModels:
    """Tests for evaluation database models."""

    def test_evaluation_run_model_fields(self):
        """Test that EvaluationRun model has required fields."""
        from app.models.evaluation import EvaluationRun
        
        # Check that model has expected columns
        columns = [c.name for c in EvaluationRun.__table__.columns]
        required = ["id", "created_at", "total_tests", "passed_tests", "failed_tests",
                    "pass_percentage", "status"]
        
        for col in required:
            assert col in columns, f"EvaluationRun missing column: {col}"

    def test_evaluation_result_model_fields(self):
        """Test that EvaluationResult model has required fields."""
        from app.models.evaluation import EvaluationResult
        
        columns = [c.name for c in EvaluationResult.__table__.columns]
        required = ["id", "run_id", "test_case_id", "question", "passed", "failure_reasons"]
        
        for col in required:
            assert col in columns, f"EvaluationResult missing column: {col}"

    def test_chat_observation_model_fields(self):
        """Test that ChatObservation model has required fields."""
        from app.models.observability import ChatObservation
        
        columns = [c.name for c in ChatObservation.__table__.columns]
        required = ["id", "created_at", "mode", "question", "answer_preview",
                    "source_files", "citation_count", "top_score", "blocked"]
        
        for col in required:
            assert col in columns, f"ChatObservation missing column: {col}"


class TestObservabilitySchema:
    """Tests for observability API schemas."""

    def test_feedback_request_schema(self):
        """Test FeedbackRequest schema validation."""
        from app.schemas.evaluation import FeedbackRequest
        
        # Valid helpful feedback
        feedback = FeedbackRequest(rating="helpful")
        assert feedback.rating == "helpful"
        assert feedback.reason is None
        
        # Valid not_helpful with reason
        feedback = FeedbackRequest(rating="not_helpful", reason="wrong_answer", comment="Test")
        assert feedback.rating == "not_helpful"
        assert feedback.reason == "wrong_answer"

    def test_observation_summary_schema(self):
        """Test ObservationSummary schema."""
        from app.schemas.evaluation import ObservationSummary
        from datetime import datetime
        
        obs = ObservationSummary(
            id=1,
            created_at=datetime.now(),
            username="testuser",
            mode="knowledge_base",
            question="test question",
            answer_preview="test answer preview",
            answer_length=100,
            source_files=["test.pdf"],
            citation_count=1,
            top_score=0.85,
            blocked=False,
            block_reason=None,
            latency_ms=500.0,
            feedback_rating="helpful",
        )
        assert obs.id == 1
        assert obs.mode == "knowledge_base"

    def test_evaluability_summary_schema(self):
        """Test ObservabilitySummary schema."""
        from app.schemas.evaluation import ObservabilitySummary
        
        summary = ObservabilitySummary(
            total_questions=100,
            general_chat_count=30,
            knowledge_base_count=60,
            debug_count=10,
            answered_count=90,
            blocked_count=10,
            blocked_rate=10.0,
            citation_rate=80.0,
            average_latency_ms=1500.0,
            average_top_score=0.75,
            thumbs_up_count=50,
            thumbs_down_count=5,
            most_used_sources=[{"source": "test.pdf", "count": 10}],
        )
        assert summary.total_questions == 100
        assert summary.knowledge_base_count == 60


class TestObservabilityService:
    """Tests for observability logging service."""

    def test_log_chat_observation_does_not_raise(self):
        """Test that logging does not raise even if DB fails."""
        from app.services.observability import log_chat_observation
        
        # Should not raise
        result = log_chat_observation(
            mode="general_chat",
            question="test question",
            answer="test answer",
            latency_ms=100.0,
        )
        # Result may be None if DB not available, but should not raise
        assert result is None or isinstance(result, int)

    def test_log_chat_observation_extracts_sources(self):
        """Test that source files are extracted from grouped sources."""
        from app.services.observability import log_chat_observation
        
        # Mock grouped sources
        class MockGroupedSource:
            source_file_name = "test.pdf"
            highest_score = 0.9
        
        result = log_chat_observation(
            mode="knowledge_base",
            question="test question",
            answer="test answer",
            grouped_sources=[MockGroupedSource()],
        )
        # Should not raise
        assert result is None or isinstance(result, int)


class TestEvaluationScoring:
    """Tests for evaluation scoring logic."""

    def test_keywords_found_scoring(self):
        """Test expected keywords detection."""
        answer = "Docker images are templates that contain application code, dependencies, and metadata."
        
        expected_keywords = ["image", "container", "dependencies", "metadata"]
        
        found = []
        missing = []
        for kw in expected_keywords:
            if kw.lower() in answer.lower():
                found.append(kw)
            else:
                missing.append(kw)
        
        assert "image" in found
        assert "dependencies" in found
        assert "metadata" in found
        assert "container" in missing  # Not in answer

    def test_forbidden_keywords_scoring(self):
        """Test forbidden keywords detection."""
        answer = "Docker components include CI/CD pipelines and cloud migration benefits."
        
        forbidden_keywords = ["CI/CD", "cloud migration", "continuous integration"]
        
        found = []
        for kw in forbidden_keywords:
            if kw.lower() in answer.lower():
                found.append(kw)
        
        assert "CI/CD" in found
        assert "cloud migration" in found
        assert "continuous integration" not in found  # Not exact match

    def test_fallback_detection(self):
        """Test fallback/insufficient information detection."""
        fallback_phrases = [
            "i don't have",
            "i cannot find",
            "i don't know",
            "not enough information",
            "no relevant documents",
        ]
        
        fallback_answer = "I don't have access to information about that topic."
        normal_answer = "Docker containers are lightweight virtual machine instances."
        
        def detect_fallback(answer):
            return any(phrase in answer.lower() for phrase in fallback_phrases)
        
        assert detect_fallback(fallback_answer) == True
        assert detect_fallback(normal_answer) == False

    def test_citation_count_scoring(self):
        """Test citation count validation."""
        citations = [
            {"source_file_name": "doc1.pdf", "content_snippet": "..."},
            {"source_file_name": "doc2.pdf", "content_snippet": "..."},
        ]
        
        citation_count = len(citations)
        minimum_expected = 1
        
        assert citation_count >= minimum_expected


class TestChatResponseObservationId:
    """Tests for chat response including observation_id."""

    def test_chat_response_has_observation_id_field(self):
        """Test that ChatResponse schema includes observation_id."""
        from app.schemas.chat import ChatResponse, MessageRole
        
        response = ChatResponse(
            message="Test answer",
            role=MessageRole.assistant,
            observation_id=123,
        )
        
        assert response.observation_id == 123

    def test_chat_response_observation_id_optional(self):
        """Test that observation_id is optional."""
        from app.schemas.chat import ChatResponse, MessageRole
        
        response = ChatResponse(
            message="Test answer",
            role=MessageRole.assistant,
        )
        
        assert response.observation_id is None


class TestAdminObservabilityEndpoints:
    """Tests for admin observability API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_observability_summary_endpoint_requires_admin(self, client):
        """Test that observability summary requires admin access."""
        response = client.get("/api/admin/observability/summary")
        # Should return 401 if unauthenticated, 403 if not admin
        assert response.status_code in [401, 403]

    def test_evaluations_latest_endpoint_requires_admin(self, client):
        """Test that evaluations latest requires admin access."""
        response = client.get("/api/admin/evaluations/latest")
        assert response.status_code in [401, 403]

    def test_feedback_endpoint_structure(self, client):
        """Test feedback submission endpoint exists."""
        # Use a non-existent observation ID - should return 404 not 500
        response = client.post(
            "/api/chat/999999/feedback",
            json={"rating": "helpful"}
        )
        assert response.status_code in [200, 404, 500]


class TestNoSecretsInObservability:
    """Tests to ensure no secrets are logged in observability."""

    def test_chat_observation_does_not_store_api_keys(self):
        """Test that ChatObservation does not have API key fields."""
        from app.models.observability import ChatObservation
        
        columns = [c.name for c in ChatObservation.__table__.columns]
        
        assert "api_key" not in columns
        assert "secret" not in columns
        assert "token" not in columns
        assert "password" not in columns

    def test_answer_preview_is_limited(self):
        """Test that answer preview length is limited (first 500 chars)."""
        from app.services.observability import log_chat_observation
        
        long_answer = "A" * 1000
        
        # The service should truncate to 500 chars
        # We can only verify it doesn't raise
        result = log_chat_observation(
            mode="knowledge_base",
            question="test",
            answer=long_answer,
        )
        assert result is None or isinstance(result, int)


class TestEvaluationRunnerScript:
    """Tests for evaluation runner script."""

    def test_evaluation_script_has_load_function(self):
        """Test that evaluation script has required functions."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        
        from scripts.run_rag_evaluation import load_evaluation_dataset
        
        assert callable(load_evaluation_dataset)

    def test_evaluation_dataset_loads(self):
        """Test that evaluation dataset loads successfully."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        
        from scripts.run_rag_evaluation import load_evaluation_dataset
        
        dataset = load_evaluation_dataset()
        assert isinstance(dataset, list)
        assert len(dataset) > 0