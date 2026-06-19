import os
os.environ["LLM_PROVIDER"] = "mock"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app, headers={"X-Dev-User": "admin_user"})

def test_chat_with_mock_provider_hello():
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    # Mock provider echoes the prompt or returns a mock response
    assert len(data["message"]) > 0

def test_chat_with_mock_provider_echo():
    response = client.post("/api/chat", json={"message": "test message"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    # Mock provider returns the prompt which contains the original message
    assert "test message" in data["message"] or "You said:" in data["message"]

def test_chat_validation_empty_message():
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


# ==============================================================================
# Phase 20B — Suggested Follow-ups Tests
# ==============================================================================

def test_suggested_followups_returned_for_general_chat():
    """General chat returns suggested follow-ups."""
    response = client.post("/api/chat", json={"message": "hello", "mode": "general_chat"})
    assert response.status_code == 200
    data = response.json()
    assert "suggested_followups" in data
    assert isinstance(data["suggested_followups"], list)
    assert 3 <= len(data["suggested_followups"]) <= 5


def test_suggested_followups_returned_for_rag_chat():
    """RAG chat returns suggested follow-ups."""
    response = client.post("/api/chat", json={"message": "hello", "mode": "knowledge_base"})
    assert response.status_code == 200
    data = response.json()
    assert "suggested_followups" in data
    assert isinstance(data["suggested_followups"], list)
    assert 3 <= len(data["suggested_followups"]) <= 5


def test_suggested_followups_are_strings():
    """All suggested follow-ups are strings of reasonable length."""
    response = client.post("/api/chat", json={"message": "test", "mode": "general_chat"})
    assert response.status_code == 200
    data = response.json()
    for suggestion in data["suggested_followups"]:
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0
        assert len(suggestion) <= 80  # Max length check


def test_suggested_followups_no_secrets():
    """Suggested follow-ups do not expose secrets or API keys."""
    response = client.post("/api/chat", json={"message": "test", "mode": "general_chat"})
    assert response.status_code == 200
    data = response.json()
    dangerous_patterns = ["key", "secret", "password", "token", "api", "sk-", "eyJ"]
    for suggestion in data["suggested_followups"]:
        suggestion_lower = suggestion.lower()
        for pattern in dangerous_patterns:
            assert pattern not in suggestion_lower, f"Suggestion contains dangerous pattern: {pattern}"


def test_suggested_followups_session_id_returned():
    """Chat response still returns session_id correctly."""
    response = client.post("/api/chat", json={"message": "hello", "mode": "general_chat"})
    assert response.status_code == 200
    data = response.json()
    # session_id may be None if sessions are disabled, or an integer if enabled
    assert data.get("session_id") is None or isinstance(data["session_id"], int)