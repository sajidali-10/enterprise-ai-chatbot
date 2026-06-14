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