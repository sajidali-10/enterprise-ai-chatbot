import os
os.environ["LLM_PROVIDER"] = "mock"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_with_mock_provider_hello():
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert "Hello! How can I help you today?" in data["message"]

def test_chat_with_mock_provider_echo():
    response = client.post("/api/chat", json={"message": "test message"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert "You said:" in data["message"]

def test_chat_validation_empty_message():
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422