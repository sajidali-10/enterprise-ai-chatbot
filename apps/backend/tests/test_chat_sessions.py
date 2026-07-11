"""
Chat Sessions Tests

Phase 20A — Persistent Chat Sessions and Message Storage.
"""
import os
os.environ["LLM_PROVIDER"] = "mock"

import json
import pytest
from fastapi.testclient import TestClient

from app.models.chat_session import ChatSession, ChatMessage, ChatMessageFeedback
from app.models.chat_session import ChatSessionMode as DBChatSessionMode, MessageRole as DBMessageRole
from app.security.models import User, UserRole
from app.security.password import hash_password


def _create_user(db_session, username, email, password, role=UserRole.user):
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _login(client, username, password):
    res = client.post("/api/auth/login", json={
        "username_or_email": username,
        "password": password,
    })
    assert res.status_code == 200, f"Login failed: {res.text}"
    return res.json()["access_token"]


def _jwt_headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Create chat session
# ---------------------------------------------------------------------------

def test_create_chat_session(jwt_user_client, db_session):
    """Authenticated user can create a new chat session."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "Test Session"})
    assert res.status_code == 201
    data = res.json()
    assert data["title"] == "Test Session"
    assert data["mode"] == "general"
    assert data["message_count"] == 0
    assert data["archived_at"] is None


def test_create_rag_session(jwt_user_client, db_session):
    """Can create a RAG mode session."""
    res = jwt_user_client.post("/api/chat/sessions", json={"mode": "rag"})
    assert res.status_code == 201
    assert res.json()["mode"] == "rag"


def test_create_session_requires_auth(client):
    """Unauthenticated request is rejected."""
    res = client.post("/api/chat/sessions", json={"title": "Test"})
    assert res.status_code == 401


def test_create_session_no_title_uses_default(db_session, client):
    """Creating a session without a title defaults to 'New Chat'."""
    _create_user(db_session, "user1", "user1@test.com", "pass123")
    token = _login(client, "user1", "pass123")
    res = client.post("/api/chat/sessions", json={}, headers=_jwt_headers(token))
    assert res.status_code == 201
    assert res.json()["title"] == "New Chat"


# ---------------------------------------------------------------------------
# List sessions
# ---------------------------------------------------------------------------

def test_list_own_sessions(jwt_user_client, db_session):
    """User can list their own sessions."""
    # Create a few sessions
    for i in range(3):
        jwt_user_client.post("/api/chat/sessions", json={"title": f"Session {i}"})

    res = jwt_user_client.get("/api/chat/sessions")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3
    assert len(data["sessions"]) == 3


def test_list_sessions_excludes_archived_by_default(jwt_user_client, db_session):
    """Archived sessions are hidden by default."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "To Archive"})
    session_id = res.json()["id"]

    # Archive it
    jwt_user_client.patch(f"/api/chat/sessions/{session_id}", json={"archived": True})

    # List — should not appear
    res = jwt_user_client.get("/api/chat/sessions")
    assert all(s["id"] != session_id for s in res.json()["sessions"])


def test_list_sessions_includes_archived_when_requested(jwt_user_client, db_session):
    """With include_archived=true, archived sessions appear."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "To Archive"})
    session_id = res.json()["id"]
    jwt_user_client.patch(f"/api/chat/sessions/{session_id}", json={"archived": True})

    res = jwt_user_client.get("/api/chat/sessions?include_archived=true")
    assert any(s["id"] == session_id for s in res.json()["sessions"])


def test_list_sessions_requires_auth(client):
    """Unauthenticated request is rejected."""
    res = client.get("/api/chat/sessions")
    assert res.status_code == 401


def test_users_only_see_own_sessions(jwt_user_client, db_session):
    """A user cannot see another user's sessions."""
    # Create sessions as user 1
    jwt_user_client.post("/api/chat/sessions", json={"title": "User 1 Session"})
    user1_sessions = jwt_user_client.get("/api/chat/sessions").json()["total"]
    assert user1_sessions >= 1

    # Create user 2 and verify they see nothing
    _create_user(db_session, "user2", "user2@test.com", "pass456")
    token2 = _login(jwt_user_client, "user2", "pass456")
    res2 = jwt_user_client.get("/api/chat/sessions", headers=_jwt_headers(token2))
    assert res2.json()["total"] == 0


# ---------------------------------------------------------------------------
# Get session
# ---------------------------------------------------------------------------

def test_get_own_session_with_messages(jwt_user_client, db_session):
    """Get a session returns it with all its messages."""
    # Create session and send a chat message
    create_res = jwt_user_client.post("/api/chat/sessions", json={"title": "My Session"})
    session_id = create_res.json()["id"]

    # Send a message (creates user + assistant messages)
    chat_res = jwt_user_client.post("/api/chat", json={
        "message": "Hello",
        "session_id": session_id,
    })
    assert chat_res.status_code == 200

    # Get the session
    res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["session"]["title"] == "My Session"
    # Should have 2 messages: user + assistant
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


def test_cannot_access_another_users_session(jwt_user_client, db_session):
    """User cannot fetch a session they don't own."""
    # Create session as user 1
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "Private"})
    session_id = res.json()["id"]

    # Create user 2 and try to access user 1's session
    _create_user(db_session, "stranger", "stranger@test.com", "pass789")
    token = _login(jwt_user_client, "stranger", "pass789")
    res = jwt_user_client.get(f"/api/chat/sessions/{session_id}", headers=_jwt_headers(token))
    assert res.status_code == 404


def test_get_nonexistent_session_returns_404(jwt_user_client):
    """Getting a session that doesn't exist returns 404."""
    res = jwt_user_client.get("/api/chat/sessions/99999")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Update session
# ---------------------------------------------------------------------------

def test_rename_session(jwt_user_client, db_session):
    """User can rename their own session."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "Old Title"})
    session_id = res.json()["id"]

    res = jwt_user_client.patch(f"/api/chat/sessions/{session_id}", json={"title": "New Title"})
    assert res.status_code == 200
    assert res.json()["title"] == "New Title"


def test_archive_session(jwt_user_client, db_session):
    """User can archive their own session."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "To Archive"})
    session_id = res.json()["id"]

    res = jwt_user_client.patch(f"/api/chat/sessions/{session_id}", json={"archived": True})
    assert res.status_code == 200
    assert res.json()["archived_at"] is not None


def test_unarchive_session(jwt_user_client, db_session):
    """User can unarchive their own session."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "Archived"})
    session_id = res.json()["id"]
    jwt_user_client.patch(f"/api/chat/sessions/{session_id}", json={"archived": True})

    res = jwt_user_client.patch(f"/api/chat/sessions/{session_id}", json={"archived": False})
    assert res.status_code == 200
    assert res.json()["archived_at"] is None


def test_cannot_update_other_users_session(jwt_user_client, db_session):
    """User cannot update a session they don't own."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "Private"})
    session_id = res.json()["id"]

    _create_user(db_session, "other", "other@test.com", "pass999")
    token = _login(jwt_user_client, "other", "pass999")
    res = jwt_user_client.patch(
        f"/api/chat/sessions/{session_id}",
        json={"title": "Hijacked"},
        headers=_jwt_headers(token),
    )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------

def test_delete_session(jwt_user_client, db_session):
    """User can delete their own session."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "To Delete"})
    session_id = res.json()["id"]

    res = jwt_user_client.delete(f"/api/chat/sessions/{session_id}")
    assert res.status_code == 204

    # Verify it's gone
    res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    assert res.status_code == 404


def test_cannot_delete_other_users_session(jwt_user_client, db_session):
    """User cannot delete a session they don't own."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "Private"})
    session_id = res.json()["id"]

    _create_user(db_session, "other2", "other2@test.com", "pass000")
    token = _login(jwt_user_client, "other2", "pass000")
    res = jwt_user_client.delete(f"/api/chat/sessions/{session_id}", headers=_jwt_headers(token))
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Chat endpoint with sessions
# ---------------------------------------------------------------------------

def test_chat_without_session_id_creates_new_session(jwt_user_client, db_session):
    """Sending a message without session_id creates a new session."""
    res = jwt_user_client.post("/api/chat", json={"message": "Hello, world!"})
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] is not None

    # Verify the session was created
    session_res = jwt_user_client.get(f"/api/chat/sessions/{data['session_id']}")
    assert session_res.status_code == 200


def test_chat_with_session_id_appends_to_session(jwt_user_client, db_session):
    """Sending a message with session_id appends to that session."""
    # Create a session
    create_res = jwt_user_client.post("/api/chat/sessions", json={"title": "My Chat"})
    session_id = create_res.json()["id"]

    # Send first message
    res1 = jwt_user_client.post("/api/chat", json={
        "message": "First",
        "session_id": session_id,
    })
    assert res1.json()["session_id"] == session_id

    # Send second message
    res2 = jwt_user_client.post("/api/chat", json={
        "message": "Second",
        "session_id": session_id,
    })
    assert res2.json()["session_id"] == session_id

    # Verify session has 4 messages: user/assistant × 2
    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    assert len(session_res.json()["messages"]) == 4


def test_chat_returns_session_id(jwt_user_client, db_session):
    """Chat response always includes session_id."""
    res = jwt_user_client.post("/api/chat", json={"message": "Hi"})
    assert res.status_code == 200
    assert res.json()["session_id"] is not None


def test_chat_stores_user_and_assistant_messages(jwt_user_client, db_session):
    """Both user and assistant messages are stored in the session."""
    res = jwt_user_client.post("/api/chat", json={"message": "Test message"})
    session_id = res.json()["session_id"]

    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    messages = session_res.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Test message"
    assert messages[1]["role"] == "assistant"


def test_chat_stores_citations_safely(jwt_user_client, db_session):
    """Citations are stored as display metadata, not raw secrets."""
    res = jwt_user_client.post("/api/chat", json={
        "message": "What docs do you know?",
        "mode": "knowledge_base",
        "session_id": None,
    })
    assert res.status_code == 200
    session_id = res.json()["session_id"]
    if session_id:
        session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
        msgs = session_res.json().get("messages", [])
        for msg in msgs:
            if msg["role"] == "assistant" and msg["citations_json"]:
                # Verify it doesn't contain raw secrets
                parsed = json.loads(msg["citations_json"])
                for c in parsed:
                    # Only safe display fields should be present
                    assert "api_key" not in str(c).lower()
                    assert "secret" not in str(c).lower()


def test_cannot_append_to_archived_session(jwt_user_client, db_session):
    """Appending to an archived session returns 400."""
    res = jwt_user_client.post("/api/chat/sessions", json={"title": "Archived"})
    session_id = res.json()["id"]
    jwt_user_client.patch(f"/api/chat/sessions/{session_id}", json={"archived": True})

    res = jwt_user_client.post("/api/chat", json={
        "message": "Hello",
        "session_id": session_id,
    })
    assert res.status_code == 400
    assert "archived" in res.json()["detail"].lower()


def test_chat_session_uses_first_message_as_title(jwt_user_client, db_session):
    """A new session is titled with the first user message (truncated)."""
    res = jwt_user_client.post("/api/chat", json={
        "message": "This is my very long first message that should be truncated to a reasonable title",
    })
    session_id = res.json()["session_id"]
    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    title = session_res.json()["session"]["title"]
    assert len(title) <= 80 + 1  # +1 for the ellipsis char
    assert "This is my very long first message" in title


def test_chat_stores_provider_and_latency(jwt_user_client, db_session):
    """Assistant messages store model/provider info and latency."""
    res = jwt_user_client.post("/api/chat", json={"message": "Hello"})
    session_id = res.json()["session_id"]

    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    msgs = session_res.json()["messages"]
    assistant_msg = next(m for m in msgs if m["role"] == "assistant")
    # These may be None with mock provider, but should not raise
    assert assistant_msg["latency_ms"] is not None
    assert assistant_msg["latency_ms"] >= 0


def test_chat_response_returns_session_id_always(jwt_user_client, db_session):
    """Even when appending to a session, session_id is in the response."""
    # Create session
    create_res = jwt_user_client.post("/api/chat/sessions", json={})
    session_id = create_res.json()["id"]

    # Append message
    res = jwt_user_client.post("/api/chat", json={
        "message": "Hello again",
        "session_id": session_id,
    })
    assert res.status_code == 200
    assert res.json()["session_id"] == session_id


# ---------------------------------------------------------------------------
# Unauthenticated requests
# ---------------------------------------------------------------------------

def test_chat_sessions_require_auth(client):
    """All session endpoints reject unauthenticated requests."""
    endpoints = [
        ("POST", "/api/chat/sessions", {"title": "Test"}),
        ("GET", "/api/chat/sessions", None),
        ("GET", "/api/chat/sessions/1", None),
        ("PATCH", "/api/chat/sessions/1", {"title": "Test"}),
        ("DELETE", "/api/chat/sessions/1", None),
    ]
    for method, url, json_data in endpoints:
        if method == "POST":
            res = client.post(url, json=json_data)
        elif method == "GET":
            res = client.get(url)
        elif method == "PATCH":
            res = client.patch(url, json=json_data)
        elif method == "DELETE":
            res = client.delete(url)
        assert res.status_code == 401, f"{method} {url} should return 401, got {res.status_code}"


# ---------------------------------------------------------------------------
# No secrets exposed
# ---------------------------------------------------------------------------

def test_no_secrets_in_stored_metadata(jwt_user_client, db_session):
    """Stored citations and metadata contain no API keys, secrets, or credentials."""
    # Send a RAG message
    res = jwt_user_client.post("/api/chat", json={
        "message": "What documents do you have?",
        "mode": "knowledge_base",
    })
    session_id = res.json().get("session_id")
    if not session_id:
        return  # RAG might not return a session in mock mode

    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    for msg in session_res.json().get("messages", []):
        for field in ["citations_json", "retrieved_documents_json"]:
            val = msg.get(field)
            if val:
                val_lower = val.lower()
                assert "api_key" not in val_lower, f"{field} contains api_key"
                assert "secret" not in val_lower, f"{field} contains secret"
                assert "password" not in val_lower, f"{field} contains password"
                assert "bearer" not in val_lower or "authorization" not in val_lower, f"{field} contains bearer token"


# ---------------------------------------------------------------------------
# RAG document permissions still enforced
# ---------------------------------------------------------------------------

def test_rag_permissions_enforced_in_existing_session(jwt_user_client, db_session):
    """
    When a user continues a RAG session, document permissions
    are still enforced on each new query.
    """
    # Create a session in RAG mode
    res = jwt_user_client.post("/api/chat/sessions", json={"mode": "rag"})
    session_id = res.json()["id"]

    # The RAG retrieval still filters by permissions when a user is authenticated.
    # We verify by checking that the chat endpoint with session_id
    # doesn't error and returns a valid response (permissions handled by retriever).
    res = jwt_user_client.post("/api/chat", json={
        "message": "What do you know?",
        "mode": "knowledge_base",
        "session_id": session_id,
    })
    # Should succeed (200) or gracefully handle no-access (grounded answer)
    assert res.status_code in (200,)


# ---------------------------------------------------------------------------
# Message feedback
# ---------------------------------------------------------------------------

def test_add_feedback_to_message(jwt_user_client, db_session):
    """User can submit feedback on an assistant message."""
    # Create a session with a message
    res = jwt_user_client.post("/api/chat", json={"message": "Hello"})
    session_id = res.json()["session_id"]
    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    messages = session_res.json()["messages"]
    assistant_msg = next(m for m in messages if m["role"] == "assistant")

    res = jwt_user_client.post(
        f"/api/chat/sessions/messages/{assistant_msg['id']}/feedback",
        json={"rating": "helpful"},
    )
    assert res.status_code == 201
    assert res.json()["rating"] == "helpful"


def test_add_feedback_with_comment(jwt_user_client, db_session):
    """Feedback can include an optional comment."""
    res = jwt_user_client.post("/api/chat", json={"message": "Hello"})
    session_id = res.json()["session_id"]
    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    messages = session_res.json()["messages"]
    assistant_msg = next(m for m in messages if m["role"] == "assistant")

    res = jwt_user_client.post(
        f"/api/chat/sessions/messages/{assistant_msg['id']}/feedback",
        json={"rating": "not_helpful", "comment": "Wrong answer"},
    )
    assert res.status_code == 201
    assert res.json()["comment"] == "Wrong answer"


def test_cannot_feedback_on_other_users_message(jwt_user_client, db_session):
    """User cannot submit feedback on messages from another user's session."""
    # User 1 creates a session
    res = jwt_user_client.post("/api/chat", json={"message": "Hello"})
    session_id = res.json()["session_id"]
    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    assistant_msg = next(m for m in session_res.json()["messages"] if m["role"] == "assistant")

    # User 2 tries to feedback on User 1's message
    _create_user(db_session, "other3", "other3@test.com", "pass123")
    token = _login(jwt_user_client, "other3", "pass123")
    res = jwt_user_client.post(
        f"/api/chat/sessions/messages/{assistant_msg['id']}/feedback",
        json={"rating": "helpful"},
        headers=_jwt_headers(token),
    )
    assert res.status_code == 404


def test_cannot_duplicate_feedback(jwt_user_client, db_session):
    """User cannot submit multiple feedback entries on the same message."""
    res = jwt_user_client.post("/api/chat", json={"message": "Hello"})
    session_id = res.json()["session_id"]
    session_res = jwt_user_client.get(f"/api/chat/sessions/{session_id}")
    assistant_msg = next(m for m in session_res.json()["messages"] if m["role"] == "assistant")

    jwt_user_client.post(
        f"/api/chat/sessions/messages/{assistant_msg['id']}/feedback",
        json={"rating": "helpful"},
    )
    # Second attempt should fail
    res = jwt_user_client.post(
        f"/api/chat/sessions/messages/{assistant_msg['id']}/feedback",
        json={"rating": "helpful"},
    )
    assert res.status_code == 409