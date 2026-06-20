"""
Conversation Context Service

Phase 20C — Limited Conversation Context (refined).

Provides recent conversation history for chat sessions.
Used to give the LLM limited context when continuing a conversation.

Security:
- Only fetches messages for the authenticated user
- Excludes archived sessions
- Never includes metadata, citations, or secrets
- Limited by message count and character count
"""

from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from app.models.chat_session import ChatSession, ChatMessage
from app.models.chat_session import ChatSessionMode as DBChatSessionMode, MessageRole as DBMessageRole


# ==============================================================================
# Configuration
# ==============================================================================

# Tighter limits to prevent verbose context from polluting retrieval
MAX_RECENT_MESSAGES = 6  # Maximum number of recent messages to include
MAX_CONTEXT_CHARACTERS = 3500  # Maximum characters in context
MAX_ASSISTANT_CONTENT_CHARS = 1200  # Truncate long assistant answers


# ==============================================================================
# Data Classes
# ==============================================================================

class ConversationMessage:
    """A simplified message for context."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# ==============================================================================
# Context Fetching
# ==============================================================================

def get_recent_conversation_context(
    db: Session,
    session_id: int,
    user_id: int,
    max_messages: int = MAX_RECENT_MESSAGES,
    max_characters: int = MAX_CONTEXT_CHARACTERS,
) -> Tuple[List[ConversationMessage], bool]:
    """
    Fetch recent conversation context for a chat session.

    Security:
    - Only returns messages owned by the authenticated user
    - Excludes archived sessions
    - Does not include metadata, citations, or secrets

    Args:
        db: Database session
        session_id: ID of the chat session
        user_id: ID of the authenticated user (ownership check)
        max_messages: Maximum number of recent messages to return
        max_characters: Maximum total characters in context

    Returns:
        Tuple of (messages, has_context) where messages is a list of
        ConversationMessage objects and has_context indicates if
        there were older messages beyond what was returned.
    """
    # Fetch session and verify ownership
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id,
    ).first()

    if not session:
        return [], False

    # Exclude archived sessions
    if session.archived_at:
        return [], False

    # Fetch recent messages (excluding the current user message that hasn't been stored yet)
    # We fetch max_messages + 1 to check if there are more messages
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id,
        ChatMessage.user_id == user_id,
    ).order_by(
        ChatMessage.created_at.desc()
    ).limit(max_messages + 1).all()

    if not messages:
        return [], False

    # Reverse to get chronological order
    messages = list(reversed(messages))

    # Check if there are more messages beyond what we fetched
    has_more = len(messages) > max_messages

    # Take only the most recent max_messages
    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    # Convert to simplified format, excluding internal metadata
    context_messages = []
    for msg in messages:
        role = "user" if msg.role == DBMessageRole.USER else "assistant"
        content = msg.content
        # Truncate long assistant messages to prevent verbose context
        if role == "assistant" and len(content) > MAX_ASSISTANT_CONTENT_CHARS:
            content = content[:MAX_ASSISTANT_CONTENT_CHARS] + "..."
        context_messages.append(ConversationMessage(role=role, content=content))

    # Truncate if too long
    if max_characters > 0:
        context_messages = _truncate_context(context_messages, max_characters)

    return context_messages, has_more


def format_conversation_context_for_prompt(
    messages: List[ConversationMessage],
) -> str:
    """
    Format conversation messages for inclusion in an LLM prompt.

    Uses a structured format that helps the LLM understand references
    like "this", "that", "second point" without treating conversation
    history as document evidence.

    Args:
        messages: List of ConversationMessage objects

    Returns:
        Formatted string for prompt inclusion
    """
    if not messages:
        return ""

    # Build structured context with clear labels
    parts = []
    for msg in messages:
        if msg.role == "user":
            parts.append(f"Previous user question: {msg.content}")
        else:
            # Truncate assistant content more aggressively for prompt
            content = msg.content
            if len(content) > MAX_ASSISTANT_CONTENT_CHARS:
                content = content[:MAX_ASSISTANT_CONTENT_CHARS] + "..."
            parts.append(f"Previous assistant answer: {content}")

    formatted = "\n\n".join(parts)

    return f"""{formatted}

---

Rules for using conversation context:
- Use this context to resolve references like "this", "that", "the previous answer", or "second point"
- Do NOT treat the conversation above as document evidence — use retrieved documents for factual claims
- If the conversation context conflicts with retrieved documents, prefer the retrieved documents
- For summaries/checklists based on a previous answer, be concise (3-6 bullets) and do not repeat the full previous answer"""


def _truncate_context(
    messages: List[ConversationMessage],
    max_characters: int,
) -> List[ConversationMessage]:
    """
    Truncate context to fit within character limit.

    Truncates from the oldest messages first, keeping the most recent.
    """
    total_chars = sum(len(msg.content) for msg in messages)

    if total_chars <= max_characters:
        return messages

    # Start from the most recent and work backwards
    truncated = []
    current_chars = 0

    for msg in reversed(messages):
        # Estimate overhead for formatting
        overhead = 30 if msg.role == "user" else 35
        if current_chars + len(msg.content) + overhead > max_characters:
            break
        truncated.insert(0, msg)
        current_chars += len(msg.content) + overhead

    return truncated


# ==============================================================================
# Context Helpers
# ==============================================================================

def can_show_contextual_suggestions(
    db: Session,
    session_id: int,
    user_id: int,
) -> bool:
    """
    Check if contextual suggestions can be shown for a session.

    Contextual suggestions (like "Summarize this for management") are only
    shown when there's recent assistant context to reference.

    Returns:
        True if there are at least 2 messages (1 user + 1 assistant) in context
    """
    messages, _ = get_recent_conversation_context(
        db=db,
        session_id=session_id,
        user_id=user_id,
        max_messages=4,  # Just need enough to verify context exists
    )

    # Need at least 1 user and 1 assistant message for context
    has_user = any(m.role == "user" for m in messages)
    has_assistant = any(m.role == "assistant" for m in messages)

    return has_user and has_assistant