"""
Conversation Context Service

Phase 20C — Limited Conversation Context.

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

MAX_RECENT_MESSAGES = 8  # Maximum number of recent messages to include
MAX_CONTEXT_CHARACTERS = 6000  # Maximum characters in context


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

    def to_prompt_format(self) -> str:
        prefix = "user" if self.role == "user" else "assistant"
        return f"[{prefix}] {self.content}"


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
        context_messages.append(ConversationMessage(role=role, content=msg.content))

    # Truncate if too long
    if max_characters > 0:
        context_messages = _truncate_context(context_messages, max_characters)

    return context_messages, has_more


def format_conversation_context_for_prompt(
    messages: List[ConversationMessage],
) -> str:
    """
    Format conversation messages for inclusion in an LLM prompt.

    Args:
        messages: List of ConversationMessage objects

    Returns:
        Formatted string for prompt inclusion
    """
    if not messages:
        return ""

    formatted = "\n".join(msg.to_prompt_format() for msg in messages)
    return f"\n\nRecent conversation context:\n{formatted}\n\nRules:\n- The conversation above provides context for follow-up questions.\n- Retrieved document context (if available) is authoritative for factual answers.\n- If the conversation conflicts with retrieved documents, use the retrieved documents.\n"


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
        if current_chars + len(msg.content) + 50 > max_characters:  # +50 for formatting
            break
        truncated.insert(0, msg)
        current_chars += len(msg.content) + 50  # Approximate formatting overhead

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