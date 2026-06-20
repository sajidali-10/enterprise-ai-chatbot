"""
Suggestion Generator Service

Phase 20B — Suggested Follow-up Buttons (UX Fix).
Phase 20C: Add typed suggestions to prevent bad fallback loops.
Phase 20C (refined): Enable contextual_action suggestions when conversation context exists.

Suggestion types:
- question: A standalone question that can be sent to the LLM directly
- frontend_action: Handled by frontend (scroll, navigate, expand)
- contextual_action: Depends on previous answer — enabled when conversation context exists

No LLM call is made — all suggestions are pre-defined and context-aware.
"""

from typing import List, Optional, Literal
from dataclasses import dataclass


# ==============================================================================
# Suggestion Type Definitions
# ==============================================================================

SuggestionType = Literal["question", "frontend_action", "contextual_action"]


@dataclass
class Suggestion:
    """A typed suggestion with label, prompt, and type."""
    label: str
    prompt: str  # What to send to LLM if type is 'question' or 'contextual_action'
    type: SuggestionType


# ==============================================================================
# Suggestion Templates by Context
# ==============================================================================

# Contextual action suggestions - shown when conversation context exists
# Phase 20C (refined): More focused, concise prompts for contextual actions
CONTEXTUAL_ACTIONS: List[Suggestion] = [
    Suggestion(
        label="Summarize for management",
        prompt="Summarize the previous answer in 3-5 concise bullet points for management. Use executive tone. Do not repeat the full previous answer unless explicitly asked. Do not include long citations.",
        type="contextual_action",
    ),
    Suggestion(
        label="Create action checklist",
        prompt="Create a concise action checklist (4-7 checkbox-style items) based on the previous answer. Keep each item to 1 line. Be specific and actionable.",
        type="contextual_action",
    ),
    Suggestion(
        label="What evidence supports this?",
        prompt="List the key evidence from the source documents that supports the previous answer. Do not rerun the full answer. Keep it to 3-5 bullet points with source references.",
        type="contextual_action",
    ),
    Suggestion(
        label="What risks or gaps exist?",
        prompt="Based on the previous answer and sources, list any missing information, uncertainties, or gaps. If there is not enough source info, say what specific information is missing. Keep it concise.",
        type="contextual_action",
    ),
]

# RAG answers WITH citations
# Phase 20C (refined): Focused suggestions for RAG with citations
RAG_WITH_CITATIONS: List[Suggestion] = [
    Suggestion(
        label="Show cited sources",
        prompt="Scroll to and display the cited sources section",
        type="frontend_action",
    ),
    Suggestion(
        label="Open Documents",
        prompt="Navigate to the Documents page",
        type="frontend_action",
    ),
    Suggestion(
        label="Summarize for management",
        prompt="Summarize the previous answer in 3-5 concise bullet points for management. Use executive tone. Do not repeat the full previous answer.",
        type="contextual_action",
    ),
    Suggestion(
        label="Create action checklist",
        prompt="Create a concise action checklist (4-7 checkbox-style items) based on the previous answer. Keep each item to 1 line.",
        type="contextual_action",
    ),
]

# RAG fallback / no-context / unsupported answers
# Phase 20C (refined): Limited suggestions - no contextual actions for fallback
RAG_FALLBACK: List[Suggestion] = [
    Suggestion(
        label="Rephrase the question",
        prompt="",  # frontend_action, handled specially by frontend
        type="frontend_action",
    ),
    Suggestion(
        label="Upload a document",
        prompt="",  # frontend_action, no prompt needed
        type="frontend_action",
    ),
    Suggestion(
        label="Open Documents",
        prompt="Navigate to the Documents page",
        type="frontend_action",
    ),
]

# General chat suggestions
# Phase 20C (refined): Concise suggestions with no overlap
GENERAL_CHAT: List[Suggestion] = [
    Suggestion(
        label="Explain in simpler terms",
        prompt="Explain this in simpler terms with a brief example if helpful",
        type="question",
    ),
    Suggestion(
        label="Create step-by-step instructions",
        prompt="Create clear step-by-step instructions based on this",
        type="question",
    ),
    Suggestion(
        label="Turn into an email",
        prompt="Turn this into a professional email",
        type="question",
    ),
    Suggestion(
        label="Create a checklist",
        prompt="Create a concise checklist (4-7 items) based on this",
        type="question",
    ),
    Suggestion(
        label="What should I do next?",
        prompt="What specific next steps would you recommend based on this?",
        type="question",
    ),
]


# ==============================================================================
# Main Generator Function
# ==============================================================================

def generate_suggestions(
    mode: str,
    has_citations: bool = False,
    is_fallback: bool = False,
    citations: Optional[List] = None,
    has_conversation_context: bool = False,
) -> List[dict]:
    """
    Generate 3-5 typed suggested follow-ups based on context.

    Returns 'question', 'frontend_action', and 'contextual_action' types.
    'contextual_action' suggestions are shown when:
    - has_conversation_context is True (Phase 20C)
    - response is not fallback/no-context

    Phase 20C (refined): When context exists, contextual actions appear alongside
    regular suggestions, making follow-ups like "Summarize this for management" available.

    Args:
        mode: Chat mode ("general_chat", "knowledge_base", "debug")
        has_citations: Whether the response includes RAG citations
        is_fallback: Whether the response is a fallback/no-context answer
        citations: Optional list of citations (used for safety checks)
        has_conversation_context: Whether there's recent conversation context (Phase 20C)

    Returns:
        List of suggestion dicts with: label, prompt, type
    """
    # Normalize mode
    mode = mode.lower().strip() if mode else "general_chat"
    is_rag_mode = mode in ("knowledge_base", "debug", "rag")

    # Choose base suggestion set
    if is_rag_mode:
        if is_fallback or not has_citations:
            suggestions = RAG_FALLBACK
        else:
            suggestions = RAG_WITH_CITATIONS
    else:
        suggestions = GENERAL_CHAT

    # Phase 20C: Add contextual actions when conversation context exists AND not a fallback
    # These provide clean, concise follow-ups referencing "the previous answer"
    if has_conversation_context and not is_fallback:
        # Prepend contextual actions to give them priority
        suggestions = list(CONTEXTUAL_ACTIONS) + suggestions

    # Filter suggestions based on Phase 20C rules:
    # contextual_action only shown when there's conversation context AND not a fallback
    safe_suggestions = [
        s for s in suggestions
        if s.type != "contextual_action" or (has_conversation_context and not is_fallback)
    ][:6]  # Allow a few more since contextual ones are prioritized

    # Convert to dict format for JSON serialization
    return [
        {
            "label": s.label,
            "prompt": s.prompt,
            "type": s.type,
        }
        for s in safe_suggestions
    ]


def is_fallback_response(answer: str, citations: Optional[List] = None) -> bool:
    """
    Detect if an answer is a fallback/no-context response.

    A fallback typically contains phrases like:
    - "I don't have", "no documents", "can't find"
    - "upload", "provide", "add documents"
    - "no relevant", "not found in"

    Args:
        answer: The assistant's response text
        citations: Optional list of citations

    Returns:
        True if the response appears to be a fallback/no-context answer
    """
    if not answer:
        return True

    answer_lower = answer.lower()

    # Check for fallback indicators
    fallback_phrases = [
        "i don't have",
        "i don't know",
        "no documents",
        "no relevant",
        "can't find",
        "couldn't find",
        "not found in",
        "not available in",
        "no information in",
        "upload",
        "add your documents",
        "provide the document",
        "no context",
        "without more information",
    ]

    for phrase in fallback_phrases:
        if phrase in answer_lower:
            return True

    # No citations when in RAG mode could indicate fallback
    if citations is not None and len(citations) == 0:
        return True

    return False