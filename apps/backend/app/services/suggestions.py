"""
Suggestion Generator Service

Phase 20B — Suggested Follow-up Buttons (UX Fix).
Phase 20C: Add typed suggestions to prevent bad fallback loops.

Suggestion types:
- question: A standalone question that can be sent to the LLM directly
- frontend_action: Handled by frontend (scroll, navigate, expand)
- contextual_action: Depends on previous answer — HIDDEN until Phase 20C

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
    prompt: str  # What to send to LLM if type is 'question'
    type: SuggestionType


# ==============================================================================
# Suggestion Templates by Context
# ==============================================================================

# RAG answers WITH citations
# - frontend_action: scroll to sources, navigate to documents
# - question: standalone questions about the documents
# - contextual_action: depends on previous answer — HIDDEN until Phase 20C
RAG_WITH_CITATIONS: List[Suggestion] = [
    Suggestion(
        label="Show cited sources",
        prompt="Show the cited source documents",
        type="frontend_action",
    ),
    Suggestion(
        label="Show documents I can access",
        prompt="Show documents I can access",
        type="frontend_action",
    ),
    Suggestion(
        label="What topics are covered?",
        prompt="What topics are covered in the uploaded documents?",
        type="question",
    ),
    Suggestion(
        label="Ask about another document",
        prompt="What information is in another document?",
        type="question",
    ),
    # CONTEXTUAL — hidden until Phase 20C:
    # Suggestion(
    #     label="Summarize for management",
    #     prompt="Summarize the previous answer for management",
    #     type="contextual_action",
    # ),
]

# RAG fallback / no-context / unsupported answers
# Only safe standalone actions — no contextual suggestions
RAG_FALLBACK: List[Suggestion] = [
    Suggestion(
        label="Upload a document",
        prompt="",  # frontend_action, no prompt needed
        type="frontend_action",
    ),
    Suggestion(
        label="Rephrase the question",
        prompt="",  # Will be handled specially by frontend
        type="frontend_action",
    ),
    Suggestion(
        label="Show my documents",
        prompt="Show documents I can access",
        type="frontend_action",
    ),
    Suggestion(
        label="Ask about another topic",
        prompt="What topics are covered in the uploaded documents?",
        type="question",
    ),
]

# General chat suggestions
# All are standalone questions — no context dependency
GENERAL_CHAT: List[Suggestion] = [
    Suggestion(
        label="Explain in simpler terms",
        prompt="Explain this in simpler terms",
        type="question",
    ),
    Suggestion(
        label="Step-by-step instructions",
        prompt="Create step-by-step instructions based on this",
        type="question",
    ),
    Suggestion(
        label="Turn into an email",
        prompt="Turn this into a professional email",
        type="question",
    ),
    Suggestion(
        label="Create a checklist",
        prompt="Create a checklist based on this",
        type="question",
    ),
    Suggestion(
        label="What should I do next?",
        prompt="What should I do next based on this?",
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
) -> List[dict]:
    """
    Generate 3-5 typed suggested follow-ups based on context.

    Only returns 'question' and 'frontend_action' types.
    'contextual_action' suggestions are hidden until Phase 20C.

    Args:
        mode: Chat mode ("general_chat", "knowledge_base", "debug")
        has_citations: Whether the response includes RAG citations
        is_fallback: Whether the response is a fallback/no-context answer
        citations: Optional list of citations (used for safety checks)

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

    # Filter out contextual_action types (hidden until Phase 20C)
    # Also limit to 5 suggestions
    safe_suggestions = [
        s for s in suggestions
        if s.type != "contextual_action"
    ][:5]

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