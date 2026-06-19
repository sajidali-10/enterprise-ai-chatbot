"""
Suggestion Generator Service

Phase 20B — Suggested Follow-up Buttons.

Rule-based suggestion generator that creates 3-5 safe follow-up prompts
based on chat mode, citations presence, and response characteristics.

No LLM call is made — all suggestions are pre-defined and context-aware.
"""

from typing import List, Optional


# ==============================================================================
# Suggestion Templates by Context
# ==============================================================================

# RAG answers WITH citations
RAG_WITH_CITATIONS = [
    "Show the cited source documents",
    "Summarize this answer for management",
    "Create an action checklist",
    "What evidence supports this answer?",
    "What risks or gaps are missing?",
]

# RAG fallback / no-context / unsupported answers
RAG_FALLBACK = [
    "Upload a related document",
    "Rephrase the question",
    "Ask about available documents",
    "Show documents I can access",
    "Explain what information is missing",
]

# General chat suggestions
GENERAL_CHAT = [
    "Explain this in simpler terms",
    "Create step-by-step instructions",
    "Turn this into an email",
    "Create a checklist",
    "What should I do next?",
]


# ==============================================================================
# Main Generator Function
# ==============================================================================

def generate_suggestions(
    mode: str,
    has_citations: bool = False,
    is_fallback: bool = False,
    citations: Optional[List] = None,
) -> List[str]:
    """
    Generate 3-5 suggested follow-up prompts based on context.

    Args:
        mode: Chat mode ("general_chat", "knowledge_base", "debug")
        has_citations: Whether the response includes RAG citations
        is_fallback: Whether the response is a fallback/no-context answer
        citations: Optional list of citations (used for safety checks)

    Returns:
        List of 3-5 suggestion strings (max 80 chars each)
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

    # Return a copy (3-5 suggestions)
    return list(suggestions[:5])


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