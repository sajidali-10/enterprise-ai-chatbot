"""
Prompt Builder for RAG and General Chat

Provides functions to build prompts for different chat modes with
appropriate instructions for the LLM.
"""


def build_rag_prompt(
    query: str,
    chunks: list[dict],
    include_citations: bool = True,
    conversation_context: str = "",
) -> str:
    """
    Build a RAG prompt from user query and retrieved chunks.
    Includes context instructions, citations format, and the user's question.

    Args:
        query: User's question.
        chunks: List of retrieved context chunks.
        include_citations: If True, instruct the model to cite sources.
        conversation_context: Optional formatted conversation context for follow-up questions.
    """
    if not chunks:
        return f"""You are a helpful support assistant. The user asked: {query}

I could not find enough relevant information in the knowledge base to answer this question.

If you don't know the answer, say so clearly and honestly. Do not make up information."""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file_name", "Unknown")
        content = chunk.get("content", "")[:500]  # Truncate for prompt size
        context_parts.append(f"[{i}] Source: {source}\n{content}")

    context = "\n\n".join(context_parts)

    citation_instruction = """- Cite your answer using [1], [2], etc. to reference the numbered sources
- Format citations like: "According to [1], ..." or "[1] states that ..."
- Place citations near the claim they support" """ if include_citations else ""

    # Build conversation context section if provided
    context_section = f"""

CONVERSATION CONTEXT:
{conversation_context}

For follow-up questions about "this answer", "the previous response", or similar references,
use the conversation context above to understand what is being asked about.
Base your answer on both the conversation context and the retrieved document information.
""" if conversation_context else ""

    # Phase 20C: Enhanced formatting rules for RAG answers
    formatting_rules = """
ANSWER FORMATTING RULES:
- For list/component questions (e.g., "What are the components of X?"): use numbered lists
- For summaries, comparisons, or overviews: use bullet points
- Keep each bullet/numbered item concise (1-3 sentences max)
- Avoid long paragraphs - break information into scannable chunks
- Do not repeat citations after every phrase - cite once per major point
- If answering "What are the components of X?", structure as:
  1. Component Name - brief explanation
  2. Component Name - brief explanation
  etc.
"""

    return f"""You are a helpful, concise support assistant. Answer the user's question using ONLY the information provided below.{context_section}

INFORMATION:
{context}

USER QUESTION: {query}

IMPORTANT GUIDELINES:
- Answer ONLY from the information provided above in the INFORMATION section
- Do NOT guess, infer, or make up information that is not directly in the sources
- If the information does not directly support a specific answer, say: "I could not find enough information in the provided sources to answer this question."
- Do NOT say "based on my knowledge" or "in general" - only use information from [1], [2], etc.
- Do NOT reference the sources as "the provided information" or "the context" - use [1], [2], etc.
{formatting_rules}
{citation_instruction}
- Focus on being helpful to a user seeking support
- If you're unsure, admit it rather than guessing

ANSWER:"""


def build_general_chat_prompt(query: str) -> str:
    """
    Build a prompt for General Chat mode (no RAG, no sources).

    Provides instructions for clean, concise, business-user friendly answers.

    Args:
        query: User's question.

    Returns:
        Formatted prompt for general chat response.
    """
    return f"""You are a helpful, professional AI assistant. Answer the user's question clearly and concisely.

USER QUESTION: {query}

IMPORTANT GUIDELINES:
- Answer clearly and concisely
- Use bullet points when helpful for list-style responses
- Avoid overly long technical explanations unless the user asks for detail
- Avoid raw markdown tables unless they significantly improve readability
- Do NOT mention sources, documents, or knowledge bases
- Write in a professional, business-friendly tone
- If you're unsure, admit it rather than guessing

ANSWER:"""


def build_knowledge_base_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a prompt for Knowledge Base mode (RAG with citations).

    This is an alias for build_rag_prompt with citations enabled.

    Args:
        query: User's question.
        chunks: List of retrieved context chunks.

    Returns:
        Formatted prompt for knowledge base response.
    """
    return build_rag_prompt(query, chunks, include_citations=True)


def build_strict_citation_prompt(
    query: str,
    chunks: list[dict],
    conversation_context: str = "",
) -> str:
    """
    Build a strict RAG prompt requiring explicit citations for every factual claim.

    This is used for retry when the initial LLM response omitted required citations.
    The prompt enforces:
    - Citation markers [1], [2], [3] on every factual statement
    - Insufficient-information fallback if no citation can be provided
    - Only information from retrieved source chunks (no hallucination)

    Args:
        query: User's question.
        chunks: List of retrieved context chunks.
        conversation_context: Optional formatted conversation context for follow-up questions.

    Returns:
        Formatted strict prompt requiring citations.
    """
    if not chunks:
        return f"""You are a helpful support assistant. The user asked: {query}

I could not find enough relevant information in the knowledge base to answer this question.

If you don't know the answer, say so clearly and honestly. Do not make up information."""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file_name", "Unknown")
        content = chunk.get("content", "")[:500]  # Truncate for prompt size
        context_parts.append(f"[{i}] Source: {source}\n{content}")

    context = "\n\n".join(context_parts)

    # Build conversation context section if provided
    context_section = f"""

CONVERSATION CONTEXT:
{conversation_context}

For follow-up questions about "this answer", "the previous response", or similar references,
use the conversation context above to understand what is being asked about.
Base your answer on both the conversation context and the retrieved document information.
""" if conversation_context else ""

    # Phase 20C: Enhanced formatting rules for RAG answers
    formatting_rules = """
ANSWER FORMATTING RULES:
- For list/component questions (e.g., "What are the components of X?"): use numbered lists
- For summaries, comparisons, or overviews: use bullet points
- Keep each bullet/numbered item concise (1-3 sentences max)
- Avoid long paragraphs - break information into scannable chunks
- Do not repeat citations after every phrase - cite once per major point
- If answering "What are the components of X?", structure as:
  1. Component Name - brief explanation
  2. Component Name - brief explanation
  etc.
"""

    return f"""You are a precise support assistant. Answer the user's question using ONLY the information provided below.{context_section}

INFORMATION:
{context}

USER QUESTION: {query}

STRICT CITATION REQUIREMENTS:
- You MUST cite every factual statement using [1], [2], [3] etc. to reference the numbered sources above
- Every bullet point in your answer MUST include at least one citation like [1] or [2]
- Format citations like: "According to [1], ..." or "[1] states that ..."
- Every distinct factual claim needs its own citation
- If you cannot support a statement with a citation from the sources above, do NOT make that statement

FALLBACK RULE:
- If the information does not directly support a specific answer, you MUST say: "I could not find enough information in the provided sources to answer this question."
- Do NOT guess or infer information not explicitly in the sources
- Do NOT say "based on my knowledge" or "in general"

RULES:
- Answer ONLY from the information provided above in the INFORMATION section
- Use ONLY the numbered citations [1], [2], etc. - do not use other citation formats
- Each bullet point must contain at least one [N] citation
{formatting_rules}
- If you're unsure, admit it rather than guessing

ANSWER:"""