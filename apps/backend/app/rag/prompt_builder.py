"""
Prompt Builder for RAG and General Chat

Provides functions to build prompts for different chat modes with
appropriate instructions for the LLM.

Phase 30E Hotfix v2 — Evidence-aware prompt construction:
- Strong evidence: standard prompt with full citation guidance
- Medium evidence: prompt includes a "Based on the retrieved sources..."
  caveat guidance so the LLM does not overstate confidence
- Weak evidence: caller should not call this function (fallback path used)

Phase 31A — LangSmith tracing:
- build_rag_prompt and build_strict_citation_prompt open a
  `prompt_building` span so the prompt construction is visible
  in the LangSmith trace hierarchy.
"""

# Evidence level constants. Kept as plain string literals so this module does
# not need to import the grounding enum (avoids circular imports).
EVIDENCE_STRONG = "strong"
EVIDENCE_MEDIUM = "medium"

# Phase 31A — import tracing helpers lazily to avoid impacting
# import-time side effects in the FastAPI startup path.
try:
    from app.services.langsmith_tracing import trace_span, redact_filenames, safe_chunk_content
except Exception:  # pragma: no cover - tracing never required
    def trace_span(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield None
        return _noop()
    def redact_filenames(value):  # type: ignore
        return list(value or [])
    def safe_chunk_content(value):  # type: ignore
        return value or ""


def _evidence_caveat_section(evidence_level: str) -> str:
    """
    Build the medium-evidence caveat section for the prompt.

    Phase 30E Hotfix v2: when evidence is medium, the LLM is explicitly told
    to answer cautiously and not overstate confidence.
    """
    if evidence_level == EVIDENCE_MEDIUM:
        return """

EVIDENCE QUALITY (CAUTION):
- The retrieved sources only PARTIALLY support a direct answer to this question.
- Answer cautiously, starting with "Based on the retrieved sources, ..." or similar phrasing.
- If a specific detail is not directly supported by the numbered sources, say so explicitly
  ("The retrieved documents indicate X, but they do not clearly state Y.").
- Do not invent or extrapolate details that are not in the numbered sources.
- Still cite every factual claim with [N] referencing the numbered sources.

"""
    return ""


def build_rag_prompt(
    query: str,
    chunks: list[dict],
    include_citations: bool = True,
    conversation_context: str = "",
    evidence_level: str = EVIDENCE_STRONG,
) -> str:
    """
    Build a RAG prompt from user query and retrieved chunks.
    Includes context instructions, citations format, and the user's question.

    Args:
        query: User's question.
        chunks: List of retrieved context chunks.
        include_citations: If True, instruct the model to cite sources.
        conversation_context: Optional formatted conversation context for follow-up questions.
        evidence_level: "strong" (default) or "medium". When "medium", a caveat
            section is included to encourage cautious answering.
    """
    # Phase 31A — open a `prompt_building` span. The span records the
    # number of chunks in context, whether a conversation context is
    # present, and the prompt length (only when LANGSMITH_LOG_FULL_PROMPT
    # is enabled). The prompt body itself is NEVER sent unless that flag
    # is explicitly on, in which case the operator has opted in.
    with trace_span(
        "prompt_building",
        metadata={
            "phase": "prompt_building",
            "prompt_type": "rag",
            "include_citations": include_citations,
            "evidence_level": evidence_level,
            "context_chunk_count": len(chunks or []),
            "has_conversation_context": bool(conversation_context),
        },
    ) as prompt_span:
        result = _build_rag_prompt_impl(
            query=query,
            chunks=chunks,
            include_citations=include_citations,
            conversation_context=conversation_context,
            evidence_level=evidence_level,
            prompt_span=prompt_span,
        )
        if prompt_span is not None:
            try:
                prompt_span.set_meta("prompt_length", len(result or ""))
                prompt_span.set_meta(
                    "source_file_names",
                    redact_filenames(list({c.get("source_file_name") for c in chunks if c.get("source_file_name")})),
                )
            except Exception:
                pass
        return result


def _build_rag_prompt_impl(
    query: str,
    chunks: list[dict],
    include_citations: bool = True,
    conversation_context: str = "",
    evidence_level: str = EVIDENCE_STRONG,
    prompt_span=None,
) -> str:
    if not chunks:
        return f"""You are a helpful support assistant. The user asked: {query}

I could not find enough information in the provided sources to answer this question. Please upload the relevant guide if available, or rephrase the question.

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

    # Phase 30E Hotfix v2: medium-evidence caveat is appended after context.
    evidence_section = _evidence_caveat_section(evidence_level)

    # Phase 30E: Cleaner answer structure for business users
    formatting_rules = """
ANSWER STRUCTURE (general):
- Lead with the direct answer first (one or two sentences).
- Then add supporting details, using bullets or short numbered lists only when they improve clarity.
- Avoid dumping raw source text. Paraphrase and summarize.
- Do not invent details that are not supported by the sources.
- If sources are partial or only weakly related, say so explicitly (e.g., "Based on the available documentation, ...").
- If the sources appear to conflict, mention the conflict briefly.
- If no reliable source supports an answer, fall back to: "I could not find enough information in the provided sources to answer this question. Please upload the relevant guide if available, or rephrase the question."

FORMATTING RULES:
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
{evidence_section}
IMPORTANT GUIDELINES:
- Answer ONLY from the information provided above in the INFORMATION section
- Do NOT guess, infer, or make up information that is not directly in the sources
- If the information does not directly support a specific answer, say: "I could not find enough information in the provided sources to answer this question. Please upload the relevant guide if available, or rephrase the question."
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


def build_knowledge_base_prompt(
    query: str,
    chunks: list[dict],
    evidence_level: str = EVIDENCE_STRONG,
) -> str:
    """
    Build a prompt for Knowledge Base mode (RAG with citations).

    This is an alias for build_rag_prompt with citations enabled.

    Args:
        query: User's question.
        chunks: List of retrieved context chunks.
        evidence_level: "strong" (default) or "medium". See build_rag_prompt.

    Returns:
        Formatted prompt for knowledge base response.
    """
    return build_rag_prompt(
        query,
        chunks,
        include_citations=True,
        evidence_level=evidence_level,
    )


def build_strict_citation_prompt(
    query: str,
    chunks: list[dict],
    conversation_context: str = "",
    evidence_level: str = EVIDENCE_STRONG,
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
        evidence_level: "strong" (default) or "medium". When "medium", the prompt
            additionally tells the model to caveat with "Based on the retrieved sources, ...".

    Returns:
        Formatted strict prompt requiring citations.
    """
    # Phase 31A — separate span for the strict/retry prompt path so the
    # LangSmith hierarchy clearly shows when a retry prompt was used.
    with trace_span(
        "prompt_building",
        metadata={
            "phase": "prompt_building",
            "prompt_type": "strict_citation",
            "evidence_level": evidence_level,
            "context_chunk_count": len(chunks or []),
            "has_conversation_context": bool(conversation_context),
        },
    ) as prompt_span:
        result = _build_strict_citation_prompt_impl(
            query=query,
            chunks=chunks,
            conversation_context=conversation_context,
            evidence_level=evidence_level,
        )
        if prompt_span is not None:
            try:
                prompt_span.set_meta("prompt_length", len(result or ""))
                prompt_span.set_meta(
                    "source_file_names",
                    redact_filenames(list({c.get("source_file_name") for c in chunks if c.get("source_file_name")})),
                )
            except Exception:
                pass
        return result


def _build_strict_citation_prompt_impl(
    query: str,
    chunks: list[dict],
    conversation_context: str = "",
    evidence_level: str = EVIDENCE_STRONG,
) -> str:
    if not chunks:
        return f"""You are a helpful support assistant. The user asked: {query}

I could not find enough information in the provided sources to answer this question. Please upload the relevant guide if available, or rephrase the question.

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

    # Phase 30E Hotfix v2: medium-evidence caveat appended after the question.
    evidence_section = _evidence_caveat_section(evidence_level)

    # Phase 30E: Strict citation prompt with cleaner answer structure
    formatting_rules = """
ANSWER STRUCTURE (general):
- Lead with the direct answer first (one or two sentences).
- Then add supporting details using bullets or short numbered lists only when they improve clarity.
- Avoid dumping raw source text. Paraphrase and summarize.
- If sources conflict, mention the conflict briefly.
- If sources are weak, say so explicitly (e.g., "Based on the available documentation, ...").

FORMATTING RULES:
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
{evidence_section}
STRICT CITATION REQUIREMENTS:
- You MUST cite every factual statement using [1], [2], [3] etc. to reference the numbered sources above
- Every bullet point in your answer MUST include at least one citation like [1] or [2]
- Format citations like: "According to [1], ..." or "[1] states that ..."
- Every distinct factual claim needs its own citation
- If you cannot support a statement with a citation from the sources above, do NOT make that statement

FALLBACK RULE:
- If the information does not directly support a specific answer, you MUST say: "I could not find enough information in the provided sources to answer this question. Please upload the relevant guide if available, or rephrase the question."
- Do NOT guess or infer information not explicitly in the sources
- Do NOT say "based on my knowledge" or "in general"

RULES:
- Answer ONLY from the information provided above in the INFORMATION section
- Use ONLY the numbered citations [1], [2], etc. - do not use other citation formats
- Each bullet point must contain at least one [N] citation
{formatting_rules}
- If you're unsure, admit it rather than guessing

ANSWER:"""