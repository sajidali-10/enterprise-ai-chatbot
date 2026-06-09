def build_rag_prompt(query: str, chunks: list[dict], include_citations: bool = True) -> str:
    """
    Build a RAG prompt from user query and retrieved chunks.
    Includes context instructions, citations format, and the user's question.
    
    Args:
        query: User's question.
        chunks: List of retrieved context chunks.
        include_citations: If True, instruct the model to cite sources.
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
- Format citations like: "According to [1], ..." or "[1] states that ..." """ if include_citations else ""
    
    return f"""You are a helpful, concise support assistant. Answer the user's question using ONLY the information provided below.

INFORMATION:
{context}

USER QUESTION: {query}

GUIDELINES:
- Answer ONLY from the information provided above
- If the information is insufficient or doesn't contain the answer, say: "I could not find enough information in the provided sources to answer this question."
- Do NOT make up, speculate, or infer information not present in the sources
- Do NOT reference the sources as "the provided information" or "the context" - use [1], [2], etc.
{citation_instruction}
- Be concise and direct in your answer
- Focus on being helpful to a user seeking support

ANSWER:"""