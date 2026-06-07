def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a RAG prompt from user query and retrieved chunks.
    Includes context instructions, citations format, and the user's question.
    """
    if not chunks:
        return f"You are a helpful assistant. The user asked: {query}\n\nNo relevant information was found in the knowledge base to answer this question."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file_name", "Unknown")
        content = chunk.get("content", "")[:500]  # Truncate for prompt size
        context_parts.append(f"[{i}] Source: {source}\n{content}")
    
    context = "\n\n".join(context_parts)
    
    return f"""You are a helpful assistant. Use the following information from the approved knowledge base to answer the user's question. If you cannot find sufficient information, say so clearly.

INFORMATION:
{context}

USER QUESTION: {query}

GUIDELINES:
- Only answer based on the information provided above
- Cite your sources using [1], [2], etc. to refer to the numbered sources
- If the information is insufficient, say "I could not find enough information..."
- Do not make up information not present in the provided sources
- Be concise and direct

ANSWER:"""