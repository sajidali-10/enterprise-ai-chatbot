"""
Answer Generation for RAG Chat

Provides functions to generate answers with or without RAG.
"""

from app.rag.retriever import retrieve_chunks, retrieve_chunks_with_settings
from app.rag.prompt_builder import build_rag_prompt
from app.rag.citations import format_citations
from app.services.llm import get_llm_provider
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole


def generate_answer_with_rag(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    use_hybrid: bool = True,
    debug: bool = False,
) -> tuple[str, list[dict], dict]:
    """
    Full RAG pipeline: retrieve chunks, build prompt, call LLM, return answer + citations.
    
    Args:
        query: User's question.
        top_k: Number of chunks to retrieve (used when use_hybrid=False).
        score_threshold: Minimum score threshold (used when use_hybrid=False).
        use_hybrid: If True, use hybrid retrieval with all Phase 5 features.
                    If False, use original vector-only retrieval.
        debug: If True, return additional debug metadata about retrieval.
        
    Returns:
        Tuple of (answer, citations, metadata).
        Metadata is empty when debug=False.
    """
    if use_hybrid:
        chunks, retrieval_metadata = retrieve_chunks_with_settings(query, debug=debug)
    else:
        chunks = retrieve_chunks(query=query, limit=top_k, score_threshold=score_threshold)
        retrieval_metadata = {}
    
    if not chunks:
        return (
            "I could not find enough information in the approved knowledge base to answer confidently.",
            [],
            retrieval_metadata,
        )
    
    prompt = build_rag_prompt(query, chunks)
    
    provider = get_llm_provider()
    llm_request = ChatRequest(message=prompt)
    llm_response = provider.chat(llm_request)
    
    answer = llm_response.message
    citations = format_citations(chunks)
    
    return answer, citations, retrieval_metadata


def generate_answer_without_rag(query: str) -> str:
    """Normal chat without RAG."""
    provider = get_llm_provider()
    llm_request = ChatRequest(message=query)
    llm_response = provider.chat(llm_request)
    return llm_response.message