from typing import List

def format_citations(chunks: list[dict]) -> List[dict]:
    """
    Format retrieved chunks as citations for the response.
    Returns list of citation dicts with index, source, and content snippet.
    """
    citations = []
    for i, chunk in enumerate(chunks, 1):
        citations.append({
            "index": i,
            "source_file_name": chunk.get("source_file_name", "Unknown"),
            "content_snippet": (chunk.get("content", "")[:200] + "...") if len(chunk.get("content", "")) > 200 else chunk.get("content", ""),
            "relevance_score": chunk.get("score"),
        })
    return citations