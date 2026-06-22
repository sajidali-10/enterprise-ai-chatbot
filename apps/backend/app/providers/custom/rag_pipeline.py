"""
Custom RAG Pipeline Adapter

Wraps the existing answer_generator.py generate_answer function.
Orchestrates: retrieve → optional rerank → LLM answer generation → citation attachment.
Phase 22 only supports the custom pipeline.
Future phases will add LangChain and LangGraph pipeline adapters.
"""

from app.rag.answer_generator import generate_answer_with_rag_audit
from app.providers.base import RagPipelineProvider


class CustomRagPipelineProvider(RagPipelineProvider):
    """
    Phase 22 active pipeline: delegates to answer_generator.generate_answer.

    The underlying generate_answer handles:
    - Hybrid retrieval (vector + keyword, with settings)
    - Optional reranking (when RETRIEVAL_RERANKER_TYPE != noop)
    - LLM answer generation with temperature=0 for deterministic output
    - Citation attachment (Phase 10.6)
    - Conversation context (Phase 20C)
    - Permission filtering via AuthContext (Phase 6)
    """

    def generate(
        self,
        query: str,
        auth=None,
        conversation_context: str = "",
        debug: bool = False,
    ):
        answer, citations, metadata = generate_answer_with_rag_audit(
            query=query,
            auth=auth,
            conversation_context=conversation_context,
            debug=debug,
        )
        return answer, citations, metadata