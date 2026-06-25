"""
Observability logging service.

Logs chat interactions for admin monitoring and evaluation.
Does not store secrets, API keys, or full document chunks.
"""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.observability import ChatObservation
from app.rag.citations import group_citations_by_source

logger = logging.getLogger(__name__)


def log_chat_observation(
    mode: str,
    question: str,
    answer: str,
    auth_context: Optional[Dict] = None,
    citations: Optional[List[Dict]] = None,
    grouped_sources: Optional[List] = None,
    metadata: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    blocked: bool = False,
    block_reason: Optional[str] = None,
    grounding_reason: Optional[str] = None,
) -> Optional[int]:
    """
    Log a chat interaction to the observability database.
    
    Returns the observation ID, or None if logging failed.
    
    Security: Does NOT log:
    - API keys or secrets
    - Full prompts (stores question only)
    - Full document chunks (stores preview and source names only)
    - Raw vector data
    """
    db = SessionLocal()
    try:
        # Determine if answer was blocked
        fallback_phrases = [
            "i don't have",
            "i cannot find",
            "i don't know",
            "not enough information",
            "no relevant documents",
            "don't have access",
            "cannot answer based on",
            "insufficient information"
        ]
        if not blocked:
            blocked = any(phrase in answer.lower() for phrase in fallback_phrases)
        
        # Get source file names from grouped sources.
        # grouped_sources can be either dicts (from group_citations_by_source) or
        # objects with attributes; handle both formats for safety.
        source_files = None
        if grouped_sources:
            files = []
            for gs in grouped_sources:
                if isinstance(gs, dict):
                    name = gs.get("source_file_name")
                else:
                    name = getattr(gs, "source_file_name", None)
                if name:
                    files.append(name)
            source_files = files or None
        elif citations:
            source_files = list(set(c.get("source_file_name") for c in citations if c.get("source_file_name")))

        # Get top score from metadata or grouped sources
        top_score = None
        if metadata and "top_score" in metadata:
            top_score = metadata.get("top_score")
        elif grouped_sources and len(grouped_sources) > 0:
            scores = []
            for gs in grouped_sources:
                if isinstance(gs, dict):
                    s = gs.get("highest_score")
                else:
                    s = getattr(gs, "highest_score", None)
                if s is not None:
                    scores.append(s)
            top_score = max(scores) if scores else None
        
        # Get citation count
        citation_count = len(citations) if citations else 0
        
        # Extract retrieval metadata summary (no full chunks)
        retrieval_metadata = None
        citation_repair_metadata = None
        if metadata:
            retrieval_metadata = {
                "retrieval_method": metadata.get("retrieval_method"),
                "relevance_threshold": metadata.get("relevance_threshold"),
                "grounding_decision": metadata.get("grounding_decision"),
                "chunks_retrieved": metadata.get("chunks_retrieved"),
                "top_score": metadata.get("top_score"),
            }
            if metadata.get("citation_repair"):
                citation_repair_metadata = {
                    "citations_added": metadata.get("citation_repair", {}).get("citations_added", 0),
                    "citations_removed": metadata.get("citation_repair", {}).get("citations_removed", 0),
                    "repair_type": metadata.get("citation_repair", {}).get("repair_type"),
                }
        
        # Extract user info from auth context
        user_id = auth_context.get("user_id") if auth_context else None
        username = auth_context.get("username") if auth_context else None
        user_role = auth_context.get("role") if auth_context else None
        
        # Create observation record
        observation = ChatObservation(
            user_id=user_id,
            username=username,
            user_role=user_role,
            mode=mode,
            question=question,
            answer_preview=answer[:500] if answer else None,  # First 500 chars only
            answer_length=len(answer) if answer else 0,
            source_files=source_files,
            citation_count=citation_count,
            top_score=top_score,
            blocked=blocked,
            block_reason=block_reason,
            grounding_reason=grounding_reason,
            latency_ms=latency_ms,
            provider=provider,
            model=model,
            retrieval_metadata=retrieval_metadata,
            citation_repair_metadata=citation_repair_metadata,
        )
        
        db.add(observation)
        db.commit()
        db.refresh(observation)
        
        return observation.id
        
    except Exception as e:
        logger.error(f"Failed to log chat observation: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def update_feedback(
    observation_id: int,
    rating: str,
    reason: Optional[str] = None,
    comment: Optional[str] = None,
) -> bool:
    """Update feedback for an observation."""
    db = SessionLocal()
    try:
        observation = db.query(ChatObservation).filter(ChatObservation.id == observation_id).first()
        
        if not observation:
            return False
        
        observation.feedback_rating = rating
        observation.feedback_reason = reason
        observation.feedback_comment = comment
        
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to update feedback: {e}")
        db.rollback()
        return False
    finally:
        db.close()