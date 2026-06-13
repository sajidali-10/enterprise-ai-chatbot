"""
Observability model for tracking live chat interactions.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, JSON, func
from app.db.base import Base


class ChatObservation(Base):
    """
    Stores observability data for each chat interaction.
    
    Security considerations:
    - Does NOT store API keys, secrets, or full prompts
    - Stores answer preview (first 500 chars) not full answers
    - Stores source file names, not full document chunks
    - Stores metadata and scores, not raw vectors
    """
    __tablename__ = "chat_observations"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    
    # User info (nullable for anonymous users)
    user_id = Column(Integer, nullable=True)
    username = Column(String, nullable=True)
    user_role = Column(String, nullable=True)
    
    # Interaction details
    mode = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer_preview = Column(Text, nullable=True)  # First 500 chars only
    answer_length = Column(Integer, nullable=True)
    
    # Source information
    source_files = Column(JSON, nullable=True)  # List of source file names
    
    # Quality metrics
    citation_count = Column(Integer, nullable=True)
    top_score = Column(Float, nullable=True)  # Top retrieval score
    blocked = Column(Boolean, nullable=False, default=False)
    block_reason = Column(String, nullable=True)
    grounding_reason = Column(String, nullable=True)
    
    # Performance
    latency_ms = Column(Float, nullable=True)
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    
    # Detailed metadata (no secrets, no full chunks)
    retrieval_metadata = Column(JSON, nullable=True)  # Debug info summary
    citation_repair_metadata = Column(JSON, nullable=True)  # Citation repair summary
    
    # Feedback
    feedback_rating = Column(String, nullable=True)  # "helpful" or "not_helpful"
    feedback_reason = Column(String, nullable=True)
    feedback_comment = Column(Text, nullable=True)