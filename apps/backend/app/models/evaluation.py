"""
Evaluation models for tracking evaluation runs and results.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON, func
from sqlalchemy.orm import relationship
from app.db.base import Base


class EvaluationRun(Base):
    """Stores metadata about a single evaluation run."""
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    total_tests = Column(Integer, nullable=False, default=0)
    passed_tests = Column(Integer, nullable=False, default=0)
    failed_tests = Column(Integer, nullable=False, default=0)
    pass_percentage = Column(Float, nullable=False, default=0.0)
    average_latency_ms = Column(Float, nullable=True)
    average_top_score = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="pending")
    notes = Column(Text, nullable=True)

    results = relationship("EvaluationResult", back_populates="run", cascade="all, delete-orphan")


class EvaluationResult(Base):
    """Stores individual test case results within an evaluation run."""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False, index=True)
    test_case_id = Column(String, nullable=False, index=True)
    question = Column(Text, nullable=False)
    expected_source_file = Column(String, nullable=True)
    actual_source_files = Column(JSON, nullable=True)
    expected_keywords = Column(JSON, nullable=True)
    found_keywords = Column(JSON, nullable=True)
    missing_keywords = Column(JSON, nullable=True)
    forbidden_keywords = Column(JSON, nullable=True)
    forbidden_keywords_found = Column(JSON, nullable=True)
    should_answer = Column(Boolean, nullable=True)
    expected_fallback = Column(Boolean, nullable=True)
    actual_blocked = Column(Boolean, nullable=True)
    answer_preview = Column(Text, nullable=True)
    citation_count = Column(Integer, nullable=True)
    top_score = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    passed = Column(Boolean, nullable=False, default=False)
    failure_reasons = Column(JSON, nullable=True)
    extra_metadata = Column(JSON, nullable=True)

    run = relationship("EvaluationRun", back_populates="results")