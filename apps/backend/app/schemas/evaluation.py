"""
Pydantic schemas for evaluation API responses.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


class EvaluationRunSummary(BaseModel):
    """Summary of an evaluation run."""
    run_id: int
    created_at: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_percentage: float
    average_latency_ms: Optional[float]
    average_top_score: Optional[float]
    status: str
    notes: Optional[str] = None


class EvaluationResultDetail(BaseModel):
    """Detailed result for a single test case."""
    test_case_id: str
    question: str
    expected_source_file: Optional[str]
    actual_source_files: Optional[List[str]]
    expected_keywords: Optional[List[str]]
    found_keywords: Optional[List[str]]
    missing_keywords: Optional[List[str]]
    forbidden_keywords: Optional[List[str]]
    forbidden_keywords_found: Optional[List[str]]
    should_answer: bool
    expected_fallback: bool
    actual_blocked: bool
    answer_preview: Optional[str]
    citation_count: int
    top_score: Optional[float]
    latency_ms: float
    passed: bool
    failure_reasons: Optional[List[str]]
    grouped_sources: Optional[List[dict]] = None


class EvaluationRunDetail(BaseModel):
    """Full evaluation run with all test results."""
    summary: EvaluationRunSummary
    results: List[EvaluationResultDetail]
    failed_results: List[EvaluationResultDetail]


class EvaluationLatestResponse(BaseModel):
    """Response for latest evaluation endpoint."""
    latest_run: Optional[EvaluationRunSummary]
    failed_cases: List[EvaluationResultDetail]
    pass_percentage: float
    average_latency_ms: Optional[float]
    average_top_score: Optional[float]
    timestamp: Optional[datetime]


class EvaluationRunsListResponse(BaseModel):
    """Response for listing evaluation runs."""
    runs: List[EvaluationRunSummary]
    total: int


class FeedbackRequest(BaseModel):
    """Request body for feedback submission."""
    rating: str = Field(..., description="Rating: 'helpful' or 'not_helpful'")
    reason: Optional[str] = Field(
        None,
        description="Reason for not_helpful: wrong_answer, wrong_source, missing_information, too_long, unclear, other"
    )
    comment: Optional[str] = Field(None, description="Optional free-text comment")


class ObservationSummary(BaseModel):
    """Summary of a chat observation for admin view."""
    id: int
    created_at: datetime
    username: Optional[str]
    mode: str
    question: str
    answer_preview: Optional[str]
    answer_length: Optional[int]
    source_files: Optional[List[str]]
    citation_count: Optional[int]
    top_score: Optional[float]
    blocked: bool
    block_reason: Optional[str]
    latency_ms: Optional[float]
    feedback_rating: Optional[str]


class ObservabilitySummary(BaseModel):
    """Summary statistics for observability dashboard."""
    total_questions: int
    general_chat_count: int
    knowledge_base_count: int
    debug_count: int
    answered_count: int
    blocked_count: int
    blocked_rate: float
    citation_rate: float
    average_latency_ms: float
    average_top_score: float
    thumbs_up_count: int
    thumbs_down_count: int
    most_used_sources: List[dict]


class BlockedObservation(BaseModel):
    """Recent blocked observation."""
    id: int
    created_at: datetime
    question: str
    block_reason: Optional[str]
    top_score: Optional[float]
    mode: str


class LowConfidenceObservation(BaseModel):
    """Low confidence observation needing review."""
    id: int
    created_at: datetime
    question: str
    top_score: Optional[float]


class RAGASScoreMetrics(BaseModel):
    """Individual metric scores from a RAGAS report."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None


class RAGASSummaryResponse(BaseModel):
    """RAGAS summary for the admin Evaluations page.

    Safe, read-only. No secrets, no full paths.
    """
    available: bool
    enabled: bool
    evaluator_provider: str
    evaluator_model: str
    report_dir_configured: bool
    latest_report_found: bool
    latest_report_name: Optional[str] = None
    latest_timestamp: Optional[str] = None
    metrics: Optional[RAGASScoreMetrics] = None
    skipped_metrics: List[str] = Field(default_factory=list)
    threshold_faithfulness: float
    threshold_answer_relevancy: float
    threshold_context_precision: float
    warnings: List[str] = Field(default_factory=list)


class LangSmithSummaryResponse(BaseModel):
    """LangSmith tracing summary for the admin Observability page.

    Safe, read-only. No API keys, secrets, or raw trace content.
    """
    available: bool
    tracing_enabled: bool
    project: str
    endpoint_host: str
    has_tracing_key: bool
    sample_rate: float
    log_full_prompt: bool
    log_document_text: bool
    log_user_input: bool
    log_retrieved_context: bool
    privacy_mode: str
    warnings: List[str] = Field(default_factory=list)


class RAGASReportSummary(BaseModel):
    """Summary of a single RAGAS report for history listing.

    Safe, read-only fields. No secrets, no full paths.
    """
    report_name: str  # filename only, not full path
    timestamp: Optional[str] = None
    metrics: Optional[RAGASScoreMetrics] = None
    skipped_metrics: List[str] = Field(default_factory=list)
    evaluator_provider: str
    evaluator_model: str
    total_cases: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)


class CustomEvalRunSummary(BaseModel):
    """Summary of a custom evaluation run for history listing.

    Same as EvaluationRunSummary but uses report_name for file-based runs.
    """
    run_id: int
    timestamp: datetime
    status: str
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    avg_latency_ms: Optional[float] = None
    avg_top_score: Optional[float] = None
    report_name: Optional[str] = None


class EvaluationHistoryResponse(BaseModel):
    """Response for evaluation report history endpoint.

    Returns read-only history of custom evaluation runs and RAGAS reports.
    Always returns 200: empty lists are a normal state, not an error.
    """
    custom_eval_runs: List[CustomEvalRunSummary] = Field(default_factory=list)
    ragas_reports: List[RAGASReportSummary] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
