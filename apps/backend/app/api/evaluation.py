"""
Evaluation and Observability API Endpoints

Admin-only endpoints for viewing evaluation results and live observability data.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta

from app.db.session import get_db
from app.schemas.evaluation import (
    EvaluationRunSummary,
    EvaluationResultDetail,
    EvaluationRunDetail,
    EvaluationLatestResponse,
    EvaluationRunsListResponse,
    ObservationSummary,
    ObservabilitySummary,
    BlockedObservation,
    LowConfidenceObservation,
    FeedbackRequest,
)

router = APIRouter(prefix="/api", tags=["Evaluation & Observability"])


def require_admin(auth) -> bool:
    """Check if user is admin, raise 403 if not."""
    if not auth or not auth.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    return True


def observation_to_summary(obs) -> ObservationSummary:
    """Convert observation model to summary schema."""
    return ObservationSummary(
        id=obs.id,
        created_at=obs.created_at,
        username=obs.username,
        mode=obs.mode,
        question=obs.question,
        answer_preview=obs.answer_preview,
        answer_length=obs.answer_length,
        source_files=obs.source_files,
        citation_count=obs.citation_count,
        top_score=obs.top_score,
        blocked=obs.blocked,
        block_reason=obs.block_reason,
        latency_ms=obs.latency_ms,
        feedback_rating=obs.feedback_rating,
    )


# ============================================================================
# Observability Endpoints
# ============================================================================

@router.get("/admin/observability/summary", response_model=ObservabilitySummary)
def get_observability_summary(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
    auth: dict = Depends(lambda: {"is_admin": True}),  # Simplified for dev mode
    db: Session = Depends(get_db),
):
    """
    Get observability summary statistics for the admin dashboard.
    
    Returns aggregate statistics about chat interactions.
    """
    require_admin(auth)
    
    from app.models.observability import ChatObservation
    
    since = datetime.utcnow() - timedelta(days=days)
    
    # Base query for the time period
    base_query = db.query(ChatObservation).filter(ChatObservation.created_at >= since)
    
    total_questions = base_query.count()
    
    # Mode counts
    general_chat_count = base_query.filter(ChatObservation.mode == "general_chat").count()
    knowledge_base_count = base_query.filter(ChatObservation.mode == "knowledge_base").count()
    debug_count = base_query.filter(ChatObservation.mode == "debug").count()
    
    # Answer/blocked stats
    blocked_count = base_query.filter(ChatObservation.blocked == True).count()
    blocked_rate = (blocked_count / total_questions * 100) if total_questions > 0 else 0
    
    # Citation stats
    with_citations = base_query.filter(ChatObservation.citation_count > 0).count()
    citation_rate = (with_citations / total_questions * 100) if total_questions > 0 else 0
    
    # Averages
    avg_latency = db.query(func.avg(ChatObservation.latency_ms)).filter(
        ChatObservation.created_at >= since
    ).scalar() or 0
    
    avg_top_score = db.query(func.avg(ChatObservation.top_score)).filter(
        ChatObservation.created_at >= since,
        ChatObservation.top_score.isnot(None),
    ).scalar() or 0
    
    # Feedback stats
    thumbs_up_count = base_query.filter(ChatObservation.feedback_rating == "helpful").count()
    thumbs_down_count = base_query.filter(ChatObservation.feedback_rating == "not_helpful").count()
    
    # Most used sources
    # Get all observations with source files and count in Python
    observations_with_sources = db.query(ChatObservation.source_files).filter(
        ChatObservation.created_at >= since,
        ChatObservation.source_files.isnot(None),
    ).all()
    
    # Count source usage
    source_freq = {}
    for row in observations_with_sources:
        if row[0]:
            for src in row[0]:
                source_freq[src] = source_freq.get(src, 0) + 1
    
    most_used_sources = sorted(
        [{"source": k, "count": v} for k, v in source_freq.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]
    
    return ObservabilitySummary(
        total_questions=total_questions,
        general_chat_count=general_chat_count,
        knowledge_base_count=knowledge_base_count,
        debug_count=debug_count,
        answered_count=total_questions - blocked_count,
        blocked_count=blocked_count,
        blocked_rate=blocked_rate,
        citation_rate=citation_rate,
        average_latency_ms=round(avg_latency, 2),
        average_top_score=round(avg_top_score, 4),
        thumbs_up_count=thumbs_up_count,
        thumbs_down_count=thumbs_down_count,
        most_used_sources=most_used_sources,
    )


@router.get("/admin/observability/recent", response_model=List[ObservationSummary])
def get_recent_observations(
    limit: int = Query(default=50, ge=1, le=200),
    mode: Optional[str] = None,
    auth: dict = Depends(lambda: {"is_admin": True}),
    db: Session = Depends(get_db),
):
    """
    Get recent chat observations.
    
    Returns recent questions with their metadata.
    """
    require_admin(auth)
    
    from app.models.observability import ChatObservation
    
    query = db.query(ChatObservation).order_by(desc(ChatObservation.created_at))
    
    if mode:
        query = query.filter(ChatObservation.mode == mode)
    
    observations = query.limit(limit).all()
    return [observation_to_summary(obs) for obs in observations]


@router.get("/admin/observability/blocked", response_model=List[BlockedObservation])
def get_blocked_observations(
    limit: int = Query(default=50, ge=1, le=200),
    auth: dict = Depends(lambda: {"is_admin": True}),
    db: Session = Depends(get_db),
):
    """
    Get recently blocked questions.
    
    Returns questions that were blocked due to insufficient information.
    """
    require_admin(auth)
    
    from app.models.observability import ChatObservation
    
    observations = db.query(ChatObservation).filter(
        ChatObservation.blocked == True
    ).order_by(desc(ChatObservation.created_at)).limit(limit).all()
    
    return [
        BlockedObservation(
            id=obs.id,
            created_at=obs.created_at,
            question=obs.question,
            block_reason=obs.block_reason,
            top_score=obs.top_score,
            mode=obs.mode,
        )
        for obs in observations
    ]


@router.get("/admin/observability/low-confidence", response_model=List[LowConfidenceObservation])
def get_low_confidence_observations(
    limit: int = Query(default=50, ge=1, le=200),
    min_latency: int = Query(default=5000, ge=0, description="Minimum latency in ms"),
    auth: dict = Depends(lambda: {"is_admin": True}),
    db: Session = Depends(get_db),
):
    """
    Get low confidence observations that may need review.
    
    Returns answers with:
    - Low retrieval score (below threshold)
    - Low citation count
    - Negative feedback
    - Blocked but should have answered
    """
    require_admin(auth)
    
    from app.models.observability import ChatObservation
    
    observations = db.query(ChatObservation).filter(
        (ChatObservation.top_score < 0.7) |
        (ChatObservation.citation_count < 1) |
        (ChatObservation.feedback_rating == "not_helpful") |
        ((ChatObservation.blocked == True) & (ChatObservation.latency_ms < min_latency))
    ).order_by(desc(ChatObservation.created_at)).limit(limit).all()
    
    return [
        LowConfidenceObservation(
            id=obs.id,
            created_at=obs.created_at,
            question=obs.question,
            top_score=obs.top_score,
            citation_count=obs.citation_count,
            feedback_rating=obs.feedback_rating,
            block_reason=obs.block_reason,
        )
        for obs in observations
    ]


@router.post("/chat/{observation_id}/feedback")
def submit_feedback(
    observation_id: int,
    feedback: FeedbackRequest,
    db: Session = Depends(get_db),
):
    """
    Submit feedback for a chat observation.
    
    Updates the observation record with user feedback.
    """
    from app.models.observability import ChatObservation
    
    observation = db.query(ChatObservation).filter(ChatObservation.id == observation_id).first()
    
    if not observation:
        raise HTTPException(status_code=404, detail="Observation not found")
    
    # Validate rating
    if feedback.rating not in ("helpful", "not_helpful"):
        raise HTTPException(status_code=400, detail="Rating must be 'helpful' or 'not_helpful'")
    
    # Validate reason if not_helpful
    if feedback.rating == "not_helpful" and feedback.reason:
        valid_reasons = ["wrong_answer", "wrong_source", "missing_information", "too_long", "unclear", "other"]
        if feedback.reason not in valid_reasons:
            raise HTTPException(status_code=400, detail=f"Invalid reason. Must be one of: {valid_reasons}")
    
    # Update observation
    observation.feedback_rating = feedback.rating
    observation.feedback_reason = feedback.reason
    observation.feedback_comment = feedback.comment
    
    db.commit()
    
    return {"status": "ok", "message": "Feedback recorded"}


# ============================================================================
# Evaluation Endpoints
# ============================================================================

@router.get("/admin/evaluations/latest", response_model=EvaluationLatestResponse)
def get_latest_evaluation(
    auth: dict = Depends(lambda: {"is_admin": True}),
    db: Session = Depends(get_db),
):
    """
    Get the latest evaluation run and its results.
    """
    require_admin(auth)
    
    from app.models.evaluation import EvaluationRun, EvaluationResult
    
    # Get latest run
    latest_run = db.query(EvaluationRun).order_by(desc(EvaluationRun.created_at)).first()
    
    if not latest_run:
        return EvaluationLatestResponse(
            latest_run=None,
            failed_cases=[],
            pass_percentage=0.0,
            average_latency_ms=None,
            average_top_score=None,
            timestamp=None,
        )
    
    # Get failed results
    failed_results = db.query(EvaluationResult).filter(
        EvaluationResult.run_id == latest_run.id,
        EvaluationResult.passed == False,
    ).all()
    
    return EvaluationLatestResponse(
        latest_run=EvaluationRunSummary(
            run_id=latest_run.id,
            created_at=latest_run.created_at,
            total_tests=latest_run.total_tests,
            passed_tests=latest_run.passed_tests,
            failed_tests=latest_run.failed_tests,
            pass_percentage=latest_run.pass_percentage,
            average_latency_ms=latest_run.average_latency_ms,
            average_top_score=latest_run.average_top_score,
            status=latest_run.status,
            notes=latest_run.notes,
        ),
        failed_cases=[
            EvaluationResultDetail(
                test_case_id=r.test_case_id,
                question=r.question,
                expected_source_file=r.expected_source_file,
                actual_source_files=r.actual_source_files,
                expected_keywords=r.expected_keywords,
                found_keywords=r.found_keywords,
                missing_keywords=r.missing_keywords,
                forbidden_keywords=r.forbidden_keywords,
                forbidden_keywords_found=r.forbidden_keywords_found,
                should_answer=r.should_answer,
                expected_fallback=r.expected_fallback,
                actual_blocked=r.actual_blocked,
                answer_preview=r.answer_preview,
                citation_count=r.citation_count,
                top_score=r.top_score,
                latency_ms=r.latency_ms,
                passed=r.passed,
                failure_reasons=r.failure_reasons,
            )
            for r in failed_results
        ],
        pass_percentage=latest_run.pass_percentage,
        average_latency_ms=latest_run.average_latency_ms,
        average_top_score=latest_run.average_top_score,
        timestamp=latest_run.created_at,
    )


@router.get("/admin/evaluations/runs", response_model=EvaluationRunsListResponse)
def get_evaluation_runs(
    limit: int = Query(default=20, ge=1, le=100),
    auth: dict = Depends(lambda: {"is_admin": True}),
    db: Session = Depends(get_db),
):
    """
    Get list of evaluation runs.
    """
    require_admin(auth)
    
    from app.models.evaluation import EvaluationRun
    
    runs = db.query(EvaluationRun).order_by(desc(EvaluationRun.created_at)).limit(limit).all()
    
    return EvaluationRunsListResponse(
        runs=[
            EvaluationRunSummary(
                run_id=r.id,
                created_at=r.created_at,
                total_tests=r.total_tests,
                passed_tests=r.passed_tests,
                failed_tests=r.failed_tests,
                pass_percentage=r.pass_percentage,
                average_latency_ms=r.average_latency_ms,
                average_top_score=r.average_top_score,
                status=r.status,
                notes=r.notes,
            )
            for r in runs
        ],
        total=len(runs),
    )


@router.get("/admin/evaluations/{run_id}", response_model=EvaluationRunDetail)
def get_evaluation_run(
    run_id: int,
    auth: dict = Depends(lambda: {"is_admin": True}),
    db: Session = Depends(get_db),
):
    """
    Get detailed results for a specific evaluation run.
    """
    require_admin(auth)
    
    from app.models.evaluation import EvaluationRun, EvaluationResult
    
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    
    results = db.query(EvaluationResult).filter(EvaluationResult.run_id == run_id).all()
    
    return EvaluationRunDetail(
        summary=EvaluationRunSummary(
            run_id=run.id,
            created_at=run.created_at,
            total_tests=run.total_tests,
            passed_tests=run.passed_tests,
            failed_tests=run.failed_tests,
            pass_percentage=run.pass_percentage,
            average_latency_ms=run.average_latency_ms,
            average_top_score=run.average_top_score,
            status=run.status,
            notes=run.notes,
        ),
        results=[
            EvaluationResultDetail(
                test_case_id=r.test_case_id,
                question=r.question,
                expected_source_file=r.expected_source_file,
                actual_source_files=r.actual_source_files,
                expected_keywords=r.expected_keywords,
                found_keywords=r.found_keywords,
                missing_keywords=r.missing_keywords,
                forbidden_keywords=r.forbidden_keywords,
                forbidden_keywords_found=r.forbidden_keywords_found,
                should_answer=r.should_answer,
                expected_fallback=r.expected_fallback,
                actual_blocked=r.actual_blocked,
                answer_preview=r.answer_preview,
                citation_count=r.citation_count,
                top_score=r.top_score,
                latency_ms=r.latency_ms,
                passed=r.passed,
                failure_reasons=r.failure_reasons,
            )
            for r in results
        ],
        failed_results=[
            EvaluationResultDetail(
                test_case_id=r.test_case_id,
                question=r.question,
                expected_source_file=r.expected_source_file,
                actual_source_files=r.actual_source_files,
                expected_keywords=r.expected_keywords,
                found_keywords=r.found_keywords,
                missing_keywords=r.missing_keywords,
                forbidden_keywords=r.forbidden_keywords,
                forbidden_keywords_found=r.forbidden_keywords_found,
                should_answer=r.should_answer,
                expected_fallback=r.expected_fallback,
                actual_blocked=r.actual_blocked,
                answer_preview=r.answer_preview,
                citation_count=r.citation_count,
                top_score=r.top_score,
                latency_ms=r.latency_ms,
                passed=r.passed,
                failure_reasons=r.failure_reasons,
            )
            for r in results if not r.passed
        ],
    )


@router.post("/admin/evaluations/run")
def trigger_evaluation_run(
    auth: dict = Depends(lambda: {"is_admin": True}),
):
    """
    Trigger a new evaluation run.
    
    Note: This is a placeholder. The actual evaluation should be run from CLI.
    """
    require_admin(auth)
    
    raise HTTPException(
        status_code=501,
        detail="Evaluation must be run from CLI: docker compose exec backend python /app/scripts/run_rag_evaluation.py"
    )