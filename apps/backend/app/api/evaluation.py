"""
Evaluation and Observability API Endpoints

Admin-only endpoints for viewing evaluation results and live observability data.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

#: Absolute path to custom evaluation results directory in the container.
#: Override via TEST_EVALUATION_RESULTS_DIR env var in tests.
_EVALUATION_RESULTS_DIR = Path(
    os.getenv("TEST_EVALUATION_RESULTS_DIR", "/app/evaluations/results")
)

from app.db.session import get_db
from app.security.dependencies import require_admin
from app.security.auth import AuthContext
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
    RAGASSummaryResponse,
    LangSmithSummaryResponse,
    EvaluationHistoryResponse,
    CustomEvalRunSummary,
    CustomEvalReportArtifact,
    RAGASReportSummary,
)

router = APIRouter(prefix="/api", tags=["Evaluation & Observability"])


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
    auth: AuthContext = Depends(require_admin),  # Simplified for dev mode
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
    auth: AuthContext = Depends(require_admin),
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
    auth: AuthContext = Depends(require_admin),
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
    auth: AuthContext = Depends(require_admin),
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


@router.get("/admin/observability/langsmith-summary", response_model=LangSmithSummaryResponse)
def get_langsmith_summary(
    auth: AuthContext = Depends(require_admin),
):
    """
    Return LangSmith tracing summary for the admin Observability page.

    Reads safe configuration fields only — no API keys, no trace content,
    no raw .env values. Privacy-sensitive fields are shown as booleans only.
    Always returns 200: disabled/missing-key is a normal state, not an error.
    """
    from app.services.langsmith_tracing import get_langsmith_status
    from app.core.config import settings

    _ = auth  # admin check

    status = get_langsmith_status()
    warnings: list[str] = []

    # Check langsmith package availability using find_spec like other places
    import importlib.util
    langsmith_available = importlib.util.find_spec("langsmith") is not None

    if not langsmith_available:
        warnings.append("LangSmith package is not installed. Run: pip install langsmith")

    if not status["langsmith_tracing"]:
        warnings.append("LangSmith tracing is disabled. Set LANGSMITH_TRACING=true to enable.")
    elif status["langsmith_tracing"] and not status["api_key_configured"]:
        warnings.append("LangSmith tracing is enabled but LANGSMITH_API_KEY is not configured. Traces will not be sent.")

    if not status["log_full_prompt"]:
        warnings.append("Full prompt logging is disabled for privacy.")
    if not status["log_document_text"]:
        warnings.append("Document text logging is disabled for privacy.")
    if not status["log_retrieved_context"]:
        warnings.append("Retrieved context logging is disabled for privacy.")

    # Determine privacy mode label
    enabled_privacy_count = sum([
        status["log_full_prompt"],
        status["log_document_text"],
        status["log_user_input"],
        status["log_retrieved_context"],
    ])
    if enabled_privacy_count == 0:
        privacy_mode = "Restricted"
    elif enabled_privacy_count >= 3:
        privacy_mode = "Safe"
    else:
        privacy_mode = "Moderate"

    return LangSmithSummaryResponse(
        available=langsmith_available,
        tracing_enabled=status["langsmith_tracing"],
        project=status["project"],
        endpoint_host=status["endpoint_host"],
        has_tracing_key=status["api_key_configured"],
        sample_rate=status["sample_rate"],
        log_full_prompt=status["log_full_prompt"],
        log_document_text=status["log_document_text"],
        log_user_input=status["log_user_input"],
        log_retrieved_context=status["log_retrieved_context"],
        privacy_mode=privacy_mode,
        warnings=warnings,
    )


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
    auth: AuthContext = Depends(require_admin),
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
    auth: AuthContext = Depends(require_admin),
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


@router.get("/admin/evaluations/ragas-summary", response_model=RAGASSummaryResponse)
def get_ragas_summary(
    auth: AuthContext = Depends(require_admin),
):
    """
    Return RAGAS evaluation summary for the admin Evaluations page.

    Reads the latest RAGAS JSON report from RAGAS_REPORT_DIR.
    Returns safe, read-only fields — no secrets, no full paths.

    Always returns 200: missing report is a graceful "no report yet" state,
    not an error.
    """
    import json
    import os
    from app.core.config import settings
    from app.schemas.evaluation import RAGASScoreMetrics

    _ = auth  # admin check
    warnings: list[str] = []

    # 1. Basic availability
    # RAGAS package detection — use find_spec like factory.py to avoid uvloop issue
    import importlib.util
    ragas_available = importlib.util.find_spec("ragas") is not None

    enabled = settings.RAGAS_ENABLED and ragas_available
    if not ragas_available:
        warnings.append("RAGAS package is not installed. Run: pip install ragas>=0.2.0,<0.3.0")
    if settings.RAGAS_ENABLED and not ragas_available:
        warnings.append("RAGAS is enabled (RAGAS_ENABLED=true) but the ragas package is not installed.")

    report_dir_configured = bool(settings.RAGAS_REPORT_DIR)

    # 2. Try to read latest report
    latest_report_name: Optional[str] = None
    latest_timestamp: Optional[str] = None
    metrics: Optional[RAGASScoreMetrics] = None
    skipped_metrics: list[str] = ["context_recall", "answer_correctness"]
    latest_report_found = False

    if ragas_available and report_dir_configured:
        report_dir = settings.RAGAS_REPORT_DIR
        # Only expose the configured dir name, not full system paths
        report_dir_short = report_dir.split("/")[-1] if "/" in report_dir else report_dir

        try:
            report_path = os.path.join(report_dir, "latest.json")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report = json.load(f)

                latest_report_found = True
                # Strip directory prefix from report name for safe display
                report_basename = os.path.basename(report_path)
                latest_report_name = report_basename
                latest_timestamp = report.get("timestamp", None)

                # Extract metric averages
                avg_scores = report.get("avg_scores", {})
                threshold_summary = report.get("threshold_summary", {})

                def _safe_float(val) -> Optional[float]:
                    if val is None:
                        return None
                    try:
                        f = float(val)
                        if f != f:  # NaN
                            return None
                        return round(f, 4)
                    except (TypeError, ValueError):
                        return None

                metrics = RAGASScoreMetrics(
                    faithfulness=_safe_float(avg_scores.get("faithfulness", {}).get("avg")),
                    answer_relevancy=_safe_float(avg_scores.get("answer_relevancy", {}).get("avg")),
                    context_precision=_safe_float(avg_scores.get("context_precision", {}).get("avg")),
                    # context_recall and answer_correctness are always skipped
                    context_recall=None,
                    answer_correctness=None,
                )

                # Add warning if no real evaluation has been run yet
                if not latest_report_found:
                    warnings.append(
                        "No RAGAS report found yet. Run RAGAS evaluation without --dry-run "
                        "to generate scores: python scripts/run_ragas_evaluation.py"
                    )
            else:
                warnings.append(
                    "No RAGAS report found yet. Run RAGAS evaluation without --dry-run "
                    "to generate scores: python scripts/run_ragas_evaluation.py"
                )
        except json.JSONDecodeError:
            warnings.append("RAGAS report exists but could not be parsed.")
            latest_report_name = None
            latest_timestamp = None
        except OSError as exc:
            warnings.append(f"Could not read RAGAS report directory: {exc}")
            latest_report_name = None
            latest_timestamp = None
    elif not report_dir_configured:
        warnings.append("RAGAS report directory is not configured (RAGAS_REPORT_DIR is empty).")

    # 3. Load thresholds from settings
    # Import here to avoid heavy import chain at module load
    _THRESHOLDS = {
        "faithfulness": float(os.getenv("RAGAS_THRESHOLD_FAITHFULNESS", "0.5")),
        "answer_relevancy": float(os.getenv("RAGAS_THRESHOLD_ANSWER_RELEVANCY", "0.5")),
        "context_precision": float(os.getenv("RAGAS_THRESHOLD_CONTEXT_PRECISION", "0.5")),
    }

    return RAGASSummaryResponse(
        available=ragas_available,
        enabled=enabled,
        evaluator_provider=settings.RAGAS_EVALUATOR_PROVIDER,
        evaluator_model=settings.RAGAS_EVALUATOR_MODEL,
        report_dir_configured=report_dir_configured,
        latest_report_found=latest_report_found,
        latest_report_name=latest_report_name,
        latest_timestamp=latest_timestamp,
        metrics=metrics,
        skipped_metrics=skipped_metrics,
        threshold_faithfulness=_THRESHOLDS["faithfulness"],
        threshold_answer_relevancy=_THRESHOLDS["answer_relevancy"],
        threshold_context_precision=_THRESHOLDS["context_precision"],
        warnings=warnings,
    )


@router.get("/admin/evaluations/history", response_model=EvaluationHistoryResponse)
def get_evaluation_history(
    limit: int = Query(default=20, ge=1, le=50, description="Max custom eval runs to return"),
    auth: AuthContext = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Return read-only history of custom evaluation runs and RAGAS reports.

    - Custom eval runs come from the database (EvaluationRun table) and
      optionally from JSON files under RAGAS_REPORT_DIR.
    - RAGAS reports are scanned from RAGAS_REPORT_DIR.
    - No secrets, no full filesystem paths, no raw prompts or document text.
    - Always returns 200: empty lists are a normal state, not an error.
    """
    import json
    import os
    from app.core.config import settings
    from app.schemas.evaluation import RAGASScoreMetrics

    _ = auth  # admin check
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Custom evaluation runs from database
    # ------------------------------------------------------------------
    from app.models.evaluation import EvaluationRun

    db_runs = (
        db.query(EvaluationRun)
        .order_by(desc(EvaluationRun.created_at))
        .limit(limit)
        .all()
    )

    custom_eval_runs: list[CustomEvalRunSummary] = []
    for r in db_runs:
        custom_eval_runs.append(
            CustomEvalRunSummary(
                run_id=r.id,
                timestamp=r.created_at,
                status=r.status,
                total_tests=r.total_tests,
                passed=r.passed_tests,
                failed=r.failed_tests,
                pass_rate=r.pass_percentage,
                avg_latency_ms=r.average_latency_ms,
                avg_top_score=r.average_top_score,
                report_name=None,  # DB runs don't have a file name; run_id is enough
            )
        )

    # ------------------------------------------------------------------
    # 2. Custom evaluation report artifacts from /app/evaluations/results/
    # ------------------------------------------------------------------
    custom_eval_reports: list[CustomEvalReportArtifact] = []

    results_base_dir = _EVALUATION_RESULTS_DIR
    if results_base_dir.is_dir():
        # Collect timestamped evaluation JSON files, newest first
        eval_files = sorted(
            [
                f
                for f in results_base_dir.iterdir()
                if f.is_file()
                and f.name.startswith("evaluation_")
                and f.name.endswith(".json")
                and f.name != "latest_results.json"
            ],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]

        # Also check for latest_results.json separately (latest alias)
        latest_results_path = results_base_dir / "latest_results.json"
        has_latest = latest_results_path.is_file()

        def _parse_eval_file(f_path: Path) -> Optional[CustomEvalReportArtifact]:
            """Parse an evaluation JSON file and return artifact or None on error."""
            try:
                with open(f_path, "r") as fh:
                    data = json.load(fh)

                ts = data.get("timestamp")
                if not ts:
                    ts = datetime.fromtimestamp(
                        f_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat()

                results = data.get("results", [])
                total = len(results)
                if total == 0:
                    return CustomEvalReportArtifact(
                        report_name=f_path.name,
                        report_type="timestamped_results_json",
                        timestamp=ts,
                        total_tests=0,
                        passed=0,
                        failed=0,
                        pass_rate=0.0,
                        avg_latency_ms=None,
                        avg_top_score=None,
                    )

                passed = sum(1 for r in results if r.get("passed", False))
                failed = total - passed
                pass_rate = round(passed / total * 100, 1) if total > 0 else 0.0

                latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
                avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

                top_scores = [r["top_score"] for r in results if r.get("top_score") is not None]
                avg_top_score = round(sum(top_scores) / len(top_scores), 4) if top_scores else None

                return CustomEvalReportArtifact(
                    report_name=f_path.name,
                    report_type="timestamped_results_json",
                    timestamp=ts,
                    total_tests=total,
                    passed=passed,
                    failed=failed,
                    pass_rate=pass_rate,
                    avg_latency_ms=avg_latency,
                    avg_top_score=avg_top_score,
                )
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                warnings.append(f"Could not read evaluation report '{f_path.name}' — skipping.")
                return None

        # Process timestamped files
        for f_path in eval_files:
            artifact = _parse_eval_file(f_path)
            if artifact is not None:
                custom_eval_reports.append(artifact)

        # Process latest_results.json if present (labeled as latest alias)
        if has_latest:
            artifact = _parse_eval_file(latest_results_path)
            if artifact is not None:
                # Override type to mark it as the latest alias
                artifact.report_type = "latest_results_json"
                custom_eval_reports.insert(0, artifact)

        # Process latest_report.md if present (markdown report metadata)
        latest_md_path = results_base_dir / "latest_report.md"
        if latest_md_path.is_file():
            try:
                mtime = latest_md_path.stat().st_mtime
                ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                custom_eval_reports.insert(
                    0,
                    CustomEvalReportArtifact(
                        report_name="latest_report.md",
                        report_type="latest_report_markdown",
                        timestamp=ts,
                        total_tests=None,
                        passed=None,
                        failed=None,
                        pass_rate=None,
                        avg_latency_ms=None,
                        avg_top_score=None,
                    ),
                )
            except OSError:
                # Cannot stat file — skip silently
                pass
    else:
        warnings.append(
            f"Custom evaluation results directory not found."
        )

    # ------------------------------------------------------------------
    # 3. RAGAS reports from RAGAS_REPORT_DIR
    # ------------------------------------------------------------------
    ragas_reports: list[RAGASReportSummary] = []

    if settings.RAGAS_REPORT_DIR:
        report_dir = Path(settings.RAGAS_REPORT_DIR)
        if report_dir.is_dir():
            # Collect all ragas_report_*.json files (skip latest.json alias)
            report_files = sorted(
                [
                    f
                    for f in report_dir.iterdir()
                    if f.is_file()
                    and f.name.startswith("ragas_report_")
                    and f.name.endswith(".json")
                ],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )[:limit]

            for report_file in report_files:
                try:
                    with open(report_file, "r") as fh:
                        report = json.load(fh)

                    # Parse timestamp — prefer the embedded one, fall back to file mtime
                    ts = report.get("timestamp")
                    if not ts:
                        ts = datetime.fromtimestamp(
                            report_file.stat().st_mtime, tz=timezone.utc
                        ).isoformat()

                    # Metric averages
                    avg_scores = report.get("avg_scores", {})

                    def _safe_float(val):
                        if val is None:
                            return None
                        try:
                            f = float(val)
                            return round(f, 4) if not (f != f) else None  # guard NaN
                        except (TypeError, ValueError):
                            return None

                    metrics = RAGASScoreMetrics(
                        faithfulness=_safe_float(
                            avg_scores.get("faithfulness", {}).get("avg")
                        ),
                        answer_relevancy=_safe_float(
                            avg_scores.get("answer_relevancy", {}).get("avg")
                        ),
                        context_precision=_safe_float(
                            avg_scores.get("context_precision", {}).get("avg")
                        ),
                        context_recall=None,
                        answer_correctness=None,
                    )

                    ragas_reports.append(
                        RAGASReportSummary(
                            report_name=report_file.name,
                            timestamp=ts,
                            metrics=metrics,
                            skipped_metrics=report.get(
                                "skipped_metrics", ["context_recall", "answer_correctness"]
                            ),
                            evaluator_provider=report.get("evaluator_provider", ""),
                            evaluator_model=report.get("evaluator_model", ""),
                            total_cases=report.get("case_count"),
                            warnings=[],
                        )
                    )
                except (json.JSONDecodeError, OSError):
                    # One bad file must not poison the whole list
                    warnings.append(
                        f"Could not read RAGAS report '{report_file.name}' — skipping."
                    )
                    continue
        else:
            ragas_dir_name = report_dir.name
            # Check if dir exists at all (symlink broken, volume not mounted, etc.)
            if not report_dir.exists():
                warnings.append(
                    f"RAGAS report directory not found ({ragas_dir_name}). "
                    "Run RAGAS evaluation to create it."
                )
            else:
                warnings.append(
                    f"RAGAS report directory is not accessible: {ragas_dir_name}"
                )
    else:
        warnings.append(
            "RAGAS report directory is not configured (RAGAS_REPORT_DIR is empty)."
        )

    # ------------------------------------------------------------------
    # 4. Empty-state warning if nothing was found
    # ------------------------------------------------------------------
    # No RAGAS reports friendly warning
    if settings.RAGAS_REPORT_DIR and not ragas_reports:
        ragas_dir = Path(settings.RAGAS_REPORT_DIR)
        if ragas_dir.is_dir():
            ragas_file_count = sum(
                1
                for f in ragas_dir.iterdir()
                if f.is_file() and f.name.startswith("ragas_report_") and f.name.endswith(".json")
            )
            if ragas_file_count == 0:
                warnings.append(
                    "No RAGAS reports found yet. Run RAGAS evaluation without --dry-run "
                    "to generate report history."
                )

    # Empty-state warning if nothing was found
    if not custom_eval_runs and not custom_eval_reports and not ragas_reports:
        warnings.append(
            "No evaluation history found yet. "
            "Run custom evaluation: docker compose exec backend python /app/scripts/run_rag_evaluation.py\n"
            "Run RAGAS without --dry-run: docker compose exec backend python /app/scripts/run_ragas_evaluation.py"
        )

    return EvaluationHistoryResponse(
        custom_eval_runs=custom_eval_runs,
        custom_eval_reports=custom_eval_reports,
        ragas_reports=ragas_reports,
        warnings=warnings,
    )


@router.get("/admin/evaluations/{run_id}", response_model=EvaluationRunDetail)
def get_evaluation_run(
    run_id: int,
    auth: AuthContext = Depends(require_admin),
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
    auth: AuthContext = Depends(require_admin),
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


