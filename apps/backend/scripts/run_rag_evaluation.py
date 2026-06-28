#!/usr/bin/env python3
"""
RAG Evaluation Runner

Runs evaluation cases against the Knowledge Base mode and scores results.
Saves results to database and JSON files.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add backend app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.evaluation import EvaluationRun, EvaluationResult
from app.models.observability import ChatObservation
from app.rag.answer_generator import generate_answer_with_rag
from app.rag.citations import group_citations_by_source

# Phase 31A — LangSmith tracing for evaluation runs. Safe wrappers no-op when
# tracing is disabled (default), and never raise on failure.
try:
    from app.services.langsmith_tracing import (
        trace_span,
        trace_chat_request,
        redact_filenames,
        sanitize_metadata,
    )
except Exception:  # pragma: no cover - tracing never required
    from contextlib import contextmanager

    @contextmanager
    def trace_span(*args, **kwargs):
        yield None

    @contextmanager
    def trace_chat_request(*args, **kwargs):
        yield None

    def redact_filenames(value):
        return list(value or [])

    def sanitize_metadata(value):
        return value or {}


def load_evaluation_dataset():
    """Load evaluation cases from JSON dataset."""
    dataset_path = Path(__file__).parent.parent / "evaluations" / "dataset.json"
    with open(dataset_path, "r") as f:
        return json.load(f)


def check_source_documents_indexed(db: Session) -> list[str]:
    """Check which required source documents are indexed in Qdrant."""
    from app.models.document import Document
    
    # Get all indexed document names
    indexed_docs = db.query(Document).filter(
        Document.status == "indexed"
    ).all()
    
    return [doc.original_name for doc in indexed_docs]


def run_evaluation_case(case: dict, db: Session) -> dict:
    """Run a single evaluation case and return results."""
    question = case["question"]
    expected_source = case.get("expected_source_file")
    case_id = case.get("id") or case.get("name") or (question[:50] if question else "unknown")

    # Phase 31A — wrap the whole eval-case run in a single span so each
    # case becomes a queryable LangSmith trace with retrieval strategy,
    # evidence level, fallback reason, pass/fail, and failure reasons.
    case_metadata: dict = {
        "phase": "rag_evaluation_case",
        "evaluation_case_id": str(case_id),
        "expected_behavior": "answer" if case.get("should_answer", True) else "fallback",
        "expected_source_file": expected_source,
        "expected_min_citations": case.get("minimum_expected_citations", 0),
        "expected_fallback": bool(case.get("expected_fallback", False)),
        "should_have_citations": bool(case.get("should_have_citations", False)),
    }
    # `expected_keywords` may contain sensitive phrases from private docs;
    # only forward a count, never the strings themselves, unless the
    # operator has explicitly opted in. Default: counts only.
    ek = case.get("expected_keywords") or []
    if ek:
        case_metadata["expected_keyword_count"] = len(ek)

    with trace_span("rag_evaluation_case", metadata=case_metadata) as case_span:
        return _run_evaluation_case_impl(case, db, case_id, case_span)


def _run_evaluation_case_impl(case: dict, db: Session, case_id, case_span) -> dict:
    question = case["question"]
    expected_source = case.get("expected_source_file")

    start_time = time.time()

    # Run through Knowledge Base mode
    answer, citations, metadata = generate_answer_with_rag(
        query=question,
        use_hybrid=True,
        debug=False,
    )

    latency_ms = (time.time() - start_time) * 1000
    
    # Get source files from citations
    actual_sources = []
    if citations:
        for cite in citations:
            src = cite.get("source_file_name")
            if src and src not in actual_sources:
                actual_sources.append(src)
    
    # Create grouped sources for analysis
    grouped_sources = None
    if citations:
        grouped_sources = group_citations_by_source(
            citations,
            question=question,
            answer=answer,
            max_excerpts=3,
            debug_mode=False
        )
    
    # Determine if answer was blocked (fallback behavior)
    # If answer contains insufficient information patterns, it may be a fallback
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
    is_blocked = any(phrase in answer.lower() for phrase in fallback_phrases)
    actual_blocked = case.get("should_answer", True) == False and is_blocked
    
    # Check for expected keywords
    expected_keywords = case.get("expected_keywords", [])
    found_keywords = []
    missing_keywords = []
    
    if expected_keywords:
        answer_lower = answer.lower()
        for kw in expected_keywords:
            if kw.lower() in answer_lower:
                found_keywords.append(kw)
            else:
                missing_keywords.append(kw)
    
    # Check for forbidden keywords
    forbidden_keywords = case.get("forbidden_keywords", [])
    forbidden_found = []
    
    if forbidden_keywords:
        answer_lower = answer.lower()
        for kw in forbidden_keywords:
            if kw.lower() in answer_lower:
                forbidden_found.append(kw)
    
    # Check citation count
    citation_count = len(citations) if citations else 0
    min_citations = case.get("minimum_expected_citations", 0)
    
    # Get top retrieval score
    top_score = None
    if metadata and "top_score" in metadata:
        top_score = metadata.get("top_score")
    elif grouped_sources and len(grouped_sources) > 0:
        # Handle both dict and object formats for grouped_sources
        scores = []
        for gs in grouped_sources:
            if isinstance(gs, dict):
                scores.append(gs.get("highest_score", 0))
            else:
                scores.append(getattr(gs, "highest_score", 0))
        top_score = max(scores) if scores else None
    
    # Score the test case
    failure_reasons = []
    
    # 1. Answer present check
    answer_present = bool(answer and len(answer.strip()) > 0)
    if case.get("should_answer", True) and not answer_present:
        failure_reasons.append("expected_answer_but_empty")
    
    # 2. Fallback correct
    fallback_correct = True
    if case.get("expected_fallback") == True and not is_blocked:
        fallback_correct = False
        failure_reasons.append("expected_fallback_but_answered")
    elif case.get("expected_fallback") == False and is_blocked:
        fallback_correct = False
        failure_reasons.append("expected_answer_but_fallback")
    
    # 3. Citation present
    # If should_have_citations is False, we don't require citations (citation_present is True)
    citation_present = citation_count > 0 if case.get("should_have_citations", False) else True
    if case.get("should_have_citations", False) and not citation_present:
        failure_reasons.append("missing_citations")
    
    # 4. Minimum citations met
    minimum_citations_met = citation_count >= min_citations
    
    # 5. Expected keywords found
    keywords_found_check = len(missing_keywords) == 0 if expected_keywords else True
    
    # 6. Forbidden keywords absent
    forbidden_absent_check = len(forbidden_found) == 0
    
    # 7. Expected source found
    source_found = True
    if expected_source:
        source_found = expected_source in actual_sources
        if not source_found:
            failure_reasons.append("wrong_source")
    
    # 8. Blocked correctly
    blocked_correct = True
    if case.get("should_answer") == True and is_blocked:
        blocked_correct = False
        failure_reasons.append("incorrectly_blocked")
    elif case.get("should_answer") == False and not is_blocked:
        # This is only a failure if it actually answered when it shouldn't
        if is_blocked:  # Correctly blocked
            blocked_correct = True
        else:
            blocked_correct = False
            failure_reasons.append("should_have_blocked")
    
    # Calculate overall pass/fail
    passed = (
        answer_present and
        fallback_correct and
        citation_present and
        minimum_citations_met and
        keywords_found_check and
        forbidden_absent_check and
        source_found and
        blocked_correct
    )
    
    if missing_keywords:
        failure_reasons.append("missing_keywords")
    if forbidden_found:
        failure_reasons.append("forbidden_keywords_found")

    # Phase 31A — attach final pass/fail result to the eval-case span so
    # LangSmith surfaces pass/fail counts, failure reasons, retrieval
    # strategy, evidence level, and fallback reason per case.
    if case_span is not None:
        try:
            grounding = (metadata or {}).get("grounding") or {}
            retrieval_strategy = (metadata or {}).get("retrieval_strategy") or (
                (metadata or {}).get("strategy")
            )
            case_span.set_meta("passed", bool(passed))
            case_span.set_meta("latency_ms", int(latency_ms or 0))
            case_span.set_meta("citation_count", int(citation_count or 0))
            case_span.set_meta("failure_reasons", failure_reasons if not passed else None)
            case_span.set_meta(
                "source_file_names",
                redact_filenames([s for s in actual_sources if s]),
            )
            case_span.set_meta("retrieval_strategy", retrieval_strategy)
            case_span.set_meta("evidence_level", grounding.get("evidence_level"))
            case_span.set_meta(
                "fallback_reason",
                grounding.get("blocked_reason") or (metadata or {}).get("block_reason"),
            )
            case_span.set_meta("blocked", bool((metadata or {}).get("blocked")))
            case_span.set_meta("actual_blocked", bool(is_blocked))
            case_span.set_meta("top_score", top_score)
        except Exception:
            pass

    return {
        "test_case_id": case["id"],
        "question": question,
        "expected_source_file": expected_source,
        "actual_source_files": actual_sources,
        "expected_keywords": expected_keywords,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "forbidden_keywords": forbidden_keywords,
        "forbidden_keywords_found": forbidden_found,
        "should_answer": case.get("should_answer", True),
        "expected_fallback": case.get("expected_fallback", False),
        "actual_blocked": is_blocked,
        "answer_preview": answer[:500] if answer else None,
        "citation_count": citation_count,
        "top_score": top_score,
        "latency_ms": latency_ms,
        "passed": passed,
        "failure_reasons": failure_reasons if not passed else None,
        "metadata": {
            "grouped_sources": [
                {
                    "source_file_name": gs.get("source_file_name") if isinstance(gs, dict) else getattr(gs, "source_file_name", None),
                    "sections_used": gs.get("sections_used") if isinstance(gs, dict) else getattr(gs, "sections_used", None),
                    "highest_score": gs.get("highest_score") if isinstance(gs, dict) else getattr(gs, "highest_score", None),
                    "confidence": gs.get("confidence") if isinstance(gs, dict) else getattr(gs, "confidence", None),
                }
                for gs in (grouped_sources or [])
            ]
        }
    }


def save_results_to_db(run_id: int, results: list[dict]):
    """Save evaluation results to database."""
    db = SessionLocal()
    try:
        for result in results:
            db_result = EvaluationResult(
                run_id=run_id,
                test_case_id=result["test_case_id"],
                question=result["question"],
                expected_source_file=result["expected_source_file"],
                actual_source_files=result["actual_source_files"],
                expected_keywords=result["expected_keywords"],
                found_keywords=result["found_keywords"],
                missing_keywords=result["missing_keywords"],
                forbidden_keywords=result["forbidden_keywords"],
                forbidden_keywords_found=result["forbidden_keywords_found"],
                should_answer=result["should_answer"],
                expected_fallback=result["expected_fallback"],
                actual_blocked=result["actual_blocked"],
                answer_preview=result["answer_preview"],
                citation_count=result["citation_count"],
                top_score=result["top_score"],
                latency_ms=result["latency_ms"],
                passed=result["passed"],
                failure_reasons=result["failure_reasons"],
                extra_metadata=result.get("metadata"),
            )
            db.add(db_result)
        db.commit()
    finally:
        db.close()


def save_run_to_db(total_tests: int, passed_tests: int, avg_latency: float, avg_score: float, status: str) -> int:
    """Create evaluation run record and return run_id."""
    db = SessionLocal()
    try:
        run = EvaluationRun(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=total_tests - passed_tests,
            pass_percentage=(passed_tests / total_tests * 100) if total_tests > 0 else 0.0,
            average_latency_ms=avg_latency,
            average_top_score=avg_score,
            status=status,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run.id
    finally:
        db.close()


def save_results_to_json(results: list[dict], run_id: int):
    """Save results to JSON files."""
    results_dir = Path(__file__).parent.parent / "evaluations" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Latest results
        latest_path = results_dir / "latest_results.json"
        with open(latest_path, "w") as f:
            json.dump({
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
        
        # Timestamped results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = results_dir / f"evaluation_{timestamp}.json"
        with open(timestamped_path, "w") as f:
            json.dump({
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
        
        return latest_path, timestamped_path
    except PermissionError as e:
        # If we can't write to the mounted volume, just print a warning
        print(f"  Warning: Could not write JSON results to {results_dir}: {e}")
        print(f"  Results are still saved to database (run_id={run_id})")
        return None, None
    
    return latest_path, timestamped_path


def generate_markdown_report(results: list[dict], summary: dict) -> str:
    """Generate a markdown report of evaluation results."""
    lines = [
        "# RAG Evaluation Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Summary",
        f"- Total Tests: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Pass Rate: {summary['pass_rate']:.1f}%",
        f"- Average Latency: {summary['avg_latency']:.2f}ms",
        f"- Average Top Score: {summary['avg_score']:.4f}",
        "\n## Failed Tests",
    ]
    
    for r in results:
        if not r["passed"]:
            lines.append(f"\n### {r['test_case_id']}: {r['question'][:60]}...")
            lines.append(f"- Failure Reasons: {', '.join(r['failure_reasons'])}")
            if r.get("missing_keywords"):
                lines.append(f"- Missing Keywords: {', '.join(r['missing_keywords'])}")
            if r.get("forbidden_keywords_found"):
                lines.append(f"- Forbidden Keywords Found: {', '.join(r['forbidden_keywords_found'])}")
            lines.append(f"- Expected Source: {r['expected_source_file']}")
            lines.append(f"- Actual Sources: {', '.join(r['actual_source_files'] or [])}")
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("RAG Evaluation Runner")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/5] Loading evaluation dataset...")
    test_cases = load_evaluation_dataset()
    print(f"  Loaded {len(test_cases)} test cases")
    
    # Check indexed documents
    db = SessionLocal()
    try:
        indexed_docs = check_source_documents_indexed(db)
        print(f"\n[2/5] Checking indexed documents...")
        print(f"  Indexed documents: {indexed_docs}")
        
        # Check if required source documents are present
        required_docs = set()
        for case in test_cases:
            if case.get("expected_source_file"):
                required_docs.add(case["expected_source_file"])
        
        missing_docs = required_docs - set(indexed_docs)
        if missing_docs:
            print(f"\n  WARNING: Missing required documents: {missing_docs}")
            print("  Evaluation will still run but source-related tests may fail.")
    finally:
        db.close()
    
    # Run evaluations
    print(f"\n[3/5] Running {len(test_cases)} evaluation cases...")
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] Running: {case['id']}...", end=" ")
        try:
            result = run_evaluation_case(case, db)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} ({result['latency_ms']:.0f}ms)")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                "test_case_id": case["id"],
                "question": case["question"],
                "expected_source_file": case.get("expected_source_file"),
                "actual_source_files": [],
                "expected_keywords": case.get("expected_keywords", []),
                "found_keywords": [],
                "missing_keywords": case.get("expected_keywords", []),
                "forbidden_keywords": case.get("forbidden_keywords", []),
                "forbidden_keywords_found": [],
                "should_answer": case.get("should_answer", True),
                "expected_fallback": case.get("expected_fallback", False),
                "actual_blocked": False,
                "answer_preview": None,
                "citation_count": 0,
                "top_score": None,
                "latency_ms": 0,
                "passed": False,
                "failure_reasons": [f"evaluation_error: {str(e)}"],
                "metadata": None,
            })
    
    # Calculate summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0
    avg_latency = sum(r["latency_ms"] for r in results) / total if total > 0 else 0
    avg_score = sum(r["top_score"] or 0 for r in results) / total if total > 0 else 0
    
    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "avg_latency": avg_latency,
        "avg_score": avg_score,
    }
    
    # Save to database
    print(f"\n[4/5] Saving results to database...")
    status = "completed" if failed < total else "failed"
    run_id = save_run_to_db(
        total_tests=total,
        passed_tests=passed,
        avg_latency=avg_latency,
        avg_score=avg_score,
        status=status,
    )
    save_results_to_db(run_id, results)
    print(f"  Saved run {run_id} with {total} results")
    
    # Save to JSON
    print(f"\n[5/5] Saving results to JSON...")
    latest_path, timestamped_path = save_results_to_json(results, run_id)
    print(f"  Latest: {latest_path}")
    print(f"  Timestamped: {timestamped_path}")
    
    # Generate markdown report
    report = generate_markdown_report(results, summary)
    report_path = Path(__file__).parent.parent / "evaluations" / "results" / "latest_report.md"
    try:
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Report: {report_path}")
    except PermissionError:
        print(f"  Report: Could not write to {report_path} (permission denied)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total Tests:    {total}")
    print(f"  Passed:         {passed}")
    print(f"  Failed:         {failed}")
    print(f"  Pass Rate:      {pass_rate:.1f}%")
    print(f"  Avg Latency:    {avg_latency:.2f}ms")
    print(f"  Avg Top Score:  {avg_score:.4f}")
    
    if failed > 0:
        print("\n  FAILED TESTS:")
        for r in results:
            if not r["passed"]:
                print(f"    - {r['test_case_id']}: {', '.join(r['failure_reasons'])}")
    
    print("\n" + "=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())