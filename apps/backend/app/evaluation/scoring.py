"""
Phase 31C — Shared RAG evaluation scoring

Pure-function scoring that takes a test case dict + the raw outputs of
a RAG pipeline run (answer, citations, metadata) and produces a
structured result dict with pass/fail, failure reasons, and the
individual checks.

Rules mirror `_run_evaluation_case_impl` from
`scripts/run_rag_evaluation.py` so the comparison script and the
existing classic evaluation agree on what "pass" means. Any drift here
is a bug.

Failure-reason vocabulary (stable, used as keys in JSON reports):

    expected_answer_but_empty       — case.expected.should_answer=True but answer is empty
    expected_fallback_but_answered  — case.expected_fallback=True but a real answer was produced
    expected_answer_but_fallback    — case.expected_fallback=False but a fallback phrase was produced
    missing_citations               — case.should_have_citations=True but citation_count==0
    missing_keywords                — at least one expected_keyword not found in answer
    forbidden_keywords_found        — at least one forbidden_keyword appeared in answer
    wrong_source                    — case.expected_source_file not in actual source list
    incorrectly_blocked             — case.should_answer=True but answer looks like a fallback
    should_have_blocked             — case.should_answer=False but a real answer was produced
    evaluation_error                — the run raised before producing an answer
    provider_rate_limit             — answer/error mentions OpenRouter rate limit / 429 / quota

The scoring function is intentionally side-effect free so it can be
unit-tested without any database, network, or LangSmith setup.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Phrases that indicate the answer is a fallback ("I don't have enough
# information..."). These mirror the patterns already used by
# `run_rag_evaluation.py` so the comparison stays consistent.
_FALLBACK_PHRASES = (
    "i don't have",
    "i cannot find",
    "i don't know",
    "not enough information",
    "no relevant documents",
    "don't have access",
    "cannot answer based on",
    "insufficient information",
)

# Provider / network failure patterns that should be flagged as
# rate-limit noise rather than scored as a normal failure.
_RATE_LIMIT_PATTERNS = [
    re.compile(r"openrouter.*rate\s*limit", re.IGNORECASE),
    re.compile(r"\b429\b"),
    re.compile(r"\bquota\b", re.IGNORECASE),
    re.compile(r"\brate\s*limit(ed)?\b", re.IGNORECASE),
    re.compile(r"too\s+many\s+requests", re.IGNORECASE),
]


def _is_fallback_answer(answer: Optional[str]) -> bool:
    """True when the answer text contains a known fallback phrase."""
    if not answer:
        return False
    lower = answer.lower()
    return any(phrase in lower for phrase in _FALLBACK_PHRASES)


def _detect_provider_rate_limit(text: Optional[str]) -> bool:
    """True when the answer (or error string) shows provider rate-limit noise."""
    if not text:
        return False
    return any(p.search(text) for p in _RATE_LIMIT_PATTERNS)


def _answer_present(answer: Optional[str]) -> bool:
    return bool(answer and len(answer.strip()) > 0)


def _actual_sources(citations: Optional[list[dict]]) -> list[str]:
    """Extract distinct source filenames from a citation list."""
    sources: list[str] = []
    if not citations:
        return sources
    for c in citations:
        if not isinstance(c, dict):
            continue
        src = c.get("source_file_name")
        if src and src not in sources:
            sources.append(src)
    return sources


def _find_missing_keywords(answer: Optional[str], expected_keywords: list[str]) -> list[str]:
    if not answer or not expected_keywords:
        return []
    lower = answer.lower()
    return [kw for kw in expected_keywords if kw and kw.lower() not in lower]


def _find_forbidden_keywords(answer: Optional[str], forbidden_keywords: list[str]) -> list[str]:
    if not answer or not forbidden_keywords:
        return []
    lower = answer.lower()
    return [kw for kw in forbidden_keywords if kw and kw.lower() in lower]


def _grounding_meta(metadata: Optional[dict]) -> dict:
    """Safely extract grounding metadata."""
    if not isinstance(metadata, dict):
        return {}
    return metadata.get("grounding") or {}


def _top_score(metadata: Optional[dict], citations: Optional[list[dict]]) -> Optional[float]:
    """Best-effort retrieval top score from pipeline metadata or citation list."""
    if isinstance(metadata, dict):
        # Phase 30E path exposes top_score directly; legacy may use
        # grounding.top_score or retrieval.top_score.
        for key in ("top_score", "highest_score"):
            if key in metadata and metadata[key] is not None:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass
        g = metadata.get("grounding") or {}
        for key in ("top_score",):
            if key in g and g[key] is not None:
                try:
                    return float(g[key])
                except (TypeError, ValueError):
                    pass
    if citations:
        scores = []
        for c in citations:
            if not isinstance(c, dict):
                continue
            s = c.get("relevance_score") or c.get("score")
            if s is None:
                continue
            try:
                scores.append(float(s))
            except (TypeError, ValueError):
                continue
        if scores:
            return max(scores)
    return None


def score_case(
    case: dict,
    answer: Optional[str],
    citations: Optional[list[dict]],
    metadata: Optional[dict],
    *,
    latency_ms: float = 0.0,
    rag_path: str = "classic",
    error: Optional[str] = None,
) -> dict:
    """
    Score one evaluation case against the RAG pipeline output.

    Args:
        case: dataset case dict (id, question, expected_*, keywords, ...)
        answer: generated answer text (may be None / empty).
        citations: list of citation dicts from the pipeline.
        metadata: pipeline metadata dict (grounding, retrieval, etc.).
        latency_ms: measured latency for this case (rounded downstream).
        rag_path: "classic" or "agentic" — used for reporting only.
        error: optional error string if the pipeline raised.

    Returns:
        Structured result dict. Keys mirror the legacy
        `_run_evaluation_case_impl` return shape so downstream consumers
        (DB rows, JSON reports, frontend) don't need to change.
    """
    case_id = case.get("id") or case.get("name") or (case.get("question", "")[:50] or "unknown")
    question = case.get("question", "")
    expected_source = case.get("expected_source_file")
    expected_keywords = list(case.get("expected_keywords") or [])
    forbidden_keywords = list(case.get("forbidden_keywords") or [])
    should_answer = bool(case.get("should_answer", True))
    expected_fallback = bool(case.get("expected_fallback", False))
    should_have_citations = bool(case.get("should_have_citations", False))
    min_citations = int(case.get("minimum_expected_citations", 0) or 0)

    failure_reasons: list[str] = []
    answer_present = _answer_present(answer)
    is_fallback = _is_fallback_answer(answer)

    # 0. Provider error / rate limit — flagged but doesn't auto-fail.
    rate_limited = _detect_provider_rate_limit(answer) or _detect_provider_rate_limit(error)
    if error:
        failure_reasons.append("evaluation_error")
    if rate_limited:
        failure_reasons.append("provider_rate_limit")

    # 1. Answer present (when expected to answer).
    if should_answer and not answer_present:
        failure_reasons.append("expected_answer_but_empty")

    # 2. Fallback correct.
    if expected_fallback and not is_fallback:
        failure_reasons.append("expected_fallback_but_answered")
    if (not expected_fallback) and is_fallback and answer_present:
        failure_reasons.append("expected_answer_but_fallback")

    # 3. Citation requirement.
    citation_count = len(citations) if citations else 0
    if should_have_citations and citation_count == 0:
        failure_reasons.append("missing_citations")
    minimum_citations_met = citation_count >= min_citations
    if not minimum_citations_met:
        failure_reasons.append("minimum_citations_not_met")

    # 4. Keywords.
    missing_keywords = _find_missing_keywords(answer, expected_keywords)
    forbidden_found = _find_forbidden_keywords(answer, forbidden_keywords)
    if missing_keywords:
        failure_reasons.append("missing_keywords")
    if forbidden_found:
        failure_reasons.append("forbidden_keywords_found")

    # 5. Source.
    actual_sources = _actual_sources(citations)
    source_found = True
    if expected_source:
        source_found = expected_source in actual_sources
        if not source_found:
            failure_reasons.append("wrong_source")

    # 6. Block / answer direction.
    if should_answer and is_fallback:
        failure_reasons.append("incorrectly_blocked")
    if (not should_answer) and (not is_fallback) and answer_present:
        failure_reasons.append("should_have_blocked")

    # Final pass: NO failure reasons (rate_limit alone does not fail,
    # but it is surfaced for reporting).
    passed = len([r for r in failure_reasons if r not in {"provider_rate_limit"}]) == 0

    grounding = _grounding_meta(metadata)
    evidence_level = grounding.get("evidence_level")
    if not evidence_level and isinstance(metadata, dict):
        evidence_level = metadata.get("evidence_level")

    fallback_reason = (
        grounding.get("blocked_reason")
        or (metadata or {}).get("block_reason")
        or (metadata or {}).get("fallback_reason")
    )

    top_score = _top_score(metadata, citations)

    blocked = bool(isinstance(metadata, dict) and metadata.get("blocked"))
    if is_fallback:
        blocked = True

    return {
        "test_case_id": str(case_id),
        "rag_path": rag_path,
        "question": question,
        "expected_source_file": expected_source,
        "actual_source_files": actual_sources,
        "expected_keywords": expected_keywords,
        "found_keywords": [kw for kw in expected_keywords if kw not in missing_keywords],
        "missing_keywords": missing_keywords,
        "forbidden_keywords": forbidden_keywords,
        "forbidden_keywords_found": forbidden_found,
        "should_answer": should_answer,
        "expected_fallback": expected_fallback,
        "actual_blocked": bool(is_fallback),
        "answer_preview": (answer[:500] if answer else None),
        "citation_count": citation_count,
        "source_file_names": list(actual_sources),
        "evidence_level": evidence_level,
        "fallback_reason": fallback_reason,
        "top_score": top_score,
        "latency_ms": float(latency_ms or 0.0),
        "passed": bool(passed),
        "blocked": bool(blocked),
        "rate_limited": bool(rate_limited),
        "error": (str(error)[:200] if error else None),
        "failure_reasons": failure_reasons or None,
        "metadata": {
            "agentic": bool(isinstance(metadata, dict) and metadata.get("agentic")),
            "agentic_framework": (
                metadata.get("agentic_framework") if isinstance(metadata, dict) else None
            ),
            "retry_count": (
                metadata.get("retry_count") if isinstance(metadata, dict) else None
            ),
            "rewrite_attempted": (
                metadata.get("rewrite_attempted") if isinstance(metadata, dict) else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# Cross-path comparison
# ---------------------------------------------------------------------------

def classify_comparison(classic_result: dict, agentic_result: dict) -> str:
    """
    Compare classic and agentic outcomes for the same case.

    Returns one of:
        "improved"   — classic failed, agentic passed
        "regressed"  — classic passed, agentic failed
        "same_pass"  — both passed
        "same_fail"  — both failed
    """
    c_pass = bool(classic_result.get("passed"))
    a_pass = bool(agentic_result.get("passed"))
    if c_pass and a_pass:
        return "same_pass"
    if (not c_pass) and (not a_pass):
        return "same_fail"
    if (not c_pass) and a_pass:
        return "improved"
    if c_pass and (not a_pass):
        return "regressed"
    return "same_fail"


def build_summary(
    classic_results: list[dict],
    agentic_results: list[dict],
    *,
    pairs: list[tuple[str, str, str]],
    dataset_path: str,
    git_commit: Optional[str] = None,
) -> dict:
    """
    Build the top-level summary for the comparison JSON report.

    Args:
        classic_results: per-case result dicts for the classic path.
        agentic_results: per-case result dicts for the agentic path.
        pairs: list of (case_id, classic_outcome, agentic_outcome) tuples
            — produced by `classify_comparison`.
        dataset_path: dataset.json path (for the report header).
        git_commit: current HEAD commit hash (best effort; may be None).
    """
    def _avg(values: list[float]) -> Optional[float]:
        return round(sum(values) / len(values), 2) if values else None

    def _count_passed(results: list[dict]) -> int:
        return sum(1 for r in results if r.get("passed"))

    total_cases = len(classic_results)
    classic_passed = _count_passed(classic_results)
    agentic_passed = _count_passed(agentic_results)
    classic_failed = total_cases - classic_passed
    agentic_failed = total_cases - agentic_passed

    outcomes = [p[2] for p in pairs]
    improved = outcomes.count("improved")
    regressed = outcomes.count("regressed")
    same_pass = outcomes.count("same_pass")
    same_fail = outcomes.count("same_fail")

    classic_avg_latency = _avg([r.get("latency_ms", 0.0) for r in classic_results])
    agentic_avg_latency = _avg([r.get("latency_ms", 0.0) for r in agentic_results])
    latency_delta_ms = (
        round(agentic_avg_latency - classic_avg_latency, 2)
        if (classic_avg_latency is not None and agentic_avg_latency is not None)
        else None
    )
    latency_delta_percent = (
        round((latency_delta_ms / classic_avg_latency) * 100.0, 1)
        if (
            latency_delta_ms is not None
            and classic_avg_latency
            and classic_avg_latency > 0
        )
        else None
    )

    classic_avg_citations = _avg([float(r.get("citation_count", 0) or 0) for r in classic_results])
    agentic_avg_citations = _avg([float(r.get("citation_count", 0) or 0) for r in agentic_results])

    rate_limit_count = sum(
        1
        for r in (classic_results + agentic_results)
        if r.get("rate_limited")
    )

    classic_pass_rate = (classic_passed / total_cases * 100.0) if total_cases else 0.0
    agentic_pass_rate = (agentic_passed / total_cases * 100.0) if total_cases else 0.0

    recommendation = build_recommendation(
        classic_pass_rate=classic_pass_rate,
        agentic_pass_rate=agentic_pass_rate,
        classic_avg_latency=classic_avg_latency,
        agentic_avg_latency=agentic_avg_latency,
        rate_limit_count=rate_limit_count,
        total_cases=total_cases,
        improved=improved,
        regressed=regressed,
    )

    return {
        "generated_at": _now_iso(),
        "git_commit": git_commit,
        "dataset_path": dataset_path,
        "total_cases": total_cases,
        "classic_passed": classic_passed,
        "classic_failed": classic_failed,
        "classic_pass_rate": round(classic_pass_rate, 2),
        "agentic_passed": agentic_passed,
        "agentic_failed": agentic_failed,
        "agentic_pass_rate": round(agentic_pass_rate, 2),
        "improved_cases": improved,
        "regressed_cases": regressed,
        "same_pass_cases": same_pass,
        "same_fail_cases": same_fail,
        "classic_avg_latency_ms": classic_avg_latency,
        "agentic_avg_latency_ms": agentic_avg_latency,
        "latency_delta_ms": latency_delta_ms,
        "latency_delta_percent": latency_delta_percent,
        "classic_avg_citation_count": classic_avg_citations,
        "agentic_avg_citation_count": agentic_avg_citations,
        "openrouter_rate_limit_count": rate_limit_count,
        "recommendation": recommendation,
    }


def build_recommendation(
    *,
    classic_pass_rate: float,
    agentic_pass_rate: float,
    classic_avg_latency: Optional[float],
    agentic_avg_latency: Optional[float],
    rate_limit_count: int,
    total_cases: int,
    improved: int,
    regressed: int,
) -> str:
    """
    Translate raw metrics into a human-readable recommendation.

    Rules (in order):
        * If >20% of cases hit rate limits → noisy run; recommend rerun
          with a stable / mocked provider.
        * If agentic pass rate is meaningfully higher AND latency increase
          is < 2x → recommend a limited admin-only pilot.
        * If agentic pass rate is equal/within 2pp AND agentic is slower →
          keep agentic experimental; no change to default.
        * If agentic is worse → keep disabled; improve graph logic.
        * Default → keep experimental pending more data.
    """
    if total_cases > 0 and (rate_limit_count / (total_cases * 2)) > 0.20:
        return (
            "Results are noisy due to provider rate limits. "
            "Rerun with a stable provider or a mocked deterministic provider "
            "before drawing conclusions."
        )

    delta_pp = agentic_pass_rate - classic_pass_rate
    latency_ratio = (
        (agentic_avg_latency / classic_avg_latency)
        if (classic_avg_latency and agentic_avg_latency and classic_avg_latency > 0)
        else None
    )

    if delta_pp >= 2.0 and (latency_ratio is None or latency_ratio <= 2.0):
        return (
            "Agentic pass rate is meaningfully higher with acceptable latency. "
            "Recommend limited admin-only pilot (do NOT make default)."
        )
    if delta_pp >= 2.0 and latency_ratio is not None and latency_ratio > 2.0:
        return (
            "Agentic pass rate is higher but latency increase is large (>2x). "
            "Keep agentic experimental until latency is optimized."
        )
    if abs(delta_pp) <= 2.0 and latency_ratio is not None and latency_ratio > 1.5:
        return (
            "Agentic pass rate is similar but slower. Keep agentic experimental; "
            "do NOT make default."
        )
    if abs(delta_pp) <= 2.0:
        return (
            "Agentic pass rate is comparable to classic. Keep agentic experimental; "
            "do NOT make default."
        )
    if delta_pp < -2.0:
        return (
            "Agentic pass rate is worse than classic. Keep disabled; improve graph "
            "logic before re-evaluating."
        )
    return (
        "Inconclusive — collect more data before changing recommendations."
    )


def _now_iso() -> str:
    """ISO-8601 timestamp in UTC (no external imports required)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
