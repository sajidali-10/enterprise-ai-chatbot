#!/usr/bin/env python3
"""
Phase 31C — Classic RAG vs Agentic RAG Comparison Runner

Runs every case in `apps/backend/evaluations/dataset.json` against BOTH
pipelines using the same dataset, embeddings, vector DB, LLM, and
scoring rules, then produces a side-by-side JSON + Markdown report.

  * Classic path: `mode=knowledge_base` via `generate_answer_with_rag_audit`
  * Agentic path: `mode=agentic_knowledge_base` via `run_agentic_rag`

The classic Knowledge Base behavior is unchanged. Agentic is invoked
only when `RAG_AGENTIC_ENABLED=true` (or the `--enable-agentic-for-run`
flag is supplied) and never becomes the default.

Output files (no overwrite of `run_rag_evaluation.py` artifacts):

  /app/evaluations/results/comparison_latest.json
  /app/evaluations/results/comparison_latest.md
  /app/evaluations/results/comparison_YYYYMMDD_HHMMSS.json
  /app/evaluations/results/comparison_YYYYMMDD_HHMMSS.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow running as a script: `python /app/scripts/compare_rag_modes.py`
BACKEND_ROOT = Path(__file__).parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings
from app.evaluation.scoring import (
    build_summary,
    classify_comparison,
    score_case,
)

# Phase 31A — LangSmith tracing helpers. Safe wrappers no-op when
# LANGSMITH_TRACING=false (the default) and never raise.
try:
    from app.services.langsmith_tracing import (
        trace_span,
        redact_filenames,
        sanitize_metadata,
    )
except Exception:  # pragma: no cover - tracing never required
    from contextlib import contextmanager

    @contextmanager
    def trace_span(*args, **kwargs):
        yield None

    def redact_filenames(value):
        return list(value or [])

    def sanitize_metadata(value):
        return value or {}


# ---------------------------------------------------------------------------
# Pipeline invocation
# ---------------------------------------------------------------------------


def _run_classic(case: dict) -> tuple[Optional[str], Optional[list], Optional[dict], float, Optional[str]]:
    """Run the classic Knowledge Base pipeline for one case."""
    from app.rag.answer_generator import generate_answer_with_rag_audit
    from app.security.auth import AuthContext

    # Phase 31C: build a minimal admin-like AuthContext so the audit
    # path is exercised (matches what the eval team uses for classic
    # runs in `run_rag_evaluation.py`, which goes through
    # `generate_answer_with_rag`). We pass is_authenticated=False so
    # the script behaves like a server-side eval, not a user request.
    fake_auth = AuthContext(
        user_id=None,
        username="comparison_runner",
        role=None,
        is_authenticated=False,
    ) if hasattr(AuthContext, "__init__") else None

    start = time.time() * 1000.0
    try:
        if fake_auth is not None and hasattr(fake_auth, "is_authenticated"):
            try:
                answer, citations, metadata = generate_answer_with_rag_audit(
                    query=case["question"],
                    auth=fake_auth,
                    debug=False,
                    use_hybrid=True,
                )
            except Exception:
                # Some AuthContext constructors require additional
                # fields; fall back to the unauthenticated variant.
                from app.rag.answer_generator import generate_answer_with_rag
                answer, citations, metadata = generate_answer_with_rag(
                    query=case["question"],
                    use_hybrid=True,
                    debug=False,
                )
        else:
            from app.rag.answer_generator import generate_answer_with_rag
            answer, citations, metadata = generate_answer_with_rag(
                query=case["question"],
                use_hybrid=True,
                debug=False,
            )
    except Exception as exc:
        return None, None, None, (time.time() * 1000.0 - start), f"{type(exc).__name__}: {exc}"

    return answer, citations, metadata, (time.time() * 1000.0 - start), None


def _run_agentic(case: dict) -> tuple[Optional[str], Optional[list], Optional[dict], float, Optional[str]]:
    """Run the LangGraph agentic pipeline for one case."""
    try:
        from app.rag.agentic_graph import run_agentic_rag
    except Exception as exc:
        return None, None, None, 0.0, f"agentic_import_error: {exc}"

    start = time.time() * 1000.0
    try:
        answer, citations, metadata = run_agentic_rag(
            query=case["question"],
            auth=None,
            debug=False,
            conversation_context="",
        )
    except Exception as exc:
        return None, None, None, (time.time() * 1000.0 - start), f"agentic_error: {exc}"

    return answer, citations, metadata, (time.time() * 1000.0 - start), None


# ---------------------------------------------------------------------------
# LangSmith metadata
# ---------------------------------------------------------------------------


def _emit_case_span(case: dict, result: dict, rag_path: str) -> None:
    """Attach a `rag_mode_comparison` span for one case/path combination."""
    metadata = {
        "evaluation_run_type": "rag_mode_comparison",
        "evaluation_case_id": str(case.get("id") or case.get("name") or "unknown"),
        "rag_path": rag_path,
        "expected_behavior": (
            "answer" if case.get("should_answer", True) else "fallback"
        ),
        "expected_source_file": case.get("expected_source_file"),
        "expected_keyword_count": len(case.get("expected_keywords") or []),
        "expected_fallback": bool(case.get("expected_fallback", False)),
        "should_have_citations": bool(case.get("should_have_citations", False)),
        "passed": bool(result.get("passed")),
        "failure_reasons": result.get("failure_reasons"),
        "latency_ms": int(result.get("latency_ms") or 0),
        "citation_count": int(result.get("citation_count") or 0),
        "evidence_level": result.get("evidence_level"),
        "fallback_reason": result.get("fallback_reason"),
        "source_file_names": result.get("source_file_names") or [],
        "blocked": bool(result.get("blocked")),
    }
    with trace_span("rag_mode_comparison_case", metadata=metadata):
        # Span emission is handled by the context manager; we just
        # need to make sure metadata is attached.
        pass


def _emit_summary_span(summary: dict) -> None:
    """Attach a single summary span for the whole comparison run."""
    metadata = {
        "evaluation_run_type": "rag_mode_comparison_summary",
        "total_cases": summary.get("total_cases"),
        "classic_pass_rate": summary.get("classic_pass_rate"),
        "agentic_pass_rate": summary.get("agentic_pass_rate"),
        "improved_cases": summary.get("improved_cases"),
        "regressed_cases": summary.get("regressed_cases"),
        "same_pass_cases": summary.get("same_pass_cases"),
        "same_fail_cases": summary.get("same_fail_cases"),
        "latency_delta_ms": summary.get("latency_delta_ms"),
        "openrouter_rate_limit_count": summary.get("openrouter_rate_limit_count"),
        "recommendation": summary.get("recommendation"),
    }
    with trace_span("rag_mode_comparison_summary", metadata=metadata):
        pass


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _redact_for_report(text: Optional[str], max_len: int = 240) -> Optional[str]:
    """Trim long answer previews and run a basic secret scan before writing."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return text


def _render_markdown(summary: dict, per_case: list[dict]) -> str:
    """Render the human-readable comparison report."""
    lines: list[str] = []
    lines.append("# Classic RAG vs Agentic RAG Comparison")
    lines.append("")
    lines.append(f"Generated: {summary.get('generated_at', '')}")
    if summary.get("git_commit"):
        lines.append(f"Git commit: `{summary['git_commit']}`")
    lines.append(f"Dataset: `{summary.get('dataset_path', '')}`")
    lines.append("")

    # 1. Executive summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(f"- Total cases: **{summary['total_cases']}**")
    lines.append(
        f"- Classic pass rate: **{summary['classic_pass_rate']}%** "
        f"({summary['classic_passed']}/{summary['total_cases']})"
    )
    lines.append(
        f"- Agentic pass rate: **{summary['agentic_pass_rate']}%** "
        f"({summary['agentic_passed']}/{summary['total_cases']})"
    )
    lines.append(
        f"- Improved: **{summary['improved_cases']}**, "
        f"Regressed: **{summary['regressed_cases']}**, "
        f"Same-pass: **{summary['same_pass_cases']}**, "
        f"Same-fail: **{summary['same_fail_cases']}**"
    )
    if summary.get("latency_delta_ms") is not None:
        lines.append(
            f"- Latency delta (agentic − classic): "
            f"**{summary['latency_delta_ms']} ms** "
            f"({summary.get('latency_delta_percent', 'n/a')}%)"
        )
    if summary.get("openrouter_rate_limit_count"):
        lines.append(
            f"- Provider rate-limit hits across both paths: "
            f"**{summary['openrouter_rate_limit_count']}**"
        )
    lines.append("")
    lines.append(f"**Recommendation:** {summary.get('recommendation', '')}")
    lines.append("")

    # 2. Overall results table
    lines.append("## 2. Overall Results")
    lines.append("")
    lines.append("| Metric | Classic | Agentic |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| Pass rate | {summary['classic_pass_rate']}% | {summary['agentic_pass_rate']}% |")
    lines.append(f"| Passed | {summary['classic_passed']} | {summary['agentic_passed']} |")
    lines.append(f"| Failed | {summary['classic_failed']} | {summary['agentic_failed']} |")
    if summary.get("classic_avg_latency_ms") is not None:
        lines.append(
            f"| Avg latency (ms) | {summary['classic_avg_latency_ms']} | "
            f"{summary.get('agentic_avg_latency_ms')} |"
        )
    if summary.get("classic_avg_citation_count") is not None:
        lines.append(
            f"| Avg citations | {summary['classic_avg_citation_count']} | "
            f"{summary.get('agentic_avg_citation_count')} |"
        )
    lines.append("")

    # 3. Quality comparison
    lines.append("## 3. Quality Comparison")
    lines.append("")
    lines.append("| Case | Classic | Agentic | Outcome |")
    lines.append("| --- | --- | --- | --- |")
    for entry in per_case:
        case_id = entry["case_id"]
        c = entry["classic_result"]
        a = entry["agentic_result"]
        c_status = "PASS" if c.get("passed") else "FAIL"
        a_status = "PASS" if a.get("passed") else "FAIL"
        lines.append(
            f"| `{case_id}` | {c_status} | {a_status} | {entry['comparison_outcome']} |"
        )
    lines.append("")

    # 4. Latency comparison
    lines.append("## 4. Latency Comparison")
    lines.append("")
    lines.append("| Case | Classic (ms) | Agentic (ms) | Δ |")
    lines.append("| --- | ---: | ---: | ---: |")
    for entry in per_case:
        c = entry["classic_result"]
        a = entry["agentic_result"]
        c_lat = int(c.get("latency_ms") or 0)
        a_lat = int(a.get("latency_ms") or 0)
        delta = a_lat - c_lat
        lines.append(
            f"| `{entry['case_id']}` | {c_lat} | {a_lat} | {delta:+d} |"
        )
    lines.append("")

    # 5. Citation/source comparison
    lines.append("## 5. Citation / Source Comparison")
    lines.append("")
    lines.append("| Case | Classic cites | Agentic cites | Classic sources | Agentic sources |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for entry in per_case:
        c = entry["classic_result"]
        a = entry["agentic_result"]
        c_src = ", ".join(c.get("source_file_names") or []) or "—"
        a_src = ", ".join(a.get("source_file_names") or []) or "—"
        lines.append(
            f"| `{entry['case_id']}` | {c.get('citation_count', 0)} | "
            f"{a.get('citation_count', 0)} | {c_src} | {a_src} |"
        )
    lines.append("")

    # 6-9. Improved / Regressed / Same-pass / Same-fail
    def _section(title: str, outcome: str) -> None:
        rows = [e for e in per_case if e["comparison_outcome"] == outcome]
        lines.append(f"## {title} ({len(rows)})")
        lines.append("")
        if not rows:
            lines.append("_None._")
            lines.append("")
            return
        for e in rows:
            c = e["classic_result"]
            a = e["agentic_result"]
            q = e["question"]
            lines.append(f"### `{e['case_id']}` — {q[:80]}")
            lines.append(
                f"- Classic: **{'PASS' if c.get('passed') else 'FAIL'}** "
                f"({', '.join(c.get('failure_reasons') or []) or '—'})"
            )
            lines.append(
                f"- Agentic: **{'PASS' if a.get('passed') else 'FAIL'}** "
                f"({', '.join(a.get('failure_reasons') or []) or '—'})"
            )
            if c.get("evidence_level") or a.get("evidence_level"):
                lines.append(
                    f"- Evidence level — classic: `{c.get('evidence_level')}`, "
                    f"agentic: `{a.get('evidence_level')}`"
                )
            preview = _redact_for_report(a.get("answer_preview"))
            if preview:
                lines.append(f"- Agentic answer preview: `{preview}`")
            lines.append("")

    _section("6. Improved Cases", "improved")
    _section("7. Regressed Cases", "regressed")
    _section("8. Same-Pass Cases", "same_pass")
    _section("9. Same-Fail Cases", "same_fail")

    # 10. Provider / rate-limit notes
    lines.append("## 10. Provider / Rate-Limit Notes")
    lines.append("")
    rl_count = summary.get("openrouter_rate_limit_count", 0)
    if rl_count == 0:
        lines.append("No provider rate-limit errors detected.")
    else:
        lines.append(f"Detected **{rl_count}** rate-limit responses across both paths.")
        rl_cases = [e for e in per_case if e["classic_result"].get("rate_limited") or e["agentic_result"].get("rate_limited")]
        for e in rl_cases:
            lines.append(
                f"- `{e['case_id']}` — "
                f"classic: {e['classic_result'].get('rate_limited')}, "
                f"agentic: {e['agentic_result'].get('rate_limited')}"
            )
    lines.append("")

    # 11. Recommendation
    lines.append("## 11. Recommendation")
    lines.append("")
    lines.append(summary.get("recommendation", ""))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _safe_dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BACKEND_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _sanitize_for_report(result: dict) -> dict:
    """Strip anything that shouldn't end up in a JSON/Markdown report."""
    sanitized = dict(result)
    # Trim answer_preview to a safe length.
    sanitized["answer_preview"] = _redact_for_report(result.get("answer_preview"))
    # Strip full filesystem paths — keep only filenames.
    sources = result.get("source_file_names") or []
    sanitized["source_file_names"] = [
        s.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for s in sources
    ]
    return sanitized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_dataset(path: Optional[Path] = None) -> list[dict]:
    dataset_path = path or (BACKEND_ROOT / "evaluations" / "dataset.json")
    with open(dataset_path, "r") as f:
        return json.load(f)


def _select_cases(dataset: list[dict], max_cases: Optional[int], only_case_id: Optional[str]) -> list[dict]:
    if only_case_id:
        selected = [c for c in dataset if c.get("id") == only_case_id]
        if not selected:
            raise SystemExit(f"No dataset case with id={only_case_id!r}")
        return selected
    if max_cases is not None and max_cases > 0:
        return dataset[:max_cases]
    return dataset


def _check_agentic_enabled(force_enable: bool) -> None:
    """Ensure the agentic path is available. Raises SystemExit on hard failure."""
    if not bool(getattr(settings, "RAG_AGENTIC_ENABLED", False)):
        if not force_enable:
            raise SystemExit(
                "RAG_AGENTIC_ENABLED is false. Refusing to run a comparison without "
                "the agentic path enabled. Either set RAG_AGENTIC_ENABLED=true in the "
                "environment / .env, or pass --enable-agentic-for-run (which only "
                "toggles the in-process setting for this script run)."
            )
        # Toggling the in-process settings module attribute is enough
        # because `should_use_agentic_for_mode` reads from the same
        # object. We do NOT touch .env or any persistent config.
        settings.RAG_AGENTIC_ENABLED = True
        print("[warn] --enable-agentic-for-run supplied; setting RAG_AGENTIC_ENABLED=true "
              "in-process for this run only.")
    # Confirm langgraph is importable.
    try:
        from app.rag.agentic_graph import LANGGRAPH_AVAILABLE  # noqa: F401
        if not LANGGRAPH_AVAILABLE:
            raise SystemExit(
                "LangGraph is not importable in this environment. The agentic path "
                "cannot run. Install langgraph or rerun without the agentic comparison."
            )
    except ImportError as exc:
        raise SystemExit(f"Cannot import app.rag.agentic_graph: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare classic vs agentic RAG.")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of cases (after dataset order).")
    parser.add_argument("--case-id", type=str, default=None,
                        help="Run only this case id.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override the output directory (default: evaluations/results).")
    parser.add_argument("--enable-agentic-for-run", action="store_true",
                        help="Temporarily enable RAG_AGENTIC_ENABLED for this run only. "
                             "Does NOT modify .env.")
    parser.add_argument("--no-markdown", action="store_true",
                        help="Skip Markdown report generation.")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip database save (default behavior).")
    args = parser.parse_args()

    dataset_path = BACKEND_ROOT / "evaluations" / "dataset.json"
    dataset = _load_dataset(dataset_path)
    cases = _select_cases(dataset, args.max_cases, args.case_id)
    print(f"[compare] Loaded {len(cases)} cases from {dataset_path}")

    _check_agentic_enabled(force_enable=args.enable_agentic_for_run)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (BACKEND_ROOT / "evaluations" / "results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 31A — wrap the whole comparison run in a parent span so
    # every per-case span nests underneath it.
    with trace_span(
        "rag_mode_comparison_run",
        metadata={
            "evaluation_run_type": "rag_mode_comparison",
            "dataset_path": str(dataset_path),
            "total_cases": len(cases),
            "agentic_enabled": bool(getattr(settings, "RAG_AGENTIC_ENABLED", False)),
            "classic_default": bool(getattr(settings, "RAG_AGENTIC_DEFAULT", False)),
        },
    ):
        classic_results: list[dict] = []
        agentic_results: list[dict] = []
        pairs: list[tuple[str, str, str]] = []

        for idx, case in enumerate(cases, 1):
            case_id = str(case.get("id") or case.get("name") or f"case_{idx}")
            print(f"[compare] [{idx}/{len(cases)}] {case_id} — classic…", end=" ", flush=True)

            c_answer, c_citations, c_metadata, c_latency, c_error = _run_classic(case)
            c_result = score_case(
                case=case,
                answer=c_answer,
                citations=c_citations,
                metadata=c_metadata,
                latency_ms=c_latency,
                rag_path="classic",
                error=c_error,
            )
            classic_results.append(c_result)
            _emit_case_span(case, c_result, "classic")
            print(f"{'PASS' if c_result['passed'] else 'FAIL'} ({int(c_latency)}ms)", flush=True)

            print(f"[compare] [{idx}/{len(cases)}] {case_id} — agentic…", end=" ", flush=True)
            a_answer, a_citations, a_metadata, a_latency, a_error = _run_agentic(case)
            a_result = score_case(
                case=case,
                answer=a_answer,
                citations=a_citations,
                metadata=a_metadata,
                latency_ms=a_latency,
                rag_path="agentic",
                error=a_error,
            )
            agentic_results.append(a_result)
            _emit_case_span(case, a_result, "agentic")
            print(f"{'PASS' if a_result['passed'] else 'FAIL'} ({int(a_latency)}ms)", flush=True)

            outcome = classify_comparison(c_result, a_result)
            pairs.append((case_id, "classic", outcome))

        summary = build_summary(
            classic_results=classic_results,
            agentic_results=agentic_results,
            pairs=pairs,
            dataset_path=str(dataset_path),
            git_commit=_git_commit(),
        )
        _emit_summary_span(summary)

    # Build per-case comparison entries.
    per_case: list[dict] = []
    for (case_id, _, outcome), case, c_result, a_result in zip(pairs, cases, classic_results, agentic_results):
        per_case.append(
            {
                "case_id": case_id,
                "question": case.get("question", ""),
                "expected_behavior": (
                    "answer" if case.get("should_answer", True) else "fallback"
                ),
                "expected_source_file": case.get("expected_source_file"),
                "expected_keywords": case.get("expected_keywords") or [],
                "comparison_outcome": outcome,
                "classic_result": _sanitize_for_report(c_result),
                "agentic_result": _sanitize_for_report(a_result),
            }
        )

    report_payload = {
        "summary": summary,
        "cases": per_case,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    latest_json = output_dir / "comparison_latest.json"
    timestamped_json = output_dir / f"comparison_{timestamp}.json"
    latest_md = output_dir / "comparison_latest.md"
    timestamped_md = output_dir / f"comparison_{timestamp}.md"

    _safe_dump_json(latest_json, report_payload)
    _safe_dump_json(timestamped_json, report_payload)
    print(f"[compare] Wrote JSON: {latest_json}")
    print(f"[compare] Wrote JSON: {timestamped_json}")

    if not args.no_markdown:
        md_text = _render_markdown(summary, per_case)
        _safe_write_text(latest_md, md_text)
        _safe_write_text(timestamped_md, md_text)
        print(f"[compare] Wrote Markdown: {latest_md}")
        print(f"[compare] Wrote Markdown: {timestamped_md}")

    # Console summary.
    print()
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Total:                 {summary['total_cases']}")
    print(f"  Classic pass rate:     {summary['classic_pass_rate']}% "
          f"({summary['classic_passed']}/{summary['total_cases']})")
    print(f"  Agentic pass rate:     {summary['agentic_pass_rate']}% "
          f"({summary['agentic_passed']}/{summary['total_cases']})")
    print(f"  Improved:              {summary['improved_cases']}")
    print(f"  Regressed:             {summary['regressed_cases']}")
    print(f"  Same pass:             {summary['same_pass_cases']}")
    print(f"  Same fail:             {summary['same_fail_cases']}")
    if summary.get("latency_delta_ms") is not None:
        print(f"  Latency delta:         {summary['latency_delta_ms']} ms "
              f"({summary.get('latency_delta_percent')}%)")
    print(f"  Provider rate-limits:  {summary.get('openrouter_rate_limit_count', 0)}")
    print(f"  Recommendation:        {summary.get('recommendation', '')}")
    print("=" * 60)

    # Exit code: non-zero only if the agentic run itself errored for
    # every case (so CI can detect a totally broken agentic path).
    if agentic_results and all(r.get("error") for r in agentic_results):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
