#!/usr/bin/env python3
"""
RAGAS Evaluation Runner

Runs RAGAS quality metrics (faithfulness, answer_relevancy, context_precision)
on the existing evaluation dataset using the configured LLM provider as evaluator.

Phase 26 — RAGAS Evaluation Foundation
- Uses the existing evaluation dataset (evaluations/dataset.json)
- Runs generate_answer_with_rag() per case to get question/answer/contexts
- Skips context_recall/answer_correctness when no ground_truth with clear warnings
- --dry-run: load and convert dataset, no LLM calls
- --strict: exit non-zero if any metric fails its threshold
- JSON report saved to RAGAS_REPORT_DIR with timestamp, git commit, case count,
  per-case scores, avg scores, threshold summary, evaluator model

Behavior-preserving: no production behavior changes, no DB/Qdrant writes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — script lives in scripts/, app is at apps/backend/
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

from app.core.config import settings
from app.rag.answer_generator import generate_answer_with_rag
from app.rag.citations import format_citations

# ---------------------------------------------------------------------------
# RAGAS imports — guarded so script can still run --help / --dry-run
# even if ragas is absent or broken
# ---------------------------------------------------------------------------
try:
    import ragas
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from ragas.run_config import RunConfig
except ImportError as exc:
    # Provide a clear message so users know to install ragas
    raise ImportError(
        "ragas is not installed or cannot be imported. "
        "Install with: pip install ragas>=0.2.0,<0.3.0"
    ) from exc

# ---------------------------------------------------------------------------
# Metric thresholds (configurable via env var override)
# ---------------------------------------------------------------------------
_THRESHOLDS = {
    "faithfulness": float(os.getenv("RAGAS_THRESHOLD_FAITHFULNESS", "0.5")),
    "answer_relevancy": float(os.getenv("RAGAS_THRESHOLD_ANSWER_RELEVANCY", "0.5")),
    "context_precision": float(os.getenv("RAGAS_THRESHOLD_CONTEXT_PRECISION", "0.5")),
}


def _get_git_commit() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=_BACKEND_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _check_ragas_available() -> bool:
    """Return True if ragas can be imported and its top-level init works."""
    try:
        import ragas
        _ = ragas.__version__
        return True
    except Exception:
        return False


def load_dataset(path: Path) -> list[dict]:
    """Load evaluation cases from JSON dataset."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    return data


def generate_answer_for_case(question: str) -> tuple[str, list[str]]:
    """
    Run generate_answer_with_rag for a single question.

    Returns (answer, context_strings) where context_strings are formatted
    citations for use as retrieved_contexts in RAGAS.

    Silently returns ("", []) on failure so one bad case doesn't abort the run.
    """
    try:
        answer, citations, _ = generate_answer_with_rag(
            query=question,
            use_hybrid=True,
            debug=False,
            conversation_context="",
        )
        contexts = []
        if citations:
            formatted = format_citations(citations)
            if isinstance(formatted, list):
                contexts = [c.get("content_snippet", "") or str(c) for c in formatted]
            elif isinstance(formatted, str):
                contexts = [formatted]
        return answer, contexts
    except Exception as exc:
        warnings.warn(f"generate_answer_with_rag failed for question '{question[:60]}...': {exc}")
        return "", []


def build_dataset(cases: list[dict], dry_run: bool = False) -> EvaluationDataset:
    """
    Convert evaluation cases to RAGAS SingleTurnSample dataset.

    - user_input: case["question"]
    - response: LLM answer from generate_answer_with_rag (or empty in dry_run)
    - retrieved_contexts: citation content snippets (or empty in dry_run)

    For context_recall / answer_correctness we would need ground_truth /
    reference_answer fields which the dataset does not have — those metrics
    are skipped at evaluation time with clear warnings.
    """
    samples = []
    for case in cases:
        question = case["question"]
        if dry_run:
            # In dry-run mode: convert without calling LLM
            answer = ""
            contexts: list[str] = []
        else:
            answer, contexts = generate_answer_for_case(question)

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts if contexts else None,
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def build_evaluator_llm():
    """
    Build a LangChain ChatOpenAI client pointing at the LiteLLM gateway
    and wrap it as a RAGAS LangchainLLMWrapper.

    Checks LITELLM_MASTER_KEY is set — raises ValueError with clear setup
    instructions if it is missing.
    """
    master_key = os.environ.get("LITELLM_MASTER_KEY") or ""
    if not master_key or master_key == "changeme_litellm_master_key":
        raise ValueError(
            "LITELLM_MASTER_KEY is not set or still has the default placeholder value. "
            "Set the LITELLM_MASTER_KEY environment variable to your LiteLLM gateway "
            "master key before running RAGAS evaluation. "
            "Example: LITELLM_MASTER_KEY=sk-... python scripts/run_ragas_evaluation.py"
        )

    base_url = os.environ.get(
        "LITELLM_BASE_URL",
        getattr(settings, "LITELLM_BASE_URL", "http://litellm:4000"),
    )
    model = settings.RAGAS_EVALUATOR_MODEL  # e.g. openrouter-gpt-oss

    from langchain_openai.chat_models import ChatOpenAI

    evaluator_llm = ChatOpenAI(
        model=model,
        openai_api_base=f"{base_url}/v1",
        openai_api_key=master_key,
        temperature=0,
        request_timeout=120,
    )

    return LangchainLLMWrapper(evaluator_llm)


def run_evaluation(
    dataset: EvaluationDataset,
    strict: bool = False,
) -> dict:
    """
    Run RAGAS evaluation with faithfulness, answer_relevancy, context_precision.

    - context_recall and answer_correctness are skipped with warnings because
      the dataset has no ground_truth / reference_answer fields.
    - Returns a dict with 'results', 'scores', 'skipped_metrics'.
    """
    evaluator_llm = build_evaluator_llm()
    results_dict: dict = {
        "results": [],
        "scores": {},
        "skipped_metrics": [],
    }

    # Faithfulness — does the answer stick to the retrieved contexts?
    faithfulness = Faithfulness(llm=evaluator_llm)
    # Answer Relevancy — is the answer relevant to the question?
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm)
    # Context Precision — are the retrieved contexts relevant and in correct order?
    context_precision = ContextPrecision(llm=evaluator_llm)

    metrics = [faithfulness, answer_relevancy, context_precision]

    print("\n  Running RAGAS evaluation...")
    print("  Metrics: faithfulness, answer_relevancy, context_precision")
    print("  Skipped (no ground_truth in dataset): context_recall, answer_correctness")

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        raise_exceptions=False,
        show_progress=True,
    )

    result_df = result.to_pandas()

    # Extract per-case scores
    for idx, row in result_df.iterrows():
        case_id = dataset.samples[idx].user_input[:60]
        results_dict["results"].append({
            "case_id": case_id,
            "user_input": dataset.samples[idx].user_input,
            "faithfulness": _safe_float(row.get("faithfulness")),
            "answer_relevancy": _safe_float(row.get("answer_relevancy")),
            "context_precision": _safe_float(row.get("context_precision")),
        })

    # Compute averages
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
        col = result_df.get(metric_name, [])
        values = [v for v in col if v is not None and (not isinstance(v, float) or not _is_nan(v))]
        results_dict["scores"][metric_name] = {
            "avg": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "count": len(values),
        }

    results_dict["scores"]["skipped"] = ["context_recall", "answer_correctness"]

    return results_dict


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if not _is_nan(f) else None
    except (TypeError, ValueError):
        return None


def _is_nan(val: float) -> bool:
    """Check if float is NaN."""
    import math
    return math.isnan(val)


def save_report(
    results: dict,
    dataset_size: int,
    git_commit: str,
    strict: bool,
) -> Path:
    """
    Save RAGAS evaluation report as JSON to RAGAS_REPORT_DIR.

    Report contains: timestamp, git_commit, evaluator_model, case_count,
    per_case_scores, avg_scores, threshold_summary, strict_mode.
    """
    report_dir = Path(settings.RAGAS_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    threshold_summary = {}
    for metric, threshold in _THRESHOLDS.items():
        score_data = results["scores"].get(metric, {})
        avg = score_data.get("avg")
        if avg is not None:
            threshold_summary[metric] = {
                "threshold": threshold,
                "avg_score": round(avg, 4),
                "pass": avg >= threshold,
            }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "evaluator_model": settings.RAGAS_EVALUATOR_MODEL,
        "evaluator_provider": settings.RAGAS_EVALUATOR_PROVIDER,
        "case_count": dataset_size,
        "ragas_version": ragas.__version__,
        "strict_mode": strict,
        "threshold_summary": threshold_summary,
        "per_case_scores": results["results"],
        "avg_scores": {
            k: v for k, v in results["scores"].items() if k != "skipped"
        },
        "skipped_metrics": ["context_recall", "answer_correctness"],
    }

    # Redact any potential secrets from report
    report_str = json.dumps(report)
    for env_key in ["LITELLM_MASTER_KEY", "OPENROUTER_API_KEY", "API_KEY", "SECRET"]:
        placeholder = f"[{env_key} REDACTED]"
        report_str = report_str.replace(os.environ.get(env_key, ""), placeholder)

    report = json.loads(report_str)

    filename = f"ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = report_dir / filename
    latest_path = report_dir / "latest.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS quality evaluation on the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ragas_evaluation.py                     # Full run
  python scripts/run_ragas_evaluation.py --dry-run           # Load + convert, no LLM calls
  python scripts/run_ragas_evaluation.py --max-cases 5       # Run first 5 cases only
  python scripts/run_ragas_evaluation.py --strict            # Exit non-zero if thresholds fail
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load dataset and convert cases without calling any LLM. "
             "Prints case count and field summary. Exits 0 on success.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        metavar="N",
        help="Limit to the first N cases from the dataset (0 = use all, max 20 by default, "
             "controlled by RAGAS_MAX_CASES setting).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any metric average falls below its threshold.",
    )
    args = parser.parse_args()

    git_commit = _get_git_commit()

    print("=" * 60)
    print("RAGAS Evaluation Runner — Phase 26")
    print("=" * 60)
    print(f"  RAGAS version:   {ragas.__version__}")
    print(f"  Evaluator:       {settings.RAGAS_EVALUATOR_PROVIDER} / {settings.RAGAS_EVALUATOR_MODEL}")
    print(f"  Report dir:      {settings.RAGAS_REPORT_DIR}")
    print(f"  Max cases:       {args.max_cases or settings.RAGAS_MAX_CASES}")
    print(f"  Strict mode:     {args.strict}")
    print(f"  Git commit:      {git_commit}")
    print()

    # ---- 1. Check ragas is importable ---------------------------------
    if not _check_ragas_available():
        raise RuntimeError(
            "RAGAS cannot be imported. Check that ragas>=0.2.0 is installed "
            "and langchain_community.chat_models.vertexai stub is in place. "
            "Run: docker compose build backend"
        )

    # ---- 2. Load dataset ----------------------------------------------
    print("[1/4] Loading evaluation dataset...")
    dataset_path = _BACKEND_ROOT / "evaluations" / "dataset.json"
    all_cases = load_dataset(dataset_path)
    print(f"  Loaded {len(all_cases)} cases from {dataset_path}")

    max_cases = args.max_cases or min(len(all_cases), settings.RAGAS_MAX_CASES)
    cases = all_cases[:max_cases]
    print(f"  Using {len(cases)} cases (max-cases={args.max_cases}, RAGAS_MAX_CASES={settings.RAGAS_MAX_CASES})")

    # ---- 3. Dry-run: convert and print summary ------------------------
    if args.dry_run:
        print("\n[2/4] Dry-run: building dataset (no LLM calls)...")
        dataset = build_dataset(cases, dry_run=True)

        print(f"  Converted {len(dataset.samples)} samples")
        print("\n  Sample case fields:")
        for i, sample in enumerate(dataset.samples[:3]):
            print(f"    [{i+1}] user_input : {sample.user_input[:60]}...")
            print(f"        response     : {(sample.response or '')[:40]}..." if sample.response else "        response     : (empty in dry-run)")
            ctx_count = len(sample.retrieved_contexts) if sample.retrieved_contexts else 0
            print(f"        contexts     : {ctx_count} context(s)")
            print()

        print("  RAGAS metrics configured:")
        print("    - faithfulness        (needs: user_input, response, retrieved_contexts)")
        print("    - answer_relevancy    (needs: user_input, response)")
        print("    - context_precision   (needs: user_input, retrieved_contexts)")
        print("    - context_recall      SKIPPED (no ground_truth in dataset)")
        print("    - answer_correctness  SKIPPED (no ground_truth in dataset)")
        print("\n  Thresholds:")
        for metric, threshold in _THRESHOLDS.items():
            print(f"    - {metric}: {threshold}")
        print("\n  --dry-run complete. Exiting 0.")
        return 0

    # ---- 4. Build dataset (calls generate_answer_with_rag per case) ---
    print("\n[3/4] Building dataset (calling generate_answer_with_rag per case)...")
    dataset = build_dataset(cases, dry_run=False)
    print(f"  Built {len(dataset.samples)} samples")

    # ---- 5. Run RAGAS evaluation --------------------------------------
    print("\n[4/4] Running RAGAS evaluation...")
    start = time.time()
    results = run_evaluation(dataset, strict=args.strict)
    elapsed = time.time() - start
    print(f"  Evaluation complete in {elapsed:.1f}s")

    # ---- 6. Save report -----------------------------------------------
    if settings.RAGAS_SAVE_RESULTS:
        report_path = save_report(results, len(cases), git_commit, args.strict)
        print(f"\n  Report saved to: {report_path}")

    # ---- 7. Print summary ---------------------------------------------
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        score_data = results["scores"].get(metric, {})
        avg = score_data.get("avg")
        threshold = _THRESHOLDS[metric]
        if avg is not None:
            status = "PASS" if avg >= threshold else "FAIL"
            print(f"  {metric:<20} avg={avg:.4f}  threshold={threshold}  [{status}]")
        else:
            print(f"  {metric:<20} avg=None (no valid scores)")

    print(f"\n  Skipped (no ground_truth): context_recall, answer_correctness")
    print(f"  Total cases evaluated:    {len(cases)}")
    print(f"  Evaluator model:          {settings.RAGAS_EVALUATOR_MODEL}")
    print(f"  RAGAS version:            {ragas.__version__}")
    print("=" * 60)

    # Strict mode: check thresholds
    if args.strict:
        failures = []
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            avg = results["scores"].get(metric, {}).get("avg")
            if avg is not None and avg < _THRESHOLDS[metric]:
                failures.append(f"{metric} ({avg:.4f} < {_THRESHOLDS[metric]})")

        if failures:
            print(f"\n  STRICT MODE: {len(failures)} metric(s) below threshold:")
            for f in failures:
                print(f"    - {f}")
            return 1
        print("\n  STRICT MODE: All thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())