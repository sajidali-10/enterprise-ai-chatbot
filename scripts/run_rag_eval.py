#!/usr/bin/env python3
"""
RAG Evaluation Runner

Evaluates retrieval quality using a question dataset.
Supports hybrid retrieval evaluation with configurable settings.

Usage:
    python scripts/run_rag_eval.py [--questions path/to/questions.jsonl] [--output path/to/results.json]

The questions file should be in JSONL format with fields:
- id: Unique question identifier
- question: The question text
- category: Question category (optional)
- expected_keywords: List of keywords that should appear in retrieved content (optional)
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "backend"))

# Set up Django/settings if needed for Flask-based apps
os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent.parent / "apps" / "backend"))


def load_questions(path: str) -> list[dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def evaluate_retrieval(
    question: str,
    expected_keywords: list[str],
) -> dict[str, Any]:
    """
    Evaluate retrieval for a single question.
    
    Uses the hybrid retriever and checks if expected keywords
    appear in the top retrieved chunks.
    """
    from app.rag.retriever import retrieve_chunks_with_settings
    from app.rag.hybrid_retriever import RetrievalConfig
    from app.core.config import settings
    
    # Retrieve chunks
    chunks, metadata = retrieve_chunks_with_settings(question, debug=True)
    
    result = {
        "question": question,
        "num_chunks_retrieved": len(chunks),
        "top_chunk_content_preview": chunks[0]["content"][:200] + "..." if chunks else "",
        "top_chunk_score": chunks[0]["score"] if chunks else None,
        "retrieval_metadata": metadata,
    }
    
    # Check if expected keywords appear in retrieved chunks
    if expected_keywords:
        all_content = " ".join(c["content"].lower() for c in chunks)
        keyword_matches = []
        for keyword in expected_keywords:
            if keyword.lower() in all_content:
                keyword_matches.append(keyword)
        result["keyword_coverage"] = len(keyword_matches) / len(expected_keywords)
        result["matched_keywords"] = keyword_matches
        result["unmatched_keywords"] = list(set(expected_keywords) - set(keyword_matches))
    else:
        result["keyword_coverage"] = None
        result["matched_keywords"] = []
        result["unmatched_keywords"] = []
    
    # Calculate retrieval metrics
    if chunks:
        # Average score of top 5 chunks
        scores = [c["score"] for c in chunks[:5]]
        result["avg_top5_score"] = sum(scores) / len(scores)
        result["min_top5_score"] = min(scores)
    else:
        result["avg_top5_score"] = 0.0
        result["min_top5_score"] = 0.0
    
    return result


def run_evaluation(
    questions_path: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Run full evaluation on a question dataset.
    
    Args:
        questions_path: Path to JSONL questions file.
        output_path: Optional path to write results JSON.
        
    Returns:
        Summary dict with aggregate metrics.
    """
    print(f"Loading questions from: {questions_path}")
    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")
    print()
    
    results = []
    for q in questions:
        qid = q.get("id", "unknown")
        question = q["question"]
        category = q.get("category", "unknown")
        expected_keywords = q.get("expected_keywords", [])
        
        print(f"[{qid}] {question}")
        print(f"       Category: {category}")
        
        try:
            result = evaluate_retrieval(question, expected_keywords)
            result["id"] = qid
            result["category"] = category
            result["success"] = True
            result["error"] = None
            
            print(f"       Retrieved: {result['num_chunks_retrieved']} chunks")
            print(f"       Top score: {result['top_chunk_score']:.4f}")
            print(f"       Avg top-5: {result['avg_top5_score']:.4f}")
            
            if result["keyword_coverage"] is not None:
                print(f"       Keyword coverage: {result['keyword_coverage']:.1%}")
                if result["unmatched_keywords"]:
                    print(f"       Missing keywords: {result['unmatched_keywords']}")
            
        except Exception as e:
            print(f"       ERROR: {str(e)}")
            result = {
                "id": qid,
                "question": question,
                "category": category,
                "success": False,
                "error": str(e),
            }
        
        results.append(result)
        print()
    
    # Compute aggregate statistics
    successful = [r for r in results if r.get("success", False)]
    summary = {
        "total_questions": len(questions),
        "successful": len(successful),
        "failed": len(questions) - len(successful),
        "avg_chunks_retrieved": (
            sum(r["num_chunks_retrieved"] for r in successful) / len(successful)
            if successful else 0
        ),
        "avg_top_score": (
            sum(r["top_chunk_score"] for r in successful if r.get("top_chunk_score")) / len(successful)
            if successful else 0
        ),
        "avg_avg_top5_score": (
            sum(r["avg_top5_score"] for r in successful) / len(successful)
            if successful else 0
        ),
        "avg_keyword_coverage": (
            sum(r["keyword_coverage"] for r in successful if r.get("keyword_coverage") is not None) /
            max(1, len([r for r in successful if r.get("keyword_coverage") is not None]))
        ),
        "category_breakdown": {},
    }
    
    # Category breakdown
    categories = set(r.get("category", "unknown") for r in results)
    for cat in categories:
        cat_results = [r for r in successful if r.get("category") == cat]
        if cat_results:
            summary["category_breakdown"][cat] = {
                "count": len(cat_results),
                "avg_score": sum(r.get("top_chunk_score", 0) for r in cat_results) / len(cat_results),
                "avg_keyword_coverage": (
                    sum(r.get("keyword_coverage", 0) for r in cat_results) / len(cat_results)
                ),
            }
    
    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average chunks retrieved: {summary['avg_chunks_retrieved']:.1f}")
    print(f"Average top score: {summary['avg_top_score']:.4f}")
    print(f"Average avg-top-5 score: {summary['avg_avg_top5_score']:.4f}")
    print(f"Average keyword coverage: {summary['avg_keyword_coverage']:.1%}")
    print()
    print("Category Breakdown:")
    for cat, stats in summary["category_breakdown"].items():
        print(f"  {cat}: count={stats['count']}, avg_score={stats['avg_score']:.4f}, "
              f"keyword_coverage={stats['avg_keyword_coverage']:.1%}")
    
    # Write results
    if output_path:
        output_data = {
            "summary": summary,
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG retrieval evaluation on a question dataset."
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="evals/questions.jsonl",
        help="Path to questions JSONL file (default: evals/questions.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write results JSON file (optional)",
    )
    
    args = parser.parse_args()
    
    # Resolve relative path
    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = Path(__file__).parent.parent / args.questions
    
    if not questions_path.exists():
        print(f"Error: Questions file not found: {questions_path}")
        sys.exit(1)
    
    summary = run_evaluation(str(questions_path), args.output)
    
    # Exit with error if any failures
    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()