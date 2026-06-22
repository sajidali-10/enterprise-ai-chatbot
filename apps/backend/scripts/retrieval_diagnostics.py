#!/usr/bin/env python3
"""
Retrieval & Reranker Diagnostics — Phase 25

Dry-run diagnostic script for retriever and reranker upgrade foundation.

This script is READ-ONLY. It does NOT perform any retrieval, write to
Qdrant, MinIO, the database, or any file. It prints a snapshot of the
current retrieval and reranker configuration for admin visibility.

Reports:
  - Active retriever provider and retrieval mode
  - Top K, score threshold, and candidate K settings
  - Hybrid search enabled/disabled and weights
  - Active reranker provider, enabled flag, top N, and model
  - Future reranker providers and their planned/available status

Usage:
  python scripts/retrieval_diagnostics.py
"""

import sys
import os

# Add app to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def _print_row(label: str, value) -> None:
    print(f"  {label:<35} {value}")


def main() -> None:
    # Load settings (read-only — no side effects)
    from app.core.config import settings

    _print_section("Retrieval & Reranker Diagnostics — Phase 25")
    print("\nThis script is READ-ONLY. No writes to Qdrant, MinIO, or database.")

    # Active providers
    _print_section("Active Providers")
    _print_row("Retriever Provider", settings.RETRIEVER_PROVIDER)
    _print_row("Reranker Provider", settings.RERANKER_PROVIDER)

    # Retrieval strategy
    _print_section("Retrieval Strategy")
    _print_row("Retrieval Mode", settings.RETRIEVAL_MODE)
    _print_row("Top K (final display)", settings.RAG_TOP_K)
    _print_row("Score Threshold", settings.RAG_SCORE_THRESHOLD)
    _print_row("Candidate K", settings.RETRIEVAL_CANDIDATE_K)

    # Hybrid search
    _print_section("Hybrid Search")
    _print_row("Hybrid Search Enabled", settings.HYBRID_SEARCH_ENABLED)
    _print_row("Hybrid Vector Weight", settings.RETRIEVAL_VECTOR_WEIGHT)
    _print_row("Hybrid Keyword Weight", settings.RETRIEVAL_KEYWORD_WEIGHT)

    # Reranker
    _print_section("Reranker")
    _print_row("Reranker Enabled", settings.RERANKER_ENABLED)
    _print_row("Reranker Top N", settings.RERANKER_TOP_N)
    _print_row("Reranker Model", settings.RERANKER_MODEL)

    # Future reranker providers
    _print_section("Future Reranker Providers")
    from app.providers.factory import FUTURE_RERANKER_PROVIDERS

    for name, status in FUTURE_RERANKER_PROVIDERS.items():
        active_marker = " ← ACTIVE" if name == "none" else ""
        print(f"  {name:<20} {status}{active_marker}")

    _print_section("End of Diagnostics")
    print("\nPhase 25: This is a configuration snapshot only.")
    print("To change retrieval behavior, update environment variables and restart.")


if __name__ == "__main__":
    main()