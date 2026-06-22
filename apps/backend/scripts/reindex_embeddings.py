#!/usr/bin/env python3
"""
Embedding Reindex Script — Phase 23

Dry-run diagnostic tool for embedding provider upgrades.

Reports:
  - Active embedding provider, model, and dimension
  - Qdrant collection name and current vector dimension
  - Whether a reindex is required (dimension mismatch)
  - Approximate document/chunk count in the collection

Usage:
  python scripts/reindex_embeddings.py              # dry-run (default)
  python scripts/reindex_embeddings.py --execute    # actual reindex (guarded)

WARNING: This script does NOT automatically reindex documents.
Changing the embedding model or dimension requires a full document reindex,
which must be done carefully to avoid downtime. Use --execute only after
backing up your data and verifying the new provider/configuration.

Requirements for --execute:
  - New embedding provider/model/dimension must be fully configured
  - New Qdrant collection will be created automatically
  - All indexed documents must be re-embedded with the new model
"""

import argparse
import sys
import os

# Add app to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_settings():
    from app.core.config import settings
    return settings


def _get_embedding_info():
    """Return a safe dict describing the active embedding configuration."""
    settings = _load_settings()
    return {
        "provider": settings.EMBEDDING_PROVIDER,
        "model": settings.EMBEDDING_MODEL,
        "dimension": settings.EMBEDDING_DIMENSION,
        "normalize": settings.EMBEDDING_NORMALIZE,
        "batch_size": settings.EMBEDDING_BATCH_SIZE,
        "device": settings.EMBEDDING_DEVICE,
    }


def _get_collection_info():
    """Inspect Qdrant collection. Returns safe dict, never raises."""
    settings = _load_settings()
    result = {
        "collection_name": settings.QDRANT_COLLECTION,
        "collection_dimension": "unknown",
        "vector_count": "unknown",
        "reindex_required": "unknown",
        "error": None,
    }
    try:
        from app.services.vector.qdrant_service import get_qdrant_client
        client = get_qdrant_client()
        info = client.get_collection(collection_name=settings.QDRANT_COLLECTION)
        # vectors is a VectorParams object (not a dict)
        vectors_config = info.config.params.vectors
        if vectors_config and hasattr(vectors_config, "size"):
            result["collection_dimension"] = vectors_config.size
        # Try to get point count
        try:
            result["vector_count"] = info.points_count
        except Exception:
            pass
        configured_dim = settings.EMBEDDING_DIMENSION
        if isinstance(result["collection_dimension"], int):
            result["reindex_required"] = result["collection_dimension"] != configured_dim
    except Exception as e:
        result["error"] = str(e)
    return result


def print_header():
    print("=" * 60)
    print("  Embedding Reindex Diagnostic Tool")
    print("=" * 60)
    print()


def print_embedding_config(info: dict):
    print(f"  Active Embedding Provider : {info['provider']}")
    print(f"  Model                     : {info['model']}")
    print(f"  Dimension                 : {info['dimension']}")
    print(f"  Normalize Embeddings      : {info['normalize']}")
    print(f"  Batch Size                : {info['batch_size']}")
    print(f"  Device                    : {info['device']}")
    print()


def print_collection_info(info: dict):
    print(f"  Collection Name           : {info['collection_name']}")
    print(f"  Collection Dimension      : {info['collection_dimension']}")
    print(f"  Vector Count              : {info['vector_count']}")
    reindex = info["reindex_required"]
    if reindex == "unknown":
        print(f"  Reindex Required          : Unknown (Qdrant inspection failed)")
    elif reindex is True:
        print(f"  Reindex Required          : YES — dimensions differ!")
    else:
        print(f"  Reindex Required          : No")
    if info["error"]:
        print(f"  Qdrant Warning            : {info['error']}")
    print()


def print_dry_run_warning():
    print("  [DRY RUN] No changes have been made.")
    print()
    print("  To perform an actual reindex, you must:")
    print("  1. Ensure the new embedding provider/model is configured")
    print("  2. Run with: python scripts/reindex_embeddings.py --execute")
    print()
    print("  WARNING: Reindexing will:")
    print("    - Delete the existing Qdrant collection")
    print("    - Re-embed all documents with the new model")
    print("    - Require significant compute resources")
    print("    - Cause temporary unavailability of Knowledge Base search")
    print()


def print_execute_guard():
    print("  ERROR: --execute requires explicit confirmation.")
    print()
    print("  To perform an actual reindex, run:")
    print("    python scripts/reindex_embeddings.py --execute --confirm")
    print()
    print("  Or back up and reindex by hand using the document indexing API.")
    print()


def run_dry_run():
    from app.providers.factory import FUTURE_EMBEDDING_PROVIDERS

    print_header()
    print("[DRY RUN MODE]")
    print()

    # Embedding config
    print("Active Embedding Configuration:")
    print("-" * 40)
    emb_info = _get_embedding_info()
    print_embedding_config(emb_info)

    # Qdrant collection
    print("Qdrant Collection Status:")
    print("-" * 40)
    col_info = _get_collection_info()
    print_collection_info(col_info)

    # Dimension mismatch warning
    if isinstance(col_info["reindex_required"], bool) and col_info["reindex_required"]:
        print("  *** DIMENSION MISMATCH DETECTED ***")
        print(f"  Configured dimension : {emb_info['dimension']}")
        print(f"  Collection dimension : {col_info['collection_dimension']}")
        print(f"  You MUST reindex before the new dimension is active.")
        print()

    # Future providers
    print("Future Embedding Providers (Phase 23+):")
    print("-" * 40)
    for provider, status in FUTURE_EMBEDDING_PROVIDERS.items():
        print(f"  {provider:<12} : {status}")
    print()

    # Reindex warning
    print("IMPORTANT:")
    print("-" * 40)
    print("  Changing the embedding model or dimension requires a FULL REINDEX.")
    print("  All existing document vectors will be deleted and recreated.")
    print("  Knowledge Base will be unavailable during reindex.")
    print()

    print_dry_run_warning()


def run_execute(confirm: bool):
    if not confirm:
        print_execute_guard()
        return

    print("[EXECUTE MODE — NOT YET IMPLEMENTED]")
    print()
    print("  Actual reindexing requires coordination with the document indexing")
    print("  pipeline and must be performed as a separate operation to avoid")
    print("  data loss. This script is a dry-run diagnostic tool only.")
    print()
    print("  To reindex manually:")
    print("  1. Set new EMBEDDING_PROVIDER / EMBEDDING_MODEL / EMBEDDING_DIMENSION")
    print("  2. Use the /api/admin/documents/reindex endpoint (if available)")
    print("     or write a custom reindex script using the document ingestion API.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Embedding reindex diagnostic tool (dry-run by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Attempt actual reindex (requires --confirm, not yet implemented)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm reindex operation (used with --execute, not yet implemented)",
    )
    args = parser.parse_args()

    if args.execute:
        run_execute(confirm=args.confirm)
    else:
        run_dry_run()


if __name__ == "__main__":
    main()