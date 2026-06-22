#!/usr/bin/env python3
"""
Embedding Reindex Diagnostic Tool — Phase 23

Dry-run diagnostic tool for embedding provider upgrades.

This script is READ-ONLY / DRY-RUN. It does NOT perform any actual reindexing.
Actual reindex execution is intentionally not implemented in Phase 23.
It will be handled in a future dedicated phase with backup, rollback,
collection naming, and validation gates.

Reports:
  - Active embedding provider, model, and dimension
  - Qdrant collection name and current vector dimension
  - Whether a reindex is required (dimension mismatch)
  - Approximate vector count in the collection
  - Future embedding providers and their status

Usage:
  python scripts/reindex_embeddings.py
"""

import sys
import os

# Add app to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_settings():
    from app.core.config import settings
    return settings


def _get_embedding_info():
    """Return a dict describing the active embedding configuration."""
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
        # vectors is a VectorParams object with .size attribute
        vectors_config = info.config.params.vectors
        if vectors_config and hasattr(vectors_config, "size"):
            result["collection_dimension"] = vectors_config.size
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


def main():
    from app.providers.factory import FUTURE_EMBEDDING_PROVIDERS

    print("=" * 60)
    print("  Embedding Reindex Diagnostic Tool")
    print("=" * 60)
    print()

    # Embedding config
    print("Active Embedding Configuration:")
    print("-" * 40)
    emb = _get_embedding_info()
    print(f"  Active Embedding Provider : {emb['provider']}")
    print(f"  Model                     : {emb['model']}")
    print(f"  Dimension                 : {emb['dimension']}")
    print(f"  Normalize Embeddings      : {emb['normalize']}")
    print(f"  Batch Size                : {emb['batch_size']}")
    print(f"  Device                    : {emb['device']}")
    print()

    # Qdrant collection
    print("Qdrant Collection Status:")
    print("-" * 40)
    col = _get_collection_info()
    print(f"  Collection Name           : {col['collection_name']}")
    print(f"  Collection Dimension      : {col['collection_dimension']}")
    print(f"  Vector Count              : {col['vector_count']}")
    reindex = col["reindex_required"]
    if reindex == "unknown":
        print(f"  Reindex Required          : Unknown (Qdrant inspection failed)")
    elif reindex is True:
        print(f"  Reindex Required          : YES — dimensions differ!")
    else:
        print(f"  Reindex Required          : No")
    if col["error"]:
        print(f"  Qdrant Warning            : {col['error']}")
    print()

    # Dimension mismatch warning
    if isinstance(col["reindex_required"], bool) and col["reindex_required"]:
        print("  *** DIMENSION MISMATCH DETECTED ***")
        print(f"  Configured dimension : {emb['dimension']}")
        print(f"  Collection dimension : {col['collection_dimension']}")
        print()

    # Future providers
    print("Future Embedding Providers (Phase 23+):")
    print("-" * 40)
    for provider, status in FUTURE_EMBEDDING_PROVIDERS.items():
        print(f"  {provider:<12} : {status}")
    print()

    # Warning
    print("IMPORTANT:")
    print("-" * 40)
    print("  This script is a DRY-RUN diagnostic tool only.")
    print("  It does NOT perform any actual reindexing.")
    print()
    print("  Changing the embedding model or dimension requires a")
    print("  full document reindex, which will be implemented in a")
    print("  future dedicated phase with proper backup, rollback,")
    print("  collection naming, and validation gates.")
    print()
    print("  No Qdrant collection has been modified.")
    print()


if __name__ == "__main__":
    main()