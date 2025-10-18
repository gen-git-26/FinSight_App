# mcp_connection/startup.py

from __future__ import annotations

import os
from typing import Callable, Optional

def _log(msg: str) -> None:
    print(f"[startup] {msg}")

def _get_embedder() -> Optional[Callable[[str], list[float]]]:
    """
    Return the embedding function used by the RAG pipeline.
    This must match your actual project. The function should take a string and return a vector (list[float]).
    If not available at import time, return None and we will skip alignment safely.
    """
    try:
        from tools import get_embedder  # you should expose this in tools.py
        fn = getattr(get_embedder, "__call__", None)
        return get_embedder if fn is None else get_embedder  # both cases return a callable
    except Exception:
        return None

def _embedding_dimension(embed_fn: Callable[[str], list[float]]) -> int:
    vec = embed_fn("dimension probe")
    return len(vec)

def _ensure_qdrant_alignment(collection_name: str = "finsight") -> None:
    """
    Ensure Qdrant collection vector size matches the active embedder dimension.
    If collection exists with wrong size, recreate it with the correct dimension.
    This is guarded with try/except to avoid blocking the app if Qdrant isn't configured.
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_key:
        _log("Qdrant env not set; skipping alignment.")
        return

    embed_fn = _get_embedder()
    if embed_fn is None:
        _log("Embedder not available at startup; skipping alignment.")
        return

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest
    except Exception:
        _log("qdrant-client not installed; skipping alignment.")
        return

    try:
        dim = _embedding_dimension(embed_fn)
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

        try:
            info = client.get_collection(collection_name)
            current = info.config.params.vectors.size  # type: ignore[attr-defined]
            if current != dim:
                _log(f"Vector size mismatch: current={current}, desired={dim}. Recreating collection.")
                client.delete_collection(collection_name)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
                )
                _log("Collection recreated with correct dimension. Please re-ingest documents.")
            else:
                _log(f"Collection '{collection_name}' already aligned at dim={dim}.")
        except Exception:
            # Create if missing
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
            )
            _log(f"Collection '{collection_name}' created at dim={dim}.")
    except Exception as e:
        _log(f"Failed to align Qdrant collection: {e}")

def app_startup() -> None:
    """
    Entry point to be called once when the app boots.
    Add health checks here if needed. Alignment is safe/no-op unless Qdrant+embedder exist.
    """
    _ensure_qdrant_alignment()
