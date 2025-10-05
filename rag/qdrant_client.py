# rag/qdrant_client.py
from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

log = logging.getLogger(__name__)

# If your project exposes a config loader – great. Otherwise fallback to envs.
try:
    from utils.config import load_settings
except Exception:
    load_settings = None

def _get_cfg():
    if load_settings:
        try:
            return load_settings()
        except Exception:
            pass
    import os
    class Cfg:
        qdrant_url = os.getenv("QDRANT_URL", "")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "finsight")
    return Cfg()

class HybridQdrant:
    """
    A resilient, hybrid (dense+sparse) Qdrant wrapper.
    If Qdrant is unreachable or misconfigured, methods return empty results instead of raising.
    """
    def __init__(self) -> None:
        cfg = _get_cfg()
        self.collection = cfg.qdrant_collection or "finsight"
        self._ensured = False
        self._client: Optional[QdrantClient] = None

        try:
            # Avoid version check that may 404 behind certain proxies
            self._client = QdrantClient(
                url=cfg.qdrant_url,
                api_key=cfg.qdrant_api_key or None,
                timeout=10.0,
                prefer_grpc=False,
                # important: do not try to verify compatibility on init
                # newer qdrant_client does this lazily; if your version supports, pass check_compatibility=False
            )
            # lightweight ping: get collections as a health check
            _ = self._client.get_collections()
            log.info("Qdrant connected.")
        except Exception as e:
            log.warning("Qdrant unavailable (%s). Running without vector store.", e)
            self._client = None

    @property
    def client(self) -> Optional[QdrantClient]:
        return self._client

    def ensure_collections(self, dense_dim: Optional[int] = None) -> None:
        """
        Create/ensure a collection with a dense vector 'text' and sparse 'bm25'.
        No-ops if client is None.
        """
        if self._ensured or self._client is None:
            return
        try:
            exists = False
            cols = self._client.get_collections()
            for c in (cols.collections or []):
                if getattr(c, "name", "") == self.collection:
                    exists = True
                    break

            if not exists:
                vectors_config = rest.VectorParams(
                    size=int(dense_dim or 1024),  # safe default; real dim is set by your embedder
                    distance=rest.Distance.COSINE,
                )
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config={"text": vectors_config},
                    sparse_vectors_config={
                        "bm25": rest.SparseVectorParams(index=rest.SparseIndexParams())
                    },
                    optimizers_config=rest.OptimizersConfigDiff(indexing_threshold=10000),
                )
                # optional payload indexes
                try:
                    self._client.create_payload_index(self.collection, field_name="type", field_schema=rest.PayloadSchemaType.KEYWORD)
                except Exception:
                    pass
                try:
                    self._client.create_payload_index(self.collection, field_name="symbol", field_schema=rest.PayloadSchemaType.KEYWORD)
                except Exception:
                    pass
            self._ensured = True
        except Exception as e:
            log.warning("ensure_collections failed; continuing without vector store. err=%s", e)

    def hybrid_search(
        self,
        dense: List[float],
        sparse: Dict[int, float],
        limit: int = 24,
        must: Optional[List[rest.FieldCondition]] = None,
        should: Optional[List[rest.FieldCondition]] = None,
    ) -> List[rest.ScoredPoint]:
        """
        Perform a hybrid (dense+sparse) search. Returns [] if client is None or on error.
        """
        if self._client is None:
            return []
        try:
            # sparse vector wrapper
            sv = rest.SparseVector(indices=list(sparse.keys()), values=list(sparse.values())) if sparse else None

            # filter
            f = None
            if must or should:
                f = rest.Filter(must=must or [], should=should or [])

            # hybrid query
            res = self._client.query_points(
                collection_name=self.collection,
                query=rest.Query(
                    vector=rest.NamedVector(
                        name="text", vector=dense
                    ),
                    sparse_vector=sv,
                ),
                filter=f,
                limit=limit,
                with_payload=True,
            )
            return res.points or []
        except Exception as e:
            log.warning("hybrid_search failed; returning empty. err=%s", e)
            return []
