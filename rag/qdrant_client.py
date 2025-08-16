# rag/qdrant_client.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from utils.config import load_settings
import asyncio

# Local tiny RRF to avoid circular imports
def _rrf(lists: List[List[rest.ScoredPoint]], k: int = 60) -> List[rest.ScoredPoint]:
    scores: Dict[str, float] = {}
    pick: Dict[str, rest.ScoredPoint] = {}
    for lst in lists:
        for rank, p in enumerate(lst, start=1):
            pid = str(p.id)
            pick[pid] = p
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [pick[pid] for pid, _ in order]

class HybridQdrant:
    def __init__(self) -> None:
        cfg = load_settings()
        self.client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
        self.collection = cfg.qdrant_collection
        self._ensured = False

    def ensure_collections(self, dense_dim: Optional[int] = None) -> None:
        """Create the collection with both dense and sparse vectors if missing.
        If dense_dim is not provided, try to infer from existing collection or default to 3072."""
        try:
            self.client.create_payload_index(self.collection, field_name="user", field_schema=rest.PayloadSchemaType.KEYWORD)
        except Exception:
            pass
        if self._ensured:
            return
        colls = [c.name for c in self.client.get_collections().collections]
        if self.collection in colls:
            # ensure payload indexes exist
            try:
                self.client.create_payload_index(self.collection, field_name="symbol", field_schema=rest.PayloadSchemaType.KEYWORD)
            except Exception:
                pass
            try:
                self.client.create_payload_index(self.collection, field_name="type", field_schema=rest.PayloadSchemaType.KEYWORD)
            except Exception:
                pass
            try:
                self.client.create_payload_index(self.collection, field_name="date", field_schema=rest.PayloadSchemaType.TEXT)
            except Exception:
                pass
            self._ensured = True
            return

        if dense_dim is None:
            # safe default for text-embedding-3-large (3072). The exact dim will be correct in most setups.
            dense_dim = 3072
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                "text": rest.VectorParams(size=dense_dim, distance=rest.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": rest.SparseVectorParams()
            },
        )
        # payload indexes
        self.client.create_payload_index(self.collection, field_name="symbol", field_schema=rest.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, field_name="type", field_schema=rest.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, field_name="date", field_schema=rest.PayloadSchemaType.TEXT)
        self._ensured = True

    async def upsert_snippets(self, items: List[Dict[str, Any]]) -> None:
        """Upsert a list of items with fields: id?, text, symbol?, type?, date?, source? and precomputed vectors optional.
        We build both dense ('text') and sparse ('bm25') vectors."""
        from rag.embeddings import embed_texts, sparse_from_text  # lazy to avoid cycles
        self.ensure_collections()
        texts = [str(it.get("text", "")) for it in items]
        dense_vecs = await embed_texts(texts)
        points: List[rest.PointStruct] = []
        for i, it in enumerate(items):
            pid = str(it.get("id", f"snip-{i}"))
            payload = {
                "text": texts[i],
                "symbol": it.get("symbol", ""),
                "type": it.get("type", ""),
                "date": it.get("date", ""),
                "source": it.get("source", ""),
            }
            points.append(
                rest.PointStruct(
                    id=pid,
                    vector={
                        "text": dense_vecs[i],
                        "bm25": sparse_from_text(texts[i]),
                    },
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def mk_filters(
        self,
        date_gte: Optional[str] = None,
        symbol_in: Optional[List[str]] = None,
        type_in: Optional[List[str]] = None,
    ) -> rest.Filter:
        must: List[rest.FieldCondition] = []
        if date_gte:
            must.append(rest.FieldCondition(key="date", range=rest.Range(gte=date_gte)))
        if symbol_in:
            vals = [s for s in symbol_in if s]
            if vals:
                must.append(rest.FieldCondition(key="symbol", match=rest.MatchAny(any=vals)))
        if type_in:
            vals = [t for t in type_in if t]
            if vals:
                must.append(rest.FieldCondition(key="type", match=rest.MatchAny(any=vals)))
        return rest.Filter(must=must) if must else None  # type: ignore

    def hybrid_search(
        self,
        *,
        dense: List[float],
        sparse: rest.SparseVector,
        limit: int = 12,
        must: Optional[List[rest.FieldCondition]] = None,
        should: Optional[List[rest.FieldCondition]] = None,
    ) -> List[rest.ScoredPoint]:
        """Run dense and sparse searches separately and fuse with RRF."""
        self.ensure_collections()
        flt_must = rest.Filter(must=must) if must else None
        flt_should = rest.Filter(should=should) if should else None

        res_dense_must = self.client.search(
            collection_name=self.collection, query_vector=("text", dense), query_filter=flt_must, limit=limit, with_payload=True
        )
        res_sparse_must = self.client.search(
            collection_name=self.collection, query_vector=rest.NamedSparseVector(name="bm25", vector=sparse), query_filter=flt_must, limit=limit, with_payload=True
        )
        fused_must = _rrf([res_dense_must, res_sparse_must])

        if flt_should is None:
            return fused_must[:limit]

        res_dense_should = self.client.search(
            collection_name=self.collection, query_vector=("text", dense), query_filter=flt_should, limit=limit, with_payload=True
        )
        res_sparse_should = self.client.search(
            collection_name=self.collection, query_vector=rest.NamedSparseVector(name="bm25", vector=sparse), query_filter=flt_should, limit=limit, with_payload=True
        )
        fused_should = _rrf([res_dense_should, res_sparse_should])

        # Final fusion of both query modes
        return _rrf([fused_must, fused_should])[:limit]