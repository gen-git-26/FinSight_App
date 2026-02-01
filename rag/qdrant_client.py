# rag/qdrant_client.py
from __future__ import annotations

from typing import List, Optional, Dict, Any, Union
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from utils.config import load_settings


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


def _as_point_id(value: Any) -> Union[int, str]:
    """
    Accepts int, numeric-string, UUID-string, or arbitrary string.
    - int / "123" -> int
    - valid UUID string -> same string
    - any other string -> deterministic UUIDv5 derived from the string
    """
    if isinstance(value, int):
        return value

    s = "" if value is None else str(value)

    # numeric string -> int
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            pass

    # valid UUID string -> return normalized
    try:
        u = uuid.UUID(s)
        return str(u)
    except Exception:
        pass

    # fallback: deterministic UUIDv5 from the string
    base = s if s else "empty-id"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


class HybridQdrant:
    def __init__(self) -> None:
        cfg = load_settings()
        self.client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
        self.collection = cfg.qdrant_collection
        self._ensured = False

    def ensure_collections(self, dense_dim: Optional[int] = None) -> None:
        if self._ensured:
            return

        colls = [c.name for c in self.client.get_collections().collections]

        # check existing collection
        if self.collection in colls:
            try:
                info = self.client.get_collection(self.collection)
                existing_dim = None
                if hasattr(info, 'config') and hasattr(info.config, 'params'):
                    vectors = info.config.params.vectors
                    if isinstance(vectors, dict) and 'text' in vectors:
                        existing_dim = vectors['text'].size

                # Check for dimension mismatch
                if existing_dim and existing_dim != 3072:
                    print(f"[Qdrant] Deleting collection with mismatched dims: {existing_dim} → 3072")
                    self.client.delete_collection(self.collection)
                else:
                    # build payload indices
                    for field, schema in [
                        ("symbol", rest.PayloadSchemaType.KEYWORD),
                        ("type", rest.PayloadSchemaType.KEYWORD),
                        ("date", rest.PayloadSchemaType.TEXT),
                        ("user", rest.PayloadSchemaType.KEYWORD),
                    ]:
                        try:
                            self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)
                        except Exception:
                            pass
                    self._ensured = True
                    return
            except Exception as e:
                print(f"[Qdrant] Error checking collection: {e}")

        # create collection
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                "text": rest.VectorParams(size=3072, distance=rest.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": rest.SparseVectorParams(),
            },
        )

        for field, schema in [
            ("symbol", rest.PayloadSchemaType.KEYWORD),
            ("type", rest.PayloadSchemaType.KEYWORD),
            ("date", rest.PayloadSchemaType.TEXT),
            ("user", rest.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)
            except Exception:
                pass

        self._ensured = True

    # --------- Upsert ---------
    async def upsert_snippets(self, items: List[Dict[str, Any]]) -> None:
        """Upserts a list of snippets into the Qdrant collection."""
        from rag.embeddings import embed_texts, sparse_from_text

        self.ensure_collections()

        texts = [str(it.get("text", "")) for it in items]
        dense_vecs = await embed_texts(texts)

        points: List[rest.PointStruct] = []
        for i, it in enumerate(items):
            raw_id = it.get("id") or f"snip-{i}"
            pid = _as_point_id(raw_id)
            payload = {
                "text": texts[i],
                "symbol": it.get("symbol", ""),
                "type": it.get("type", ""),
                "date": it.get("date", ""),
                "source": it.get("source", ""),
                "slug": it.get("slug", ""),
                "user": it.get("user", ""),
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

    # --------- Filters helper ---------
    def mk_filters(
        self,
        date_gte: Optional[str] = None,
        symbol_in: Optional[List[str]] = None,
        type_in: Optional[List[str]] = None,
        user_in: Optional[List[str]] = None,
    ) -> Optional[rest.Filter]:
        """Creates a Qdrant filter based on the provided criteria."""
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

        if user_in:
            vals = [u for u in user_in if u]
            if vals:
                must.append(rest.FieldCondition(key="user", match=rest.MatchAny(any=vals)))

        return rest.Filter(must=must) if must else None  # type: ignore[return-value]

    # --------- Hybrid search + RRF ---------
    def hybrid_search(
        self,
        *,
        dense: List[float],
        sparse: rest.SparseVector,
        limit: int = 12,
        must: Optional[List[rest.FieldCondition]] = None,
        should: Optional[List[rest.FieldCondition]] = None,
    ) -> List[rest.ScoredPoint]:
        """Performs a hybrid search using both dense and sparse vectors with RRF fusion."""
        self.ensure_collections()

        flt_must = rest.Filter(must=must) if must else None
        flt_should = rest.Filter(should=should) if should else None

        # Pass 1: MUST
        res_dense_must = self.client.search(
            collection_name=self.collection,
            query_vector=("text", dense),
            query_filter=flt_must,
            limit=limit,
            with_payload=True,
        )
        res_sparse_must = self.client.search(
            collection_name=self.collection,
            query_vector=rest.NamedSparseVector(name="bm25", vector=sparse),
            query_filter=flt_must,
            limit=limit,
            with_payload=True,
        )
        fused_must = _rrf([res_dense_must, res_sparse_must])

        if flt_should is None:
            return fused_must[:limit]

        # Pass 2: SHOULD
        res_dense_should = self.client.search(
            collection_name=self.collection,
            query_vector=("text", dense),
            query_filter=flt_should,
            limit=limit,
            with_payload=True,
        )
        res_sparse_should = self.client.search(
            collection_name=self.collection,
            query_vector=rest.NamedSparseVector(name="bm25", vector=sparse),
            query_filter=flt_should,
            limit=limit,
            with_payload=True,
        )
        fused_should = _rrf([res_dense_should, res_sparse_should])
        # Final fusion of MUST and SHOULD
        return _rrf([fused_must, fused_should])[:limit]
