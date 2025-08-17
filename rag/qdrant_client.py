# rag/qdrant_client.py
from __future__ import annotations

from typing import List, Optional, Dict, Any, Union
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from utils.config import load_settings


# --- Helper: Reciprocal Rank Fusion (קטן ומקומי) ---
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


# --- Helper: נרמול מזהה ל־Qdrant (מספר או UUID תקני) ---
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

    # --------- יצירת קולקציה/אינדקסים ---------
    def ensure_collections(self, dense_dim: Optional[int] = None) -> None:
        """
        יוצר/מוודא קולקציה עם וקטור צפוף 'text' וספראז 'bm25' ואינדקסי payload.
        רצה פעם אחת לכל מופע.
        """
        if self._ensured:
            return

        colls = [c.name for c in self.client.get_collections().collections]
        if self.collection in colls:
            # קיימת: רק לוודא אינדקסים לשדות
            for field, schema in [
                ("symbol", rest.PayloadSchemaType.KEYWORD),
                ("type", rest.PayloadSchemaType.KEYWORD),
                ("date", rest.PayloadSchemaType.TEXT),
                ("user", rest.PayloadSchemaType.KEYWORD),
            ]:
                try:
                    self.client.create_payload_index(
                        self.collection, field_name=field, field_schema=schema
                    )
                except Exception:
                    # כנראה כבר קיים – מתעלמים
                    pass
            self._ensured = True
            return

        # לא קיימת: יוצרים מאפס
        if dense_dim is None:
            # ברירת מחדל בטוחה ל-text-embedding-3-large
            dense_dim = 3072

        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                "text": rest.VectorParams(size=dense_dim, distance=rest.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": rest.SparseVectorParams(),
            },
        )

        # אינדקסים על שדות המטא-דאטה
        for field, schema in [
            ("symbol", rest.PayloadSchemaType.KEYWORD),
            ("type", rest.PayloadSchemaType.KEYWORD),
            ("date", rest.PayloadSchemaType.TEXT),
            ("user", rest.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(
                    self.collection, field_name=field, field_schema=schema
                )
            except Exception:
                pass

        self._ensured = True

    # --------- Upsert ---------
    async def upsert_snippets(self, items: List[Dict[str, Any]]) -> None:
        """
        מצפה לרשומות בפורמט:
        { id?, text, symbol?, type?, date?, source?, slug?, user? }
        בונה גם וקטור צפוף ('text') וגם sparse ('bm25').
        """
        from rag.embeddings import embed_texts, sparse_from_text  # lazy import למניעת מעגליות

        self.ensure_collections()

        texts = [str(it.get("text", "")) for it in items]
        dense_vecs = await embed_texts(texts)

        points: List[rest.PointStruct] = []
        for i, it in enumerate(items):
            raw_id = it.get("id") or f"snip-{i}"
            pid = _as_point_id(raw_id)  # <-- כאן ההמרה הקריטית למזהה חוקי
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

        # אפשר להוסיף wait=True אם רוצים לחכות לכתיבה:
        self.client.upsert(collection_name=self.collection, points=points)

    # --------- Filters helper ---------
    def mk_filters(
        self,
        date_gte: Optional[str] = None,
        symbol_in: Optional[List[str]] = None,
        type_in: Optional[List[str]] = None,
        user_in: Optional[List[str]] = None,
    ) -> Optional[rest.Filter]:
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
        """
        מריץ חיפוש צפוף וחיפוש BM25 בנפרד, מאחה בעזרת RRF,
        ותומך בשני מצבי פילטר: MUST ו-SHOULD (לריכוך).
        """
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

        # איחוי סופי של שני המצבים
        return _rrf([fused_must, fused_should])[:limit]
