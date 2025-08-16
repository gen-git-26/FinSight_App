# rag/qdrant_client.py
from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from utils.config import load_settings
from rag.embeddings import embed_texts, sparse_from_text  

@dataclass
class SearchHit:
    payload: dict
    score: float

class HybridQdrant:
    def __init__(self) -> None:
        cfg = load_settings()
        self.client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
        self.collection = cfg.qdrant_collection

    def mk_filters(
        self,
        date_gte: Optional[str] = None,
        symbol_in: Optional[List[str]] = None,
        type_in: Optional[List[str]] = None,
    ) -> List[rest.FieldCondition]:
        must: List[rest.FieldCondition] = []

        if date_gte:
            must.append(
                rest.FieldCondition(
                    key="date",
                    range=rest.Range(gte=date_gte)
                )
            )
        if symbol_in:
            vals = [s for s in symbol_in if s]
            if vals:
                must.append(
                    rest.FieldCondition(
                        key="symbol",
                        match=rest.MatchAny(any=vals)
                    )
                )
        if type_in:
            vals = [t for t in type_in if t]
            if vals:
                must.append(
                    rest.FieldCondition(
                        key="type",
                        match=rest.MatchAny(any=vals)
                    )
                )
        return must

    def _dense(self, text: str) -> List[float]:
        return embed_texts([text])[0]

    def _sparse(self, text: str) -> dict:
        return sparse_from_text(text)

    def hybrid_search(
        self,
        query: str,
        filters: Optional[List[rest.FieldCondition]] = None,
        limit: int = 12,
    ) -> List[SearchHit]:
        must = filters or []
        dense_vec = self._dense(query)
        sparse_vec = self._sparse(query)

     
        dense_hits = self.client.search(
            collection_name=self.collection,
            query_vector=dense_vec,
            query_filter=rest.Filter(must=must) if must else None,
            limit=limit,
            with_payload=True,
            score_threshold=None,
        )
        sparse_hits = self.client.search(
            collection_name=self.collection,
            query_vector=rest.NamedSparseVector(
                name="bm25",  
                vector=sparse_vec,
            ),
            query_filter=rest.Filter(must=must) if must else None,
            limit=limit,
            with_payload=True,
            score_threshold=None,
        )

   
        pool = []
        for p in dense_hits:
            pool.append(SearchHit(payload=p.payload, score=float(p.score)))
        for p in sparse_hits:
            pool.append(SearchHit(payload=p.payload, score=float(p.score)))

        pool.sort(key=lambda h: h.score, reverse=True)
        return pool[:limit]