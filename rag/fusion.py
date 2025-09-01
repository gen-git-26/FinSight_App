# rag/fusion.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Tuple
from functools import wraps
import numpy as np

from qdrant_client.http import models as rest

# your embedding/sparse functions
try:
    from rag.embeddings import embed_texts, sparse_from_text
except Exception:
    async def _dummy_embed_texts(texts: List[str]): return [[0.0] * 1024 for _ in texts]
    def _dummy_sparse_from_text(text: str): return {}
    embed_texts = _dummy_embed_texts
    sparse_from_text = _dummy_sparse_from_text

from rag.qdrant_client import HybridQdrant

log = logging.getLogger(__name__)

def _rrf(lists: List[List[rest.ScoredPoint]], k: int = 60) -> List[rest.ScoredPoint]:
    scores: Dict[str, float] = {}
    pick: Dict[str, rest.ScoredPoint] = {}
    for lst in lists:
        for rank, p in enumerate(lst[:k], start=1):
            pid = str(p.id)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (50.0 + rank)  # RRF constant 50
            if pid not in pick:
                pick[pid] = p
    # return by score desc
    return [pick[i] for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _asyncify(fn):
    @wraps(fn)
    async def wrap(*a, **k):
        return fn(*a, **k)
    return wrap

@_asyncify
def retrieve(
    query: str,
    filters: List[rest.FieldCondition] | None = None,
    k: int = 24
) -> List[Dict[str, Any]]:
    """
    SAFE retrieve: if Qdrant or embeddings are unavailable, return [].
    Output format: list of dicts with lightweight fields {text, symbol, source, date}
    """
    try:
        qdr = HybridQdrant()
        qdr.ensure_collections()
    except Exception as e:
        log.warning("Qdrant init failed; retrieve will return empty. err=%s", e)
        return []

    try:
        sparse = sparse_from_text(query) or {}
        dense = asyncio.run(embed_texts([query]))[0]  # embed_texts is async
    except Exception as e:
        log.warning("Embeddings failed; retrieve will return empty. err=%s", e)
        return []

    try:
        # Two passes: must & should → gentle bias by filters
        r1 = qdr.hybrid_search(dense=dense, sparse=sparse, limit=k, must=filters or [])
        r2 = qdr.hybrid_search(dense=dense, sparse=sparse, limit=k, should=filters or [])
        fused = _rrf([r1, r2])

        out: List[Dict[str, Any]] = []
        for p in fused[:k]:
            payload = p.payload or {}
            out.append({
                "text": str(payload.get("text", "")),
                "symbol": payload.get("symbol"),
                "source": payload.get("source") or payload.get("type"),
                "date": payload.get("date"),
                "score": float(p.score or 0.0),
            })
        return out
    except Exception as e:
        log.warning("retrieve hybrid failed; returning empty. err=%s", e)
        return []

@_asyncify
def rerank_and_summarize(
    query: str,
    docs: List[Dict[str, Any]],
    style: str = "concise",
    extra_context: str = "",
    max_tokens: int = 512
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    SAFE rerank+summarize: if embeddings or LLM summarizer are unavailable,
    return a minimalist string assembled from the retrieved docs.
    """
    if not docs:
        return ("", [])

    try:
        texts = [str(d.get("text", ""))[:2000] for d in docs[:max(4, min(10, len(docs)))]]
        # simple cosine rerank
        q_emb = asyncio.run(embed_texts([query]))[0]
        d_embs = asyncio.run(embed_texts(texts))
        q = np.array(q_emb)
        order = sorted(range(len(d_embs)), key=lambda i: _cosine(q, np.array(d_embs[i])), reverse=True)

        top = [docs[i] for i in order[:5]]
        # very lightweight “summary”: take first 1-2 sentences per doc
        bullets: List[str] = []
        for d in top:
            t = (d.get("text") or "").split(". ")
            snippet = (". ".join(t[:2])).strip()
            if snippet:
                bullets.append(f"- {snippet}…")
        summary = (extra_context + "\n" if extra_context else "") + "\n".join(bullets)
        return (summary, top)
    except Exception as e:
        log.warning("rerank_and_summarize failed; returning minimal text. err=%s", e)
        return ("\n".join((d.get("text") or "")[:160] for d in docs[:5]), docs[:5])
