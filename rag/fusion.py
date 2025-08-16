# rag/fusion.py
# This module implements the fusion of search results using Reciprocal Rank Fusion (RRF) and
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Tuple

from qdrant_client.http import models as rest

from rag.embeddings import embed_texts, sparse_from_text
from rag.qdrant_client import HybridQdrant
from utils.config import load_settings
import httpx
import numpy as np

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(results: List[List[rest.ScoredPoint]], k: int = 60) -> List[rest.ScoredPoint]:
    scores: Dict[str, float] = {}
    lookup: Dict[str, rest.ScoredPoint] = {}
    for result in results:
        for rank, item in enumerate(result, start=1):
            rid = str(item.id)
            lookup[rid] = item
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [lookup[rid] for rid, _ in ranked]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


async def retrieve(query: str, *, filters: List[rest.FieldCondition]) -> List[rest.ScoredPoint]:
    qdr = HybridQdrant()
    qdr.ensure_collections()
    sparse = sparse_from_text(query)
    dense = (await embed_texts([query]))[0]

    r1 = qdr.hybrid_search(dense=dense, sparse=sparse, limit=20, must=filters)
    r2 = qdr.hybrid_search(dense=dense, sparse=sparse, limit=20, should=filters)
    fused = reciprocal_rank_fusion([r1, r2])

    # Lightweight heuristic reranker: cosine on embeddings between query and doc text embedding
    # We approximate by re-embedding doc texts in a small batch (top 20 only)
    texts = [str((d.payload or {}).get("text", ""))[:2000] for d in fused[:20]]
    if texts:
        doc_embs = await embed_texts(texts)
        q = np.array(dense)
        scored = [
            (i, _cosine(q, np.array(doc_embs[i]))) for i in range(len(doc_embs))
        ]
        order = sorted(range(len(scored)), key=lambda i: scored[i][1], reverse=True)
        fused = [fused[i] for i in order]

    return fused[:12]


async def rerank_and_summarize(query: str, docs: List[rest.ScoredPoint]) -> Tuple[str, List[Dict[str, Any]]]:
    cfg = load_settings()
    system = (
        "You are a precise financial research assistant. Given the user question and a set of snippets, "
        "select the most relevant 4 snippets, then answer concisely with citations (symbol/date) and include a short risk disclaimer."
    )
    snippets = [
        {
            "id": str(d.id),
            "text": (d.payload or {}).get("text", ""),
            "symbol": (d.payload or {}).get("symbol", ""),
            "date": (d.payload or {}).get("date", ""),
            "type": (d.payload or {}).get("type", ""),
        }
        for d in docs
    ]

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Question: {query}\nSnippets: {snippets}",
        },
    ]

    headers = {"Authorization": f"Bearer {cfg.openai_api_key}", "Content-Type": "application/json"}
    payload = {"model": cfg.openai_model, "messages": messages, "temperature": 0.1}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer, snippets
