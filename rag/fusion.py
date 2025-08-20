# rag/fusion.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Tuple, Callable, Optional
from functools import wraps
import re
import json

import numpy as np
import httpx
from qdrant_client.http import models as rest

from rag.embeddings import embed_texts, sparse_from_text
from rag.qdrant_client import HybridQdrant
from utils.config import load_settings


logger = logging.getLogger(__name__)

DEFAULT_K = 12
MAX_SNIPPET_CHARS = 600
LLM_MAX_TOKENS = 512

def _rrf(results: List[List[rest.ScoredPoint]], k: int = 60) -> List[rest.ScoredPoint]:
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


async def retrieve(query: str, *, filters: Optional[List[rest.FieldCondition]] = None, k: int = DEFAULT_K) -> List[rest.ScoredPoint]:
    qdr = HybridQdrant()
    qdr.ensure_collections()
    sparse = sparse_from_text(query)
    dense = (await embed_texts([query]))[0]

    # two passes: must and should to gently bias filters
    r1 = qdr.hybrid_search(dense=dense, sparse=sparse, limit=k, must=filters or [])
    r2 = qdr.hybrid_search(dense=dense, sparse=sparse, limit=k, should=filters or [])
    fused = _rrf([r1, r2])

    # Lightweight heuristic reranker: cosine between query and doc text embedding
    texts = [str((d.payload or {}).get("text", ""))[:2000] for d in fused[:k]]
    if texts:
        doc_embs = await embed_texts(texts)
        q = np.array(dense)
        order = sorted(range(len(doc_embs)), key=lambda i: _cosine(q, np.array(doc_embs[i])), reverse=True)
        fused = [fused[i] for i in order]

    return fused[:k]


async def rerank_and_summarize(query: str, docs: List[rest.ScoredPoint], *, style: str = "concise", extra_context: str = "", max_tokens: int = LLM_MAX_TOKENS) -> Tuple[str, List[Dict[str, Any]]]:
    cfg = load_settings()
    if style == "report":
        system = (
        "You are a seasoned financial expert specializing in finance, trading, investments, and cryptocurrency. "
        "Your responses must be coherent, reliable, and evidence-based. "
        "Structure your analysis with clear sections and comprehensive coverage. "
        "Limit responses to under 8 lines while ensuring depth and accuracy. "
        "Focus on actionable insights and data-driven analysis. "
        "Use precise financial terminology and maintain a professional tone. "
        "Provide current, up-to-date information and market data when available. "
        "Conclude with a single risk disclaimer line acknowledging market volatility and investment risks."
        )
    else:
        system = (
        "You are a precise financial research assistant and expert in finance, trading, investments, and cryptocurrency. "
        "Deliver coherent, reliable, and well-researched responses based on current market data and analysis. "
        "Keep answers concise yet comprehensive, limiting to under 8 lines while maintaining accuracy. "
        "Focus on actionable insights and data-driven analysis. "
        "End with one risk disclaimer line regarding market volatility and investment risks."
        )

    snippets = [
        {
            "id": str(d.id),
            "text": str((d.payload or {}).get("text", ""))[:MAX_SNIPPET_CHARS],
            "symbol": (d.payload or {}).get("symbol", ""),
            "date": (d.payload or {}).get("date", ""),
            "type": (d.payload or {}).get("type", ""),
            "source": (d.payload or {}).get("source", ""),
        }
        for d in docs[:DEFAULT_K]
    ]

    user_block = {
        "role": "user",
        "content": f"""Question: {query}
                       Memory:{extra_context}
                       Snippets: {json.dumps(snippets, ensure_ascii=False)}""",
    }

    headers = {"Authorization": f"Bearer {cfg.openai_api_key}", "Content-Type": "application/json"}
    payload = {"model": cfg.openai_model, "messages": [{"role": "system", "content": system}, user_block], "temperature": 0.1, "max_tokens": max_tokens}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer, snippets


def _chunk_text(raw: str, *, max_len: int = MAX_SNIPPET_CHARS) -> List[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    # simple chunk by sentences/lines, then merge
    parts: List[str] = []
    for seg in re.split(r"(?<=[.!?])\s+|\n+", raw):
        seg = seg.strip()
        if not seg:
            continue
        parts.append(seg)
    chunks: List[str] = []
    buf = ""
    for seg in parts:
        if len(buf) + 1 + len(seg) <= max_len:
            buf = (buf + " " + seg).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = seg[:max_len]
    if buf:
        chunks.append(buf)
    return chunks[:DEFAULT_K]


async def ingest_raw(*, tool: str, raw: Any, symbol: str = "", doc_type: str = "", date: str = "") -> List[str]:
    """Ingest raw API output as small, queryable snippets and return generated IDs."""
    qdr = HybridQdrant()
    qdr.ensure_collections()
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    chunks = _chunk_text(text)
    items = []
    ids = []
    for i, ch in enumerate(chunks):
        pid = f"{tool}-{symbol or 'NA'}-{i}"
        ids.append(pid)
        items.append({
            "id": pid,
            "text": ch,
            "symbol": symbol,
            "type": doc_type,
            "date": date,
            "source": tool,
        })
    if items:
        await qdr.upsert_snippets(items)
    return ids


def fuse(*, tool_name: str, doc_type: str, k: int = DEFAULT_K, snippet_chars: int = MAX_SNIPPET_CHARS):
    """Decorator to wrap a tool function: ingest its raw output, retrieve+fuse, and summarize.
    The wrapped tool should accept either 'query' or 'ticker' in args/kwargs."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            async def run():
                # 1) call underlying tool (supports sync/async)
                res = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                # 2) derive verify query and symbol
                verify_q = str(kwargs.get('query') or kwargs.get('ticker') or (args[0] if args else ''))
                symbol = str(kwargs.get('ticker') or kwargs.get('symbol') or (args[0] if args else ''))
                # 3) ingest to Qdrant
                await ingest_raw(tool=tool_name, raw=res, symbol=symbol, doc_type=doc_type)
                # 4) build filters and retrieve
                qdr = HybridQdrant()
                flt = []
                if symbol:
                    flt.append(rest.FieldCondition(key='symbol', match=rest.MatchAny(any=[symbol])))
                if doc_type:
                    flt.append(rest.FieldCondition(key='type', match=rest.MatchAny(any=[doc_type])))
                docs = await retrieve(verify_q or symbol or tool_name, filters=flt, k=k)
                # 5) summarize
                answer, snippets = await rerank_and_summarize(verify_q or symbol or tool_name, docs)
                return {
                    'answer': answer,
                    'snippets': snippets,
                    'meta': {'k': k, 'snippet_chars': snippet_chars, 'tool': tool_name, 'type': doc_type}
                }
            return asyncio.run(run())
        return wrapper
    return decorator