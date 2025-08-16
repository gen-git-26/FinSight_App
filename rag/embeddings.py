# rag/embeddings.py
from __future__ import annotations

import logging
from typing import List

import httpx
from qdrant_client.http import models as rest

from utils.config import load_settings
from utils.cache import FileTTLCache

logger = logging.getLogger(__name__)

# Try to use Qdrant's helper for sparse embeddings; fall back to empty vectors if not available
try:  # pragma: no cover - optional dependency path differs by version
    from qdrant_client.fastembed_sparse import DefaultSparseModel  # type: ignore

    SPARSE_MODEL = DefaultSparseModel()
    SPARSE_AVAILABLE = True
except Exception:  # noqa: BLE001
    SPARSE_MODEL = None
    SPARSE_AVAILABLE = False

EMB_CACHE = FileTTLCache(cache_dir=".cache/embeddings", ttl_seconds=60 * 60 * 24 * 7)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    cfg = load_settings()
    outputs: List[List[float]] = []
    to_fetch: List[str] = []
    mapping: List[int] = []

    for idx, t in enumerate(texts):
        cached = EMB_CACHE.get(t)
        if cached:
            outputs.append(cached)
        else:
            outputs.append([])  # placeholder
            mapping.append(idx)
            to_fetch.append(t)

    if to_fetch:
        headers = {"Authorization": f"Bearer {cfg.openai_api_key}", "Content-Type": "application/json"}
        payload = {"model": cfg.openai_embed_model, "input": to_fetch}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            for i, d in enumerate(data["data"]):
                emb = d["embedding"]
                orig_idx = mapping[i]
                outputs[orig_idx] = emb
                EMB_CACHE.set(texts[orig_idx], emb)

    return outputs


def sparse_from_text(query: str) -> rest.SparseVector:
    if not SPARSE_AVAILABLE or SPARSE_MODEL is None:
        return rest.SparseVector(indices=[], values=[])
    emb = SPARSE_MODEL.encode_queries([query])[0]
    return rest.SparseVector(indices=emb.indices, values=emb.values)