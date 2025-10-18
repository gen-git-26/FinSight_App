# answer.py
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Tuple

try:
    from tools import retrieve, rerank_and_summarize  # your real implementations
except Exception:
    async def retrieve(query: str, k: int = 8) -> List[Dict[str, Any]]:
        return []
    async def rerank_and_summarize(query: str, docs: List[Dict[str, Any]], **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        return "RAG fallback (mock).", []

def answer_core(
    query: str,
    style: str = "concise",
    extra_context: str | None = None,
    max_tokens: int = 700,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"query": query, "answer": "", "snippets": []}
    try:
        docs = asyncio.run(retrieve(query, k=12))
    except Exception as e:
        result["answer"] = (
            "RAG is currently unavailable (retrieval error). "
            f"Fallback answer only. Reason: {str(e)[:200]}"
        )
        return result
    try:
        answer_text, snippets = asyncio.run(
            rerank_and_summarize(query, docs, style=style, extra_context=extra_context, max_tokens=max_tokens)
        )
        result["answer"] = answer_text or ""
        result["snippets"] = snippets or []
        return result
    except Exception as e:
        msg = str(e)
        if "Vector dimension error" in msg or "expected dim" in msg:
            result["answer"] = (
                "Live vector search is temporarily disabled due to an embedding dimension mismatch.\n"
                "Please align the Qdrant collection vector size with the active embedding model dimension, "
                "then re-ingest the documents."
            )
            result["snippets"] = []
            return result
        result["answer"] = f"Could not run the RAG summarization. Fallback only. Reason: {msg[:200]}"
        result["snippets"] = []
        return result
