
# tools/answer.py

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

# Import your local helpers. These names should match your project.
# The functions used below are placeholders for your real implementations.
try:
    from tools import retrieve, rerank_and_summarize  # type: ignore
except Exception:
    # Minimal no-op fallbacks
    async def retrieve(query: str, k: int = 8) -> List[Dict[str, Any]]:  # type: ignore
        return []

    async def rerank_and_summarize(query: str, docs: List[Dict[str, Any]], **kwargs) -> Tuple[str, List[Dict[str, Any]]]:  # type: ignore
        return "RAG fallback (mock).", []


def answer_core(
    query: str,
    style: str = "concise",
    extra_context: str | None = None,
    max_tokens: int = 700,
) -> Dict[str, Any]:
    """
    High-level answer flow:
    1) Try to run RAG (retrieve -> rerank_and_summarize)
    2) If vector DB throws dimension mismatch or any retriever error, gracefully degrade
    3) Return a stable shape for Streamlit UI
    """
    result: Dict[str, Any] = {"query": query, "answer": "", "snippets": []}

    # Phase 1: Retrieve
    try:
        docs = asyncio.run(retrieve(query, k=12))
    except Exception as e:
        # Retrieval failed; RAG is unavailable
        result["answer"] = (
            "RAG is currently unavailable (retrieval error). "
            f"Fallback answer only. Reason: {str(e)[:200]}"
        )
        return result

    # Phase 2: Summarize with strong guards
    try:
        answer_text, snippets = asyncio.run(
            rerank_and_summarize(query, docs, style=style, extra_context=extra_context, max_tokens=max_tokens)
        )
        result["answer"] = answer_text or ""
        result["snippets"] = snippets or []
        return result
    except Exception as e:
        msg = str(e)
        # Friendly handling for Qdrant dimension mismatch
        if "Vector dimension error" in msg or "expected dim" in msg:
            result["answer"] = (
                "Live vector search is temporarily disabled due to an embedding dimension mismatch."
                "Please align the Qdrant collection vector size with the active embedding model dimension, "
                "then re-ingest the documents."
            )
            result["snippets"] = []
            return result

        # Generic fallback
        result["answer"] = f"Could not run the RAG summarization. Fallback only. Reason: {msg[:200]}"
        result["snippets"] = []
        return result
