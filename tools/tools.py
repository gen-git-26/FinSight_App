
# tools/tools.py


from __future__ import annotations

import os
import json
from typing import Callable, Iterable, List, Optional

import finnhub
import requests

from agno.tools import tool
from mcp_connection.manager import MCPManager  
from utils.config import load_settings
from rag.fusion import fuse

# ========== Embedding utilities (NEW) ==========

class EmbeddingError(RuntimeError):
    pass


def _get_openai_client():
    """
    Lazy import OpenAI client.
    Raises a friendly error if the package or environment variables are missing.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise EmbeddingError(
            "OpenAI Python package is not installed. Please install `openai` or adjust your embedding backend."
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EmbeddingError("OPENAI_API_KEY is missing in environment.")

    return OpenAI(api_key=api_key)


def _get_embed_model_name() -> str:
    """
    Read the embedding model name from env, defaulting to `text-embedding-3-small` (1536 dims).
    You can set OPENAI_EMBED_MODEL=text-embedding-3-large for 3072 dims.
    """
    return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def embed_text(text: str) -> List[float]:
    """
    Single-text embedding.
    This callable is what the startup alignment probes to infer the true embedding dimension.
    """
    if not text or not text.strip():
        return []
    client = _get_openai_client()
    model = _get_embed_model_name()
    try:
        resp = client.embeddings.create(model=model, input=text)  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        raise EmbeddingError(f"Embedding request failed: {e}") from e

    try:
        vec = resp.data[0].embedding  # type: ignore[index]
        if not isinstance(vec, list):
            raise TypeError("Invalid embedding vector type")
        return vec  # list[float]
    except Exception as e:  # pragma: no cover
        raise EmbeddingError(f"Invalid embedding response schema: {e}") from e


def embed_many(texts: Iterable[str]) -> List[List[float]]:
    """
    Batch embed. Falls back to per-item embedding if API errors occur.
    """
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    client = _get_openai_client()
    model = _get_embed_model_name()
    try:
        resp = client.embeddings.create(model=model, input=texts)  # type: ignore[attr-defined]
        out = []
        for item in resp.data:  # type: ignore[attr-defined]
            out.append(item.embedding)  # type: ignore[attr-defined]
        return out
    except Exception:
        # Fallback to single requests to salvage partial batches
        return [embed_text(t) for t in texts]


def get_embedder() -> Callable[[str], List[float]]:
    """
    Return the callable used by the RAG stack to embed a single string.
    Startup alignment will call this once to probe the dimension.
    Keep this stable for the rest of the codebase.
    """
    return embed_text


# ========== Finnhub and other financial tools ==========

# --- Finnhub: quote ---
@tool(name="stock_prices", description="Get near real-time quote for a ticker (open/high/low/close, volume). Input: ticker symbol, e.g., TSLA.")
@fuse(tool_name="finnhub_quote", doc_type="quote")
def finnhub_stock_info(ticker: str):
    try:
        settings = load_settings()
        client = finnhub.Client(api_key=settings.finnhub_api_key)
        info = client.quote(ticker)
        return info  # dict
    except finnhub.exceptions.ApiException as e:
        print(f"Error use stock_info tool: {e}")
        raise

# --- Finnhub: basic financials ---
@tool(name="basic_financials", description="Get company basic financials for a metric. Inputs: ticker, metric (price/valuation/growth/margin).")
@fuse(tool_name="finnhub_basic_financials", doc_type="basic_financials")
def finnhub_basic_financials(ticker: str, metric: str):
    try:
        settings = load_settings()
        client = finnhub.Client(api_key=settings.finnhub_api_key)
        info = client.company_basic_financials(symbol=ticker, metric=metric)
        return info  # dict
    except finnhub.exceptions.ApiException as e:
        print(f"Error use basic_financials tool: {e}")
        raise

# --- Finnhub: financials as reported ---
@tool(name="financials_as_reported", description="Get financials as reported for a ticker (Finhub). Input: ticker symbol.")
@fuse(tool_name="finnhub_financials", doc_type="as_reported")
def finnhub_financials_as_reported(ticker: str):
    try:
        settings = load_settings()
        client = finnhub.Client(api_key=settings.finnhub_api_key)
        info = client.financials_reported(symbol=ticker)
        return info  # dict
    except finnhub.exceptions.ApiException as e:
        print(f"Error use financials_as_reported tool: {e}")
        raise

# --- AlphaVantage: company overview ---
@tool(name="get_company_info", description="AlphaVantage company overview (ratios & profile). Input: ticker symbol.")
@fuse(tool_name="alpha_overview", doc_type="overview")
def company_overview(ticker: str):
    try:
        settings = load_settings()
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={settings.alphavantage_api_key}'
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data  # dict
    except requests.RequestException as e:
        print(f"Error use company_overview tool: {e}")
        raise

# --- Finnhub: general news (kept as in your file name even if description mismatched) ---
@tool(name="finnhub_financials_new", description="Get general news from Finnhub. Input: query string.")
@fuse(tool_name="finnhub_financials_new", doc_type="as_reported")
def finnhub_financials_new(query: str):
    try:
        settings = load_settings()
        client = finnhub.Client(api_key=settings.finnhub_api_key)
        info = client.general_news(query, 'general', min_id=0)
        # Ensure the return type is JSON-serializable
        try:
            json.dumps(info)
            return info
        except Exception:
            return str(info)
    except finnhub.exceptions.ApiException as e:
        print(f"Error use finnhub_financials_new tool: {e}")
        raise
