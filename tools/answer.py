# tools/answer.py
from __future__ import annotations
import re, json, asyncio, logging
from typing import Dict, Any, List
from agno.tools import tool
from qdrant_client.http import models as rest

try:
    from utils.config import load_settings
except Exception:
    load_settings = None

try:
    from tools.tools import (
        finnhub_stock_info,
        finnhub_basic_financials,
        finnhub_financials_as_reported,
        company_overview,
        finnhub_financials_new,
    )
except Exception:
    def finnhub_stock_info(*a, **k): return None
    def finnhub_basic_financials(*a, **k): return None
    def finnhub_financials_as_reported(*a, **k): return None
    def company_overview(*a, **k): return None
    def finnhub_financials_new(*a, **k): return None

# RAG layer 
try:
    from rag.fusion import retrieve, rerank_and_summarize
except Exception:
    # ultra-safe fallbacks
    async def _dummy_retrieve(*a, **k): return []
    async def _dummy_rerank(*a, **k): return ("", [])
    def retrieve(*a, **k): return asyncio.run(_dummy_retrieve())
    def rerank_and_summarize(*a, **k): return asyncio.run(_dummy_rerank())

# MCP auto-router
from tools.mcp_router import mcp_auto
from mcp_connection.manager import MCPServer

log = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z0-9]{1,6})?\b")
_STOP = {
    'AND','THE','FOR','WITH','OVER','FROM','THIS','THAT','INTO','YOUR','US','USA',
    'NYSE','NASDAQ','SP','ETF','PE','EV','EBITDA','EPS','CAGR','ROE','P','E'
}

def _extract_tickers(text: str) -> List[str]:
    return [c for c in _TICKER_RE.findall(text or "") if c not in _STOP]

def _detect_style(q: str, explicit: str = "") -> str:
    if explicit: 
        return explicit
    ql = (q or "").lower()
    return "report" if any(k in ql for k in [
        "executive summary","report","comprehensive","deep dive","overview including",
        "investment memo","thesis"
    ]) else "concise"

def _should_use_mcp(query: str) -> bool:
    q = (query or "").lower()
    if any(k in q for k in ["current","now","latest","today","live","real-time","quote","price now"]): 
        return True
    if any(k in q for k in ["crypto","bitcoin","btc","ethereum","eth","coin","token"]): 
        return True
    if any(k in q for k in ["recent","breaking","news","announcement","update"]): 
        return True
    return False

def _safe_retrieve(query: str, filters: List[rest.FieldCondition], k: int) -> List[Dict[str, Any]]:
    """
    RAG must never crash the agent. Any exception → return [].
    """
    try:
        return asyncio.run(retrieve(query, filters=filters, k=k))
    except Exception as e:
        log.warning("RAG retrieve failed; falling back without RAG. err=%s", e)
        return []

def _safe_rerank_summarize(query: str, docs, style: str, extra_context: str, max_tokens: int):
    """
    Summarization must never crash the agent. Any exception → ("", []).
    """
    try:
        return asyncio.run(rerank_and_summarize(
            query, docs, style=style, extra_context=extra_context, max_tokens=max_tokens
        ))
    except Exception as e:
        log.warning("RAG rerank/summarize failed; continuing without. err=%s", e)
        return ("", [])

# --------- Stable core the UI calls ---------
def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    style = _detect_style(query, style)
    tickers = list(dict.fromkeys(([t.strip().upper() for t in (ticker.split() if ticker else [])] + _extract_tickers(query))))

    want_live = _should_use_mcp(query)
    servers = MCPServer.from_env()
    mcp_attempted = False
    mcp_success = False
    mcp_payload: Dict[str, Any] | None = None
    mcp_text: str = ""

    # 1) Try live MCP if appropriate
    if want_live and servers:
        try:
            mcp_raw = mcp_auto(query)      # route_and_call under the hood
            mcp_attempted = True
            # Keep a consistent envelope: if router already returns a string, we wrap it
            if isinstance(mcp_raw, dict):
                mcp_text = mcp_raw.get("answer") or ""
            else:
                mcp_text = str(mcp_raw)
            mcp_success = True
        except Exception as e:
            log.warning("MCP live call failed; continuing with fallback. err=%s", e)
            mcp_attempted = True
            mcp_success = False
            mcp_text = ""

    # 2) Best-effort prefetch (legacy) if we have explicit tickers and MCP failed or wasn't requested
    if tickers and (not mcp_success):
        for t in tickers[:3]:
            try:
                finnhub_stock_info(t)
                finnhub_basic_financials(t, metric="price")
                finnhub_basic_financials(t, metric="valuation")
                finnhub_basic_financials(t, metric="growth")
                finnhub_financials_as_reported(t)
                company_overview(t)
            except Exception:
                pass

    flt: List[rest.FieldCondition] = []
    if tickers:
        try:
            flt.append(rest.FieldCondition(key="symbol", match=rest.MatchAny(any=tickers[:3])))
        except Exception:
            pass

    doc_types = ["quote", "overview", "basic_financials", "as_reported"]
    if servers:
        doc_types.append("mcp")
    try:
        flt.append(rest.FieldCondition(key="type", match=rest.MatchAny(any=doc_types)))
    except Exception:
        pass

    # 4) SAFE RAG: never throw
    docs = _safe_retrieve(query, filters=flt, k=24)

    # 5) Extra context
    extra_context = ""
    if mcp_attempted:
        extra_context += "\n[Data Source Status: {}]".format(
            "Live data included" if mcp_success else "Live data unavailable, using cached data"
        )
        if not mcp_success and want_live:
            extra_context += "\n[Note: Live data requested but MCP unavailable]"

    # 6) Summarize or pass-through
    #    If MCP succeeded and produced text, prefer it as the main answer.
    max_tok = 1200 if (style == "report" and mcp_success) else (900 if style == "report" else (650 if mcp_success else 512))
    if mcp_success and mcp_text:
        answer_text = mcp_text
        snippets = []  # MCP usually is “live”; you can append docs if you want hybrid
        # If you want hybrid summarization on top of MCP text + docs, you could call _safe_rerank_summarize here.
    else:
        answer_text, snippets = _safe_rerank_summarize(query, docs, style, extra_context, max_tok)

    # 7) Persist memory (turn-level)
    try:
        from memory.manager import fetch_memory, persist_turn
        mem_ctx = asyncio.run(fetch_memory(query=query, k=3))
        asyncio.run(persist_turn(user_text=query, assistant_text=answer_text or ""))
    except Exception:
        pass

    meta = {
        "style": style,
        "tickers": tickers,
        "mcp_attempted": mcp_attempted,
        "mcp_success": mcp_success,
        "available_servers": list((servers or {}).keys()),
        "docs_retrieved": len(docs),
        "should_use_mcp": want_live,
    }
    return {"answer": answer_text or (mcp_text or "I couldn’t find relevant data."), "snippets": snippets, "meta": meta}


@tool(name="answer", description="Fusion RAG + MCP with graceful fallbacks")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    return answer_core(query, ticker, style)
