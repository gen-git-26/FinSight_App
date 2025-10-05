# tools/answer.py
from __future__ import annotations
import re, json, asyncio
from typing import Dict, Any, List
from agno.tools import tool
from qdrant_client.http import models as rest
from utils.config import load_settings
from rag.fusion import retrieve, rerank_and_summarize
from tools.tools import (
    finnhub_stock_info, finnhub_basic_financials,
    finnhub_financials_as_reported, company_overview, finnhub_financials_new
)
from memory.manager import fetch_memory, persist_turn
from mcp_connection.manager import MCPServer
from tools.mcp_router import mcp_auto   

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z]{1,3})?\b")
_STOP = {'AND','THE','FOR','WITH','OVER','FROM','THIS','THAT','INTO','YOUR','US','USA','NYSE','NASDAQ','SP','ETF','PE','EV','EBITDA','EPS','CAGR','ROE','P','E'}

def _extract_tickers(text: str) -> List[str]:
    return [c for c in _TICKER_RE.findall(text or "") if c not in _STOP]

def _detect_style(q: str, explicit: str = "") -> str:
    if explicit: return explicit
    ql = (q or "").lower()
    return "report" if any(k in ql for k in [
        "executive summary","report","comprehensive","deep dive","overview including","investment memo","thesis"
    ]) else "concise"

def _should_use_mcp(query: str) -> bool:
    q = (query or "").lower()
    if any(k in q for k in ["current","now","latest","today","live","real-time","quote","price now"]): return True
    if any(k in q for k in ["crypto","bitcoin","btc","ethereum","eth","coin","token"]): return True
    if any(k in q for k in ["recent","breaking","news","announcement","update"]): return True
    return False

# Main answer function
def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    style = _detect_style(query, style)
    tickers = list(dict.fromkeys(([t.strip().upper() for t in (ticker.split() if ticker else [])] + _extract_tickers(query))))

    should_use = _should_use_mcp(query)
    servers = MCPServer.from_env()
    mcp_attempted = False
    mcp_success = False
    mcp_payload = None

    if should_use and servers:
        try:

            mcp_payload = mcp_auto(query)
            mcp_attempted = True
            mcp_success = True
        except Exception:
            mcp_attempted = True
            mcp_success = False

    if not mcp_success and tickers:
        for t in tickers[:3]:
            try:
                finnhub_stock_info(t)
                finnhub_basic_financials(t, metric="price")
                finnhub_basic_financials(t, metric="valuation")
                finnhub_basic_financials(t, metric="growth")
                finnhub_financials_as_reported(t)
                finnhub_financials_new(t)
                company_overview(t)
            except Exception:
                pass

    flt: List[rest.FieldCondition] = []
    if tickers:
        flt.append(rest.FieldCondition(key="symbol", match=rest.MatchAny(any=tickers[:3])))
    doc_types = ["quote","overview","basic_financials","as_reported"]
    if servers: doc_types.append("mcp")
    flt.append(rest.FieldCondition(key="type", match=rest.MatchAny(any=doc_types)))

    docs = asyncio.run(retrieve(query, filters=flt, k=24))
    mem_ctx = asyncio.run(fetch_memory(query=query, k=3))

    extra_context = mem_ctx
    if mcp_attempted:
        extra_context += "\n[Data Source Status: {}]".format("Live data included" if mcp_success else "Live data unavailable, using cached data")
        if not mcp_success and should_use:
            extra_context += "\n[Note: Live data requested but MCP unavailable]"

    max_tok = 1200 if (style == "report" and mcp_success) else (900 if style == "report" else (650 if mcp_success else 512))

    if mcp_success and isinstance(mcp_payload, dict) and mcp_payload.get("answer"):
        answer_text = mcp_payload["answer"]
        snippets = mcp_payload.get("snippets", [])
    else:
        answer_text, snippets = asyncio.run(rerank_and_summarize(
            query, docs, style=style, extra_context=extra_context, max_tokens=max_tok
        ))

    asyncio.run(persist_turn(user_text=query, assistant_text=answer_text))

    meta = {
        "style": style,
        "tickers": tickers,
        "mcp_attempted": mcp_attempted,
        "mcp_success": mcp_success,
        "available_servers": list((servers or {}).keys()),
        "docs_retrieved": len(snippets) if mcp_success else len(docs),
        "should_use_mcp": should_use
    }
    return {"answer": answer_text, "snippets": snippets, "meta": meta}


@tool(name="answer", description="Fusion RAG + MCP")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    return answer_core(query, ticker, style)
