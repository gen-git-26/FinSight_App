# tools/answer.py
from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, Any, List

from agno.tools import tool
from qdrant_client.http import models as rest

from utils.config import load_settings
from rag.fusion import retrieve, rerank_and_summarize
from tools.tools import (
    finnhub_stock_info,
    finnhub_basic_financials,
    finnhub_financials_as_reported,
    company_overview,
    finnhub_financials_new
)
from tools.mcp_router import route_and_call
from memory.manager import fetch_memory, persist_turn
from mcp_connection.manager import MCPServer
from tools.mcp_router import mcp_auto



# Regex to match common stock ticker patterns
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z]{1,3})?\b")
_STOP = { 'AND','THE','FOR','WITH','OVER','FROM','THIS','THAT','INTO','YOUR','US','USA','NYSE','NASDAQ','SP','ETF','PE','EV','EBITDA','EPS','CAGR','ROE','P','E' }

def _extract_tickers(text: str) -> List[str]:
    cands = _TICKER_RE.findall(text or "")
    return [c for c in cands if c not in _STOP]

def _detect_style(q: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    ql = (q or "").lower()
    if any(k in ql for k in ["executive summary", "report", "comprehensive", "deep dive", "overview including", "investment memo", "thesis"]):
        return "report"
    return "concise"

def _should_use_mcp(query: str) -> bool:
    """Determine if MCP should be prioritized for this query."""
    q_lower = query.lower()
    
    # Real-time indicators
    realtime_keywords = ["current", "now", "latest", "today", "live", "real-time", "quote", "price now"]
    if any(keyword in q_lower for keyword in realtime_keywords):
        return True
    
    # Crypto indicators
    crypto_keywords = ["crypto", "bitcoin", "btc", "ethereum", "eth", "coin", "token"]
    if any(keyword in q_lower for keyword in crypto_keywords):
        return True
    
    # Recent/fresh data indicators
    fresh_keywords = ["recent", "breaking", "news", "announcement", "update"]
    if any(keyword in q_lower for keyword in fresh_keywords):
        return True
    
    return False

@tool(name="answer", description="Enhanced Fusion RAG answerer with intelligent MCP integration. Automatically chooses between live MCP data and cached data. Input: free-form question; optional ticker(s) and style.")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    """
    Enhanced answer tool that intelligently combines MCP live data with RAG retrieval.
    Prioritizes fresh data when appropriate, falls back to cached data when needed.
    """
    
    # 0) Resolve style and tickers
    style = _detect_style(query, style)
    tickers = list(dict.fromkeys(([t.strip().upper() for t in (ticker.split() if ticker else [])] + _extract_tickers(query))))
    
    # 1) Determine data strategy
    should_use_mcp = _should_use_mcp(query)
    available_mcp_servers = MCPServer.from_env()
    
    mcp_data_attempted = False
    mcp_success = False
    
    # 2) Try MCP first if conditions are met
    # mcp_payload = None  # will hold fused output when available
    if should_use_mcp and available_mcp_servers:
        try:
            print(f"Attempting MCP data fetch for: {query}")
            # IMPORTANT: call the TOOL, not the raw router
            mcp_pack = mcp_auto(query)  # <- this is decorated with @fuse and INGESTS 'mcp' docs
            mcp_data_attempted = True

            # Treat as success if no exception; we ignore mcp_pack["answer"] here
            mcp_success = True
            print("MCP data fetch successful and ingested into vector store")

        except Exception as e:
            print(f"MCP call failed: {e}")
            mcp_data_attempted = True
            mcp_success = False
    
    
    # 3) Refresh cache with static tools if needed (especially if MCP failed or not used)
    if not mcp_success and tickers:
        print("Refreshing static data cache...")
        for t in tickers[:3]:  # limit to 3 to cap latency
            try:
                finnhub_stock_info(t)
                finnhub_basic_financials(t, metric="price")
                finnhub_basic_financials(t, metric="valuation") 
                finnhub_basic_financials(t, metric="growth")
                finnhub_financials_as_reported(t)
                company_overview(t)
            except Exception as e:
                print(f"Static tool error for {t}: {e}")
                pass

    # 4) Retrieve from vector store with enhanced filters
    flt: List[rest.FieldCondition] = []
    if tickers:
        flt.append(rest.FieldCondition(key="symbol", match=rest.MatchAny(any=tickers[:3])))

    # Include MCP data in search if we have servers configured
    doc_types = ["quote", "overview", "basic_financials", "as_reported"]
    if available_mcp_servers:
        doc_types.append("mcp")
        
    flt.append(
        rest.FieldCondition(
            key="type",
            match=rest.MatchAny(any=doc_types)
        )
    )

    docs = asyncio.run(retrieve(query, filters=flt, k=24))
    print(f"Retrieved {len(docs)} documents from vector store")

    # 5) Bring in user memory context
    mem_ctx = asyncio.run(fetch_memory(query=query, k=3))

    # 6) Enhanced context with MCP status
    extra_context = mem_ctx
    if mcp_data_attempted:
        mcp_status = "Live data included" if mcp_success else "Live data unavailable, using cached data"
        extra_context += f"\n[Data Source Status: {mcp_status}]"
        
        if not mcp_success and should_use_mcp:
            extra_context += "\n[Note: Query requested live data but MCP servers unavailable - using most recent cached data]"

    # 7) Dynamic token allocation based on style and data freshness
    if style == "report":
        max_tok = 1200 if mcp_success else 900  # More tokens for live data reports
    else:
        max_tok = 650 if mcp_success else 512   # More tokens for live data responses

    # 8) Summarize with enhanced prompt (or reuse MCP fused answer)
    if mcp_success and isinstance(mcp_payload, dict) and mcp_payload.get("answer"):
        answer_text = mcp_payload["answer"]
        snippets = mcp_payload.get("snippets", [])
    else:
        answer_text, snippets = asyncio.run(rerank_and_summarize(
            query,
            docs,
            style=style,
            extra_context=extra_context,
            max_tokens=max_tok
        ))

    # 9) Persist this turn in long-term memory
    asyncio.run(persist_turn(user_text=query, assistant_text=answer_text))

    # 10) Enhanced metadata
    metadata = {
    "style": style,
    "tickers": tickers,
    "mcp_attempted": mcp_data_attempted,
    "mcp_success": mcp_success,
    "available_servers": list(available_mcp_servers.keys()),
    "docs_retrieved": len(docs) if not mcp_success else len(snippets),
    "should_use_mcp": should_use_mcp
}

    return {
        "answer": answer_text, 
        "snippets": snippets, 
        "meta": metadata
    }