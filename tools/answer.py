# tools/answer.py
from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, Any, List

from agno.tools import tool
from qdrant_client.http import models as rest

from rag.fusion import retrieve, rerank_and_summarize
from memory.manager import fetch_memory, persist_turn
from mcp_connection.manager import MCPServer
from tools.mcp_router import mcp_auto


# Helpers
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z]{1,3})?\b")
_STOP = {
    "AND", "THE", "FOR", "WITH", "OVER", "FROM", "THIS", "THAT", "INTO", "YOUR",
    "US", "USA", "NYSE", "NASDAQ", "SP", "ETF", "PE", "EV", "EBITDA", "EPS",
}


def _extract_tickers(text: str) -> List[str]:
    matches = _TICKER_RE.findall(text or "")
    out = []
    for m in matches:
        if m in _STOP:
            continue
        if m.isupper() or "." in m or "-" in m or (f"${m}" in (text or "")):
            out.append(m)
    return out


def _detect_style(q: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    ql = (q or "").lower()
    return (
        "report"
        if any(
            k in ql
            for k in [
                "executive summary", "report", "comprehensive", "deep dive",
                "overview including", "investment memo", "thesis",
            ]
        )
        else "concise"
    )


def _should_use_mcp(query: str) -> bool:
    """Detect if query needs live data."""
    q = (query or "").lower()
    if any(k in q for k in ["current", "now", "latest", "today", "live", "real-time", "quote", "price now"]):
        return True
    if any(k in q for k in ["crypto", "bitcoin", "btc", "ethereum", "eth", "coin", "token"]):
        return True
    if any(k in q for k in ["recent", "breaking", "news", "announcement", "update"]):
        return True
    if any(k in q for k in ["option", "options", "chain", "calls", "puts", "expiration", "strike"]):
        return True
    if any(k in q for k in ["income statement", "balance sheet", "cashflow", "cash flow", "fundamentals", "recommendations", "holders", "filings"]):
        return True
    return False


def _normalize_mcp_payload(payload: Any) -> tuple[bool, Dict]:
    """Normalize route_and_call outputs into a dict form. Returns (success, normalized_dict)."""
    if isinstance(payload, dict):
        if payload.get("error"):
            return False, payload
        has_content = payload.get("parsed") is not None or bool(payload.get("raw"))
        return has_content, payload
    
    # Backward compatibility: string payload
    raw = payload
    parsed = None
    if isinstance(raw, str):
        try:
            start_candidates = [i for i in [raw.find("{"), raw.find("[")] if i != -1]
            if start_candidates:
                start = min(start_candidates)
                parsed = json.loads(raw[start:])
        except Exception:
            parsed = None
    
    norm = {"route": {}, "raw": raw, "parsed": parsed}
    return bool(parsed or raw), norm


def _summarize_mcp_payload(norm: Dict, max_rows: int = 5) -> str:
    """Build a short summary from normalized MCP payload."""
    route = norm.get("route", {}) or {}
    server = route.get("server", "?")
    tool = route.get("tool", "?")
    ticker = route.get("primary_ticker", "?")
    intent = route.get("intent", "?")
    
    header = f"📊 Data from {server}/{tool} | Symbol: {ticker} | Intent: {intent}"
    
    parsed = norm.get("parsed")
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        cols = list(parsed[0].keys())
        preview_rows = parsed[:max_rows]
        shown_cols = cols[:8]
        lines = [
            header,
            "",
            f"📈 Rows: {len(parsed)} | Columns: {len(cols)}",
            "Columns: " + ", ".join(shown_cols[:5]),
        ]
        for i, row in enumerate(preview_rows[:3], 1):
            compact = {k: row.get(k) for k in shown_cols[:4]}
            lines.append(f"{i}. {json.dumps(compact, ensure_ascii=False)}")
        return "\n".join(lines)
    
    if isinstance(parsed, dict):
        keys = list(parsed.keys())[:10]
        return "\n".join([header, "", f"Object with {len(parsed)} keys: {', '.join(keys)}"])
    
    raw = norm.get("raw")
    if isinstance(raw, str):
        snippet = raw[:300]
        return "\n".join([header, "", "Raw response:", snippet])
    
    return header


# ============================================================================
# MAIN ANSWER FUNCTION
# ============================================================================

def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    """
    Main answer function - safe for Streamlit.
    Tries MCP first, falls back to RAG.
    """
    style = _detect_style(query, style)
    
    # Extract tickers from query
    tickers = list(
        dict.fromkeys(
            ([t.strip().upper() for t in (ticker.split() if ticker else [])] + _extract_tickers(query))
        )
    )
    
    # Decide whether to use MCP
    should_use = _should_use_mcp(query)
    servers = MCPServer.from_env()
    mcp_attempted = False
    mcp_success = False
    mcp_norm: Dict = {}
    
    print(f"[answer_core] Query: {query[:100]}...")
    print(f"[answer_core] Should use MCP: {should_use}, Servers available: {bool(servers)}")
    
    if should_use and servers:
        try:
            print(f"[answer_core] Calling mcp_auto...")
            mcp_payload = mcp_auto(query)
            mcp_attempted = True
            mcp_success, mcp_norm = _normalize_mcp_payload(mcp_payload)
            print(f"[answer_core] MCP success: {mcp_success}")
        except Exception as e:
            print(f"[answer_core] MCP failed: {e}")
            mcp_attempted = True
            mcp_success = False
            mcp_norm = {"error": str(e)}
    
    # RAG fallback if MCP failed
    docs = []
    if not mcp_success:
        try:
            flt: List[rest.FieldCondition] = []
            if tickers:
                flt.append(rest.FieldCondition(key="symbol", match=rest.MatchAny(any=tickers[:3])))
            
            doc_types = ["quote", "overview", "basic_financials", "as_reported"]
            if servers:
                doc_types.append("mcp")
            flt.append(rest.FieldCondition(key="type", match=rest.MatchAny(any=doc_types)))
            
            docs = asyncio.run(retrieve(query, filters=flt if flt else None, k=12))
            print(f"[answer_core] Retrieved {len(docs)} docs from RAG")
        except Exception as e:
            print(f"[answer_core] RAG retrieval failed: {e}")
            docs = []
    
    # Memory context
    try:
        mem_ctx = asyncio.run(fetch_memory(query=query, k=3))
    except Exception as e:
        print(f"[answer_core] Memory fetch failed: {e}")
        mem_ctx = ""
    
    # Summarize
    extra_context = mem_ctx
    if mcp_attempted:
        extra_context += "\n[Data Source: {}]".format(
            "Live MCP data" if mcp_success else "Cached/RAG data"
        )
    
    if mcp_success:
        answer_text = _summarize_mcp_payload(mcp_norm)
        snippets = []
    else:
        try:
            max_tok = 900 if style == "report" else 512
            answer_text, snippets = asyncio.run(
                rerank_and_summarize(query, docs, style=style, extra_context=extra_context, max_tokens=max_tok)
            )
        except Exception as e:
            print(f"[answer_core] Summarization failed: {e}")
            answer_text = f"Could not generate answer: {e}"
            snippets = []
    
    # Persist to memory
    try:
        asyncio.run(persist_turn(user_text=query, assistant_text=answer_text))
    except Exception as e:
        print(f"[answer_core] Memory persist failed: {e}")
    
    meta = {
        "style": style,
        "tickers": tickers,
        "mcp_attempted": mcp_attempted,
        "mcp_success": mcp_success,
        "available_servers": list((servers or {}).keys()),
        "should_use_mcp": should_use,
    }
    
    return {"answer": answer_text, "snippets": snippets, "meta": meta}


@tool(name="answer", description="Deliver a comprehensive answer to a financial query")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    return answer_core(query, ticker, style)