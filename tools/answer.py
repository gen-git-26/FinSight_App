# tools/answer.py

from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, Any, List, Tuple

from agno.tools import tool
from qdrant_client.http import models as rest

# RAG / memory
from utils.config import load_settings
from rag.fusion import retrieve, rerank_and_summarize
from memory.manager import fetch_memory, persist_turn

# Tooling backstops (non-MCP)
from tools.tools import (
    finnhub_stock_info,
    finnhub_basic_financials,
    finnhub_financials_as_reported,
    company_overview,
    finnhub_financials_new,
)

# MCP routing (live data)
from mcp_connection.manager import MCPServer
from tools.mcp_router import mcp_auto, resolve_tickers  # reuse the unified resolver


# -----------------------------
# Lightweight intent & style
# -----------------------------

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z0-9]{1,6})?\b")
_STOP = {
    "AND", "THE", "FOR", "WITH", "OVER", "FROM", "THIS", "THAT", "INTO", "YOUR",
    "US", "USA", "NYSE", "NASDAQ", "SP", "ETF", "PE", "EV", "EBITDA", "EPS",
    "CAGR", "ROE", "P", "E", "PRICE", "NEWS", "OPTIONS", "CHAIN", "PUTS", "CALLS",
}

def _extract_tickers(text: str) -> List[str]:
    return [c for c in _TICKER_RE.findall(text or "") if c not in _STOP]

def _detect_style(q: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    ql = (q or "").lower()
    if any(k in ql for k in [
        "executive summary", "report", "comprehensive", "deep dive",
        "overview including", "investment memo", "thesis",
    ]):
        return "report"
    return "concise"

def _should_use_mcp(query: str) -> bool:
    q = (query or "").lower()
    if any(k in q for k in ["current", "now", "latest", "today", "live", "real-time", "quote", "price now"]):
        return True
    if any(k in q for k in ["crypto", "bitcoin", "btc", "ethereum", "eth", "coin", "token"]):
        return True
    if any(k in q for k in ["recent", "breaking", "news", "announcement", "update"]):
        return True
    return False


# -----------------------------
# Backfill fetch (non-MCP) on demand
# -----------------------------

def _warm_backstops(tickers: List[str]) -> None:
    """
    Opportunistically populate the vector store with multi-source docs
    so RAG has something to fuse if MCP is unavailable.
    No hard-coded assets; operate on the provided tickers only.
    """
    for t in tickers[:3]:
        try:
            finnhub_stock_info(t)
        except Exception:
            pass
        # Basic financials (multiple metric groups)
        for metric in ("price", "valuation", "growth"):
            try:
                finnhub_basic_financials(t, metric=metric)
            except Exception:
                pass
        # As reported & overview
        try:
            finnhub_financials_as_reported(t)
        except Exception:
            pass
        try:
            company_overview(t)
        except Exception:
            pass
        # Additional modern endpoint if available
        try:
            finnhub_financials_new(t)
        except Exception:
            pass


# -----------------------------
# Core answering logic
# -----------------------------

def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _collect_candidate_tickers(query: str, explicit_ticker: str = "") -> List[str]:
    raw = []
    if explicit_ticker:
        raw.extend([s.strip().upper() for s in explicit_ticker.split() if s.strip()])
    raw.extend(_extract_tickers(query))
    # Use unified resolver to normalize free-text (e.g., "Tesla" -> "TSLA", "bitcoin" -> "BTC-USD")
    resolved = resolve_tickers(query)
    raw.extend(resolved)
    return _dedupe([t for t in raw if t])

def _build_filters(tickers: List[str], mcp_enabled: bool) -> List[rest.FieldCondition]:
    flt: List[rest.FieldCondition] = []
    if tickers:
        flt.append(rest.FieldCondition(key="symbol", match=rest.MatchAny(any=tickers[:3])))
    doc_types = ["quote", "overview", "basic_financials", "as_reported"]
    if mcp_enabled:
        doc_types.append("mcp")
    flt.append(rest.FieldCondition(key="type", match=rest.MatchAny(any=doc_types)))
    return flt

def _choose_max_tokens(style: str, mcp_success: bool) -> int:
    if style == "report" and mcp_success:
        return 1200
    if style == "report":
        return 900
    return 650 if mcp_success else 512


def _mcp_try(query: str) -> Tuple[bool, Dict[str, Any] | None, str]:
    """
    Try live MCP call via auto-router.
    Returns: (success, parsed_payload_or_None, raw_text)
    """
    try:
        raw = mcp_auto(query)
        if isinstance(raw, str) and raw.startswith("MCP routing failed:"):
            return False, None, raw
        # Best-effort parse: prefer JSON in content if router returned it
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else None
        except Exception:
            parsed = None
        return True, parsed, raw
    except Exception as e:
        return False, None, f"MCP Error: {e}"


# --------- Stable core that UI always calls ---------
def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    style = _detect_style(query, style)

    # Unify ticker resolution (regex + resolver + explicit param)
    tickers = _collect_candidate_tickers(query, explicit_ticker=ticker)

    # Live data?
    use_mcp = _should_use_mcp(query)
    servers = MCPServer.from_env()
    mcp_attempted = False
    mcp_success = False
    mcp_raw = ""
    mcp_payload: Dict[str, Any] | None = None

    if use_mcp and servers:
        mcp_attempted = True
        ok, payload, raw = _mcp_try(query)
        mcp_success, mcp_payload, mcp_raw = ok, payload, raw

    # If MCP failed and we have candidate tickers: warm up the cache from static APIs
    if not mcp_success and tickers:
        _warm_backstops(tickers)

    # Build RAG filter and memory
    flt = _build_filters(tickers, mcp_enabled=bool(servers))
    docs = asyncio.run(retrieve(query, filters=flt, k=24))

    mem_ctx = asyncio.run(fetch_memory(query=query, k=3))
    extra_context = mem_ctx
    if mcp_attempted:
        extra_context += "\n[Data Source Status: {}]".format(
            "Live data included" if mcp_success else "Live data unavailable, using cached data"
        )
        if not mcp_success and use_mcp:
            extra_context += "\n[Note: Live data requested but MCP unavailable]"

    max_tok = _choose_max_tokens(style, mcp_success)

    # Prefer the MCP textual answer if router returned a clear body; otherwise do RAG synth
    if mcp_success and isinstance(mcp_payload, dict) and mcp_payload.get("answer"):
        answer_text = str(mcp_payload["answer"])
        snippets = mcp_payload.get("snippets", [])
    else:
        answer_text, snippets = asyncio.run(
            rerank_and_summarize(
                query,
                docs,
                style=style,
                extra_context=extra_context,
                max_tokens=max_tok,
            )
        )

    # Persist turn
    asyncio.run(persist_turn(user_text=query, assistant_text=answer_text))

    # Diagnostics/meta
    meta = {
        "style": style,
        "tickers": tickers,
        "mcp_attempted": mcp_attempted,
        "mcp_success": mcp_success,
        "available_servers": list((servers or {}).keys()),
        "docs_retrieved": len(snippets) if mcp_success else len(docs),
        "should_use_mcp": use_mcp,
        "mcp_raw": mcp_raw[:1000] if isinstance(mcp_raw, str) else "",
    }

    return {"answer": answer_text, "snippets": snippets, "meta": meta}


@tool(name="answer", description="Fusion RAG + MCP with unified symbol resolution and dynamic fallbacks.")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    return answer_core(query, ticker, style)
