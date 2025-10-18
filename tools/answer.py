# tools/answer.py
from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, Any, List, Tuple

from agno.tools import tool
from qdrant_client.http import models as rest

from utils.config import load_settings  # noqa: F401  # kept if needed elsewhere
from rag.fusion import retrieve, rerank_and_summarize
from tools.tools import (
    finnhub_stock_info,
    finnhub_basic_financials,
    finnhub_financials_as_reported,
    company_overview,
    finnhub_financials_new,
)
from memory.manager import fetch_memory, persist_turn
from mcp_connection.manager import MCPServer
from tools.mcp_router import mcp_auto

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z]{1,3})?\b")
_STOP = {
    "AND", "THE", "FOR", "WITH", "OVER", "FROM", "THIS", "THAT", "INTO", "YOUR",
    "US", "USA", "NYSE", "NASDAQ", "SP", "ETF", "PE", "EV", "EBITDA", "EPS",
    "CAGR", "ROE", "P", "E",
}


def _extract_tickers(text: str) -> List[str]:
    # Keep original casing to avoid capturing common words that would become uppercase
    matches = _TICKER_RE.findall(text or "")
    out: List[str] = []
    for m in matches:
        if m in _STOP:
            continue
        # Keep tokens that look like real symbols
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
                "executive summary",
                "report",
                "comprehensive",
                "deep dive",
                "overview including",
                "investment memo",
                "thesis",
            ]
        )
        else "concise"
    )


def _should_use_mcp(query: str) -> bool:
    q = (query or "").lower()
    if any(k in q for k in ["current", "now", "latest", "today", "live", "real-time", "quote", "price now"]):
        return True
    if any(k in q for k in ["crypto", "bitcoin", "btc", "ethereum", "eth", "coin", "token"]):
        return True
    if any(k in q for k in ["recent", "breaking", "news", "announcement", "update"]):
        return True
    if any(k in q for k in ["option", "options", "chain", "calls", "puts", "expiration", "strike"]):
        return True
    if any(k in q for k in ["income statement", "balance sheet", "cashflow", "cash flow", "fundamentals", "recommendations", "holders", "filings", "sec"]):
        return True
    return False


# -----------------------------------------------------
# MCP payload normalization and summary
# -----------------------------------------------------

Payload = Dict[str, Any]


def _normalize_mcp_payload(payload: Any) -> Tuple[bool, Payload]:
    """Normalize route_and_call/mcp_auto outputs into a dict form.
    Returns (success, normalized_dict).
    """
    # Already a dict from the new router
    if isinstance(payload, dict):
        if payload.get("error"):
            return False, payload
        # Consider it successful if we have either parsed or raw content
        has_content = payload.get("parsed") is not None or bool(payload.get("raw"))
        return has_content, payload

    # Backward compatibility: string payload
    raw = payload
    parsed = None
    if isinstance(raw, str):
        try:
            # Try to parse JSON from the first JSON-looking segment
            start_candidates = [i for i in [raw.find("{"), raw.find("[")] if i != -1]
            if start_candidates:
                start = min(start_candidates)
                parsed = json.loads(raw[start:])
        except Exception:
            parsed = None
    norm = {"route": {}, "raw": raw, "parsed": parsed}
    return bool(parsed or raw), norm


def _summarize_mcp_payload(norm: Payload, max_rows: int = 5) -> str:
    """Build a short human-readable summary from normalized MCP payload."""
    route = norm.get("route", {}) or {}
    server = route.get("server", "?")
    tool = route.get("tool", "?")
    tickers = route.get("tickers", []) or []

    header = f"Route: {server}/{tool} | Tickers: {tickers}"

    parsed = norm.get("parsed")
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        cols = list(parsed[0].keys())
        preview_rows = parsed[:max_rows]
        # Choose a concise set of columns if too many
        shown_cols = cols[:8]
        # Create a simple table-like preview as text
        lines = [
            header,
            "",
            f"Rows: {len(parsed)} | Columns: {len(cols)}",
            "Columns: " + ", ".join(shown_cols),
        ]
        # Add the first few rows serialized compactly
        for i, row in enumerate(preview_rows, 1):
            compact = {k: row.get(k) for k in shown_cols}
            lines.append(f"{i}. {json.dumps(compact, ensure_ascii=False)}")
        return "\n".join(lines)

    if isinstance(parsed, dict):
        keys = list(parsed.keys())
        return "\n".join([header, "", f"Object with keys: {', '.join(keys[:15])}"])  # type: ignore

    # Fallback to raw text size
    raw = norm.get("raw")
    if isinstance(raw, str):
        snippet = raw[:500]
        return "\n".join([header, "", "Raw snippet:", snippet])
    return header


# -----------------------------------------------------
# Main answer function
# -----------------------------------------------------


# tools/answer.py - החלף את answer_core

def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    style = _detect_style(query, style)
    
    # Try MCP first (with new parser)
    servers = MCPServer.from_env()
    mcp_success = False
    mcp_norm = {}
    
    if servers:
        try:
            mcp_payload = mcp_auto(query)  # Uses new route_and_call_v2 internally
            mcp_success, mcp_norm = _normalize_mcp_payload(mcp_payload)
        except Exception as e:
            print(f"[answer] MCP failed: {e}")
            mcp_success = False
    
    # RAG fallback
    docs = []
    if not mcp_success:
        try:
            flt = []
            docs = asyncio.run(retrieve(query, filters=flt if not flt else None, k=24))
        except Exception:
            docs = []
    
    # Summarize
    if mcp_success:
        answer_text = _summarize_mcp_payload(mcp_norm)
        snippets = []
    else:
        answer_text, snippets = asyncio.run(
            rerank_and_summarize(query, docs, style=style, max_tokens=900)
        )
    
    return {"answer": answer_text, "snippets": snippets, "meta": {"mcp_success": mcp_success}}


@tool(name="answer", description="deliver a comprehensiv answer to a query")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    return answer_core(query, ticker, style)
