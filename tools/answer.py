# tools/answer.py - Simplified: MCP first, summarize MCP data only
from __future__ import annotations

import re
import json
from typing import Dict, Any, List

from agno.tools import tool
from mcp_connection.manager import MCPServer
from tools.mcp_router import route_and_call


# Helpers
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z]{1,3})?\b")


def _should_use_mcp(query: str) -> bool:
    """Detect if query needs live data."""
    q = (query or "").lower()
    # Always try MCP for financial queries
    return True


def _normalize_mcp_payload(payload: Any) -> tuple[bool, Dict]:
    """Normalize route_and_call outputs into a dict form."""
    if isinstance(payload, dict):
        if payload.get("error"):
            return False, payload
        has_content = payload.get("parsed") is not None or bool(payload.get("raw"))
        return has_content, payload
    
    return False, {"error": "Invalid payload"}


def _summarize_mcp_payload(norm: Dict) -> str:
    """Build a summary from normalized MCP payload."""
    route = norm.get("route", {}) or {}
    server = route.get("server", "?")
    tool = route.get("tool", "?")
    ticker = route.get("primary_ticker", "?")
    intent = route.get("intent", "?")
    
    header = f"{server}/{tool} | {ticker} | {intent}\n"
    
    parsed = norm.get("parsed")
    
    # If it's a list (e.g., options chain)
    if isinstance(parsed, list) and parsed:
        if isinstance(parsed[0], dict):
            cols = list(parsed[0].keys())[:5]
            return f"{header} Retrieved {len(parsed)} records with columns: {', '.join(cols)}"
        else:
            return f"{header} Retrieved {len(parsed)} items"
    
    # If it's a dict (e.g., stock info)
    if isinstance(parsed, dict):
        keys = list(parsed.keys())[:5]
        items_count = len(parsed)
        return f"{header} Retrieved data with {items_count} fields (sample: {', '.join(keys)})"
    
    # Fallback to raw
    raw = norm.get("raw", "")
    if isinstance(raw, str) and raw:
        snippet = raw[:200]
        return f"{header} Retrieved raw data ({len(raw)} bytes)"
    
    return header + "No data returned"


# ============================================================================
# MAIN ANSWER FUNCTION - MCP ONLY
# ============================================================================

def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    """
    Answer using MCP ONLY - no RAG fallback.
    For all financial queries, route to the best MCP server/tool.
    """
    print(f"[answer_core] Query: {query[:100]}...")
    
    servers = MCPServer.from_env()
    if not servers:
        return {
            "answer": "No MCP servers configured.",
            "snippets": [],
            "meta": {"mcp_attempted": False, "mcp_success": False}
        }
    
    # Call MCP router
    mcp_payload = route_and_call(query)
    mcp_success, mcp_norm = _normalize_mcp_payload(mcp_payload)
    
    print(f"[answer_core] MCP success: {mcp_success}")
    
    if mcp_success:
        answer_text = _summarize_mcp_payload(mcp_norm)
    else:
        error_msg = mcp_norm.get("error", "Unknown error")
        answer_text = f"Could not retrieve data: {error_msg}\n\n💡 Try a more specific query with a ticker symbol (e.g., AAPL, MSFT, BTC-USD)"
    
    meta = {
        "mcp_attempted": True,
        "mcp_success": mcp_success,
        "available_servers": list((servers or {}).keys()),
    }
    
    return {"answer": answer_text, "snippets": [], "meta": meta}


@tool(name="answer", description="Get financial data from MCP servers")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    return answer_core(query, ticker, style)