# tools/answer.py
from __future__ import annotations

import re
import json
from typing import Dict, Any, List

from agno.tools import tool
from mcp_connection.manager import MCPServer
from tools.mcp_router import route_and_call
from tools.smart_formatter import _format_data_as_text, _detect_content_type


def _normalize_mcp_payload(payload: Any) -> tuple[bool, Dict]:
    """Normalize route_and_call outputs into a dict form."""
    if isinstance(payload, dict):
        if payload.get("error"):
            return False, payload
        has_content = payload.get("parsed") is not None or bool(payload.get("raw"))
        return has_content, payload
    
    return False, {"error": "Invalid payload"}


def _summarize_mcp_payload(norm: Dict) -> tuple[str, str, bool]:
    """
    Build a summary from normalized MCP payload with smart formatting.
    
    Returns:
        (answer_text, content_type, is_dataframe)
    """
    route = norm.get("route", {}) or {}
    server = route.get("server", "?")
    tool = route.get("tool", "?")
    ticker = route.get("primary_ticker", "?")
    intent = route.get("intent", "?")
    
    header = f"📊 **Data Source**: {server}/{tool}\n"
    header += f"🔹 **Symbol**: {ticker} | **Intent**: {intent}\n"
    header += f"{'='*60}\n\n"
    
    parsed = norm.get("parsed")
    
    if parsed:
        # Auto-detect best display format
        content_type = _detect_content_type(parsed)
        formatted_text, detected_type, is_df = _format_data_as_text(parsed, content_type)
        
        return header + formatted_text, detected_type, is_df
    
    # Fallback to raw
    raw = norm.get("raw", "")
    if isinstance(raw, str) and raw:
        # Try to parse raw as JSON
        try:
            if raw.startswith('[') or raw.startswith('{'):
                parsed_raw = json.loads(raw)
                content_type = _detect_content_type(parsed_raw)
                formatted_text, detected_type, is_df = _format_data_as_text(parsed_raw, content_type)
                return header + formatted_text, detected_type, is_df
        except:
            pass
        
        # Raw text response
        return header + f"Retrieved {len(raw)} bytes:\n\n{raw[:2000]}", "text", False
    
    return header + " No data returned", "error", False


def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    """
    Answer using MCP with intelligent formatting.
    Returns formatted text that's optimized for Streamlit display.
    """
    print(f"[answer_core] Query: {query[:100]}...")
    
    servers = MCPServer.from_env()
    if not servers:
        return {
            "answer": "No MCP servers configured.",
            "snippets": [],
            "display_type": "error",
            "meta": {"mcp_attempted": False, "mcp_success": False}
        }
    
    # Call MCP router
    mcp_payload = route_and_call(query)
    mcp_success, mcp_norm = _normalize_mcp_payload(mcp_payload)
    
    print(f"[answer_core] MCP success: {mcp_success}")
    
    if mcp_success:
        answer_text, display_type, is_dataframe = _summarize_mcp_payload(mcp_norm)
    else:
        error_msg = mcp_norm.get("error", "Unknown error")
        answer_text = f"Could not retrieve data: {error_msg}\n\n💡 **Tip**: Try a more specific query with a ticker symbol (e.g., AAPL, MSFT, SOL-USD)"
        display_type = "error"
        is_dataframe = False
    
    meta = {
        "mcp_attempted": True,
        "mcp_success": mcp_success,
        "available_servers": list((servers or {}).keys()),
        "display_type": display_type,
        "is_dataframe": is_dataframe,
    }
    
    return {
        "answer": answer_text,
        "snippets": [],
        "display_type": display_type,
        "is_dataframe": is_dataframe,
        "meta": meta
    }


@tool(name="answer", description="Get financial data from MCP servers with smart formatting")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    return answer_core(query, ticker, style)