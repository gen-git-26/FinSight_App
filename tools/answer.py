# tools/answer.py - MCP Only with Full Data Display
from __future__ import annotations

import re
import json
from typing import Dict, Any, List

from agno.tools import tool
from mcp_connection.manager import MCPServer
from tools.mcp_router import route_and_call


def _normalize_mcp_payload(payload: Any) -> tuple[bool, Dict]:
    """Normalize route_and_call outputs into a dict form."""
    if isinstance(payload, dict):
        if payload.get("error"):
            return False, payload
        has_content = payload.get("parsed") is not None or bool(payload.get("raw"))
        return has_content, payload
    
    return False, {"error": "Invalid payload"}


def _format_data_as_text(parsed: Any, max_chars: int = 3000) -> str:
    """Format parsed data as readable text for display."""
    
    # List of records (e.g., options chain, financial statements)
    if isinstance(parsed, list) and parsed:
        if isinstance(parsed[0], dict):
            lines = []
            lines.append(f"Retrieved {len(parsed)} records:\n")
            
            # Show first 5 records with all fields
            for i, record in enumerate(parsed[:5], 1):
                lines.append(f"Record {i}:")
                for key, value in record.items():
                    # Format value nicely
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            val_str = f"{value:,.2f}" if abs(value) > 0.01 else f"{value}"
                        else:
                            val_str = f"{value:,}"
                    else:
                        val_str = str(value)
                    lines.append(f"  {key}: {val_str}")
                lines.append("")
            
            if len(parsed) > 5:
                lines.append(f"... and {len(parsed) - 5} more records")
            
            text = "\n".join(lines)
            return text[:max_chars]
        else:
            return "\n".join(str(item) for item in parsed[:20])
    
    # Single dict (e.g., stock info)
    if isinstance(parsed, dict):
        lines = []
        lines.append(f"Data Summary ({len(parsed)} fields):\n")
        
        # Show key metrics (common financial fields)
        key_fields = [
            "currentPrice", "regularMarketPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "marketCap", "volume", "averageVolume", "beta", "pe", "trailingPE", "forwardPE",
            "dividend", "yield", "earnings", "eps", "epsTrailingTwelveMonths",
            "priceToBook", "priceToSalesTrailing12Months", "debtToEquity", "returnOnAssets",
            "shortName", "longName", "sector", "industry", "website"
        ]
        
        for field in key_fields:
            if field in parsed:
                value = parsed[field]
                # Format numbers nicely
                if isinstance(value, (int, float)):
                    if isinstance(value, float) and value > 100:
                        val_str = f"{value:,.2f}"
                    elif isinstance(value, float):
                        val_str = f"{value:.4f}"
                    else:
                        val_str = f"{value:,}"
                else:
                    val_str = str(value)
                lines.append(f"  {field}: {val_str}")
        
        # Show any remaining important fields
        lines.append("\nOther fields:")
        shown_fields = set(key_fields)
        for key in sorted(parsed.keys())[:20]:
            if key not in shown_fields and not key.startswith("_"):
                value = parsed[key]
                if isinstance(value, (int, float, str, bool)):
                    if isinstance(value, float):
                        val_str = f"{value:.2f}" if abs(value) > 0.01 else str(value)
                    else:
                        val_str = str(value)
                    lines.append(f"  {key}: {val_str}")
        
        text = "\n".join(lines)
        return text[:max_chars]
    
    # Fallback
    return json.dumps(parsed, indent=2, ensure_ascii=False)[:max_chars]


def _summarize_mcp_payload(norm: Dict) -> str:
    """Build a summary from normalized MCP payload with full data."""
    route = norm.get("route", {}) or {}
    server = route.get("server", "?")
    tool = route.get("tool", "?")
    ticker = route.get("primary_ticker", "?")
    intent = route.get("intent", "?")
    
    header = f"Data from {server}/{tool}\n🔹 Symbol: {ticker} | Intent: {intent}\n{'='*60}\n"
    
    parsed = norm.get("parsed")
    
    if parsed:
        data_text = _format_data_as_text(parsed)
        return header + data_text
    
    # Fallback to raw
    raw = norm.get("raw", "")
    if isinstance(raw, str) and raw:
        return header + f"Retrieved {len(raw)} bytes of data:\n{raw[:2000]}"
    
    return header + "No data returned"


# ============================================================================
# MAIN ANSWER FUNCTION - MCP ONLY
# ============================================================================

def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    """
    Answer using MCP ONLY with full data display.
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