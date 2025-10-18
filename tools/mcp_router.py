# tools/mcp_router.py
from __future__ import annotations

import re
import json
import asyncio
from functools import lru_cache
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager, MCPServer
from tools.async_utils import run_async_safe, sync_callable
from tools.query_parser import ParsedQuery, parse_query_with_llm
from tools.time_parser import parse_time_range_to_days

# Constants
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3}|-[A-Z0-9]{1,6})?\b")
_STOP = {
    "USD", "USDT", "USDC", "PE", "EV", "EPS", "ETF",
    "AND", "OR", "THE", "A", "AN", "IS", "ARE", "FOR", "TO", "OF", "IN",
}

_CRYPTO_KEYWORDS = {
    "CRYPTO", "BITCOIN", "BTC", "ETH", "ETHEREUM", "COIN", "TOKEN", 
    "BLOCKCHAIN", "DEFI", "XRP", "DOGE", "LTC", "SOL", "ADA"
}

VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

KNOWN_TOOLSETS: Dict[str, List[str]] = {
    "yfinance": [
        "get_stock_info", "get_historical_stock_prices", "get_yahoo_finance_news",
        "get_stock_actions", "get_financial_statement", "get_holder_info",
        "get_option_expiration_dates", "get_option_chain", "get_recommendations",
    ],
    "financial-datasets": [
        "get_current_stock_price", "get_historical_stock_prices",
        "get_current_crypto_price", "get_historical_crypto_prices",
    ]
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\.\-]+", (text or ""))


def _extract_regex_tickers(text: str) -> List[str]:
    matches = _TICKER_RE.findall((text or "").upper())
    return [m for m in matches if m not in _STOP]


def _detect_crypto_intent(query: str) -> bool:
    q_upper = (query or "").upper()
    if any(k in q_upper for k in _CRYPTO_KEYWORDS):
        return True
    if re.search(r"\b[A-Z0-9]{2,10}-USD\b", q_upper):
        return True
    return False


def _safe_json_loads(txt: str) -> Optional[Dict]:
    try:
        if not isinstance(txt, str):
            return None
        return json.loads(txt)
    except Exception:
        return None


def _normalize_tool_name(raw: Any) -> Optional[str]:
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("(") and "," in s and "'" in s:
            m = re.search(r"'([^']+)'", s)
            if m:
                s = m.group(1)
        return s or None
    if isinstance(raw, (list, tuple)) and raw:
        return _normalize_tool_name(raw[0])
    s = str(raw).strip()
    if s.startswith("(") and "," in s and "'" in s:
        m = re.search(r"'([^']+)'", s)
        if m:
            s = m.group(1)
    return s or None


# ============================================================================
# TOOL LISTING (FIXED for async)
# ============================================================================

def _list_server_tools_sync(manager: MCPManager, server: str) -> Dict[str, dict]:
    """
    FIXED: Safely list tools from server, handling both sync and async.
    Never call asyncio.run() here - caller handles async context.
    """
    try:
        # Try sync method first
        if hasattr(manager, "list_tools_sync") and callable(getattr(manager, "list_tools_sync")):
            data = manager.list_tools_sync(server)
        else:
            # If no sync method, return known tools as fallback
            print(f"[_list_server_tools_sync] No list_tools_sync method, using fallback")
            data = {"tools": [{"name": n} for n in KNOWN_TOOLSETS.get(server, [])]}
    
    except Exception as e:
        print(f"[_list_server_tools_sync] Error listing tools from {server}: {e}")
        # Fallback to known toolsets
        data = {"tools": [{"name": n} for n in KNOWN_TOOLSETS.get(server, [])]}
    
    # Parse the response
    tools: Dict[str, dict] = {}
    raw_tools = []
    
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        raw_tools = data["tools"]
    
    for t in raw_tools:
        name_val = t.get("name") if isinstance(t, dict) else getattr(t, "name", None) or t
        name = _normalize_tool_name(name_val) or ""
        if not name or not VALID_TOOL_NAME.match(name) or name.lower() in {"meta", "health", "status"}:
            continue
        tool_obj = t if isinstance(t, dict) else {"name": name}
        tool_obj["name"] = name
        tools[name] = tool_obj
    
    # Prefer known tools if available
    known = set(KNOWN_TOOLSETS.get(server, []))
    if tools and known:
        inter = {k: v for k, v in tools.items() if k in known}
        if inter:
            return inter
    
    if not tools and known:
        return {k: {"name": k, "inputSchema": {"properties": {}}} for k in known}
    
    return tools


# ============================================================================
# TOOL SELECTION
# ============================================================================

def _pick_tool_by_intent(intent: str, tools_map: Dict[str, dict], server: str) -> Optional[str]:
    intent_map = {
        "price_quote": ["get_current_stock_price", "get_stock_info"],
        "news": ["get_yahoo_finance_news", "get_company_news"],
        "analysis": ["get_stock_info", "get_financial_statement"],
        "fundamentals": ["get_financial_statement", "get_balance_sheets", "get_income_statements"],
        "options": ["get_option_chain"],
        "recommendation": ["get_recommendations"],
        "dividend": ["get_stock_actions"],
        "insider": ["get_holder_info"],
        "historical": ["get_historical_stock_prices"],
    }
    
    candidates = intent_map.get(intent, [])
    for tool_name in candidates:
        if tool_name in tools_map:
            return tool_name
    
    return next(iter(tools_map)) if tools_map else None


# ============================================================================
# ARGS BUILDING
# ============================================================================

def _parse_yyyy_mm_dd(d: str) -> Optional[date]:
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None


def _build_args_from_parsed_query(
    manager: MCPManager,
    server: str,
    tool: dict,
    parsed_query: "ParsedQuery"
) -> Dict[str, Any]:
    """Build arguments from structured parsed query."""
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    
    args = {}
    
    # Ticker/symbol
    if "ticker" in props and parsed_query.primary_ticker:
        args["ticker"] = parsed_query.primary_ticker
    elif "symbol" in props and parsed_query.primary_ticker:
        args["symbol"] = parsed_query.primary_ticker
    
    # Time windows
    if "period" in props:
        if parsed_query.time_range and "6 month" in parsed_query.time_range.lower():
            args["period"] = "6mo"
        elif parsed_query.time_range and "year" in parsed_query.time_range.lower():
            args["period"] = "1y"
        else:
            args["period"] = "1y"
    
    if "interval" in props:
        args["interval"] = "1d"
    
    # Financial statement type
    if tool.get("name") == "get_financial_statement":
        if "balance" in parsed_query.raw_intent.lower():
            args["financial_type"] = "balance_sheet"
        elif "cash" in parsed_query.raw_intent.lower():
            args["financial_type"] = "cashflow"
        else:
            args["financial_type"] = "income_stmt"
    
    # Options expiration
    if tool.get("name") == "get_option_chain":
        if "option_type" in props:
            args["option_type"] = parsed_query.options_type or "calls"
        if "expiration_date" in props and parsed_query.primary_ticker:
            try:
                raw = _call_tool_blocking(manager, server, "get_option_expiration_dates", {"ticker": parsed_query.primary_ticker})
                dates = _safe_json_loads(raw)
                if isinstance(dates, list) and dates:
                    args["expiration_date"] = dates[0]
            except Exception:
                pass
    
    return args


# ============================================================================
# TOOL CALLING
# ============================================================================

def _call_tool_blocking(manager: MCPManager, server: str, tool_name: str, args: dict) -> str:
    """Call tool synchronously (blocks)."""
    try:
        if hasattr(manager, "call_sync") and callable(getattr(manager, "call_sync")):
            return manager.call_sync(server, tool_name, args)
        else:
            print(f"[_call_tool_blocking] No call_sync method, trying async...")
            # Fallback - shouldn't reach here in production
            return f"Tool call failed: {tool_name}"
    except Exception as e:
        print(f"[_call_tool_blocking] Error calling {tool_name}: {e}")
        return f"Tool call error: {str(e)}"


# ============================================================================
# MAIN ROUTING LOGIC
# ============================================================================

def route_and_call(query: str) -> Dict[str, Any]:
    """
    Route a query to the best MCP server/tool and call it.
    
    Returns:
        {
            "route": {"server": ..., "tool": ..., "tickers": ...},
            "parsed": parsed_json_or_none,
            "raw": raw_response_string,
            "error": error_message_or_none
        }
    """
    try:
        # Step 1: Parse query with LLM
        parsed_query = parse_query_with_llm(query)
        
        print(f"[route_and_call] Parsed: ticker={parsed_query.primary_ticker}, intent={parsed_query.intent}")
        
        # Step 2: Check available servers
        available_servers = MCPServer.from_env()
        if not available_servers:
            return {"error": "No MCP servers configured", "route": {}}
        
        # Step 3: Pick server
        manager = MCPManager()
        server = _pick_server(parsed_query, available_servers)
        if not server:
            return {"error": f"No suitable server found. Available: {list(available_servers.keys())}", "route": {}}
        
        print(f"[route_and_call] Selected server: {server}")
        
        # Step 4: List and pick tool
        tools_map = _list_server_tools_sync(manager, server)
        if not tools_map:
            return {"error": f"No tools found in server: {server}", "route": {}}
        
        tool_name = _pick_tool_by_intent(parsed_query.intent, tools_map, server)
        if not tool_name:
            return {"error": f"No tool matching intent: {parsed_query.intent}", "route": {}}
        
        print(f"[route_and_call] Selected tool: {tool_name}")
        
        # Step 5: Build arguments
        tool_obj = tools_map.get(tool_name, {"name": tool_name})
        args = _build_args_from_parsed_query(manager, server, tool_obj, parsed_query)
        
        # Check required args
        schema = tool_obj.get("inputSchema", {})
        required = schema.get("required", []) if isinstance(schema, dict) else []
        requires_ticker = any(r in ("ticker", "symbol") for r in required)
        
        if requires_ticker and not (args.get("ticker") or args.get("symbol")):
            return {
                "error": "Could not resolve a ticker from the query. Please specify a symbol (e.g., AAPL or BTC-USD).",
                "route": {"server": server, "tool": tool_name}
            }
        
        print(f"[route_and_call] Calling with args: {args}")
        
        # Step 6: Call tool
        result = _call_tool_blocking(manager, server, tool_name, args)
        
        # Step 7: Parse result
        parsed_result = _safe_json_loads(result)
        
        return {
            "route": {
                "server": server,
                "tool": tool_name,
                "primary_ticker": parsed_query.primary_ticker,
                "intent": parsed_query.intent,
            },
            "parsed": parsed_result,
            "raw": result,
            "error": None
        }
        
    except Exception as e:
        print(f"[route_and_call] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Routing failed: {str(e)}", "route": {}}


def _pick_server(parsed_query: "ParsedQuery", available_servers: Dict[str, MCPServer]) -> Optional[str]:
    """
    Pick best server based on parsed query.
    
    Rules:
    1. Crypto intent (BTC, ETH, -USD suffix) → financial-datasets
    2. Options/fundamentals → yfinance
    3. Stock price/historical → yfinance (better coverage)
    4. Default → yfinance
    """
    primary_ticker = (parsed_query.primary_ticker or "").upper()
    
    # Crypto symbols (well-known)
    crypto_symbols = {"BTC", "ETH", "XRP", "DOGE", "LTC", "SOL", "ADA", "MATIC", "AVAX", "LINK", "UNI", "AAVE", "ATOM"}
    
    # Rule 1: Crypto detection
    is_crypto = (
        primary_ticker.endswith("-USD") or 
        primary_ticker in crypto_symbols
    )
    
    if is_crypto:
        if "financial-datasets" in available_servers:
            print(f"[_pick_server] Crypto detected ({primary_ticker}) → financial-datasets")
            return "financial-datasets"
        # Fallback to yfinance if financial-datasets unavailable
        if "yfinance" in available_servers:
            print(f"[_pick_server] Crypto but financial-datasets unavailable, using yfinance")
            return "yfinance"
    
    # Rule 2: Options/fundamentals prefer yfinance
    if parsed_query.intent in ("options", "fundamentals", "recommendation", "dividend", "insider"):
        if "yfinance" in available_servers:
            print(f"[_pick_server] Intent {parsed_query.intent} → yfinance")
            return "yfinance"
    
    # Rule 3: Default preference order (yfinance better coverage)
    if "yfinance" in available_servers:
        print(f"[_pick_server] Default → yfinance")
        return "yfinance"
    if "financial-datasets" in available_servers:
        print(f"[_pick_server] yfinance unavailable → financial-datasets")
        return "financial-datasets"
    
    # Last resort
    result = next(iter(available_servers)) if available_servers else None
    print(f"[_pick_server] Last resort → {result}")
    return result


# ============================================================================
# AGNO TOOL
# ============================================================================

@tool(
    name="mcp_auto",
    description="Auto-router for MCP servers. Intelligently routes queries to the best financial data source."
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    """Entry point for agent."""
    result = route_and_call(query)
    
    # If error, return error message
    if result.get("error"):
        return f"Error: {result['error']}"
    
    # Return raw response (can be parsed later)
    raw = result.get("raw", "")
    route_info = result.get("route", {})
    
    info_str = f"[Route: {route_info.get('server')}/{route_info.get('tool')} | Ticker: {route_info.get('primary_ticker')} | Intent: {route_info.get('intent')}]"
    
    return f"{info_str}\n\n{raw}"