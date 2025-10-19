# tools/mcp_router.py 
from __future__ import annotations

import re
import json
from typing import Dict, List, Optional, Any

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager, MCPServer
from tools.async_utils import run_async_safe
from tools.query_parser import ParsedQuery, parse_query_with_llm
from tools.crypto_resolver import get_crypto_resolver, is_crypto_query

# Constants
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

VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


def _parse_mcp_response(raw: str) -> tuple[Optional[Dict], str]:
    """
    Try to extract structured data from MCP response.
    Returns: (parsed_dict_or_none, cleaned_text)
    """
    if not isinstance(raw, str):
        return None, str(raw)
    
    # Try 1: Direct JSON
    parsed = _safe_json_loads(raw)
    if parsed:
        return parsed, raw
    
    # Try 2: JSON within text (extract from [...] or {...})
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            extracted = match.group(0)
            parsed = _safe_json_loads(extracted)
            if parsed:
                return parsed, extracted
    
    # Try 3: Parse error messages
    if "Error:" in raw or "error:" in raw.lower():
        return None, raw
    
    return None, raw


# ============================================================================
# TOOL LISTING
# ============================================================================

def _list_server_tools_sync(manager: MCPManager, server: str) -> Dict[str, dict]:
    """List available tools from server."""
    try:
        if hasattr(manager, "list_tools_sync"):
            data = manager.list_tools_sync(server)
        else:
            print(f"[mcp_router] No list_tools_sync, using fallback")
            data = {"tools": [{"name": n} for n in KNOWN_TOOLSETS.get(server, [])]}
    
    except Exception as e:
        print(f"[mcp_router] Error listing tools: {e}")
        data = {"tools": [{"name": n} for n in KNOWN_TOOLSETS.get(server, [])]}
    
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
    """Pick tool based on intent."""
    intent_map = {
        "price_quote": ["get_current_stock_price", "get_current_crypto_price", "get_stock_info"],
        "news": ["get_yahoo_finance_news", "get_company_news"],
        "analysis": ["get_stock_info", "get_financial_statement"],
        "fundamentals": ["get_financial_statement", "get_balance_sheets", "get_income_statements"],
        "options": ["get_option_chain"],
        "recommendation": ["get_recommendations"],
        "dividend": ["get_stock_actions"],
        "insider": ["get_holder_info"],
        "historical": ["get_historical_stock_prices", "get_historical_crypto_prices"],
    }
    
    candidates = intent_map.get(intent, [])
    for tool_name in candidates:
        if tool_name in tools_map:
            return tool_name
    
    return next(iter(tools_map)) if tools_map else None


# ============================================================================
# ARGS BUILDING
# ============================================================================

# tools/mcp_router.py - ARGS BUILDING SECTION ONLY (replace lines 155-210)

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
    
    # Resolve ticker with crypto support
    ticker = parsed_query.primary_ticker or ""
    
    # Check if crypto
    crypto_resolver = get_crypto_resolver()
    crypto_result = crypto_resolver.resolve(ticker or parsed_query.raw_intent, use_cache_only=True)
    
    if crypto_result['found']:
        ticker = crypto_result['symbol']
        # Add -USD suffix for crypto in financial-datasets
        if server == "financial-datasets" and not ticker.endswith("-USD"):
            ticker = f"{ticker}-USD"
    
    # Set ticker/symbol
    if "ticker" in props and ticker:
        args["ticker"] = ticker
    elif "symbol" in props and ticker:
        args["symbol"] = ticker
    
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
    
    # Recommendations - ADD THIS SECTION
    if tool.get("name") == "get_recommendations":
        if "recommendation_type" in props:
            # Default to analyst recommendations
            args["recommendation_type"] = "analyst"
    
    # Options
    if tool.get("name") == "get_option_chain":
        if "option_type" in props:
            args["option_type"] = parsed_query.options_type or "calls"
    
    return args


# ============================================================================
# TOOL CALLING
# ============================================================================

def _call_tool_blocking(manager: MCPManager, server: str, tool_name: str, args: dict) -> str:
    """Call tool synchronously."""
    try:
        if hasattr(manager, "call_sync") and callable(getattr(manager, "call_sync")):
            return manager.call_sync(server, tool_name, args)
        else:
            print(f"[mcp_router] No call_sync method")
            return f"Tool call failed: {tool_name}"
    except Exception as e:
        print(f"[mcp_router] Error calling {tool_name}: {e}")
        return f"Tool call error: {str(e)}"


# ============================================================================
# SERVER SELECTION (IMPROVED with Crypto Detection)
# ============================================================================

def _pick_server(parsed_query: "ParsedQuery", available_servers: Dict[str, MCPServer]) -> Optional[str]:
    """
    Pick best server with dynamic crypto detection.
    
    Priority:
    1. Crypto detection → financial-datasets (if available)
    2. Specific intents (options, fundamentals) → yfinance
    3. Default → yfinance, then financial-datasets
    """
    # Try to detect crypto from query or ticker
    crypto_resolver = get_crypto_resolver()
    
    query_text = parsed_query.primary_ticker or parsed_query.raw_intent
    crypto_result = crypto_resolver.resolve(query_text, use_cache_only=True)
    
    is_crypto = crypto_result['found']
    
    if is_crypto:
        print(f"🪙 [mcp_router] Crypto detected: {crypto_result['symbol']} ({crypto_result['name']}) from {crypto_result['source']}")
        
        if "financial-datasets" in available_servers:
            return "financial-datasets"
        
        # Fallback if financial-datasets not available
        if "yfinance" in available_servers:
            print(f"⚠️ [mcp_router] financial-datasets unavailable, trying yfinance")
            return "yfinance"
    
    # Non-crypto: prefer yfinance for better coverage
    if "yfinance" in available_servers:
        return "yfinance"
    
    if "financial-datasets" in available_servers:
        return "financial-datasets"
    
    return next(iter(available_servers)) if available_servers else None


# ============================================================================
# MAIN ROUTING LOGIC
# ============================================================================

def route_and_call(query: str) -> Dict[str, Any]:
    """
    Route query to best MCP server/tool and call it.
    
    Returns:
        {
            "route": {"server": ..., "tool": ..., "primary_ticker": ...},
            "parsed": parsed_json_or_none,
            "raw": raw_response_string,
            "error": error_message_or_none
        }
    """
    try:
        # Step 1: Parse query
        parsed_query = parse_query_with_llm(query)
        print(f"[route_and_call] Parsed: ticker={parsed_query.primary_ticker}, intent={parsed_query.intent}")
        
        # Step 2: Check servers
        available_servers = MCPServer.from_env()
        if not available_servers:
            return {"error": "No MCP servers configured", "route": {}}
        
        # Step 3: Pick server
        manager = MCPManager()
        server = _pick_server(parsed_query, available_servers)
        if not server:
            return {"error": f"No suitable server. Available: {list(available_servers.keys())}", "route": {}}
        
        print(f"[route_and_call] Selected server: {server}")
        
        # Step 4: List and pick tool
        tools_map = _list_server_tools_sync(manager, server)
        if not tools_map:
            return {"error": f"No tools in {server}", "route": {}}
        
        tool_name = _pick_tool_by_intent(parsed_query.intent, tools_map, server)
        if not tool_name:
            return {"error": f"No tool for intent: {parsed_query.intent}", "route": {}}
        
        print(f"[route_and_call] Selected tool: {tool_name}")
        
        # Step 5: Build args
        tool_obj = tools_map.get(tool_name, {"name": tool_name})
        args = _build_args_from_parsed_query(manager, server, tool_obj, parsed_query)
        
        # Check required args
        schema = tool_obj.get("inputSchema", {})
        required = schema.get("required", []) if isinstance(schema, dict) else []
        requires_ticker = any(r in ("ticker", "symbol") for r in required)
        
        if requires_ticker and not (args.get("ticker") or args.get("symbol")):
            return {
                "error": "Could not resolve ticker. Please specify (e.g., AAPL or SOL).",
                "route": {"server": server, "tool": tool_name}
            }
        
        print(f"[route_and_call] Args: {args}")
        
        # Step 6: Call tool
        result = _call_tool_blocking(manager, server, tool_name, args)
        
        # Step 7: Parse result WITH robust fallback
        parsed_result, cleaned_raw = _parse_mcp_response(result)
        
        return {
            "route": {
                "server": server,
                "tool": tool_name,
                "primary_ticker": parsed_query.primary_ticker,
                "intent": parsed_query.intent,
            },
            "parsed": parsed_result,
            "raw": cleaned_raw,
            "error": None
        }
        
    except Exception as e:
        print(f"[route_and_call] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Routing failed: {str(e)}", "route": {}}


# ============================================================================
# AGNO TOOL
# ============================================================================

@tool(
    name="mcp_auto",
    description="Auto-router for MCP servers with crypto detection"
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    """Entry point for agent."""
    result = route_and_call(query)
    
    if result.get("error"):
        return f"Error: {result['error']}"
    
    raw = result.get("raw", "")
    route_info = result.get("route", {})
    
    info_str = f"[{route_info.get('server')}/{route_info.get('tool')} | {route_info.get('primary_ticker')} | {route_info.get('intent')}]"
    
    return f"{info_str}\n\n{raw}"