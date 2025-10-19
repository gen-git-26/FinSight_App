# tools/mcp_router.py
from __future__ import annotations

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager, MCPServer
from tools.async_utils import run_async_safe
from tools.query_parser import ParsedQuery, parse_query_with_llm
from tools.crypto_resolver import get_crypto_resolver, is_crypto_query

# ============================================================================
# CONSTANTS
# ============================================================================

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
    """Safely parse JSON string."""
    try:
        if not isinstance(txt, str):
            return None
        return json.loads(txt)
    except Exception:
        return None


def _normalize_tool_name(raw: Any) -> Optional[str]:
    """Normalize tool name from various formats."""
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


def _get_next_monthly_expiration() -> str:
    """
    Get the next monthly options expiration date (3rd Friday of the month).
    This is the most liquid options expiration.
    
    Returns:
        Date string in YYYY-MM-DD format
    """
    today = datetime.now()
    
    # Start with current or next month
    if today.day > 15:  # If past mid-month, go to next month
        target_month = today.month + 1 if today.month < 12 else 1
        target_year = today.year if today.month < 12 else today.year + 1
    else:
        target_month = today.month
        target_year = today.year
    
    # Find 3rd Friday of target month
    first_day = datetime(target_year, target_month, 1)
    
    # Find first Friday (weekday 4 = Friday)
    days_until_friday = (4 - first_day.weekday()) % 7
    if days_until_friday == 0:  # If 1st is Friday
        days_until_friday = 0
    first_friday = first_day + timedelta(days=days_until_friday)
    
    # 3rd Friday is 2 weeks after first Friday
    third_friday = first_friday + timedelta(weeks=2)
    
    # If 3rd Friday is in the past, get next month's
    if third_friday < today:
        target_month = target_month + 1 if target_month < 12 else 1
        target_year = target_year if target_month < 12 else target_year + 1
        first_day = datetime(target_year, target_month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 0
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(weeks=2)
    
    return third_friday.strftime("%Y-%m-%d")


def _round_to_next_friday(date: datetime) -> datetime:
    """Round a date to the next Friday (options expiration day)."""
    days_until_friday = (4 - date.weekday()) % 7
    if days_until_friday == 0:  # Already Friday
        return date
    return date + timedelta(days=days_until_friday)


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
    """Pick tool based on intent with smart prioritization."""
    intent_map = {
        "price_quote": ["get_current_stock_price", "get_current_crypto_price", "get_stock_info"],
        "news": ["get_yahoo_finance_news", "get_company_news"],
        "analysis": ["get_stock_info", "get_financial_statement"],
        "fundamentals": ["get_financial_statement", "get_balance_sheets", "get_income_statements"],
        "options": ["get_option_chain", "get_option_expiration_dates"],
        "recommendation": ["get_recommendations"],
        "dividend": ["get_stock_actions"],
        "insider": ["get_holder_info"],
        "historical": ["get_historical_stock_prices", "get_historical_crypto_prices"],
    }
    
    candidates = intent_map.get(intent, [])
    
    # Special case: for options, prefer get_option_chain
    if intent == "options" and "get_option_chain" in tools_map:
        return "get_option_chain"
    
    # Try each candidate in priority order
    for tool_name in candidates:
        if tool_name in tools_map:
            return tool_name
    
    # Fallback: return first available tool
    return next(iter(tools_map)) if tools_map else None


# ============================================================================
# ARGS BUILDING
# ============================================================================

def _build_args_from_parsed_query(
    manager: MCPManager,
    server: str,
    tool: dict,
    parsed_query: "ParsedQuery"
) -> Dict[str, Any]:
    """Build arguments from structured parsed query with crypto support."""
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
        print(f"[mcp_router] Crypto resolved: {ticker} ({crypto_result['name']})")
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
        elif parsed_query.time_range and "month" in parsed_query.time_range.lower():
            args["period"] = "1mo"
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
    
    # Recommendations
    if tool.get("name") == "get_recommendations":
        if "recommendation_type" in props:
            args["recommendation_type"] = "analyst"
    
    # Options - COMPLETE FIX
    if tool.get("name") == "get_option_chain":
        # Set option type
        if "option_type" in props:
            raw_intent = parsed_query.raw_intent.lower()
            if "call" in raw_intent:
                args["option_type"] = "calls"
            elif "put" in raw_intent:
                args["option_type"] = "puts"
            else:
                args["option_type"] = parsed_query.options_type or "chain"
        
        # CRITICAL: Always provide expiration_date
        if "expiration_date" in props:
            # Priority 1: Use specific date from query
            if parsed_query.specific_date:
                args["expiration_date"] = parsed_query.specific_date
                print(f"[mcp_router] Using specific date: {args['expiration_date']}")
            
            # Priority 2: Calculate from expiration_days
            elif parsed_query.expiration_days:
                expiry = datetime.now() + timedelta(days=parsed_query.expiration_days)
                expiry = _round_to_next_friday(expiry)
                args["expiration_date"] = expiry.strftime("%Y-%m-%d")
                print(f"[mcp_router] Calculated expiry from days: {args['expiration_date']}")
            
            # Priority 3: Default to next monthly expiration
            else:
                args["expiration_date"] = _get_next_monthly_expiration()
                print(f"[mcp_router] Using next monthly expiration: {args['expiration_date']}")
    
    # Option expiration dates
    if tool.get("name") == "get_option_expiration_dates":
        # No additional args needed beyond ticker
        pass
    
    return args


# ============================================================================
# TOOL CALLING
# ============================================================================

def _call_tool_blocking(manager: MCPManager, server: str, tool_name: str, args: dict) -> str:
    """Call tool synchronously with error handling."""
    try:
        print(f"[mcp_router] Calling {server}/{tool_name} with args: {args}")
        
        if hasattr(manager, "call_sync") and callable(getattr(manager, "call_sync")):
            result = manager.call_sync(server, tool_name, args)
            return result
        else:
            print(f"[mcp_router] No call_sync method available")
            return f"Tool call failed: {tool_name} - manager lacks call_sync"
    
    except Exception as e:
        error_msg = str(e)
        print(f"[mcp_router] Error calling {tool_name}: {error_msg}")
        
        # Parse Pydantic validation errors
        if "validation error" in error_msg.lower():
            return f"Tool call validation error: {error_msg}"
        
        return f"Tool call error: {error_msg}"


# ============================================================================
# SERVER SELECTION (with Crypto Detection)
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
        print(f"[mcp_router] Crypto detected: {crypto_result['symbol']} ({crypto_result['name']}) from {crypto_result['source']}")
        
        if "financial-datasets" in available_servers:
            return "financial-datasets"
        
        # Fallback if financial-datasets not available
        if "yfinance" in available_servers:
            print(f"[mcp_router] financial-datasets unavailable, trying yfinance")
            return "yfinance"
    
    # Non-crypto: prefer yfinance for better coverage
    if "yfinance" in available_servers:
        return "yfinance"
    
    if "financial-datasets" in available_servers:
        return "financial-datasets"
    
    # Fallback to any available server
    return next(iter(available_servers)) if available_servers else None


# ============================================================================
# MAIN ROUTING LOGIC
# ============================================================================

def route_and_call(query: str) -> Dict[str, Any]:
    """
    Route query to best MCP server/tool and call it.
    
    Returns:
        {
            "route": {"server": ..., "tool": ..., "primary_ticker": ..., "intent": ...},
            "parsed": parsed_json_or_none,
            "raw": raw_response_string,
            "error": error_message_or_none
        }
    """
    try:
        print(f"\n{'='*60}")
        print(f"[route_and_call] Query: {query}")
        print(f"{'='*60}")
        
        # Step 1: Parse query with LLM
        parsed_query = parse_query_with_llm(query)
        print(f"[route_and_call] Parsed:")
        print(f"  - Ticker: {parsed_query.primary_ticker}")
        print(f"  - Intent: {parsed_query.intent}")
        print(f"  - Is Crypto: {parsed_query.is_crypto}")
        
        # Step 2: Check available servers
        available_servers = MCPServer.from_env()
        if not available_servers:
            return {
                "error": "No MCP servers configured. Please set MCP_SERVERS in .env",
                "route": {},
                "parsed": None,
                "raw": ""
            }
        
        print(f"[route_and_call] Available servers: {list(available_servers.keys())}")
        
        # Step 3: Pick best server
        manager = MCPManager()
        server = _pick_server(parsed_query, available_servers)
        if not server:
            return {
                "error": f"No suitable server found. Available: {list(available_servers.keys())}",
                "route": {},
                "parsed": None,
                "raw": ""
            }
        
        print(f"[route_and_call] Selected server: {server}")
        
        # Step 4: List and pick tool
        tools_map = _list_server_tools_sync(manager, server)
        if not tools_map:
            return {
                "error": f"No tools available in server '{server}'",
                "route": {"server": server},
                "parsed": None,
                "raw": ""
            }
        
        print(f"[route_and_call] Available tools: {list(tools_map.keys())}")
        
        tool_name = _pick_tool_by_intent(parsed_query.intent, tools_map, server)
        if not tool_name:
            return {
                "error": f"No suitable tool found for intent: {parsed_query.intent}",
                "route": {"server": server},
                "parsed": None,
                "raw": ""
            }
        
        print(f"[route_and_call] Selected tool: {tool_name}")
        
        # Step 5: Build arguments
        tool_obj = tools_map.get(tool_name, {"name": tool_name})
        args = _build_args_from_parsed_query(manager, server, tool_obj, parsed_query)
        
        # Step 6: Validate required arguments
        schema = tool_obj.get("inputSchema", {})
        required = schema.get("required", []) if isinstance(schema, dict) else []
        requires_ticker = any(r in ("ticker", "symbol") for r in required)
        
        if requires_ticker and not (args.get("ticker") or args.get("symbol")):
            return {
                "error": "Could not resolve ticker symbol. Please specify a valid ticker (e.g., AAPL, TSLA, BTC, SOL)",
                "route": {"server": server, "tool": tool_name, "intent": parsed_query.intent},
                "parsed": None,
                "raw": ""
            }
        
        print(f"[route_and_call] Final args: {args}")
        
        # Step 7: Call tool
        result = _call_tool_blocking(manager, server, tool_name, args)
        
        # Step 8: Parse result
        parsed_result, cleaned_raw = _parse_mcp_response(result)
        
        print(f"[route_and_call] Result received: {len(cleaned_raw)} chars")
        print(f"[route_and_call] Parsed: {parsed_result is not None}")
        print(f"{'='*60}\n")
        
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
        print(f"[route_and_call] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": f"Routing failed: {str(e)}",
            "route": {},
            "parsed": None,
            "raw": ""
        }


# ============================================================================
# AGNO TOOL (Agent Entry Point)
# ============================================================================

@tool(
    name="mcp_auto",
    description="Intelligent MCP router with crypto detection, options support, and multi-server orchestration"
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    """
    Entry point for agent to use MCP servers.
    Automatically routes queries to the best server and tool.
    """
    result = route_and_call(query)
    
    # Handle errors
    if result.get("error"):
        error_msg = result['error']
        route_info = result.get('route', {})
        
        if route_info:
            return f"Error ({route_info.get('server', 'unknown')}/{route_info.get('tool', 'unknown')}): {error_msg}"
        
        return f"Error: {error_msg}"
    
    # Build response
    raw = result.get("raw", "")
    route_info = result.get("route", {})
    
    # Add routing metadata for debugging
    info_str = f"[{route_info.get('server')}/{route_info.get('tool')} | {route_info.get('primary_ticker')} | {route_info.get('intent')}]"
    
    return f"{info_str}\n\n{raw}"