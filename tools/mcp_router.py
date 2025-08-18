# tools/mcp_router.py 
from __future__ import annotations
import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager, MCPServer



# Enhanced ticker detection
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3}|-[A-Z]{1,3})?\b")
_CRYPTO_SYMBOLS = {"BTC", "ETH", "SOL", "ADA", "DOT", "MATIC", "AVAX", "LINK", "UNI", "AAVE"}
_STOP = {"USD", "PE", "EV", "EPS", "ETF", "AND", "OR", "THE", "A", "AN", "IS", "ARE", "FOR", "TO", "OF", "IN"}

def _extract_tickers(text: str) -> List[str]:
    """Extract potential ticker symbols from text."""
    matches = _TICKER_RE.findall(text.upper())
    return [m for m in matches if m not in _STOP]

def _detect_crypto_intent(query: str) -> bool:
    """Detect if query is about cryptocurrency."""
    q_upper = query.upper()
    crypto_keywords = ["CRYPTO", "BITCOIN", "BTC", "ETH", "ETHEREUM", "COIN", "TOKEN", "BLOCKCHAIN", "DEFI"]
    
    # Check for crypto keywords
    if any(keyword in q_upper for keyword in crypto_keywords):
        return True
    
    # Check for known crypto symbols
    tickers = _extract_tickers(query)
    if any(ticker in _CRYPTO_SYMBOLS for ticker in tickers):
        return True
        
    return False

def _detect_data_type(query: str) -> str:
    """Detect what type of data is being requested."""
    q_lower = query.lower()
    
    # Real-time/current data
    if any(keyword in q_lower for keyword in ["current", "now", "latest", "today", "real-time", "live", "quote"]):
        return "realtime"
    
    # Historical data
    if any(keyword in q_lower for keyword in ["historical", "history", "past", "chart", "trend", "over time", "since"]):
        return "historical"
    
    # News/fundamental
    if any(keyword in q_lower for keyword in ["news", "earnings", "report", "announcement", "fundamentals"]):
        return "fundamental"
    
    # Default to realtime for price queries
    if any(keyword in q_lower for keyword in ["price", "value", "cost", "trading"]):
        return "realtime"
        
    return "general"

def _select_best_server(query: str, available_servers: Dict[str, MCPServer]) -> Optional[str]:
    """Intelligently select the best MCP server for the query."""
    is_crypto = _detect_crypto_intent(query)
    data_type = _detect_data_type(query)
    
    # Priority logic
    if is_crypto:
        # For crypto, prefer financial-datasets if available
        if "financial-datasets" in available_servers:
            return "financial-datasets"
        if "coinmarketcap" in available_servers:
            return "coinmarketcap"
    
    # For stocks, prefer yfinance for comprehensive data
    if "yfinance" in available_servers and not is_crypto:
        return "yfinance"
    
    # Fallback to any available server
    if "financial-datasets" in available_servers:
        return "financial-datasets"
    
    # Return first available server
    return next(iter(available_servers.keys())) if available_servers else None

def _choose_optimal_tool(server: str, query: str, data_type: str) -> str:
    """Choose the optimal tool based on server and query analysis."""
    q_lower = query.lower()
    is_crypto = _detect_crypto_intent(query)
    
    if server == "financial-datasets":
        if data_type == "realtime":
            return "get_current_crypto_price" if is_crypto else "get_current_stock_price"
        elif data_type == "historical":
            return "get_historical_crypto_prices" if is_crypto else "get_historical_stock_prices"
        else:
            return "get_current_crypto_price" if is_crypto else "get_current_stock_price"
    
    elif server == "yfinance":
        if "info" in q_lower or "company" in q_lower or data_type == "fundamental":
            return "get_stock_info"
        elif data_type == "historical":
            return "get_historical_data"
        else:
            return "get_current_price"
    
    elif server == "coinmarketcap":
        return "quote"
    
    # Default fallback
    return "get_stock_info" if server == "yfinance" else "get_current_stock_price"

def _build_smart_args(server: str, tool: str, query: str, tickers: List[str]) -> Dict:
    """Build smart arguments based on server, tool, and detected context."""
    is_crypto = _detect_crypto_intent(query)
    data_type = _detect_data_type(query)
    
    # Default ticker selection
    if tickers:
        primary_ticker = tickers[0]
    else:
        primary_ticker = "BTC" if is_crypto else "AAPL"
    
    if server == "financial-datasets":
        if is_crypto:
            # Ensure crypto tickers have proper format (e.g., BTC-USD)
            if "-" not in primary_ticker and primary_ticker in _CRYPTO_SYMBOLS:
                primary_ticker = f"{primary_ticker}-USD"
        
        args = {"ticker": primary_ticker}
        
        # Add historical parameters if needed
        if "historical" in tool:
            current_year = datetime.now().year
            args.update({
                "start_date": f"{current_year-1}-01-01",
                "end_date": f"{current_year}-12-31",
                "interval": "day",
                "interval_multiplier": 1
            })
        
        return args
    
    elif server == "yfinance":
        args = {"symbol" if "symbol" in query.lower() else "ticker": primary_ticker}
        
        # Add period for historical data
        if data_type == "historical":
            args["period"] = "1y"  # Default to 1 year
            
        return args
    
    elif server == "coinmarketcap":
        return {"symbol": primary_ticker, "convert": "USD"}
    
    return {"ticker": primary_ticker}

def route_and_call(query: str) -> str:
    """
    Intelligent routing function that analyzes query and calls optimal MCP server/tool.
    This is the core routing logic used by mcp_auto.
    """
    try:
        available_servers = MCPServer.from_env()
        if not available_servers:
            return "No MCP servers configured. Please check MCP_SERVERS in .env"
        
        # Extract context
        tickers = _extract_tickers(query)
        data_type = _detect_data_type(query)
        is_crypto = _detect_crypto_intent(query)
        
        # Select optimal server
        server = _select_best_server(query, available_servers)
        if not server:
            return "No suitable MCP server found"
        
        # Choose optimal tool
        tool_name = _choose_optimal_tool(server, query, data_type)
        
        # Build smart arguments
        args = _build_smart_args(server, tool_name, query, tickers)
        
        # Add routing metadata to response
        routing_info = f"Route: {server}/{tool_name} | Type: {data_type} | Crypto: {is_crypto} | Tickers: {tickers}"
        
        # Execute the call using MCPManager directly
        manager = MCPManager()
        result = manager.call_sync(server, tool_name, args)
        
        
        # Return with routing transparency
        return f"{routing_info}\n\n{result}"
        
    except Exception as e:
        return f"MCP routing failed: {str(e)}"

@tool(
    name="mcp_auto",
    description="Intelligent auto-router for MCP servers. Analyzes query and automatically selects the best server/tool combination. Handles stocks, crypto, real-time data, historical data, and news. Input: natural language query."
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    """
    Enhanced auto-routing tool with intelligent server/tool selection.
    Automatically detects query intent and routes to optimal MCP server.
    """
    return route_and_call(query)