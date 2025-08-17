# tools/mcp_router.py
from __future__ import annotations
import json
import re
from typing import Dict, List, Tuple

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager, MCPServer
from tools.mcp_bridge import mcp_run

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
_STOP = {"USD","PE","EV","EPS","ETF","AND","OR","THE","A","AN"}

def _tickers(s: str) -> List[str]:
    return [m.group(0) for m in _TICKER_RE.finditer(s.upper()) if m.group(0) not in _STOP]

def _which_server(q: str) -> str:
    servers = set(MCPServer.from_env().keys())
    U = q.upper()
    # prefer Financial Datasets for **crypto**
    if any(x in U for x in ["CRYPTO","BTC","ETH","SOL","TOKEN","ONCHAIN","COIN"]) and "financial-datasets" in servers:
        return "financial-datasets"
    if any(x in U for x in ["CRYPTO","BTC","ETH","SOL","TOKEN","ONCHAIN","COIN"]) and "coinmarketcap" in servers:
        return "coinmarketcap"
    if "yfinance" in servers:
        return "yfinance"
    return next(iter(servers), "")

def _choose_tool(server: str, query: str) -> str:
    # keep this lightweight; we inspect available tools and pick a sensible one
    # fallbacks if listing fails are hard-coded defaults
    try:
        # we don’t need names here (avoids extra spawns)
        pass
    except Exception:
        pass
    U = query.upper()
    if server == "financial-datasets":
        if any(x in U for x in ["PRICE","QUOTE","CURRENT","NOW","SNAPSHOT"]):
            return "get_current_crypto_price" if any(t in {"BTC","ETH","SOL"} for t in _tickers(U)) else "get_current_stock_price"
        if any(x in U for x in ["HIST","HISTORICAL","OHLC","CHART"]):
            return "get_historical_crypto_prices" if any(t in {"BTC","ETH","SOL"} for t in _tickers(U)) else "get_historical_stock_prices"
        return "get_current_crypto_price"
    if server == "coinmarketcap":
        return "quote"
    # yfinance (stocks)
    return "get_stock_info"

def _build_args(server: str, tool: str, q: str, ticks: List[str]) -> Dict:
    if server == "financial-datasets":
        # API expects tickers like 'BTC-USD' for crypto; if user typed BTC, default to BTC-USD
        if "crypto" in tool or "get_current_crypto_price" in tool or "get_historical_crypto_prices" in tool:
            sym = (ticks[0] if ticks else "BTC").upper()
            sym = sym if "-" in sym else f"{sym}-USD"
            args = {"ticker": sym}
            if "historical" in tool:
                args.update({"start_date":"2024-01-01","end_date":"2025-12-31","interval":"day","interval_multiplier":1})
            return args
        # stocks
        t = ticks[0] if ticks else "AAPL"
        if "historical" in tool:
            return {"ticker": t, "start_date":"2024-01-01","end_date":"2025-12-31","interval":"day","interval_multiplier":1}
        return {"ticker": t}

    if server == "coinmarketcap":
        sym = (ticks[0] if ticks else "BTC").upper()
        return {"symbol": sym, "convert": "USD"}

    # yfinance
    return {"ticker": (ticks[0] if ticks else "AAPL").upper()}

# exported helper you can call from app.py (NOT decorated)
def route_and_call(query: str) -> str:
    ticks = _tickers(query)
    server = _which_server(query)
    if not server:
        return "No MCP servers configured."
    tool = _choose_tool(server, query)
    args = _build_args(server, tool, query, ticks)
    return mcp_run(server=server, tool=tool, args_json=json.dumps(args))

@tool(
    name="mcp_auto",
    description="Auto-route a query to the best MCP server/tool (crypto→financial-datasets if available). Input: query."
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    return route_and_call(query)
