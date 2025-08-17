# tools/mcp_router.py
from __future__ import annotations
import json, re
from typing import Dict, List, Tuple

from agno.tools import tool
from mcp_connection.manager import MCPManager

mgr = MCPManager()

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
_STOP = {"USD","EPS","EV","PE","P/E","AND","OR","THE","A","AN"}

def _tickers(text: str) -> List[str]:
    return [t for t in _TICKER_RE.findall(text.upper()) if t not in _STOP]

def _choose_server(q: str, servers: List[str]) -> str:
    up = q.upper()
    if any(x in servers for x in ["coinmarketcap"]) and any(k in up for k in ["CRYPTO","BTC","ETH","COIN","TOKEN","COINMARKETCAP"]):
        return "coinmarketcap"
    return "yfinance" if "yfinance" in servers else (servers[0] if servers else "")

async def _choose_tool(server: str, q: str) -> str:
    tools = [t for t in await mgr.list_tools(server) if t]
    uq = q.upper()
    pref: List[str] = []
    if server == "yfinance":
        if any(k in uq for k in ["NEWS"]): pref = ["get_yahoo_finance_news"]
        elif any(k in uq for k in ["OPTION","CALL","PUT"]): pref = ["get_option_chain","get_option_expiration_dates"]
        elif any(k in uq for k in ["HOLDER","INSIDER","INSTITUTION"]): pref = ["get_holder_info"]
        elif any(k in uq for k in ["INCOME","BALANCE","CASH","FINANCIAL"]): pref = ["get_financial_statement"]
        elif any(k in uq for k in ["HIST","HISTORICAL","CHART","PRICE"]): pref = ["get_historical_stock_prices"]
        else: pref = ["get_stock_info"]
    else:
        if any(k in uq for k in ["PRICE","QUOTE","MARKET CAP","MARKETCAP"]): pref = ["quotes","quote","ohlcv"]
        elif any(k in uq for k in ["LIST","TOP","TREND","GAINER","LOSER"]):   pref = ["listings","trending"]
        else: pref = ["quotes","listings","quote"]
    for p in pref:
        for t in tools:
            if p.lower() in t.lower():
                return t
    return tools[0] if tools else ""

def _build_args(server: str, tool: str, q: str, tickers: List[str]) -> Dict:
    if server == "yfinance":
        t = tickers[0] if tickers else "AAPL"
        if "get_financial_statement" in tool:
            f = "income_stmt"
            ql = q.lower()
            if "balance" in ql: f = "balance_sheet"
            elif "cash" in ql:  f = "cashflow"
            if "quarter" in ql: f = "quarterly_" + f
            return {"ticker": t, "financial_type": f}
        if "get_historical_stock_prices" in tool:
            return {"ticker": t, "period": "1mo", "interval": "1d"}
        if "get_holder_info" in tool:
            k = "institutional_holders" if "institution" in q.lower() else "major_holders"
            return {"ticker": t, "holder_type": k}
        if "get_option_chain" in tool:
            return {"ticker": t, "expiration_date": "2099-01-01", "option_type": "calls"}
        return {"ticker": t}
    # coinmarketcap
    sym = (tickers[0] if tickers else "BTC")
    return {"symbol": sym, "symbols": sym, "convert": "USD", "limit": 50}

@tool(
    name="mcp_auto",
    description=(
        "Auto-select MCP server & tool based on the question and run it. "
        "Inputs: query (str), tickers (optional CSV). Returns raw text."
    ),
)
async def mcp_auto(query: str, tickers: str = "") -> str:
    servers = list(MCPManager.from_env().keys())
    if not servers:
        return "No MCP servers configured in MCP_SERVERS."
    hint = [t.strip().upper() for t in re.split(r"[ ,]", tickers) if t.strip()] if tickers else []
    server = _choose_server(query, servers)
    tool   = await _choose_tool(server, query)
    args   = _build_args(server, tool, query, (hint or _tickers(query)))
    return await mgr.call(server, tool, args)
