# tools/mcp_router.py
from __future__ import annotations
import json
import re
from typing import Dict, List, Tuple

from agno.tools import tool
from rag.fusion import fuse
from mcp.manager import MCPServer, list_tool_names_sync
from tools.mcp_bridge import mcp_run

_TICKER_RE = re.compile(r"[A-Z]{1,5}")
_STOPWORDS = {"USD","EPS","EV","PE","P/E","AND","OR","THE","A","AN"}

CRYPTO_HINTS = {
    "CRYPTO","BITCOIN","BTC","ETH","ETHEREUM","ALTCOIN","TOKEN","ONCHAIN","COIN","COINMARKETCAP",
}

# Simple keyword → preferred tool substrings (scores) per server
YF_TOOL_HINTS = [
    ({"NEWS"}, ["get_yahoo_finance_news"]),
    ({"OPTION","OPTIONS","CALL","PUT"}, ["get_option_chain","get_option_expiration_dates"]),
    ({"HOLDER","HOLDERS","INSTITUTIONAL","INSIDER"}, ["get_holder_info"]),
    ({"INCOME","BALANCE","CASHFLOW","CASH","STATEMENT","FINANCIALS"}, ["get_financial_statement"]),
    ({"HIST","HISTORICAL","CHART","PRICE","PRICES"}, ["get_historical_stock_prices"]),
]
# Default for Yahoo
YF_DEFAULT = "get_stock_info"

CMC_TOOL_KEYWORDS = [
    ({"PRICE","PRICES","QUOTE","QUOTES","MARKETCAP","MARKET CAP","VOLUME"}, ["quote","quotes","listings","ohlcv"]),
    ({"LIST","LISTINGS","TOP","TREND","GAINERS","LOSERS"}, ["listings","trending","gainers","losers"]),
    ({"META","METADATA","INFO","DETAILS"}, ["metadata","info"]),
]

CMC_DEFAULT_FALLBACKS = ["quotes","listings","quote","ohlcv"]


def _find_tickers(text: str) -> List[str]:
    cand = [m.group(0) for m in _TICKER_RE.finditer(text.upper())]
    return [c for c in cand if c not in _STOPWORDS]


def _choose_server(query: str) -> str:
    servers = {name for name in MCPServer.from_env().keys()}
    upper = query.upper()
    if any(w in upper for w in CRYPTO_HINTS) and "coinmarketcap" in servers:
        return "coinmarketcap"
    # If query contains crypto tickers like BTC/ETH treat as crypto
    if any(tok in {"BTC","ETH","SOL","BNB","ADA"} for tok in _find_tickers(upper)) and "coinmarketcap" in servers:
        return "coinmarketcap"
    # else prefer yahoo if available
    if "yfinance" in servers:
        return "yfinance"
    # otherwise first available
    return next(iter(servers)) if servers else ""


def _score_tool(tools: List[str], wanted_substrings: List[str]) -> str:
    wanted = [w.lower() for w in wanted_substrings]
    for w in wanted:
        for t in tools:
            if w in t.lower():
                return t
    return tools[0] if tools else ""


def _choose_yahoo_tool(query: str) -> str:
    tools = list_tool_names_sync("yfinance") or []
    uq = query.upper()
    for kws, subs in YF_TOOL_HINTS:
        if any(k in uq for k in kws):
            return _score_tool(tools, subs)
    return _score_tool(tools, [YF_DEFAULT])


def _choose_cmc_tool(query: str) -> str:
    tools = list_tool_names_sync("coinmarketcap") or []
    uq = query.upper()
    for kws, subs in CMC_TOOL_KEYWORDS:
        if any(k in uq for k in kws):
            return _score_tool(tools, subs)
    return _score_tool(tools, CMC_DEFAULT_FALLBACKS)


def _build_args(server: str, tool: str, query: str, tickers: List[str]) -> Dict:
    if server == "yfinance":
        if "get_historical_stock_prices" in tool:
            t = tickers[0] if tickers else ("AAPL")
            return {"ticker": t, "period": "1mo", "interval": "1d"}
        if "get_financial_statement" in tool:
            t = tickers[0] if tickers else ("AAPL")
            q = query.lower()
            ftype = "income_stmt"
            if "balance" in q:
                ftype = "balance_sheet"
            elif "cash" in q or "cashflow" in q or "cash flow" in q:
                ftype = "cashflow"
            if "quarter" in q:
                ftype = "quarterly_" + ftype
            return {"ticker": t, "financial_type": ftype}
        if "get_holder_info" in tool:
            t = tickers[0] if tickers else ("AAPL")
            q = query.lower()
            h = "institutional_holders" if "institution" in q else "major_holders"
            return {"ticker": t, "holder_type": h}
        if "get_yahoo_finance_news" in tool:
            t = tickers[0] if tickers else ("AAPL")
            return {"ticker": t}
        if "get_option_chain" in tool:
            # default: pick first expiration date via another call would be ideal; keep minimal here
            t = tickers[0] if tickers else ("AAPL")
            return {"ticker": t, "expiration_date": "2099-01-01", "option_type": "calls"}
        # default stock info
        t = tickers[0] if tickers else ("AAPL")
        return {"ticker": t}
    # coinmarketcap — most tools want a symbol or slug; pass first token if looks like crypto
    if server == "coinmarketcap":
        up = [x for x in tickers if x not in {"USD"}]
        sym = up[0] if up else "BTC"
        # generic arg patterns (tool names vary; JSON will be ignored if fields aren't used)
        return {"symbol": sym, "symbols": sym, "convert": "USD", "limit": 100}
    return {}


def _route(query: str, tickers_hint: List[str] | None = None) -> Tuple[str,str,Dict]:
    server = _choose_server(query)
    if not server:
        raise ValueError("No MCP servers configured in environment (MCP_SERVERS)")
    if server == "yfinance":
        tool = _choose_yahoo_tool(query)
    else:
        tool = _choose_cmc_tool(query)
    tickers = tickers_hint or _find_tickers(query)
    args = _build_args(server, tool, query, tickers)
    return server, tool, args

@tool(
    name="mcp_auto",
    description=(
        "Automatically choose an MCP server/tool based on the user's query and run it. "
        "Inputs: query (str), optional tickers (comma/space separated). Returns raw text/json and ingests into RAG."
    ),
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str, tickers: str = "") -> str:
    hint = [t.strip().upper() for t in re.split(r"[ ,]", tickers) if t.strip()] if tickers else None
    server, tool, args = _route(query, hint)
    return mcp_run(server=server, tool=tool, args_json=json.dumps(args))