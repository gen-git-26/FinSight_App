# tools/mcp_router.py
from __future__ import annotations

import os
import re
import json
import requests
from functools import lru_cache
from datetime import datetime, date
from typing import Dict, List, Optional, Any

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager, MCPServer

# -----------------------------
# Constants & heuristics
# -----------------------------
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3}|-[A-Z]{1,6})?\b")
_STOP = {
    "USD","PE","EV","EPS","ETF","AND","OR","THE","A","AN","IS","ARE","FOR","TO","OF","IN",
    "PRICE","NEWS","OPTIONS","CHAIN","PUTS","CALLS",
    "WHAT","ABOUT","SHOW","ME","RECENT","UPGRADES","DOWNGRADES","PLEASE","GET","GIVE","WITH",
}
_CRYPTO_SYMBOLS = {
    "BTC","ETH","SOL","ADA","DOT","MATIC","AVAX","LINK","UNI","AAVE",
    "XRP","DOGE","LTC","BCH","ATOM","ETC","FIL","XLM","NEAR","APT","ARB",
}

VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

# Known toolsets for each MCP server (used to intersect with dynamic discovery)
KNOWN_TOOLSETS: Dict[str, List[str]] = {
    "yfinance": [
        "get_stock_info",
        "get_historical_stock_prices",
        "get_yahoo_finance_news",
        "get_stock_actions",
        "get_financial_statement",
        "get_holder_info",
        "get_option_expiration_dates",
        "get_option_chain",
        "get_recommendations",
    ],
    "financial-datasets": [
        "get_current_stock_price",
        "get_historical_stock_prices",
        "get_current_crypto_price",
        "get_historical_crypto_prices",
    ],
    "coinmarketcap": ["quote"],
}

# -----------------------------
# Tokenization & detection
# -----------------------------
def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\.\-]+", (text or ""))

def _extract_tickers_regex_only(text: str) -> List[str]:
    matches = _TICKER_RE.findall((text or "").upper())
    return [m for m in matches if m not in _STOP]

def _detect_crypto_intent(query: str) -> bool:
    q_upper = (query or "").upper()
    crypto_keywords = [
        "CRYPTO","BITCOIN","BTC","ETH","ETHEREUM","COIN","TOKEN","BLOCKCHAIN","DEFI",
        "XRP","DOGE","LTC","BCH","ALTCOIN","ALTCOINS"
    ]
    if any(k in q_upper for k in crypto_keywords):
        return True
    tickers = _extract_tickers_regex_only(query)
    return any(t.replace("-USD","") in _CRYPTO_SYMBOLS for t in tickers)

def _detect_data_type(query: str) -> str:
    q_lower = (query or "").lower()
    if any(k in q_lower for k in ["current", "now", "latest", "today", "real-time", "live", "quote", "price now"]):
        return "realtime"
    if any(k in q_lower for k in ["historical", "history", "past", "chart", "trend", "over time", "since"]):
        return "historical"
    if any(k in q_lower for k in ["news", "earnings", "report", "announcement", "fundamentals", "upgrade", "downgrade"]):
        return "fundamental"
    if any(k in q_lower for k in ["price", "value", "cost", "trading"]):
        return "realtime"
    return "general"

# -----------------------------
# Dynamic symbol resolution
# -----------------------------
@lru_cache(maxsize=512)
def _alpha_vantage_symbol_search(query: str) -> Optional[str]:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return None
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "SYMBOL_SEARCH", "keywords": query, "apikey": api_key}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        best = (data.get("bestMatches") or [])
        def score(m):
            region = (m.get("4. region") or "").upper()
            t = (m.get("1. symbol") or "")
            s = float((m.get("9. matchScore") or "0") or 0)
            pref = 1.0 if region.startswith("UNITED STATES") else 0.0
            return (pref, s, -len(t))
        if best:
            best.sort(key=score, reverse=True)
            return best[0].get("1. symbol")
    except Exception:
        pass
    return None

@lru_cache(maxsize=512)
def _finnhub_symbol_lookup(query: str) -> Optional[str]:
    try:
        from utils.config import load_settings
        import finnhub
        settings = load_settings()
        client = finnhub.Client(api_key=settings.finnhub_api_key)
        res = client.symbol_lookup(query)
        items = (res or {}).get("result") or []
        if not items:
            return None
        def score(it):
            desc = (it.get("description") or "").upper()
            ty = (it.get("type") or "").upper()
            sym = (it.get("symbol") or "")
            pref = 1.0 if ty in {"EQS","COMMON STOCK"} else 0.0
            pref += 0.2 if "NYSE" in desc or "NASDAQ" in desc else 0.0
            return (pref, -len(sym))
        items.sort(key=score, reverse=True)
        return items[0].get("symbol")
    except Exception:
        return None

def _maybe_build_crypto_symbol(token: str) -> Optional[str]:
    t = (token or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,10}", t):
        return None
    if t in _STOP:
        return None
    if t in {"USD","USDT","USDC"}:
        return None
    return f"{t}-USD"

def _resolve_company_names_to_tickers_dynamic(text: str, is_crypto: bool) -> List[str]:
    words = _tokenize_words(text)
    candidates: List[str] = []

    if is_crypto:
        for w in words:
            s = _maybe_build_crypto_symbol(w)
            if s:
                candidates.append(s)
                break
        if candidates:
            return candidates

    q = " ".join([w for w in words if w.upper() not in _STOP]).strip()
    if q:
        sym = _finnhub_symbol_lookup(q)
        if sym:
            return [sym]
        sym2 = _alpha_vantage_symbol_search(q)
        if sym2:
            return [sym2]

    if is_crypto:
        for w in words:
            s = _maybe_build_crypto_symbol(w)
            if s:
                return [s]

    return []

def _extract_tickers(text: str) -> List[str]:
    tickers = _extract_tickers_regex_only(text)
    if not tickers:
        is_crypto = _detect_crypto_intent(text)
        tickers = _resolve_company_names_to_tickers_dynamic(text, is_crypto)
    seen, out = set(), []
    for t in tickers:
        if t and t not in _STOP and t not in seen:
            seen.add(t)
            out.append(t)
    return out

# -----------------------------
# Server & tool selection
# -----------------------------
def _select_best_server(query: str, available_servers: Dict[str, MCPServer]) -> Optional[str]:
    is_crypto = _detect_crypto_intent(query)
    if is_crypto:
        if "financial-datasets" in available_servers:
            return "financial-datasets"
        if "coinmarketcap" in available_servers:
            return "coinmarketcap"
    if "yfinance" in available_servers and not is_crypto:
        return "yfinance"
    if "financial-datasets" in available_servers:
        return "financial-datasets"
    return next(iter(available_servers.keys())) if available_servers else None

def _safe_json_loads(txt: str):
    try:
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

def _list_server_tools(manager: MCPManager, server: str) -> Dict[str, dict]:
    data = manager.list_tools_sync(server)
    tools: Dict[str, dict] = {}

    raw_tools = []
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        raw_tools = data["tools"]

    for t in raw_tools:
        name_val = t.get("name") if isinstance(t, dict) else getattr(t, "name", None) or t
        name = _normalize_tool_name(name_val) or ""
        if not name:
            continue
        if not VALID_TOOL_NAME.match(name):
            continue
        if name.lower() in {"meta", "health", "status"}:
            continue
        tool_obj = t if isinstance(t, dict) else {"name": name}
        tools[name] = tool_obj

    known = set(KNOWN_TOOLSETS.get(server, []))
    if tools and known:
        inter = {k: v for k, v in tools.items() if k in known}
        if inter:
            return inter

    if not tools and known:
        return {k: {"name": k, "inputSchema": {"properties": {}}} for k in known}

    return tools

def _choose_tool_from_available(server: str, query: str, data_type: str, tool_names: List[str]) -> Optional[str]:
    q = (query or "").lower()
    names = [n for n in tool_names if n.lower() not in {"meta", "health", "status"}]
    if not names:
        return None

    if server == "yfinance":
        if data_type == "historical" and "get_historical_stock_prices" in names:
            return "get_historical_stock_prices"
        if "news" in q and "get_yahoo_finance_news" in names:
            return "get_yahoo_finance_news"
        if any(w in q for w in ["dividend", "dividends", "split", "splits"]) and "get_stock_actions" in names:
            return "get_stock_actions"
        if any(w in q for w in ["financial statement","balance sheet","income statement","cashflow","cash flow"]) and "get_financial_statement" in names:
            return "get_financial_statement"
        if "option" in q and "get_option_chain" in names:
            return "get_option_chain"
        if "get_stock_info" in names:
            return "get_stock_info"

    if server == "financial-datasets":
        is_crypto = _detect_crypto_intent(query)
        if data_type == "historical":
            pref = "get_historical_crypto_prices" if is_crypto else "get_historical_stock_prices"
            if pref in names:
                return pref
        pref = "get_current_crypto_price" if is_crypto else "get_current_stock_price"
        if pref in names:
            return pref

    if "quote" in names:
        return "quote"

    return names[0]

# -----------------------------
# Options helpers
# -----------------------------
def _parse_yyyy_mm_dd(d: str) -> Optional[date]:
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None

def _pick_best_expiry(dates: List[str]) -> Optional[str]:
    today = date.today()
    parsed = [(_parse_yyyy_mm_dd(s), s) for s in dates if isinstance(s, str)]
    parsed = [(dt, raw) for dt, raw in parsed if dt is not None]
    if not parsed:
        return None
    futures = sorted([(dt, raw) for dt, raw in parsed if dt >= today], key=lambda x: x[0])
    if futures:
        return futures[0][1]
    past = sorted(parsed, key=lambda x: x[0])
    return past[-1][1] if past else None

# -----------------------------
# Args builder guided by schema
# -----------------------------
def _build_args_from_schema(manager: MCPManager, server: str, tool: dict, query: str, tickers: List[str]) -> Dict[str, Any]:
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else {}
    req = schema.get("required") if isinstance(schema, dict) else []

    is_crypto = _detect_crypto_intent(query)
    primary = tickers[0] if tickers else ("BTC" if is_crypto else "AAPL")
    if is_crypto and "-" not in primary and primary in _CRYPTO_SYMBOLS:
        primary = f"{primary}-USD"

    args: Dict[str, Any] = {}

    if isinstance(props, dict):
        if "ticker" in props:
            args["ticker"] = primary
        elif "symbol" in props:
            args["symbol"] = primary

        if "period" in props:
            args.setdefault("period", "1y")
        if "interval" in props:
            args.setdefault("interval", "1d")

        y = datetime.now().year
        if "start_date" in props:
            args.setdefault("start_date", f"{y-1}-01-01")
        if "end_date" in props:
            args.setdefault("end_date", f"{y}-12-31")
        if "interval" in props and "interval_multiplier" in props:
            args.setdefault("interval", "day")
            args.setdefault("interval_multiplier", 1)

        if tool.get("name") == "get_financial_statement" and "financial_type" in props:
            args.setdefault("financial_type", "income_stmt")

        if tool.get("name") == "get_option_chain":
            if "option_type" in props:
                qt = (query or "").lower()
                if "puts" in qt or "put" in qt:
                    args.setdefault("option_type", "puts")
                else:
                    args.setdefault("option_type", "calls")

            if "expiration_date" in props:
                ticker_arg = args.get("ticker") or args.get("symbol") or primary
                try:
                    raw = manager.call_sync(server, "get_option_expiration_dates", {"ticker": ticker_arg})
                    data = _safe_json_loads(raw)
                    if isinstance(data, list):
                        best = _pick_best_expiry(data)
                    else:
                        lines = raw.splitlines() if isinstance(raw, str) else []
                        candidates = [ln.strip() for ln in lines if _parse_yyyy_mm_dd(ln.strip())]
                        best = _pick_best_expiry(candidates)
                    if best:
                        args.setdefault("expiration_date", best)
                except Exception:
                    pass

    if isinstance(req, list):
        for r in req:
            if r not in args and r in ("ticker", "symbol"):
                args[r] = primary

    return args

# -----------------------------
# Public API
# -----------------------------
def route_and_call(query: str) -> str:
    try:
        available_servers = MCPServer.from_env()
        if not available_servers:
            return "No MCP servers configured. Please check MCP_SERVERS in .env"

        tickers = _extract_tickers(query)
        data_type = _detect_data_type(query)
        is_crypto = _detect_crypto_intent(query)

        server = _select_best_server(query, available_servers)
        if not server:
            return "No suitable MCP server found"

        manager = MCPManager()

        tools_map = _list_server_tools(manager, server)
        tool_names = list(tools_map.keys())
        if not tool_names:
            return f"No tools exposed by server '{server}'"

        tool_name = _choose_tool_from_available(server, query, data_type, tool_names)
        if not tool_name:
            return f"No matching tool found on server '{server}'"

        tool_obj = tools_map.get(tool_name, {"name": tool_name})

        args = _build_args_from_schema(manager, server, tool_obj, query, tickers)

        routing_info = f"Route: {server}/{tool_name} | Type: {data_type} | Crypto: {is_crypto} | Tickers: {tickers}"
        result = manager.call_sync(server, tool_name, args)
        return f"{routing_info}\n\n{result}"

    except Exception as e:
        return f"MCP routing failed: {str(e)}"

@tool(
    name="mcp_auto",
    description="Auto-router for MCP servers. Dynamically lists tools from the chosen server, picks the best match, and builds valid arguments from the tool schema."
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    return route_and_call(query)
