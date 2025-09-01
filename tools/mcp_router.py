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


# -----------------------------
# Constants and heuristics
# -----------------------------

# Strict but flexible ticker pattern
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3}|-[A-Z0-9]{1,6})?\b")

# Stop words never treated as symbols
_STOP = {
    "USD", "USDT", "USDC",
    "PE", "EV", "EPS", "ETF",
    "AND", "OR", "THE", "A", "AN", "IS", "ARE", "FOR", "TO", "OF", "IN",
    "PRICE", "NEWS", "OPTIONS", "CHAIN", "PUTS", "CALLS", "STOCK", "SHARE", "SHARES",
    "TODAY", "NOW", "LATEST", "CURRENT", "REAL", "TIME", "REALTIME", "LIVE",
    "UPGRADES", "DOWNGRADES", "RECOMMENDATIONS",
}

# Crypto keywords
_CRYPTO_KEYWORDS = {
    "CRYPTO", "BITCOIN", "BTC", "ETH", "ETHEREUM", "COIN", "TOKEN", "BLOCKCHAIN", "DEFI",
    "XRP", "DOGE", "LTC", "BCH", "SOL", "ADA", "MATIC", "AVAX", "LINK",
    "UNI", "AAVE", "ATOM", "ETC", "FIL", "XLM", "NEAR", "APT", "ARB",
}

VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

# Known toolsets per MCP server used for intersection with dynamic discovery
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
        "get_company_news",
        "get_income_statements",
        "get_balance_sheets",
        "get_cash_flow_statements",
        "get_sec_filings",
        "get_available_crypto_tickers",
        "get_crypto_prices",
        "get_historical_crypto_prices",
    ],
    "coinmarketcap": ["quote"],
}


# -----------------------------
# Tokenization and intent
# -----------------------------

def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\.\-]+", (text or ""))


def _extract_regex_tickers(text: str) -> List[str]:
    matches = _TICKER_RE.findall((text or "").upper())
    return [m for m in matches if m not in _STOP]


def _detect_crypto_intent(query: str) -> bool:
    q_upper = (query or "").upper()
    if any(k in q_upper for k in _CRYPTO_KEYWORDS):
        return True
    # Explicit SYMBOL-USD style
    if re.search(r"\b[A-Z0-9]{2,10}-USD\b", q_upper):
        return True
    return False


def _detect_data_type(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["current", "now", "latest", "today", "real-time", "live", "quote", "price now"]):
        return "realtime"
    if any(k in q for k in ["historical", "history", "past", "chart", "trend", "over time", "since"]):
        return "historical"
    if any(k in q for k in ["news", "earnings", "report", "announcement", "fundamentals", "upgrade", "downgrade"]):
        return "fundamental"
    if any(k in q for k in ["price", "value", "cost", "trading"]):
        return "realtime"
    return "general"


# -----------------------------
# Dynamic symbol resolution
# -----------------------------

@lru_cache(maxsize=512)
def _finnhub_symbol_lookup_single(query: str) -> Optional[str]:
    """
    Resolve a free-text company name into a canonical symbol using Finnhub.
    Prefers NASDAQ and NYSE common equities.
    """
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
            pref = 0.0
            if "NASDAQ" in desc or "NYSE" in desc or "NEW YORK" in desc:
                pref += 1.0
            if ty in {"EQS", "COMMON STOCK", "EQUITY", "STOCK"}:
                pref += 0.5
            # mildly prefer shorter and simpler symbols if tie
            return (pref, -len(sym))

        items.sort(key=score, reverse=True)
        return (items[0].get("symbol") or "").upper().strip() or None
    except Exception:
        return None


def _is_valid_equity_symbol(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,3})?", token))


def _maybe_build_crypto_symbol(token: str) -> Optional[str]:
    t = (token or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,10}", t):
        return None
    if t in _STOP:
        return None
    if t in {"USD", "USDT", "USDC"}:
        return None
    return f"{t}-USD"


def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def resolve_tickers(query: str) -> List[str]:
    """
    Unified symbol resolver.
      1) Extract regex symbols and keep only valid-looking ones that are not stop words.
      2) If crypto intent and no explicit SYMBOL-USD is present, try to build one from tokens.
      3) Else, attempt Finnhub symbol_lookup on the free-text to resolve equities (e.g., "Tesla" -> "TSLA").
      4) As a last crypto step, try to normalize a plausible coin token to SYMBOL-USD.
      5) Return deduped list, can be empty if nothing matches.
    """
    out: List[str] = []

    # 1) direct tickers
    rx = _extract_regex_tickers(query)
    rx = [t for t in rx if _is_valid_equity_symbol(t) or re.fullmatch(r"[A-Z0-9]{2,10}-USD", t)]
    out.extend(rx)

    # 2) crypto normalization
    is_crypto = _detect_crypto_intent(query)
    if is_crypto and not any(s.endswith("-USD") for s in out):
        for w in _tokenize_words(query):
            s = _maybe_build_crypto_symbol(w)
            if s:
                out.append(s)
                break  # prefer a single primary

    # 3) equity free text via Finnhub when nothing valid so far
    if not out and not is_crypto:
        sym = _finnhub_symbol_lookup_single(query)
        if sym:
            out.append(sym)

    # 4) last crypto attempt
    if not out and is_crypto:
        for w in _tokenize_words(query):
            s = _maybe_build_crypto_symbol(w)
            if s:
                out.append(s)
                break

    return _dedupe(out)


# -----------------------------
# Server and tool selection
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
    # fallback to first available
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
    if hasattr(manager, "list_tools_sync") and callable(getattr(manager, "list_tools_sync")):
        data = manager.list_tools_sync(server)
    else:
        res = manager.list_tools(server)
        data = asyncio.run(res) if asyncio.iscoroutine(res) else res

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
        if any(w in q for w in ["financial statement", "balance sheet", "income statement", "cashflow", "cash flow"]) and "get_financial_statement" in names:
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
        if "news" in q and "get_company_news" in names:
            return "get_company_news"
        if any(w in q for w in ["income", "cash flow", "balance sheet"]) and "get_income_statements" in names:
            # pick income statements as a representative fundamentals endpoint
            return "get_income_statements"
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
    """
    Choose the soonest future expiration else latest historical.
    """
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


def _call_tool_blocking(manager: MCPManager, server: str, tool_name: str, args: dict) -> str:
    if hasattr(manager, "call_sync") and callable(getattr(manager, "call_sync")):
        return manager.call_sync(server, tool_name, args)
    res = manager.call(server, tool_name, args)
    return asyncio.run(res) if asyncio.iscoroutine(res) else res


# -----------------------------
# Args builder guided by schema
# -----------------------------

def _build_args_from_schema(manager: MCPManager, server: str, tool: dict, query: str, tickers: List[str]) -> Dict[str, Any]:
    """
    Build arguments based on the tool's input schema.
    No hard-coded assets.
    If the schema requires a ticker and none was resolved, caller should fail fast.
    """
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else {}
    req = schema.get("required") if isinstance(schema, dict) else []

    args: Dict[str, Any] = {}
    primary = tickers[0] if tickers else None

    if isinstance(props, dict):
        # Primary symbol field
        if "ticker" in props and primary:
            args["ticker"] = primary
        elif "symbol" in props and primary:
            args["symbol"] = primary

        # Generic time window defaults if present
        if "period" in props:
            args.setdefault("period", "1y")
        if "interval" in props and "interval_multiplier" not in props:
            # yfinance style
            args.setdefault("interval", "1d")

        # financial-datasets style
        y = datetime.now().year
        if "start_date" in props:
            args.setdefault("start_date", f"{y-1}-01-01")
        if "end_date" in props:
            args.setdefault("end_date", f"{y}-12-31")
        if "interval" in props and "interval_multiplier" in props:
            args.setdefault("interval", "day")
            args.setdefault("interval_multiplier", 1)

        # yfinance financial statements
        if tool.get("name") == "get_financial_statement" and "financial_type" in props:
            args.setdefault("financial_type", "income_stmt")

        # Options chain helpers
        if tool.get("name") == "get_option_chain":
            if "option_type" in props:
                qt = (query or "").lower()
                args.setdefault("option_type", "puts" if ("puts" in qt or "put" in qt) else "calls")

            if "expiration_date" in props:
                ticker_arg = args.get("ticker") or args.get("symbol")
                if ticker_arg:
                    try:
                        raw = _call_tool_blocking(manager, server, "get_option_expiration_dates", {"ticker": ticker_arg})
                        data = _safe_json_loads(raw)
                        if isinstance(data, list):
                            best = _pick_best_expiry(data)
                        else:
                            # Parse lines to YYYY-MM-DD if needed
                            lines = raw.splitlines() if isinstance(raw, str) else []
                            candidates = [ln.strip() for ln in lines if _parse_yyyy_mm_dd(ln.strip())]
                            best = _pick_best_expiry(candidates)
                        if best:
                            args.setdefault("expiration_date", best)
                    except Exception:
                        pass

    return args


# -----------------------------
# Public API
# -----------------------------

def route_and_call(query: str) -> str:
    try:
        servers = MCPServer.from_env()
        if not servers:
            return "No MCP servers configured. Please check MCP_SERVERS in .env"

        # Resolve symbols as needed for tools that require ticker
        tickers = resolve_tickers(query)
        data_type = _detect_data_type(query)
        is_crypto = _detect_crypto_intent(query)

        server = _select_best_server(query, servers)
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

        # If tool schema requires a ticker and we do not have one, fail fast with a helpful message
        schema = tool_obj.get("inputSchema") or {}
        req = (schema.get("required") if isinstance(schema, dict) else None) or []
        needs_symbol = any(r in ("ticker", "symbol") for r in req) or ("ticker" in (schema.get("properties") or {}))

        if needs_symbol and not (args.get("ticker") or args.get("symbol")):
            # Try one more weak resolution using Finnhub if not attempted already
            if not tickers:
                fallback = _finnhub_symbol_lookup_single(query)
                if fallback:
                    args["ticker"] = fallback
                    tickers = [fallback]
                else:
                    return "Could not resolve a valid ticker from the query. Please specify a symbol such as TSLA or AAPL."

        routing_info = f"Route: {server}/{tool_name} | Type: {data_type} | Crypto: {is_crypto} | Tickers: {tickers or []}"
        result = _call_tool_blocking(manager, server, tool_name, args)
        return f"{routing_info}\n\n{result}"

    except Exception as e:
        return f"MCP routing failed: {str(e)}"


@tool(
    name="mcp_auto",
    description="Auto-router for MCP servers. Dynamically lists tools from the chosen server, picks a best match, and builds valid arguments from the tool schema."
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    return route_and_call(query)
