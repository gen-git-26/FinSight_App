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
from tools.query_parser import ParsedQuery

# -----------------------------
# Constants & heuristics 
# -----------------------------

# Strict ticker regex: e.g., MSFT, NVDA, GOOGL, BRK.B, XRP-USD
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3}|-[A-Z0-9]{1,6})?\b")

# Words we never treat as tickers
_STOP = {
    "USD", "USDT", "USDC",
    "PE", "EV", "EPS", "ETF",
    "AND", "OR", "THE", "A", "AN", "IS", "ARE", "FOR", "TO", "OF", "IN",
    "PRICE", "NEWS", "OPTIONS", "CHAIN", "PUTS", "CALLS",
    "WHAT", "ABOUT", "SHOW", "ME", "RECENT", "UPGRADES", "DOWNGRADES",
    "PLEASE", "GET", "GIVE", "WITH",
}

# Single crypto keyword set (no duplicate symbol lists)
_CRYPTO_KEYWORDS = {
    "CRYPTO", "BITCOIN", "BTC", "ETH", "ETHEREUM", "COIN", "TOKEN", "BLOCKCHAIN", "DEFI",
    "XRP", "DOGE", "LTC", "BCH", "SOL", "ADA", "MATIC", "AVAX", "LINK",
    "UNI", "AAVE", "ATOM", "ETC", "FIL", "XLM", "NEAR", "APT", "ARB"
}

VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

# Known toolsets per MCP server (used to intersect with dynamic discovery)
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
    ]
}

# -----------------------------
# Tokenization & intent
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
    # also consider explicit SYMBOL-USD form
    if re.search(r"\b[A-Z0-9]{2,10}-USD\b", q_upper):
        return True
    return False


def _detect_data_type(query: str) -> str:
    q_lower = (query or "").lower()
    if any(k in q_lower for k in ["current", "now", "latest", "today", "real-time", "live", "quote", "price now"]):
        return "realtime"
    if any(k in q_lower for k in ["historical", "history", "past", "chart", "trend", "over time", "since"]):
        return "historical"
    if any(k in q_lower for k in ["news", "earnings", "report", "announcement", "fundamentals", "upgrade", "downgrade", "dividend", "dividends", "split", "splits", "balance sheet", "income statement", "cashflow", "cash flow", "option", "options", "chain"]):
        return "fundamental"
    if any(k in q_lower for k in ["price", "value", "cost", "trading"]):
        return "realtime"
    return "general"

# -----------------------------
# Dynamic symbol resolution (single-source + crypto normalize)
# -----------------------------

@lru_cache(maxsize=512)
def _finnhub_symbol_lookup_single(query: str) -> Optional[str]:
    """
    Resolve a free-text company name into a symbol using Finnhub.
    Preference: US listings (NASDAQ/NYSE) and common equities.
    """
    try:
        from utils.config import load_settings
        import finnhub
        settings = load_settings()
        client = finnhub.Client(api_key=settings.finnhub_api_key)
        res = client.symbol_lookup(query)  # {'count':..., 'result':[...]}
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
            return (pref, -len(sym))

        items.sort(key=score, reverse=True)
        return items[0].get("symbol") or None
    except Exception:
        return None


def _maybe_build_crypto_symbol(token: str) -> Optional[str]:
    t = (token or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,10}", t):
        return None
    if t in _STOP:
        return None
    # ignore explicit stablecoins as a "base"
    if t in {"USD", "USDT", "USDC"}:
        return None
    return f"{t}-USD"


def resolve_tickers(query: str) -> List[str]:
    """
    Minimal dynamic resolver (no redundancy):
      1) If regex tickers exist in text, return them (deduped).
      2) If crypto intent: try to normalize a token to SYMBOL-USD.
      3) Else: use Finnhub symbol_lookup(query) to resolve (Microsoft -> MSFT).
      4) If crypto intent and still nothing: last attempt build SYMBOL-USD.
      5) Else: return [] (caller should fail-fast with a clear message).
    """
    # 1) direct tickers in text
    rx = _extract_regex_tickers(query)
    if rx:
        seen, out = set(), []
        for t in rx:
            if t not in _STOP and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # 2) crypto: normalize coin-like token
    is_crypto = _detect_crypto_intent(query)
    if is_crypto:
        for w in _tokenize_words(query):
            s = _maybe_build_crypto_symbol(w)
            if s:
                return [s]

    # 3) equity free text -> Finnhub
    sym = _finnhub_symbol_lookup_single(query)
    if sym:
        return [sym.upper().strip()]

    # 4) last crypto attempt
    if is_crypto:
        for w in _tokenize_words(query):
            s = _maybe_build_crypto_symbol(w)
            if s:
                return [s]

    # 5) nothing
    return []

# -----------------------------
# Server & tool selection (dynamic ranking)
# -----------------------------

CATEGORY_KEYWORDS = {
    "historical": {"historical", "history", "past", "chart", "trend", "since", "backtest"},
    "news": {"news", "headline", "article", "press", "upgrade", "downgrade"},
    "fundamental": {"fundamental", "financial", "balance", "income", "cash", "dividend", "split", "holders", "recommendation"},
    "options": {"option", "options", "chain", "expiration", "strike", "puts", "calls"},
    "realtime": {"current", "now", "live", "quote", "price", "today"},
}


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
        tool_obj["name"] = name
        tools[name] = tool_obj

    known = set(KNOWN_TOOLSETS.get(server, []))
    if tools and known:
        inter = {k: v for k, v in tools.items() if k in known}
        if inter:
            return inter

    if not tools and known:
        return {k: {"name": k, "inputSchema": {"properties": {}}} for k in known}

    return tools


def _extract_tool_text(tool: dict) -> str:
    parts = [tool.get("name", "")]
    desc = tool.get("description") or tool.get("desc") or ""
    if isinstance(desc, str):
        parts.append(desc)
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else {}
    if isinstance(props, dict):
        parts.extend(list(props.keys()))
    return " ".join(str(p) for p in parts if p)


def _score_match(query_tokens: List[str], text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    for qt in query_tokens:
        if not qt:
            continue
        if qt in t:
            score += 1.0
        if f" {qt} " in f" {t} ":
            score += 0.5
    return score


def _rank_tools(query: str, tools_map: Dict[str, dict], data_type: str, is_crypto: bool, have_symbol: bool) -> List[Tuple[str, float]]:
    tokens = [w.lower() for w in _tokenize_words(query)]
    ranks: List[Tuple[str, float]] = []

    for name, tool in tools_map.items():
        base = 0.0
        name_l = name.lower()
        text = _extract_tool_text(tool).lower()

        # name or description token matches
        base += _score_match(tokens, text)

        # boosts for category alignment
        if data_type == "historical" and ("historical" in name_l or "start_date" in text or "period" in text):
            base += 2.0
        if data_type == "realtime" and ("current" in name_l or "quote" in name_l or "price" in text):
            base += 2.0
        if any(k in tokens for k in CATEGORY_KEYWORDS["news"]) and ("news" in name_l or "headline" in text):
            base += 2.0
        if any(k in tokens for k in CATEGORY_KEYWORDS["options"]) and ("option" in name_l or "expiration_date" in text or "strike" in text):
            base += 2.0
        if any(k in tokens for k in CATEGORY_KEYWORDS["fundamental"]) and ("financial" in name_l or "income" in text or "balance" in text or "cash" in text or "dividend" in text or "split" in text or "holder" in text or "recommendation" in text):
            base += 2.0

        # crypto vs stock preference
        if is_crypto and ("crypto" in name_l or "symbol" in text and "-usd" in query.lower()):
            base += 1.0
        if not is_crypto and ("stock" in name_l):
            base += 0.5

        # satisfiability penalty: if tool requires ticker/symbol and we do not have it
        schema = tool.get("inputSchema") or {}
        req = (schema.get("required") if isinstance(schema, dict) else None) or []
        requires_symbol = any(r in ("ticker", "symbol") for r in req)
        if requires_symbol and not have_symbol:
            base -= 2.5

        # small boost if tool is in KNOWN_TOOLSETS for its server name embedded in tool
        base += 0.0

        ranks.append((name, base))

    ranks.sort(key=lambda x: x[1], reverse=True)
    return ranks


def _pick_best_server_and_tool(query: str, available_servers: Dict[str, MCPServer]) -> Tuple[Optional[str], Optional[str], Dict[str, dict]]:
    manager = MCPManager()
    data_type = _detect_data_type(query)
    is_crypto = _detect_crypto_intent(query)
    tickers = resolve_tickers(query)
    have_symbol = bool(tickers)

    best: Tuple[Optional[str], Optional[str], float, Dict[str, dict]] = (None, None, float("-inf"), {})

    for server in available_servers.keys():
        tools_map = _list_server_tools(manager, server)
        if not tools_map:
            continue
        ranked = _rank_tools(query, tools_map, data_type, is_crypto, have_symbol)
        if ranked:
            top_name, top_score = ranked[0]
            # prefer yfinance for stocks when not crypto, small bias
            if server == "yfinance" and not is_crypto:
                top_score += 0.25
            # prefer financial-datasets for crypto, small bias
            if server == "financial-datasets" and is_crypto:
                top_score += 0.25
            if top_score > best[2]:
                best = (server, top_name, top_score, tools_map)

    return best[0], best[1], best[3]


def _call_tool_blocking(manager: MCPManager, server: str, tool_name: str, args: dict) -> str:
    if hasattr(manager, "call_sync") and callable(getattr(manager, "call_sync")):
        return manager.call_sync(server, tool_name, args)
    res = manager.call(server, tool_name, args)
    return asyncio.run(res) if asyncio.iscoroutine(res) else res


# -----------------------------
# Args builder (guided by schema)
# -----------------------------

def _parse_yyyy_mm_dd(d: str) -> Optional[date]:
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None


def _pick_best_expiry(dates: List[str]) -> Optional[str]:
    """
    Choose the soonest future expiration, else latest historical.
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


def _build_args_from_schema(manager: MCPManager, server: str, tool: dict, query: str, tickers: List[str]) -> Dict[str, Any]:
    """
    Build arguments guided by the tool's input schema.
    - No hardcoded primary fallback (no AAPL/BTC).
    - If the schema requires a ticker/symbol and none was resolved, caller should fail-fast.
    """
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else {}

    args: Dict[str, Any] = {}

    # set ticker/symbol only if present in schema and we actually resolved one
    primary = tickers[0] if tickers else None
    if isinstance(props, dict):
        if "ticker" in props and primary:
            args["ticker"] = primary
        elif "symbol" in props and primary:
            args["symbol"] = primary

        # sensible defaults for time windows if present in schema
        if "period" in props:
            args.setdefault("period", "1y")
        if "interval" in props and "interval_multiplier" not in props:
            args.setdefault("interval", "1d")

        # financial-datasets style historical schema
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
            # light intent extraction
            ql = (query or "").lower()
            if "balance" in ql:
                args.setdefault("financial_type", "balance_sheet")
            elif "cash" in ql:
                args.setdefault("financial_type", "cashflow")
            else:
                args.setdefault("financial_type", "income_stmt")

        # options chain: choose best expiration if we have a ticker
        if tool.get("name") == "get_option_chain":
            ql = (query or "").lower()
            if "option_type" in props:
                args.setdefault("option_type", "puts" if ("puts" in ql or "put" in ql) else "calls")
            if "expiration_date" in props:
                ticker_arg = args.get("ticker") or args.get("symbol")
                if ticker_arg:
                    try:
                        raw = _call_tool_blocking(manager, server, "get_option_expiration_dates", {"ticker": ticker_arg})
                        data = _safe_json_loads(raw)
                        if isinstance(data, list):
                            best = _pick_best_expiry(data)
                        else:
                            # parse any YYYY-MM-DD lines
                            lines = raw.splitlines() if isinstance(raw, str) else []
                            candidates = [ln.strip() for ln in lines if _parse_yyyy_mm_dd(ln.strip())]
                            best = _pick_best_expiry(candidates)
                        if best:
                            args.setdefault("expiration_date", best)
                    except Exception:
                        pass

    # caller will check required fields and decide whether to fail-fast
    return args


# -----------------------------
# Public API
# -----------------------------

async def route_and_call_v2(query: str) -> Dict[str, str]:
    """
    NEW: Use LLM parser + dynamic server selection.
    Returns: {"route": {...}, "parsed": {...}, "raw": str, "error": str}
    """
    try:
        # Step 1: Parse query with LLM
        from tools.query_parser import parse_query_with_llm
        import asyncio
        
        parsed_query = await parse_query_with_llm(query)
        
        available_servers = MCPServer.from_env()
        if not available_servers:
            return {"error": "No MCP servers configured"}
        
        # Step 2: Select best server based on parsed intent
        manager = MCPManager()
        server = None
        
        # Logic: crypto intent → financial-datasets, else → yfinance
        if parsed_query.intent in ("options", "fundamentals", "recommendation"):
            server = "yfinance" if "yfinance" in available_servers else next(iter(available_servers))
        elif any(ticker and ticker.endswith("-USD") for ticker in [parsed_query.primary_ticker] + parsed_query.secondary_tickers):
            server = "financial-datasets" if "financial-datasets" in available_servers else next(iter(available_servers))
        else:
            server = "yfinance" if "yfinance" in available_servers else "financial-datasets" if "financial-datasets" in available_servers else next(iter(available_servers))
        
        # Step 3: List tools from chosen server
        tools_map = _list_server_tools(manager, server)
        
        # Step 4: Pick best tool based on intent
        tool_name = _pick_tool_by_intent(parsed_query.intent, tools_map, server)
        if not tool_name:
            return {"error": f"No suitable tool found for intent: {parsed_query.intent}"}
        
        # Step 5: Build arguments
        tool_obj = tools_map.get(tool_name, {"name": tool_name})
        args = _build_args_from_parsed_query(
            manager, server, tool_obj, parsed_query
        )
        
        # Step 6: Call the tool
        result = _call_tool_blocking(manager, server, tool_name, args)
        
        return {
            "route": {
                "server": server,
                "tool": tool_name,
                "primary_ticker": parsed_query.primary_ticker,
                "intent": parsed_query.intent,
                "data_type": parsed_query.data_type
            },
            "parsed": _safe_json_loads(result),
            "raw": result,
            "error": None
        }
        
    except Exception as e:
        return {"error": f"Routing failed: {str(e)}"}


def _pick_tool_by_intent(intent: str, tools_map: Dict[str, dict], server: str) -> Optional[str]:
    """Pick the best tool based on parsed intent."""
    intent_map = {
        "price_quote": ["get_current_stock_price", "get_stock_info", "quote"],
        "news": ["get_yahoo_finance_news", "get_company_news"],
        "analysis": ["get_stock_info", "get_financial_statement"],
        "fundamentals": ["get_financial_statement", "get_balance_sheets", "get_income_statements"],
        "options": ["get_option_chain"],
        "recommendation": ["get_recommendations"],
        "dividend": ["get_stock_actions"],
        "insider": ["get_holder_info"]
    }
    
    candidates = intent_map.get(intent, [])
    for tool_name in candidates:
        if tool_name in tools_map:
            return tool_name
    
    # Fallback: return first tool
    return next(iter(tools_map)) if tools_map else None


def _build_args_from_parsed_query(
    manager: MCPManager,
    server: str,
    tool: dict,
    parsed: "ParsedQuery"
) -> Dict[str, any]:
    """Build tool arguments from structured parsed query."""
    
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    
    args = {}
    
    # Ticker/symbol
    if "ticker" in props and parsed.primary_ticker:
        args["ticker"] = parsed.primary_ticker
    elif "symbol" in props and parsed.primary_ticker:
        args["symbol"] = parsed.primary_ticker
    
    # Time range
    if "period" in props:
        if parsed.time_range and "6 month" in parsed.time_range.lower():
            args["period"] = "6mo"
        elif parsed.time_range and "year" in parsed.time_range.lower():
            args["period"] = "1y"
        else:
            args["period"] = "1y"  # default
    
    if "interval" in props:
        args["interval"] = "1d"  # default
    
    # Financial statement type
    if tool.get("name") == "get_financial_statement":
        if "balance" in parsed.raw_intent.lower():
            args["financial_type"] = "balance_sheet"
        elif "cash" in parsed.raw_intent.lower():
            args["financial_type"] = "cashflow"
        else:
            args["financial_type"] = "income_stmt"
    
    # Options expiration
    if tool.get("name") == "get_option_chain" and parsed.expiration_days:
        if "expiration_date" in props and parsed.primary_ticker:
            try:
                # Get available expirations and pick closest to parsed.expiration_days
                raw = _call_tool_blocking(
                    manager, server, "get_option_expiration_dates",
                    {"ticker": parsed.primary_ticker}
                )
                dates = _safe_json_loads(raw) or []
                if isinstance(dates, list) and dates:
                    args["expiration_date"] = dates[0]  # Simplified: pick first
            except Exception:
                pass
        
        if "option_type" in props:
            args["option_type"] = parsed.options_type or "calls"
    
    return args


# Synchronous wrapper for use in non-async context
def route_and_call(query: str) -> Dict[str, any]:
    """Wrapper to call async function from sync context."""
    import asyncio
    try:
        return asyncio.run(route_and_call_v2(query))
    except Exception as e:
        return {"error": f"Route error: {str(e)}"}


@tool(
    name="mcp_auto",
    description="Auto-router for MCP servers. Dynamically lists tools from the chosen server, picks the best match, and builds valid arguments from the tool schema."
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    return route_and_call(query)
