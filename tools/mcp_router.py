from __future__ import annotations

import re
import json
import math
import asyncio
from functools import lru_cache
from datetime import datetime, date
from typing import Dict, List, Optional, Any

import yfinance as yf  # validation only

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager, MCPServer


# =============================
# Constants & heuristics
# =============================

# e.g., MSFT, BRK.B, TSLA, XRP-USD
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3}|-[A-Z0-9]{1,6})?\b")

_STOP = {
    "USD", "USDT", "USDC",
    "PE", "EV", "EPS", "ETF",
    "AND", "OR", "THE", "A", "AN", "IS", "ARE", "FOR", "TO", "OF", "IN",
    "PRICE", "PRICES", "NEWS", "OPTIONS", "CHAIN", "PUTS", "CALLS",
    "STOCK", "SHARE", "SHARES",
    "TODAY", "NOW", "LATEST", "CURRENT", "REAL", "TIME", "REALTIME", "LIVE",
    "UPGRADES", "DOWNGRADES", "RECOMMENDATIONS",
}

_CRYPTO_HINTS = {
    "CRYPTO", "BITCOIN", "BTC", "ETH", "ETHEREUM", "COIN", "TOKEN", "BLOCKCHAIN", "DEFI",
    "XRP", "DOGE", "LTC", "BCH", "SOL", "ADA", "MATIC", "AVAX", "LINK",
    "UNI", "AAVE", "ATOM", "ETC", "FIL", "XLM", "NEAR", "APT", "ARB",
}

VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

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


# =============================
# Intent classification
# =============================

def _tok(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\.\-]+", text or "")

def _extract_regex_tickers(text: str) -> List[str]:
    matches = _TICKER_RE.findall((text or "").upper())
    return [m for m in matches if m not in _STOP]

def _detect_crypto_intent(query: str) -> bool:
    up = (query or "").upper()
    if any(k in up for k in _CRYPTO_HINTS):
        return True
    if re.search(r"\b[A-Z0-9]{2,10}-USD\b", up):
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


# =============================
# Symbol validation (robust Yahoo)
# =============================

def _is_stop(token: str) -> bool:
    return token in _STOP

def _looks_like_equity_symbol(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,3})?", token))

def _is_number(x: Any) -> bool:
    try:
        return x is not None and not (isinstance(x, bool)) and math.isfinite(float(x))
    except Exception:
        return False

@lru_cache(maxsize=1024)
def _validate_equity_symbol_yahoo(sym: str) -> bool:
    """
    Strong Yahoo validation:
      - Success if 1-day history is non-empty.
      - Else, success only if fast_info/regularMarketPrice is numeric AND quoteType/market look sane.
    This avoids false-positives like 'APPL'.
    """
    try:
        t = yf.Ticker(sym)

        # 1) Try recent history – most reliable
        try:
            hist = t.history(period="1d", interval="1d", prepost=False)
            if hasattr(hist, "empty") and not hist.empty:
                return True
        except Exception:
            pass

        # 2) Fallback on fast_info / info with stricter checks
        price = None
        try:
            # fast_info is faster and less error-prone when available
            fi = getattr(t, "fast_info", None)
            if fi and hasattr(fi, "last_price"):
                price = getattr(fi, "last_price")
        except Exception:
            pass

        if price is None:
            try:
                info = t.info or {}
            except Exception:
                info = {}
            price = info.get("regularMarketPrice")
            quote_type = (info.get("quoteType") or "").upper()
            market = (info.get("market") or info.get("fullExchangeName") or "").upper()
            symbol_echo = (info.get("symbol") or "").upper()

            # Require BOTH: numeric price AND plausible quoteType/market,
            # and symbol echo (if present) should match candidate prefix (avoid random echoes)
            if _is_number(price) and (quote_type in {"EQUITY", "ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY"} or "NASDAQ" in market or "NYSE" in market):
                if not symbol_echo or sym.startswith(symbol_echo[:len(sym)]):
                    return True
            return False

        # fast_info price numeric is good enough
        return _is_number(price)

    except Exception:
        return False


# =============================
# Finnhub lookup with better ranking
# =============================

def _words(s: str) -> List[str]:
    return re.findall(r"[A-Z0-9]+", (s or "").upper())

def _whole_word_hits(query_words: List[str], text_words: List[str]) -> int:
    tw = set(text_words)
    return sum(1 for w in query_words if w in tw)

def _substring_hits(query: str, text: str) -> int:
    q = (query or "").upper()
    t = (text or "").upper()
    return sum(1 for w in _words(q) if w and w in t)

@lru_cache(maxsize=512)
def _candidate_queries_for_lookup(q: str) -> List[str]:
    cands: List[str] = []
    orig = (q or "").strip()
    if orig:
        cands.append(orig)

    # cleaned
    toks = [t for t in _tok(orig) if t.upper() not in _STOP]
    cleaned = " ".join(toks)
    if cleaned and cleaned != orig:
        cands.append(cleaned)

    # unigrams / bigrams / trigrams
    for t in toks:
        cands.append(t)
    for i in range(len(toks) - 1):
        cands.append(f"{toks[i]} {toks[i+1]}")
    for i in range(len(toks) - 2):
        cands.append(f"{toks[i]} {toks[i+1]} {toks[i+2]}")

    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

@lru_cache(maxsize=512)
def _lookup_equity_symbol_finnhub_one(query: str) -> Optional[str]:
    try:
        from utils.config import load_settings
        import finnhub
        settings = load_settings()
        client = finnhub.Client(api_key=settings.finnhub_api_key)
        res = client.symbol_lookup(query)
        items = (res or {}).get("result") or []
        if not items:
            return None

        qw = _words(query)

        def score(it):
            desc = (it.get("description") or "")
            ty = (it.get("type") or "")
            sym = (it.get("symbol") or "")
            dw = _words(desc)

            s = 0.0
            # exact symbol preference if user wrote symbol explicitly
            if sym.upper() in _words(query):
                s += 5.0
            # whole-word matches (avoid 'pineAPPLE')
            s += 3.0 * _whole_word_hits(qw, dw)
            # name startswith
            if dw and qw and " ".join(dw).startswith(" ".join(qw)):
                s += 1.5
            # substring matches give tiny weight only
            s += 0.25 * _substring_hits(query, desc)
            # prefer US exchanges/types
            if "NASDAQ" in desc.upper() or "NYSE" in desc.upper():
                s += 1.0
            if ty.upper() in {"EQS", "COMMON STOCK", "EQUITY", "STOCK"}:
                s += 0.5
            # shorter symbols slightly preferred
            s += max(0.0, 1.5 - 0.1 * len(sym))
            return s

        items.sort(key=score, reverse=True)
        best = (items[0].get("symbol") or "").upper().strip()
        return best or None
    except Exception:
        return None

def _lookup_equity_symbol_finnhub(q: str) -> Optional[str]:
    for cand in _candidate_queries_for_lookup(q):
        sym = _lookup_equity_symbol_finnhub_one(cand)
        if sym and _validate_equity_symbol_yahoo(sym):
            return sym
    return None


# =============================
# Crypto helpers
# =============================

def _maybe_crypto_symbol(token: str) -> Optional[str]:
    t = (token or "").upper().strip()
    if not re.fullmatch(r"[A-Z0-9]{2,10}", t):
        return None
    if t in {"USD", "USDT", "USDC"} or _is_stop(t):
        return None
    return f"{t}-USD"

def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =============================
# Public resolver
# =============================

def resolve_tickers(query: str) -> List[str]:
    """
    Robust resolver:
      1) Regex candidates are accepted ONLY if they pass strict Yahoo validation.
      2) If crypto intent, try SYMBOL-USD normalization from a token.
      3) Otherwise use Finnhub multi-candidate lookup with whole-word ranking, then validate via Yahoo.
      4) Return deduped list (possibly empty).
    """
    q_up = (query or "").upper()
    out: List[str] = []

    # 1) regex candidates with strict validation
    for tok in _extract_regex_tickers(q_up):
        if _is_stop(tok):
            continue
        if _looks_like_equity_symbol(tok) and _validate_equity_symbol_yahoo(tok):
            out.append(tok)

    out = _dedupe(out)

    # 2) crypto normalization
    is_crypto = _detect_crypto_intent(query)
    if is_crypto and not any(s.endswith("-USD") for s in out):
        for w in _tok(query):
            s = _maybe_crypto_symbol(w)
            if s:
                out.append(s)
                break

    if out:
        return _dedupe(out)

    # 3) free-text equity lookup via Finnhub
    if not is_crypto:
        sym = _lookup_equity_symbol_finnhub(query)
        if sym and _validate_equity_symbol_yahoo(sym):
            return [sym]

    # 4) last crypto attempt
    if is_crypto:
        for w in _tok(query):
            s = _maybe_crypto_symbol(w)
            if s:
                return [s]

    return []


# =============================
# Server & tool selection
# =============================

def _select_best_server(query: str, available: Dict[str, MCPServer]) -> Optional[str]:
    is_crypto = _detect_crypto_intent(query)
    if is_crypto:
        if "financial-datasets" in available:
            return "financial-datasets"
        if "coinmarketcap" in available:
            return "coinmarketcap"
    if "yfinance" in available and not is_crypto:
        return "yfinance"
    if "financial-datasets" in available:
        return "financial-datasets"
    return next(iter(available.keys())) if available else None

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
            prefer = "get_historical_crypto_prices" if is_crypto else "get_historical_stock_prices"
            if prefer in names:
                return prefer
        if "news" in q and "get_company_news" in names:
            return "get_company_news"
        if any(w in q for w in ["income", "cash flow", "balance sheet"]) and "get_income_statements" in names:
            return "get_income_statements"
        prefer = "get_current_crypto_price" if is_crypto else "get_current_stock_price"
        if prefer in names:
            return prefer

    if "quote" in names:
        return "quote"

    return names[0]


# =============================
# Options helpers
# =============================

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

def _call_tool_blocking(manager: MCPManager, server: str, tool_name: str, args: dict) -> str:
    if hasattr(manager, "call_sync") and callable(getattr(manager, "call_sync")):
        return manager.call_sync(server, tool_name, args)
    res = manager.call(server, tool_name, args)
    return asyncio.run(res) if asyncio.iscoroutine(res) else res


# =============================
# Args builder
# =============================

def _build_args_from_schema(manager: MCPManager, server: str, tool: dict, query: str, tickers: List[str]) -> Dict[str, Any]:
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else {}
    args: Dict[str, Any] = {}
    primary = tickers[0] if tickers else None

    if isinstance(props, dict):
        if "ticker" in props and primary:
            args["ticker"] = primary
        elif "symbol" in props and primary:
            args["symbol"] = primary

        if "period" in props and "start_date" not in props:
            args.setdefault("period", "1y")
        if "interval" in props and "interval_multiplier" not in props:
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
                            lines = raw.splitlines() if isinstance(raw, str) else []
                            candidates = [ln.strip() for ln in lines if _parse_yyyy_mm_dd(ln.strip())]
                            best = _pick_best_expiry(candidates)
                        if best:
                            args.setdefault("expiration_date", best)
                    except Exception:
                        pass

    return args


# =============================
# Routing
# =============================

def route_and_call(query: str) -> str:
    try:
        available_servers = MCPServer.from_env()
        if not available_servers:
            return "No MCP servers configured. Please check MCP_SERVERS in .env"

        tickers = resolve_tickers(query)  # validated or empty
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

        schema = tool_obj.get("inputSchema") or {}
        props = (schema.get("properties") if isinstance(schema, dict) else {}) or {}
        required = (schema.get("required") if isinstance(schema, dict) else []) or []
        needs_symbol = ("ticker" in props or "symbol" in props) or any(r in ("ticker", "symbol") for r in required)

        if needs_symbol and not (args.get("ticker") or args.get("symbol")):
            # last-attempt lookup
            sym = _lookup_equity_symbol_finnhub(query)
            if sym and _validate_equity_symbol_yahoo(sym):
                args["ticker"] = sym
            else:
                return "Could not resolve a valid ticker from the query. Please specify a symbol (e.g., TSLA or AAPL)."

        routing_info = (
            f"Route: {server}/{tool_name} | Type: {data_type} | Crypto: {is_crypto} | "
            f"Tickers: {args.get('ticker') or args.get('symbol') or tickers}"
        )

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
