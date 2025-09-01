# tools/mcp_router.py
from __future__ import annotations

import os
import re
import json
import string
import logging
from typing import Dict, List, Optional, Any, Tuple

import httpx
import yfinance as yf
from agno.tools import tool

from mcp_connection.manager import MCPManager, MCPServer

log = logging.getLogger(__name__)

# -----------------------------
# Normalization helpers
# -----------------------------

_PUNC_TABLE = str.maketrans({c: " " for c in string.punctuation})

def _normalize_words(text: str) -> List[str]:
    """Uppercase, remove punctuation, split, drop empties."""
    if not text:
        return []
    cleaned = text.translate(_PUNC_TABLE).upper()
    return [w for w in cleaned.split() if w.isascii() and w]

# -----------------------------
# Ticker resolution
# -----------------------------

# Allow typical ticker forms (incl. BRK.B, RDS-A, etc.)
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z0-9]{1,6})?\b")

_COMMON_CORRECTIONS = {
    # Big caps & common names
    "APPL": "AAPL",
    "APPLE": "AAPL",
    "TESLA": "TSLA",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "META": "META",
    "NVIDIA": "NVDA",
    "NVDA": "NVDA",
    "AMAZON": "AMZN",
    "MICROSOFT": "MSFT",
    "INTEL": "INTC",
    "BROADCOM": "AVGO",
    "BERKSHIRE": "BRK.B",
    "JPMORGAN": "JPM",
    "JPM": "JPM",
}

_STOPWORDS = {
    "AND", "THE", "FOR", "WITH", "OVER", "FROM", "THIS", "THAT", "INTO", "YOUR",
    "WHAT", "IS", "ARE", "LATEST", "NEWS", "PRICE", "STOCK", "QUOTE", "TODAY",
    "A", "AN", "OF", "ON", "ABOUT", "PLEASE", "SHOW", "TELL", "ME", "LIVE",
    "CURRENT", "NOW", "RECENT", "UPDATE", "UPDATES", "FYI", "MARKET", "CAPITAL",
}

def _extract_ticker_like_tokens(text: str) -> List[str]:
    # Find all short, ticker-like tokens (keeps TSLA, AAPL, BRK.B etc.)
    return _TICKER_RE.findall((text or "").upper())

def _apply_corrections(words: List[str]) -> List[str]:
    mapped = []
    for w in words:
        mapped.append(_COMMON_CORRECTIONS.get(w, w))
    # dedupe while preserving order
    seen, out = set(), []
    for t in mapped:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _yfinance_exists(symbol: str) -> bool:
    try:
        _ = yf.Ticker(symbol).fast_info
        return True
    except Exception:
        return False

def _finnhub_search_short(query_words: List[str], api_key: Optional[str]) -> List[Tuple[str, str]]:
    """Query Finnhub with a *short, clean* term (avoid full sentences -> 422)."""
    if not api_key:
        return []
    if not query_words:
        return []
    # Build a short query: up to 3 non-stopwords, longest first
    candidates = [w for w in query_words if w not in _STOPWORDS]
    # Prefer longer tokens (company names) first
    candidates.sort(key=len, reverse=True)
    q = " ".join(candidates[:3])
    if not q:
        return []
    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.get(
                "https://finnhub.io/api/v1/search",
                params={"q": q, "token": api_key},
            )
            r.raise_for_status()
            data = r.json() or {}
        out = []
        for item in data.get("result", [])[:10]:
            sym = (item.get("symbol") or "").upper()
            desc = item.get("description") or ""
            if sym:
                out.append((sym, desc))
        return out
    except Exception as e:
        log.warning("Finnhub search failed for %r: %s", q, e)
        return []

def resolve_tickers(query: str) -> List[str]:
    """
    Robust ticker resolver:
    1) Exact ticker-like tokens (TSLA, BRK.B) if exist and valid.
    2) Dictionary corrections on *words* (APPLE->AAPL, MICROSOFT->MSFT), punctuation-insensitive.
    3) Finnhub search with a short, sanitized query (avoid 422).
    4) As a last resort, validate top Finnhub candidates with yfinance.
    """
    if not query:
        return []

    # 1) direct ticker-looking tokens
    tickerish = _extract_ticker_like_tokens(query)
    direct = [t for t in tickerish if _yfinance_exists(t)]
    if direct:
        return list(dict.fromkeys(direct))[:3]

    # 2) corrections over cleaned words
    words = _normalize_words(query)
    corrected = _apply_corrections(words)
    corrected_valid = [t for t in corrected if len(t) <= 8 and _yfinance_exists(t)]
    if corrected_valid:
        return corrected_valid[:3]

    # 3) short Finnhub search
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    suggestions = _finnhub_search_short(words, finnhub_key)

    # 4) validate Finnhub results with yfinance
    validated = []
    for sym, _ in suggestions:
        if _yfinance_exists(sym):
            validated.append(sym)
    if validated:
        return list(dict.fromkeys(validated))[:3]

    return []

# -----------------------------
# Intent detection
# -----------------------------

def _detect_intent(q: str) -> str:
    ql = (q or "").lower()
    if any(k in ql for k in ["latest", "news", "headline", "headlines", "article", "story"]):
        return "news"
    if any(k in ql for k in ["option", "chain", "calls", "puts", "expiration"]):
        return "options"
    if any(k in ql for k in ["holder", "insider", "institutional", "mutualfund"]):
        return "holders"
    if any(k in ql for k in ["recommendation", "upgrades", "downgrades", "rating", "analyst"]):
        return "recs"
    if any(k in ql for k in ["income statement", "balance sheet", "cash flow", "cashflow", "financial statement"]):
        return "financials"
    if any(k in ql for k in ["history", "historical", "past", "1y", "5y", "chart"]):
        return "history"
    if any(k in ql for k in ["price", "quote", "now", "current", "live", "today"]):
        return "realtime"
    return "auto"

# -----------------------------
# MCP via MCPManager
# -----------------------------

def _mgr() -> MCPManager:
    return MCPManager()

def _available_yf_tools() -> Dict[str, dict]:
    mgr = _mgr()
    data = mgr.list_tools_sync("yfinance")
    tools: Dict[str, dict] = {}
    raw = data["tools"] if isinstance(data, dict) and isinstance(data.get("tools"), list) else data
    for t in (raw or []):
        name = (t.get("name") if isinstance(t, dict) else getattr(t, "name", None)) or ""
        if name:
            tools[name] = t if isinstance(t, dict) else {"name": name}
    return tools

def _call_yf(tool_name: str, **kwargs) -> str:
    return _mgr().call_sync("yfinance", tool_name, kwargs)

# -----------------------------
# Routing & calling
# -----------------------------

def route_and_call(query: str) -> str:
    servers = MCPServer.from_env()
    if "yfinance" not in servers:
        return "No MCP server named 'yfinance' is configured. Please check MCP_SERVERS."

    catalog = _available_yf_tools()
    intent = _detect_intent(query)
    tickers = resolve_tickers(query)

    if not tickers:
        return "Could not resolve a valid ticker from the query. Please specify a symbol (e.g., TSLA or AAPL)."

    ticker = tickers[0]

    try:
        if intent in ("news", "auto"):
            if "get_yahoo_finance_news" in catalog:
                return _call_yf("get_yahoo_finance_news", ticker=ticker)
            return _call_yf("get_stock_info", ticker=ticker)

        if intent == "realtime":
            return _call_yf("get_stock_info", ticker=ticker)

        if intent == "history":
            if "get_historical_stock_prices" in catalog:
                return _call_yf("get_historical_stock_prices", ticker=ticker, period="1mo", interval="1d")
            return _call_yf("get_stock_info", ticker=ticker)

        if intent == "options":
            if "get_option_expiration_dates" in catalog:
                return _call_yf("get_option_expiration_dates", ticker=ticker)
            return _call_yf("get_stock_info", ticker=ticker)

        if intent == "holders":
            if "get_holder_info" in catalog:
                return _call_yf("get_holder_info", ticker=ticker, holder_type="major_holders")
            return _call_yf("get_stock_info", ticker=ticker)

        if intent == "financials":
            if "get_financial_statement" in catalog:
                return _call_yf("get_financial_statement", ticker=ticker, financial_type="income_stmt")
            return _call_yf("get_stock_info", ticker=ticker)

        if intent == "recs":
            if "get_recommendations" in catalog:
                return _call_yf("get_recommendations", ticker=ticker, recommendation_type="recommendations", months_back=12)
            return _call_yf("get_stock_info", ticker=ticker)

        return _call_yf("get_stock_info", ticker=ticker)

    except Exception as e:
        msg = str(e)
        if "HTTP Error 404" in msg or "possibly delisted" in msg:
            return f"Could not fetch data for {ticker}. The symbol may be delisted or temporarily unavailable."
        log.warning("MCP call failed: %s", e)
        return f"MCP Error: {e}"

# -----------------------------
# Public helper
# -----------------------------

def mcp_auto(query: str) -> Dict[str, Any] | str:
    res = route_and_call(query)
    return {"answer": res} if isinstance(res, str) else res

@tool(name="mcp_router", description="Route a free-form financial query to Yahoo Finance MCP with robust ticker resolution.")
def mcp_router_tool(query: str) -> Any:
    return route_and_call(query)
