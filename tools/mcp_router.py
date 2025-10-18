# tools/mcp_router.py


from __future__ import annotations

import re
import json
from datetime import date, timedelta, datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# Import the MCP bridge/manager used in this project.
# These should already exist in your codebase.
try:
    from mcp_bridge import MCPManager, call_tool_blocking  # type: ignore
except Exception:  # pragma: no cover
    MCPManager = None  # type: ignore
    def call_tool_blocking(*args, **kwargs):
        raise RuntimeError("mcp_bridge not available")

_CANDIDATE_RX = re.compile(r"\$?[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?")


def _iso_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _extract_target_expiry_from_query(query: str) -> Optional[date]:
    """
    Extract target expiry as absolute date if user says "YYYY-MM-DD",
    or relative date if user says "expiration in X days".
    """
    q = (query or "").lower()
    # in X days
    m_days = re.search(r"expir\w*\s+in\s+(\d{1,3})\s+day", q)
    if m_days:
        try:
            days = int(m_days.group(1))
            return date.today() + timedelta(days=days)
        except Exception:
            pass
    # explicit date
    m_date = re.search(r"(\d{4}-\d{2}-\d{2})", q)
    if m_date:
        dt = _iso_date(m_date.group(1))
        if dt:
            return dt
    return None


def _pick_expiry_near_target(all_dates: List[str], target: date) -> Optional[str]:
    candidates: List[Tuple[int, str]] = []
    for s in all_dates or []:
        d = _iso_date(s)
        if d:
            candidates.append((abs((d - target).days), s))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _validate_symbol_via_yahoo(manager_key: str, sym: str) -> bool:
    """
    Validation via Yahoo Finance MCP: get_stock_info(symbol=...).
    Returns True only if the tool responds with valid structure (e.g., has price/marketCap or sane fields).
    Using LRU cache to avoid hitting MCP repeatedly.
    """
    try:
        manager = MCPManager.get(manager_key) if hasattr(MCPManager, "get") else MCPManager()
        raw = call_tool_blocking(manager, "yahoo-finance-mcp", "get_stock_info", {"symbol": sym})
        doc = _safe_json_loads(raw) if isinstance(raw, str) else raw
        if not isinstance(doc, dict):
            return False
        # Loose sanity checks
        has_any = any(k in doc for k in ("price", "marketCap", "regularMarketPrice", "shortName"))
        return bool(has_any)
    except Exception:
        return False


@lru_cache(maxsize=4096)
def _validate_symbol_via_finnhub(manager_key: str, sym: str) -> bool:
    """
    Optional validation via Finnhub MCP-backed tools (if present).
    If not available, returns False (router will rely on Yahoo validation only).
    """
    try:
        manager = MCPManager.get(manager_key) if hasattr(MCPManager, "get") else MCPManager()
        # Prefer a lightweight endpoint if exposed. If not, fallback to any finnhub tool you have:
        # Here we try "company_overview" (if you registered it) or "finnhub_stock_info" as examples.
        for server, tool, args in [
            ("finnhub", "company_overview", {"symbol": sym}),
            ("finnhub", "finnhub_stock_info", {"symbol": sym}),
        ]:
            try:
                raw = call_tool_blocking(manager, server, tool, args)
                if raw:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _dynamic_symbol_validator(sym: str) -> bool:
    """
    Dynamically validate a candidate symbol without hardcoded STOP lists:
    1) Attempt Yahoo MCP validation
    2) Fallback to Finnhub MCP validation if available
    3) As a last resort, apply conservative heuristics to avoid obvious false positives
    """
    # We use a singleton key for manager since most apps keep one instance.
    manager_key = "default"

    # 1) Yahoo validation
    if _validate_symbol_via_yahoo(manager_key, sym):
        return True

    # 2) Finnhub validation (optional)
    if _validate_symbol_via_finnhub(manager_key, sym):
        return True

    # 3) Conservative heuristics (last resort):
    # - Require uppercase or dot-notation (e.g., BRK.B)
    # - Length 1..5 for uppercase-only symbols; allow dot-notation up to 8
    up = sym.upper()
    if sym != up and "." not in sym:
        return False
    if "." in sym and len(sym) <= 8:
        # BRK.B-like symbols allowed
        return True
    return 1 <= len(sym) <= 5


def resolve_tickers(query: str) -> List[str]:
    """
    Extract and validate equity symbols from user query dynamically,
    without relying on hardcoded STOP words.
    """
    if not query:
        return []
    candidates: List[str] = []
    for m in _CANDIDATE_RX.finditer(query):
        raw = m.group(0)
        sym = raw.upper().lstrip("$")
        if sym in candidates:
            continue
        if _dynamic_symbol_validator(sym):
            candidates.append(sym)

    # If no equities found, you can optionally try crypto resolution here by reading from
    # a cached list built at startup via financial-datasets-mcp/get_available_crypto_tickers.
    # (Omitted here to avoid tight coupling; add if needed in your environment.)

    return candidates


def _nearest_expiration_for_symbol(manager: MCPManager, symbol: str, target: Optional[date]) -> Optional[str]:
    """
    Use Yahoo Finance MCP get_option_expiration_dates to pick the nearest expiry to target date.
    If target is None, pick the closest future date to today.
    """
    try:
        raw = call_tool_blocking(manager, "yahoo-finance-mcp", "get_option_expiration_dates", {"ticker": symbol})
        dates = _safe_json_loads(raw) if isinstance(raw, str) else raw
        if not isinstance(dates, list) or not dates:
            return None
        # If no target, use today as target (closest future date)
        ref = target or date.today()
        chosen = _pick_expiry_near_target([str(x) for x in dates], ref)
        return chosen
    except Exception:
        return None


def route_and_call(query: str) -> Any:
    """
    Example router that demonstrates:
    - dynamic ticker resolution
    - options expiry date parsing/selection
    - delegating to MCP tools

    NOTE: Adapt the "routing" logic to your real intent classification.
    This function shows one concrete path for options queries, but you should expand it as needed.
    """
    manager = MCPManager.get("default") if hasattr(MCPManager, "get") else MCPManager()
    ql = (query or "").lower()

    # 1) Resolve tickers
    syms = resolve_tickers(query)

    # 2) Handle options query (example)
    if "option" in ql or "options" in ql or "chain" in ql:
        # Choose one primary symbol if multiple
        symbol = syms[0] if syms else None
        if not symbol:
            return {"error": "No valid symbol resolved for options query."}

        expiry_target = _extract_target_expiry_from_query(query)
        expiry = _nearest_expiration_for_symbol(manager, symbol, expiry_target)

        # Calls vs puts
        option_type = "calls" if ("put" not in ql and "puts" not in ql) else "puts"

        args = {"ticker": symbol}
        if expiry:
            args["expiration_date"] = expiry
        args["option_type"] = option_type

        raw = call_tool_blocking(manager, "yahoo-finance-mcp", "get_option_chain", args)
        try:
            return _safe_json_loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return raw

    # 3) Example: current price
    if "current price" in ql or "latest price" in ql:
        symbol = syms[0] if syms else None
        if not symbol:
            return {"error": "No valid symbol resolved for price query."}
        try:
            raw = call_tool_blocking(manager, "financial-datasets-mcp", "get_current_stock_price", {"ticker": symbol})
            return _safe_json_loads(raw) if isinstance(raw, str) else raw
        except Exception:
            # fallback to Yahoo
            raw = call_tool_blocking(manager, "yahoo-finance-mcp", "get_stock_info", {"symbol": symbol})
            return _safe_json_loads(raw) if isinstance(raw, str) else raw

    # 4) Fallback: try Yahoo stock info if we have exactly one symbol
    if len(syms) == 1:
        raw = call_tool_blocking(manager, "yahoo-finance-mcp", "get_stock_info", {"symbol": syms[0]})
        try:
            return _safe_json_loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return raw

    # 5) Nothing matched
    return {"error": "No route matched. Please refine the query."}
