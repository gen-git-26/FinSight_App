# mcp_router.py
from __future__ import annotations

import re, json
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from mcp_connection.manager import MCPServerManager as MCPManager  # adjust if your class name differs
from mcp_connection.manager import call_tool_blocking  # if you expose such helper; else use manager.call_sync

_CANDIDATE_RX = re.compile(r"\$?[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?")

def _iso_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s)
    except Exception:
        return None

def _safe_json_loads(s: Any) -> Any:
    try:
        return json.loads(s) if isinstance(s, str) else s
    except Exception:
        return None

def _extract_target_expiry(query: str) -> Optional[date]:
    q = (query or "").lower()
    m = re.search(r"expir\w*\s+in\s+(\d{1,3})\s+day", q)
    if m:
        try:
            return date.today() + timedelta(days=int(m.group(1)))
        except Exception:
            pass
    m2 = re.search(r"(\d{4}-\d{2}-\d{2})", q)
    if m2:
        dt = _iso_date(m2.group(1))
        if dt:
            return dt
    return None

def _pick_expiry_near_target(all_dates: List[str], target: date) -> Optional[str]:
    cands: List[Tuple[int, str]] = []
    for s in all_dates or []:
        d = _iso_date(str(s))
        if d:
            cands.append((abs((d - target).days), str(s)))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1]

@lru_cache(maxsize=4096)
def _validate_symbol_via_yahoo(sym: str) -> bool:
    try:
        manager = MCPManager()
        raw = manager.call_sync("yfinance", "get_stock_info", {"symbol": sym})
        doc = _safe_json_loads(raw)
        return isinstance(doc, dict) and any(k in doc for k in ("price","marketCap","regularMarketPrice","shortName"))
    except Exception:
        return False

@lru_cache(maxsize=4096)
def _validate_symbol_via_finnhub(sym: str) -> bool:
    try:
        manager = MCPManager()
        # if you have a lightweight finnhub tool registered:
        raw = manager.call_sync("financial-datasets", "get_current_stock_price", {"ticker": sym})
        return bool(_safe_json_loads(raw))
    except Exception:
        return False

def _dynamic_symbol_ok(sym: str) -> bool:
    # 1) Yahoo
    if _validate_symbol_via_yahoo(sym):
        return True
    # 2) Finnhub/financial-datasets fallback
    if _validate_symbol_via_finnhub(sym):
        return True
    # 3) conservative heuristic fallback
    up = sym.upper()
    if sym != up and "." not in sym:
        return False
    if "." in sym and len(sym) <= 8:
        return True
    return 1 <= len(sym) <= 5

def resolve_tickers(query: str) -> List[str]:
    if not query:
        return []
    out: List[str] = []
    for m in _CANDIDATE_RX.finditer(query):
        sym = m.group(0).upper().lstrip("$")
        if sym not in out and _dynamic_symbol_ok(sym):
            out.append(sym)
    return out

def _nearest_expiry(symbol: str, target: Optional[date]) -> Optional[str]:
    try:
        manager = MCPManager()
        raw = manager.call_sync("yfinance", "get_option_expiration_dates", {"ticker": symbol})
        dates = _safe_json_loads(raw)
        if not isinstance(dates, list) or not dates:
            return None
        ref = target or date.today()
        return _pick_expiry_near_target([str(x) for x in dates], ref)
    except Exception:
        return None

def route_and_call(query: str) -> Any:
    manager = MCPManager()
    ql = (query or "").lower()
    syms = resolve_tickers(query)

    if "option" in ql or "options" in ql or "chain" in ql:
        symbol = syms[0] if syms else None
        if not symbol:
            return {"error": "No valid symbol resolved for options query."}
        expiry = _nearest_expiry(symbol, _extract_target_expiry(query))
        option_type = "calls" if ("put" not in ql and "puts" not in ql) else "puts"
        args = {"ticker": symbol, "option_type": option_type}
        if expiry:
            args["expiration_date"] = expiry
        raw = manager.call_sync("yfinance", "get_option_chain", args)
        return _safe_json_loads(raw)

    if "current price" in ql or "latest price" in ql:
        symbol = syms[0] if syms else None
        if not symbol:
            return {"error": "No valid symbol resolved for price query."}
        try:
            raw = manager.call_sync("financial-datasets", "get_current_stock_price", {"ticker": symbol})
            return _safe_json_loads(raw)
        except Exception:
            raw = manager.call_sync("yfinance", "get_stock_info", {"symbol": symbol})
            return _safe_json_loads(raw)

    if len(syms) == 1:
        raw = manager.call_sync("yfinance", "get_stock_info", {"symbol": syms[0]})
        return _safe_json_loads(raw)

    return {"error": "No route matched. Please refine the query."}
