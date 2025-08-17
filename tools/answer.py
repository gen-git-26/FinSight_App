# tools/answer.py
from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, Any, List

from agno.tools import tool
from qdrant_client.http import models as rest

from utils.config import load_settings
from rag.fusion import retrieve, rerank_and_summarize
from tools.tools import (
    finnhub_stock_info,
    finnhub_basic_financials,
    finnhub_financials_as_reported,
    company_overview,
    finnhub_financials_new
)
from memory.manager import fetch_memory, persist_turn

# Regex to match common stock ticker patterns, including suffixes like .US or -US
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]|-[A-Z]{1,3})?\b")
# Common stop words that are not useful as tickers
_STOP = { 'AND','THE','FOR','WITH','OVER','FROM','THIS','THAT','INTO','YOUR','US','USA','NYSE','NASDAQ','SP','ETF','PE','EV','EBITDA','EPS','CAGR','ROE','P','E' }


def _extract_tickers(text: str) -> List[str]:
    cands = _TICKER_RE.findall(text or "")
    return [c for c in cands if c not in _STOP]


def _detect_style(q: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    ql = (q or "").lower()
    if any(k in ql for k in ["executive summary", "report", "comprehensive", "deep dive", "overview including", "investment memo", "thesis"]):
        return "report"
    return "concise"

@tool(name="answer", description="General-purpose Fusion RAG answerer. Input: free-form question; optional ticker(s). Dynamically fetches data, retrieves, fuses, and summarizes.")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    # 0) resolve style and tickers
    style = _detect_style(query, style)
    tickers = list(dict.fromkeys(([t.strip().upper() for t in (ticker.split() if ticker else [])] + _extract_tickers(query))))

    # 1) If tickers found, refresh cache by calling data tools (each is wrapped by Fusion and ingests snippets)
    for t in tickers[:3]:  # limit to 3 to cap latency
        try:
            finnhub_stock_info(t)
            finnhub_basic_financials(t, metric="price")
            finnhub_basic_financials(t, metric="valuation")
            finnhub_basic_financials(t, metric="growth")
            finnhub_financials_as_reported(t)
            finnhub_financials_new(t)
            company_overview(t)
        except Exception:
            pass

    # 2) Retrieve across all types, optionally filter by symbol(s)
    flt: List[rest.FieldCondition] = []
    if tickers:
        flt.append(rest.FieldCondition(key='symbol', match=rest.MatchAny(any=tickers[:3])))
    docs = asyncio.run(retrieve(query, filters=flt, k=24))

    # 3) Bring in user memory context
    mem_ctx = asyncio.run(fetch_memory(query=query, k=3))

    # 4) Summarize with dynamic style and extra memory context
    max_tok = 900 if style == "report" else 512
    answer, snippets = asyncio.run(rerank_and_summarize(query, docs, style=style, extra_context=mem_ctx, max_tokens=max_tok))

    # 5) Persist this turn in long-term memory
    asyncio.run(persist_turn(user_text=query, assistant_text=answer))

    return {"answer": answer, "snippets": snippets, "meta": {"style": style, "tickers": tickers}}