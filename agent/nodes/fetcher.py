# agent/nodes/fetcher.py
"""
Fetcher Agent - Fetches stock data using the unified datasources layer.

Uses:
- DataFetcher for automatic source selection and fallback
- Integrates with RAG for data storage
"""
from __future__ import annotations

import asyncio
from typing import Dict, Any, List

from agent.state import AgentState, FetchedData, ParsedQuery
from datasources import DataFetcher, DataType, get_fetcher
from infrastructure.memory_manager import get_memory_manager


# Map intent strings to DataType
INTENT_TO_DATA_TYPE = {
    'price': DataType.QUOTE,
    'quote': DataType.QUOTE,
    'fundamentals': DataType.FUNDAMENTALS,
    'options': DataType.OPTIONS,
    'historical': DataType.HISTORICAL,
    'news': DataType.NEWS,
    'info': DataType.QUOTE,
    'analysis': DataType.FUNDAMENTALS,
    'trading': DataType.FUNDAMENTALS,
}


def _convert_result_to_fetched_data(result, ticker: str) -> FetchedData:
    """Convert DataResult to FetchedData for state compatibility."""
    if result.success:
        # Convert dataclass to dict if needed
        parsed_data = result.data
        if hasattr(parsed_data, "__dict__"):
            parsed_data = parsed_data.__dict__

        return FetchedData(
            source=result.source,
            tool_used=result.data_type.value if result.data_type else "unknown",
            raw_data=result.raw,
            parsed_data=parsed_data
        )
    else:
        return FetchedData(
            source=result.source or "datasources",
            error=result.error
        )


async def _fetch_ticker_async(
    fetcher: DataFetcher,
    ticker: str,
    data_type: DataType,
    **kwargs
) -> FetchedData:
    """Fetch data for a single ticker asynchronously."""
    result = await fetcher.fetch(ticker, data_type, **kwargs)
    return _convert_result_to_fetched_data(result, ticker)


async def fetcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetcher node - fetches stock data using the unified datasources layer.

    Handles: stocks, options, fundamentals, news
    Automatic fallback chain: yfinance → finnhub → alphavantage
    Checks RunCache before fetching to deduplicate A2A calls.
    """
    parsed = state.get("parsed_query")
    if not parsed:
        return {"error": "No parsed query", "fetched_data": []}

    print(f"\n[Fetcher] Intent: {parsed.intent}, Ticker: {parsed.ticker}")

    # Collect all tickers
    tickers = []
    if parsed.ticker:
        tickers.append(parsed.ticker)
    tickers.extend(parsed.additional_tickers or [])

    if not tickers:
        return {"error": "No tickers found", "fetched_data": []}

    # Map intent to data type
    data_type = INTENT_TO_DATA_TYPE.get(parsed.intent, DataType.QUOTE)
    print(f"[Fetcher] Data type: {data_type.value}")

    run_id = state.get("run_id")
    manager = get_memory_manager()
    fetcher = get_fetcher()
    fetched_data: List[FetchedData] = []

    for ticker in tickers:
        cache_key = f"{ticker}:{data_type.value}"

        # Check RunCache first (A2A deduplication)
        cached = None
        if run_id:
            try:
                cached = manager.get_cached_tool(run_id=run_id, tool_key=cache_key)
            except Exception:
                pass

        if cached and not cached.error:
            print(f"[Fetcher] RunCache hit: {cache_key}")
            fetched_data.append(cached)
            continue

        # Cache miss — call API
        try:
            result = await fetcher.fetch(ticker, data_type)
            fd = _convert_result_to_fetched_data(result, ticker)
        except Exception as e:
            fd = FetchedData(source="datasources", error=str(e))

        fetched_data.append(fd)

        # Write to RunCache on success
        if run_id and not fd.error:
            try:
                await manager.cache_tool_result(run_id=run_id, tool_key=cache_key, result=fd)
            except Exception:
                pass

    # Log results
    successful = [r for r in fetched_data if not r.error]
    print(f"[Fetcher] Fetched {len(fetched_data)} results, {len(successful)} successful")

    return {
        "fetched_data": fetched_data,
        "sources": [r.source for r in successful]
    }


# === Trading Fetcher (for A2A flow) ===

def trading_fetcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Trading Fetcher - fetches comprehensive data for trading analysis.

    Fetches multiple data types: fundamentals, news, and quote.
    """
    parsed = state.get("parsed_query")
    if not parsed:
        return {"error": "No parsed query", "fetched_data": []}

    ticker = parsed.ticker
    if not ticker:
        return {"error": "No ticker for trading analysis", "fetched_data": []}

    print(f"\n[Trading Fetcher] Comprehensive fetch for: {ticker}")

    fetcher = get_fetcher()

    async def fetch_comprehensive():
        return await fetcher.fetch_comprehensive(ticker)

    # Run async fetch
    try:
        results = asyncio.run(fetch_comprehensive())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(fetch_comprehensive())

    # Convert to FetchedData list
    fetched_data = []
    for data_type, result in results.items():
        fetched_data.append(_convert_result_to_fetched_data(result, ticker))

    successful = [r for r in fetched_data if not r.error]
    print(f"[Trading Fetcher] Fetched {len(fetched_data)} data types, {len(successful)} successful")

    return {
        "fetched_data": fetched_data,
        "sources": list(set(r.source for r in successful)),
        "ticker": ticker
    }
