# agent/nodes/crypto.py
"""
Crypto Agent - Dedicated agent for cryptocurrency queries.

Uses the unified datasources layer for crypto data fetching.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Any, Optional

from agent.state import AgentState, FetchedData, ParsedQuery
from datasources import get_fetcher, DataType


# Crypto symbol mappings
CRYPTO_SYMBOLS = {
    'bitcoin': 'BTC-USD',
    'btc': 'BTC-USD',
    'ethereum': 'ETH-USD',
    'eth': 'ETH-USD',
    'solana': 'SOL-USD',
    'sol': 'SOL-USD',
    'dogecoin': 'DOGE-USD',
    'doge': 'DOGE-USD',
    'ripple': 'XRP-USD',
    'xrp': 'XRP-USD',
    'cardano': 'ADA-USD',
    'ada': 'ADA-USD',
    'polkadot': 'DOT-USD',
    'dot': 'DOT-USD',
    'avalanche': 'AVAX-USD',
    'avax': 'AVAX-USD',
    'chainlink': 'LINK-USD',
    'link': 'LINK-USD',
    'polygon': 'MATIC-USD',
    'matic': 'MATIC-USD',
}


def _normalize_crypto_ticker(ticker: Optional[str], query: str) -> str:
    """Normalize crypto ticker to standard format."""
    if ticker and ticker.endswith('-USD'):
        return ticker

    # Check if ticker is in mapping
    if ticker:
        ticker_lower = ticker.lower()
        if ticker_lower in CRYPTO_SYMBOLS:
            return CRYPTO_SYMBOLS[ticker_lower]
        # Add -USD if not present
        return f"{ticker.upper()}-USD"

    # Try to extract from query
    query_lower = query.lower()
    for name, symbol in CRYPTO_SYMBOLS.items():
        if name in query_lower:
            return symbol

    return "BTC-USD"  # Default


def _convert_result_to_fetched_data(result, ticker: str) -> FetchedData:
    """Convert DataResult to FetchedData for state compatibility."""
    if result.success:
        parsed_data = result.data
        if hasattr(parsed_data, "__dict__"):
            parsed_data = parsed_data.__dict__

        return FetchedData(
            source=result.source,
            tool_used=result.data_type.value if result.data_type else "crypto",
            raw_data=result.raw,
            parsed_data=parsed_data
        )
    else:
        return FetchedData(
            source=result.source or "datasources",
            error=result.error
        )


def crypto_node(state: AgentState) -> Dict[str, Any]:
    """
    Crypto Agent node - handles all cryptocurrency queries.

    Uses unified datasources layer with automatic fallback:
    1. yfinance (primary)
    2. CoinGecko public API (fallback)
    """
    parsed = state.get("parsed_query")
    if not parsed:
        return {"error": "No parsed query", "fetched_data": []}

    print(f"\n[Crypto] Processing crypto query")
    print(f"[Crypto] Ticker: {parsed.ticker}, Query: {parsed.raw_query}")

    # Normalize ticker
    ticker = _normalize_crypto_ticker(parsed.ticker, parsed.raw_query)
    print(f"[Crypto] Normalized ticker: {ticker}")

    # Use unified data fetcher
    fetcher = get_fetcher()

    async def fetch_crypto():
        return await fetcher.fetch(ticker, DataType.CRYPTO)

    # Run async fetch
    try:
        result = asyncio.run(fetch_crypto())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(fetch_crypto())

    fetched_data = [_convert_result_to_fetched_data(result, ticker)]
    success = result.success

    print(f"[Crypto] Fetched from {result.source}, success={success}")

    return {
        "fetched_data": fetched_data,
        "sources": [result.source] if success else []
    }
