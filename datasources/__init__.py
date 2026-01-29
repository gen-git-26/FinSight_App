# datasources/__init__.py
"""
Unified Data Sources Layer for FinSight.

Combines:
- MCP servers (yahoo-finance-mcp, financial-datasets-mcp)
- Direct APIs (yfinance, finnhub, alpha vantage, coingecko)
- RAG integration (Qdrant vector search)

Usage:
    from datasources import DataFetcher

    fetcher = DataFetcher()
    result = await fetcher.fetch("AAPL", data_type="quote")
"""
from __future__ import annotations

import asyncio
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from datasources.models import (
    DataResult, DataType, DataSourceType,
    StockQuote, Fundamentals, OptionsData, HistoricalData, NewsItem, CryptoQuote
)
from datasources.api_clients import (
    get_client, get_all_clients,
    YFinanceClient, FinnhubClient, AlphaVantageClient, CoinGeckoClient
)
from datasources.mcp_client import get_mcp_client, register_mcp_server


class FetchStrategy(Enum):
    """Strategy for fetching data."""
    FIRST_SUCCESS = "first_success"  # Use first successful source
    ALL_SOURCES = "all_sources"      # Fetch from all sources
    PREFER_MCP = "prefer_mcp"        # Try MCP first, fallback to API
    PREFER_API = "prefer_api"        # Try API first, fallback to MCP


class DataFetcher:
    """
    Unified data fetcher that combines MCP and API sources.

    Automatically:
    1. Routes to appropriate data source
    2. Falls back if primary source fails
    3. Ingests data to RAG for future retrieval
    """

    def __init__(self, strategy: FetchStrategy = FetchStrategy.PREFER_API):
        self.strategy = strategy
        self.mcp = get_mcp_client()
        self._rag_enabled = False
        self._init_rag()

    def _init_rag(self):
        """Initialize RAG integration."""
        try:
            from rag.fusion import ingest_raw
            self._ingest_raw = ingest_raw
            self._rag_enabled = True
        except ImportError:
            self._rag_enabled = False

    async def fetch(
        self,
        symbol: str,
        data_type: Union[str, DataType] = "quote",
        **kwargs
    ) -> DataResult:
        """
        Fetch data for a symbol.

        Args:
            symbol: Stock/crypto symbol (e.g., "AAPL", "BTC-USD")
            data_type: Type of data to fetch (quote, fundamentals, options, historical, news, crypto)
            **kwargs: Additional arguments (period, expiration, etc.)

        Returns:
            DataResult with the fetched data
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type.lower())

        # Detect crypto
        is_crypto = self._is_crypto(symbol)

        # Get appropriate client
        if is_crypto:
            result = await self._fetch_crypto(symbol, data_type, **kwargs)
        else:
            result = await self._fetch_stock(symbol, data_type, **kwargs)

        # Ingest to RAG if successful
        if result.success and self._rag_enabled:
            await self._ingest_to_rag(symbol, result)

        return result

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is cryptocurrency."""
        crypto_indicators = [
            '-usd', '-usdt', 'btc', 'eth', 'sol', 'doge',
            'xrp', 'ada', 'dot', 'avax', 'link', 'matic',
            'bitcoin', 'ethereum', 'solana'
        ]
        symbol_lower = symbol.lower()
        return any(ind in symbol_lower for ind in crypto_indicators)

    async def _fetch_stock(
        self,
        symbol: str,
        data_type: DataType,
        **kwargs
    ) -> DataResult:
        """Fetch stock data with fallback chain."""
        clients_order = ["yfinance", "finnhub", "alphavantage"]

        for client_name in clients_order:
            client = get_client(client_name)
            if not client or not client.available:
                continue

            try:
                if data_type == DataType.QUOTE or data_type == DataType.PRICE:
                    result = client.get_quote(symbol)
                elif data_type == DataType.FUNDAMENTALS:
                    result = client.get_fundamentals(symbol)
                elif data_type == DataType.OPTIONS:
                    result = client.get_options(symbol, kwargs.get("expiration"))
                elif data_type == DataType.HISTORICAL:
                    result = client.get_historical(
                        symbol,
                        kwargs.get("period", "1mo"),
                        kwargs.get("interval", "1d")
                    )
                elif data_type == DataType.NEWS:
                    result = client.get_news(symbol, kwargs.get("limit", 5))
                else:
                    result = client.get_quote(symbol)

                if result.success:
                    return result

            except Exception as e:
                print(f"[DataFetcher] {client_name} failed: {e}")
                continue

        return DataResult(
            success=False,
            error="All data sources failed",
            source="datasources"
        )

    async def _fetch_crypto(
        self,
        symbol: str,
        data_type: DataType,
        **kwargs
    ) -> DataResult:
        """Fetch crypto data."""
        # Try yfinance first (supports -USD pairs)
        yf_client = get_client("yfinance")
        if yf_client and yf_client.available:
            # Normalize symbol
            if not symbol.upper().endswith('-USD'):
                symbol = f"{symbol.upper()}-USD"

            result = yf_client.get_quote(symbol)
            if result.success:
                result.data_type = DataType.CRYPTO
                return result

        # Fallback to CoinGecko
        cg_client = get_client("coingecko")
        if cg_client:
            if data_type == DataType.HISTORICAL:
                return cg_client.get_historical(symbol, kwargs.get("period", "30"))
            return cg_client.get_quote(symbol)

        return DataResult(
            success=False,
            error="No crypto data source available",
            source="datasources"
        )

    async def _ingest_to_rag(self, symbol: str, result: DataResult):
        """Ingest fetched data to RAG for future retrieval."""
        if not self._rag_enabled:
            return

        try:
            data = result.data
            if hasattr(data, "__dict__"):
                data = data.__dict__

            await self._ingest_raw(
                tool=result.source,
                raw=data,
                symbol=symbol,
                doc_type=result.data_type.value if result.data_type else "unknown"
            )
        except Exception as e:
            print(f"[DataFetcher] RAG ingest failed: {e}")

    async def fetch_multiple(
        self,
        symbols: List[str],
        data_type: Union[str, DataType] = "quote"
    ) -> Dict[str, DataResult]:
        """Fetch data for multiple symbols in parallel."""
        tasks = [self.fetch(symbol, data_type) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: (result if isinstance(result, DataResult) else DataResult(success=False, error=str(result)))
            for symbol, result in zip(symbols, results)
        }

    async def fetch_comprehensive(self, symbol: str) -> Dict[str, DataResult]:
        """Fetch all data types for a symbol."""
        data_types = [DataType.QUOTE, DataType.FUNDAMENTALS, DataType.NEWS]

        if not self._is_crypto(symbol):
            data_types.append(DataType.OPTIONS)

        tasks = [self.fetch(symbol, dt) for dt in data_types]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            dt.value: (result if isinstance(result, DataResult) else DataResult(success=False, error=str(result)))
            for dt, result in zip(data_types, results)
        }


# === Convenience Functions ===

_fetcher: Optional[DataFetcher] = None


def get_fetcher() -> DataFetcher:
    """Get the singleton DataFetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher


async def fetch(symbol: str, data_type: str = "quote", **kwargs) -> DataResult:
    """Quick fetch function."""
    fetcher = get_fetcher()
    return await fetcher.fetch(symbol, data_type, **kwargs)


def fetch_sync(symbol: str, data_type: str = "quote", **kwargs) -> DataResult:
    """Synchronous fetch wrapper."""
    return asyncio.run(fetch(symbol, data_type, **kwargs))


# === Exports ===

__all__ = [
    # Main classes
    "DataFetcher",
    "FetchStrategy",

    # Models
    "DataResult",
    "DataType",
    "DataSourceType",
    "StockQuote",
    "Fundamentals",
    "OptionsData",
    "HistoricalData",
    "NewsItem",
    "CryptoQuote",

    # API Clients
    "get_client",
    "get_all_clients",
    "YFinanceClient",
    "FinnhubClient",
    "AlphaVantageClient",
    "CoinGeckoClient",

    # MCP
    "get_mcp_client",
    "register_mcp_server",

    # Convenience
    "get_fetcher",
    "fetch",
    "fetch_sync",
]
