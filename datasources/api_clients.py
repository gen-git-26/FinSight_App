# datasources/api_clients.py
"""
Direct API clients for financial data.

Supports:
- yfinance (Yahoo Finance)
- Finnhub
- Alpha Vantage
- CoinGecko (crypto)
"""
from __future__ import annotations

import httpx
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from utils.config import load_settings
from datasources.models import (
    DataResult, DataType, DataSourceType,
    StockQuote, Fundamentals, OptionsData, HistoricalData, NewsItem, CryptoQuote
)


class BaseAPIClient(ABC):
    """Base class for API clients."""

    name: str = "base"

    @abstractmethod
    def get_quote(self, symbol: str) -> DataResult:
        """Get stock/crypto quote."""
        pass

    def get_fundamentals(self, symbol: str) -> DataResult:
        """Get company fundamentals."""
        return DataResult(success=False, error="Not implemented", source=self.name)

    def get_options(self, symbol: str, expiration: Optional[str] = None) -> DataResult:
        """Get options chain."""
        return DataResult(success=False, error="Not implemented", source=self.name)

    def get_historical(self, symbol: str, period: str = "1mo", interval: str = "1d") -> DataResult:
        """Get historical data."""
        return DataResult(success=False, error="Not implemented", source=self.name)

    def get_news(self, symbol: str, limit: int = 5) -> DataResult:
        """Get news articles."""
        return DataResult(success=False, error="Not implemented", source=self.name)


class YFinanceClient(BaseAPIClient):
    """Yahoo Finance client using yfinance library."""

    name = "yfinance"

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            self.available = False

    def get_quote(self, symbol: str) -> DataResult:
        """Get stock quote from Yahoo Finance."""
        if not self.available:
            return DataResult(success=False, error="yfinance not installed", source=self.name)

        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info

            quote = StockQuote(
                symbol=symbol,
                price=info.get("currentPrice") or info.get("regularMarketPrice"),
                change=info.get("regularMarketChange"),
                change_percent=info.get("regularMarketChangePercent"),
                open=info.get("open") or info.get("regularMarketOpen"),
                high=info.get("dayHigh") or info.get("regularMarketDayHigh"),
                low=info.get("dayLow") or info.get("regularMarketDayLow"),
                previous_close=info.get("previousClose"),
                volume=info.get("volume") or info.get("regularMarketVolume"),
                market_cap=info.get("marketCap"),
                name=info.get("shortName") or info.get("longName"),
                source=self.name
            )

            return DataResult(
                success=True,
                data=quote,
                data_type=DataType.QUOTE,
                source=self.name,
                raw=info
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def get_fundamentals(self, symbol: str) -> DataResult:
        """Get company fundamentals."""
        if not self.available:
            return DataResult(success=False, error="yfinance not installed", source=self.name)

        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info

            fundamentals = Fundamentals(
                symbol=symbol,
                name=info.get("shortName"),
                sector=info.get("sector"),
                industry=info.get("industry"),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                price_to_book=info.get("priceToBook"),
                dividend_yield=info.get("dividendYield"),
                market_cap=info.get("marketCap"),
                revenue=info.get("totalRevenue"),
                profit_margin=info.get("profitMargins"),
                debt_to_equity=info.get("debtToEquity"),
                week_52_high=info.get("fiftyTwoWeekHigh"),
                week_52_low=info.get("fiftyTwoWeekLow"),
                eps=info.get("trailingEps"),
                beta=info.get("beta"),
                source=self.name
            )

            return DataResult(
                success=True,
                data=fundamentals,
                data_type=DataType.FUNDAMENTALS,
                source=self.name,
                raw=info
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def get_options(self, symbol: str, expiration: Optional[str] = None) -> DataResult:
        """Get options chain."""
        if not self.available:
            return DataResult(success=False, error="yfinance not installed", source=self.name)

        try:
            ticker = self.yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return DataResult(success=False, error="No options available", source=self.name)

            # Use provided expiration or find nearest
            if expiration and expiration in expirations:
                exp_date = expiration
            else:
                exp_date = expirations[0]  # Nearest

            chain = ticker.option_chain(exp_date)

            options = OptionsData(
                symbol=symbol,
                expiration=exp_date,
                calls=chain.calls.head(15).to_dict('records') if not chain.calls.empty else [],
                puts=chain.puts.head(15).to_dict('records') if not chain.puts.empty else [],
                source=self.name
            )

            return DataResult(
                success=True,
                data=options,
                data_type=DataType.OPTIONS,
                source=self.name
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def get_historical(self, symbol: str, period: str = "1mo", interval: str = "1d") -> DataResult:
        """Get historical price data."""
        if not self.available:
            return DataResult(success=False, error="yfinance not installed", source=self.name)

        try:
            ticker = self.yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                return DataResult(success=False, error="No historical data", source=self.name)

            data = HistoricalData(
                symbol=symbol,
                period=period,
                interval=interval,
                data=hist.reset_index().to_dict('records'),
                source=self.name
            )

            return DataResult(
                success=True,
                data=data,
                data_type=DataType.HISTORICAL,
                source=self.name
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def get_news(self, symbol: str, limit: int = 5) -> DataResult:
        """Get news for symbol."""
        if not self.available:
            return DataResult(success=False, error="yfinance not installed", source=self.name)

        try:
            ticker = self.yf.Ticker(symbol)
            news = ticker.news or []

            items = [
                NewsItem(
                    title=n.get("title", ""),
                    url=n.get("link"),
                    summary=n.get("summary"),
                    source=n.get("publisher"),
                    published=datetime.fromtimestamp(n["providerPublishTime"]) if n.get("providerPublishTime") else None,
                    symbol=symbol
                )
                for n in news[:limit]
            ]

            return DataResult(
                success=True,
                data=items,
                data_type=DataType.NEWS,
                source=self.name,
                raw=news
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)


class FinnhubClient(BaseAPIClient):
    """Finnhub API client."""

    name = "finnhub"

    def __init__(self):
        cfg = load_settings()
        self.api_key = cfg.finnhub_api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.available = bool(self.api_key)

    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request."""
        params = params or {}
        params["token"] = self.api_key

        response = httpx.get(f"{self.base_url}/{endpoint}", params=params, timeout=15.0)
        response.raise_for_status()
        return response.json()

    def get_quote(self, symbol: str) -> DataResult:
        """Get stock quote from Finnhub."""
        if not self.available:
            return DataResult(success=False, error="No Finnhub API key", source=self.name)

        try:
            data = self._request("quote", {"symbol": symbol})

            quote = StockQuote(
                symbol=symbol,
                price=data.get("c"),  # Current price
                change=data.get("d"),  # Change
                change_percent=data.get("dp"),  # Change percent
                open=data.get("o"),
                high=data.get("h"),
                low=data.get("l"),
                previous_close=data.get("pc"),
                source=self.name
            )

            return DataResult(
                success=True,
                data=quote,
                data_type=DataType.QUOTE,
                source=self.name,
                raw=data
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def get_fundamentals(self, symbol: str) -> DataResult:
        """Get company fundamentals from Finnhub."""
        if not self.available:
            return DataResult(success=False, error="No Finnhub API key", source=self.name)

        try:
            data = self._request("stock/metric", {"symbol": symbol, "metric": "all"})
            metrics = data.get("metric", {})

            fundamentals = Fundamentals(
                symbol=symbol,
                pe_ratio=metrics.get("peBasicExclExtraTTM"),
                price_to_book=metrics.get("pbQuarterly"),
                dividend_yield=metrics.get("dividendYieldIndicatedAnnual"),
                market_cap=metrics.get("marketCapitalization"),
                revenue=metrics.get("revenuePerShareTTM"),
                profit_margin=metrics.get("netProfitMarginTTM"),
                week_52_high=metrics.get("52WeekHigh"),
                week_52_low=metrics.get("52WeekLow"),
                eps=metrics.get("epsBasicExclExtraItemsTTM"),
                beta=metrics.get("beta"),
                source=self.name
            )

            return DataResult(
                success=True,
                data=fundamentals,
                data_type=DataType.FUNDAMENTALS,
                source=self.name,
                raw=data
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def get_news(self, symbol: str, limit: int = 5) -> DataResult:
        """Get company news from Finnhub."""
        if not self.available:
            return DataResult(success=False, error="No Finnhub API key", source=self.name)

        try:
            today = datetime.now()
            from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
            to_date = today.strftime("%Y-%m-%d")

            data = self._request("company-news", {
                "symbol": symbol,
                "from": from_date,
                "to": to_date
            })

            items = [
                NewsItem(
                    title=n.get("headline", ""),
                    url=n.get("url"),
                    summary=n.get("summary"),
                    source=n.get("source"),
                    published=datetime.fromtimestamp(n["datetime"]) if n.get("datetime") else None,
                    symbol=symbol,
                    sentiment=n.get("sentiment")
                )
                for n in data[:limit]
            ]

            return DataResult(
                success=True,
                data=items,
                data_type=DataType.NEWS,
                source=self.name,
                raw=data
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)


class AlphaVantageClient(BaseAPIClient):
    """Alpha Vantage API client."""

    name = "alphavantage"

    def __init__(self):
        cfg = load_settings()
        self.api_key = cfg.alphavantage_api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.available = bool(self.api_key)

    def _request(self, function: str, params: Dict = None) -> Dict:
        """Make API request."""
        params = params or {}
        params["function"] = function
        params["apikey"] = self.api_key

        response = httpx.get(self.base_url, params=params, timeout=15.0)
        response.raise_for_status()
        return response.json()

    def get_quote(self, symbol: str) -> DataResult:
        """Get stock quote from Alpha Vantage."""
        if not self.available:
            return DataResult(success=False, error="No Alpha Vantage API key", source=self.name)

        try:
            data = self._request("GLOBAL_QUOTE", {"symbol": symbol})
            gq = data.get("Global Quote", {})

            if not gq:
                return DataResult(success=False, error="No data returned", source=self.name)

            quote = StockQuote(
                symbol=symbol,
                price=float(gq.get("05. price", 0)),
                change=float(gq.get("09. change", 0)),
                change_percent=float(gq.get("10. change percent", "0%").replace("%", "")),
                open=float(gq.get("02. open", 0)),
                high=float(gq.get("03. high", 0)),
                low=float(gq.get("04. low", 0)),
                previous_close=float(gq.get("08. previous close", 0)),
                volume=int(gq.get("06. volume", 0)),
                source=self.name
            )

            return DataResult(
                success=True,
                data=quote,
                data_type=DataType.QUOTE,
                source=self.name,
                raw=data
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def get_fundamentals(self, symbol: str) -> DataResult:
        """Get company overview from Alpha Vantage."""
        if not self.available:
            return DataResult(success=False, error="No Alpha Vantage API key", source=self.name)

        try:
            data = self._request("OVERVIEW", {"symbol": symbol})

            if "Symbol" not in data:
                return DataResult(success=False, error="No data returned", source=self.name)

            fundamentals = Fundamentals(
                symbol=symbol,
                name=data.get("Name"),
                sector=data.get("Sector"),
                industry=data.get("Industry"),
                pe_ratio=float(data.get("PERatio", 0)) if data.get("PERatio") != "None" else None,
                forward_pe=float(data.get("ForwardPE", 0)) if data.get("ForwardPE") != "None" else None,
                price_to_book=float(data.get("PriceToBookRatio", 0)) if data.get("PriceToBookRatio") != "None" else None,
                dividend_yield=float(data.get("DividendYield", 0)) if data.get("DividendYield") != "None" else None,
                market_cap=int(data.get("MarketCapitalization", 0)),
                revenue=int(data.get("RevenueTTM", 0)),
                profit_margin=float(data.get("ProfitMargin", 0)) if data.get("ProfitMargin") != "None" else None,
                week_52_high=float(data.get("52WeekHigh", 0)),
                week_52_low=float(data.get("52WeekLow", 0)),
                eps=float(data.get("EPS", 0)) if data.get("EPS") != "None" else None,
                beta=float(data.get("Beta", 0)) if data.get("Beta") != "None" else None,
                source=self.name
            )

            return DataResult(
                success=True,
                data=fundamentals,
                data_type=DataType.FUNDAMENTALS,
                source=self.name,
                raw=data
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)


class CoinGeckoClient(BaseAPIClient):
    """CoinGecko API client for crypto data."""

    name = "coingecko"

    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.available = True  # Public API

        # Symbol to CoinGecko ID mapping
        self.symbol_map = {
            'btc': 'bitcoin', 'bitcoin': 'bitcoin',
            'eth': 'ethereum', 'ethereum': 'ethereum',
            'sol': 'solana', 'solana': 'solana',
            'doge': 'dogecoin', 'dogecoin': 'dogecoin',
            'xrp': 'ripple', 'ripple': 'ripple',
            'ada': 'cardano', 'cardano': 'cardano',
            'dot': 'polkadot', 'polkadot': 'polkadot',
            'avax': 'avalanche-2', 'avalanche': 'avalanche-2',
            'link': 'chainlink', 'chainlink': 'chainlink',
            'matic': 'matic-network', 'polygon': 'matic-network',
        }

    def _get_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko ID."""
        symbol = symbol.lower().replace('-usd', '').replace('_usd', '')
        return self.symbol_map.get(symbol, symbol)

    def get_quote(self, symbol: str) -> DataResult:
        """Get crypto quote from CoinGecko."""
        try:
            coin_id = self._get_coin_id(symbol)

            response = httpx.get(
                f"{self.base_url}/simple/price",
                params={
                    "ids": coin_id,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                    "include_24hr_vol": "true",
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            if coin_id not in data:
                return DataResult(success=False, error=f"Coin {symbol} not found", source=self.name)

            coin_data = data[coin_id]

            quote = CryptoQuote(
                symbol=symbol.upper(),
                price=coin_data.get("usd"),
                change_percent_24h=coin_data.get("usd_24h_change"),
                market_cap=coin_data.get("usd_market_cap"),
                volume_24h=coin_data.get("usd_24h_vol"),
                source=self.name
            )

            return DataResult(
                success=True,
                data=quote,
                data_type=DataType.CRYPTO,
                source=self.name,
                raw=data
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)

    def get_historical(self, symbol: str, period: str = "30", interval: str = "daily") -> DataResult:
        """Get historical crypto data."""
        try:
            coin_id = self._get_coin_id(symbol)

            response = httpx.get(
                f"{self.base_url}/coins/{coin_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": period,
                },
                timeout=15.0
            )
            response.raise_for_status()
            data = response.json()

            prices = data.get("prices", [])
            formatted = [
                {"timestamp": p[0], "price": p[1]}
                for p in prices[-30:]  # Last 30 data points
            ]

            historical = HistoricalData(
                symbol=symbol,
                period=f"{period}d",
                interval=interval,
                data=formatted,
                source=self.name
            )

            return DataResult(
                success=True,
                data=historical,
                data_type=DataType.HISTORICAL,
                source=self.name,
                raw=data
            )

        except Exception as e:
            return DataResult(success=False, error=str(e), source=self.name)


# === Client Registry ===

_clients: Dict[str, BaseAPIClient] = {}


def get_client(name: str) -> Optional[BaseAPIClient]:
    """Get API client by name."""
    global _clients

    if name not in _clients:
        if name == "yfinance":
            _clients[name] = YFinanceClient()
        elif name == "finnhub":
            _clients[name] = FinnhubClient()
        elif name == "alphavantage":
            _clients[name] = AlphaVantageClient()
        elif name == "coingecko":
            _clients[name] = CoinGeckoClient()

    return _clients.get(name)


def get_all_clients() -> List[BaseAPIClient]:
    """Get all available API clients."""
    clients = []
    for name in ["yfinance", "finnhub", "alphavantage", "coingecko"]:
        client = get_client(name)
        if client and client.available:
            clients.append(client)
    return clients
