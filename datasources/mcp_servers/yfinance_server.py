# datasources/mcp_servers/yfinance_server.py
"""
Yahoo Finance MCP Server - Local implementation using FastMCP.

Run with: python -m datasources.mcp_servers.yfinance_server
"""
from __future__ import annotations

import json
from typing import Optional
from enum import Enum

from mcp.server.fastmcp import FastMCP
import yfinance as yf


# Create the MCP server
mcp = FastMCP(
    "yfinance",
    instructions="""
    Yahoo Finance MCP Server for financial data.

    Available tools:
    - get_stock_info: Get current stock information
    - get_stock_price: Get current price quote
    - get_historical_prices: Get historical price data
    - get_options_chain: Get options data
    - get_financials: Get financial statements
    - get_news: Get recent news
    """
)


class FinancialType(str, Enum):
    INCOME = "income_stmt"
    BALANCE = "balance_sheet"
    CASHFLOW = "cashflow"


class Period(str, Enum):
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    MAX = "max"


@mcp.tool()
def get_stock_info(ticker: str) -> str:
    """
    Get comprehensive stock information including company profile and key metrics.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT, TSLA)

    Returns:
        JSON string with stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        result = {
            "symbol": ticker,
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website"),
            "description": info.get("longBusinessSummary", "")[:500],
            "employees": info.get("fullTimeEmployees"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """
    Get current stock price and trading data.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT, TSLA)

    Returns:
        JSON string with price data
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        result = {
            "symbol": ticker,
            "name": info.get("shortName"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "change": info.get("regularMarketChange"),
            "change_percent": info.get("regularMarketChangePercent"),
            "open": info.get("open") or info.get("regularMarketOpen"),
            "high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "previous_close": info.get("previousClose"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "avg_volume": info.get("averageVolume"),
            "market_cap": info.get("marketCap"),
            "bid": info.get("bid"),
            "ask": info.get("ask"),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_historical_prices(
    ticker: str,
    period: Period = Period.ONE_MONTH,
    interval: str = "1d"
) -> str:
    """
    Get historical price data for a stock.

    Args:
        ticker: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        JSON string with historical price data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period.value, interval=interval)

        if hist.empty:
            return json.dumps({"error": "No historical data available", "ticker": ticker})

        # Convert to list of records
        records = []
        for date, row in hist.iterrows():
            records.append({
                "date": str(date),
                "open": round(row["Open"], 2) if row["Open"] else None,
                "high": round(row["High"], 2) if row["High"] else None,
                "low": round(row["Low"], 2) if row["Low"] else None,
                "close": round(row["Close"], 2) if row["Close"] else None,
                "volume": int(row["Volume"]) if row["Volume"] else None,
            })

        return json.dumps({
            "symbol": ticker,
            "period": period.value,
            "interval": interval,
            "data": records[-30:]  # Last 30 records
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_options_chain(
    ticker: str,
    expiration: Optional[str] = None
) -> str:
    """
    Get options chain data for a stock.

    Args:
        ticker: Stock symbol
        expiration: Optional expiration date (YYYY-MM-DD). If not provided, uses nearest.

    Returns:
        JSON string with options chain data
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            return json.dumps({"error": "No options available", "ticker": ticker})

        # Select expiration
        exp_date = expiration if expiration in expirations else expirations[0]
        chain = stock.option_chain(exp_date)

        # Get top calls and puts
        calls = chain.calls.head(10).to_dict('records') if not chain.calls.empty else []
        puts = chain.puts.head(10).to_dict('records') if not chain.puts.empty else []

        # Clean up NaN values
        def clean_record(rec):
            return {k: (None if str(v) == 'nan' else v) for k, v in rec.items()}

        return json.dumps({
            "symbol": ticker,
            "expiration": exp_date,
            "available_expirations": list(expirations[:5]),
            "calls": [clean_record(c) for c in calls],
            "puts": [clean_record(p) for p in puts]
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_financials(
    ticker: str,
    financial_type: FinancialType = FinancialType.INCOME
) -> str:
    """
    Get financial statement data for a stock.

    Args:
        ticker: Stock symbol
        financial_type: Type of financial statement (income_stmt, balance_sheet, cashflow)

    Returns:
        JSON string with financial data
    """
    try:
        stock = yf.Ticker(ticker)

        if financial_type == FinancialType.INCOME:
            data = stock.income_stmt
        elif financial_type == FinancialType.BALANCE:
            data = stock.balance_sheet
        else:
            data = stock.cashflow

        if data.empty:
            return json.dumps({"error": f"No {financial_type.value} data", "ticker": ticker})

        # Get latest column (most recent period)
        latest = data.iloc[:, 0]
        result = {
            "symbol": ticker,
            "type": financial_type.value,
            "period": str(data.columns[0]),
            "data": {str(k): v for k, v in latest.items() if str(v) != 'nan'}
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_news(ticker: str, limit: int = 5) -> str:
    """
    Get recent news articles for a stock.

    Args:
        ticker: Stock symbol
        limit: Maximum number of articles (default 5)

    Returns:
        JSON string with news articles
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []

        articles = []
        for article in news[:limit]:
            articles.append({
                "title": article.get("title"),
                "publisher": article.get("publisher"),
                "link": article.get("link"),
                "published": article.get("providerPublishTime"),
                "type": article.get("type"),
            })

        return json.dumps({
            "symbol": ticker,
            "articles": articles
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_recommendations(ticker: str) -> str:
    """
    Get analyst recommendations for a stock.

    Args:
        ticker: Stock symbol

    Returns:
        JSON string with analyst recommendations
    """
    try:
        stock = yf.Ticker(ticker)
        recs = stock.recommendations

        if recs is None or recs.empty:
            return json.dumps({"error": "No recommendations available", "ticker": ticker})

        # Get recent recommendations
        recent = recs.tail(10).to_dict('records')

        return json.dumps({
            "symbol": ticker,
            "recommendations": recent
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
