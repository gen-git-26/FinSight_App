# datasources/mcp_servers/financial_datasets_server.py
"""
Financial Datasets MCP Server - Local implementation using FastMCP.

Provides access to financial data from the Financial Datasets API:
- Financial statements (income, balance sheet, cash flow)
- Stock prices (current, historical)
- Company news
- Crypto prices
- SEC filings

Requires: FINANCIAL_DATASETS_API_KEY environment variable

Run with: python -m datasources.mcp_servers.financial_datasets_server
"""
from __future__ import annotations

import os
import json
import logging
from typing import Optional
from datetime import datetime, timedelta

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = "https://api.financialdatasets.ai"


# Create the MCP server
mcp = FastMCP(
    "financial-datasets",
    instructions="""
    Financial Datasets MCP Server for comprehensive financial data.

    Available data:
    - Financial statements (income, balance sheet, cash flow)
    - Stock prices (current and historical)
    - Company news
    - Cryptocurrency prices
    - SEC filings (10-K, 10-Q, 8-K)

    Requires FINANCIAL_DATASETS_API_KEY environment variable.
    """
)


async def make_request(endpoint: str, params: dict = None) -> dict:
    """
    Make an authenticated request to the Financial Datasets API.

    Args:
        endpoint: API endpoint (e.g., "/financials/income-statements")
        params: Query parameters

    Returns:
        JSON response or error dict
    """
    api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")

    if not api_key:
        return {"error": "FINANCIAL_DATASETS_API_KEY not set"}

    url = f"{API_BASE_URL}{endpoint}"
    headers = {"X-API-Key": api_key}

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        return {"error": "Request timed out"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ============================================
# Financial Statements Tools
# ============================================

@mcp.tool()
async def get_income_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 4
) -> str:
    """
    Get income statements for a company.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        period: 'annual', 'quarterly', or 'ttm' (trailing twelve months)
        limit: Number of statements to return (default 4)

    Returns:
        JSON string with income statement data
    """
    data = await make_request(
        "/financials/income-statements",
        {"ticker": ticker, "period": period, "limit": limit}
    )
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_balance_sheets(
    ticker: str,
    period: str = "annual",
    limit: int = 4
) -> str:
    """
    Get balance sheets for a company.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        period: 'annual' or 'quarterly'
        limit: Number of statements to return (default 4)

    Returns:
        JSON string with balance sheet data
    """
    data = await make_request(
        "/financials/balance-sheets",
        {"ticker": ticker, "period": period, "limit": limit}
    )
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_cash_flow_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 4
) -> str:
    """
    Get cash flow statements for a company.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        period: 'annual' or 'quarterly'
        limit: Number of statements to return (default 4)

    Returns:
        JSON string with cash flow data
    """
    data = await make_request(
        "/financials/cash-flow-statements",
        {"ticker": ticker, "period": period, "limit": limit}
    )
    return json.dumps(data, indent=2)


# ============================================
# Stock Price Tools
# ============================================

@mcp.tool()
async def get_current_stock_price(ticker: str) -> str:
    """
    Get the current stock price.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)

    Returns:
        JSON string with current price data
    """
    data = await make_request(
        "/prices/snapshot",
        {"ticker": ticker}
    )
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_historical_stock_prices(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "day",
    interval_multiplier: int = 1
) -> str:
    """
    Get historical stock prices.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
        end_date: End date (YYYY-MM-DD). Defaults to today.
        interval: 'minute', 'hour', 'day', 'week', 'month'
        interval_multiplier: Multiplier for interval (e.g., 5 with 'minute' = 5 minutes)

    Returns:
        JSON string with historical price data
    """
    # Default dates
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    data = await make_request(
        "/prices/historical",
        {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "interval_multiplier": interval_multiplier
        }
    )
    return json.dumps(data, indent=2)


# ============================================
# News Tools
# ============================================

@mcp.tool()
async def get_company_news(ticker: str, limit: int = 10) -> str:
    """
    Get recent news articles for a company.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        limit: Maximum number of articles (default 10)

    Returns:
        JSON string with news articles
    """
    data = await make_request(
        "/news",
        {"ticker": ticker, "limit": limit}
    )
    return json.dumps(data, indent=2)


# ============================================
# Crypto Tools
# ============================================

@mcp.tool()
async def get_available_crypto_tickers() -> str:
    """
    Get list of available cryptocurrency tickers.

    Returns:
        JSON string with available crypto tickers
    """
    data = await make_request("/crypto/tickers")
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_current_crypto_price(ticker: str) -> str:
    """
    Get the current cryptocurrency price.

    Args:
        ticker: Crypto symbol (e.g., BTC, ETH, SOL)

    Returns:
        JSON string with current crypto price
    """
    data = await make_request(
        "/crypto/snapshot",
        {"ticker": ticker}
    )
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_historical_crypto_prices(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "day",
    interval_multiplier: int = 1
) -> str:
    """
    Get historical cryptocurrency prices.

    Args:
        ticker: Crypto symbol (e.g., BTC, ETH)
        start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
        end_date: End date (YYYY-MM-DD). Defaults to today.
        interval: 'minute', 'hour', 'day', 'week', 'month'
        interval_multiplier: Multiplier for interval

    Returns:
        JSON string with historical crypto prices
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    data = await make_request(
        "/crypto/historical",
        {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "interval_multiplier": interval_multiplier
        }
    )
    return json.dumps(data, indent=2)


# ============================================
# SEC Filings Tools
# ============================================

@mcp.tool()
async def get_sec_filings(
    ticker: str,
    filing_type: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Get SEC filings for a company.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        filing_type: Filter by type: '10-K', '10-Q', '8-K', etc.
        limit: Maximum number of filings (default 10)

    Returns:
        JSON string with SEC filings data
    """
    params = {"ticker": ticker, "limit": limit}
    if filing_type:
        params["filing_type"] = filing_type

    data = await make_request("/sec/filings", params)
    return json.dumps(data, indent=2)


# ============================================
# Insider Trading Tools
# ============================================

@mcp.tool()
async def get_insider_trades(
    ticker: str,
    limit: int = 20
) -> str:
    """
    Get insider trading data for a company.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        limit: Maximum number of trades (default 20)

    Returns:
        JSON string with insider trades
    """
    data = await make_request(
        "/insider-trades",
        {"ticker": ticker, "limit": limit}
    )
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
