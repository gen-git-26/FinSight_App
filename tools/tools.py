# tools/tools.py
# This file contains various tools for financial data retrieval and analysis.

import asyncio
import json
import websockets
import os
import finnhub
import time
import requests
from agno.tools import tool
from agno.tools.yfinance import YFinanceTools
from utils.config import load_settings


async def _with_fallback(func, query: str, *args, **kwargs):
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        # Fallback logic can be added here if needed
        docs = await retrieve(query, filters=None)
        answer, used = await rerank_and_summarize(query, docs)
        return {"error": str(e), "answer": answer, "snippets": used}


@tool(
    name="stock_prices",
    description="Provides real-time data ",
)
def finnhub_stock_info(query):
    """Provides real-time data."""
    try:
        settings = load_settings()
        finnhub_client = finnhub.Client(api_key=settings.finnhub_api_key)
        info = finnhub_client.quote(query)
        return str(info)
    except Exception as e:
        return f"Error: {e}"

@tool(
    name="basic_financials",
    description="Get company basic financials for a specific metric (e.g., 'price', 'valuation', 'growth', 'margin'). Requires ticker and metric as input.",
)
def finnhub_basic_financials(ticker: str, metric: str):
    """Provides basic financial data for a given stock ticker and metric."""
    try:
        settings = load_settings()
        finnhub_client = finnhub.Client(api_key=settings.finnhub_api_key)
        info = finnhub_client.company_basic_financials(symbol=ticker, metric=all)
        return str(info)
    except Exception as e:
        return f"Error: {e}"

@tool(
    name="financials_as_reported",
    description="Get financials as reported",
)
def finnhub_financials_as_reported(query):
    try:
        settings = load_settings()
        finnhub_client = finnhub.Client(api_key=settings.finnhub_api_key)
        info = finnhub_client.financials_reported(query)
        return str(info)
    except Exception as e:
        return f"Error: {e}"

@tool(
    name="get_company_info",
    description="Get the company information, financial ratios, and other key metrics",
)
def company_overview(ticker):
    settings = load_settings()
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={settings.alphavantage_api_key}'
    try:
        response  = requests.get(url)
        response.raise_for_status()
        data = response.json()
        for fact in data:
            return(f"{fact}: {data[fact]}")
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"



