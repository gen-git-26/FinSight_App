# tools/tools.py
# Financial tools wrapped with Fusion RAG

import json
import finnhub
import requests
from agno.tools import tool

from utils.config import load_settings
from rag.fusion import fuse

# --- Finnhub: quote ---
@tool(name="stock_prices", description="Get near real-time quote for a ticker (open/high/low/close, volume). Input: ticker symbol, e.g., TSLA.")
@fuse(tool_name="finnhub_quote", doc_type="quote")
def finnhub_stock_info(ticker: str):
    settings = load_settings()
    client = finnhub.Client(api_key=settings.finnhub_api_key)
    info = client.quote(ticker)
    return info  # dict

# --- Finnhub: basic financials ---
@tool(name="basic_financials", description="Get company basic financials for a metric. Inputs: ticker, metric (price/valuation/growth/margin).")
@fuse(tool_name="finnhub_basic_financials", doc_type="basic_financials")
def finnhub_basic_financials(ticker: str, metric: str):
    settings = load_settings()
    client = finnhub.Client(api_key=settings.finnhub_api_key)
    info = client.company_basic_financials(symbol=ticker, metric=metric)
    return info  # dict

# --- Finnhub: financials as reported ---
@tool(name="financials_as_reported", description="Get financials as reported for a ticker (Finhub). Input: ticker symbol.")
@fuse(tool_name="finnhub_financials", doc_type="as_reported")
def finnhub_financials_as_reported(ticker: str):
    settings = load_settings()
    client = finnhub.Client(api_key=settings.finnhub_api_key)
    info = client.financials_reported(symbol=ticker)
    return info  # dict

# --- AlphaVantage: company overview ---
@tool(name="get_company_info", description="AlphaVantage company overview (ratios & profile). Input: ticker symbol.")
@fuse(tool_name="alpha_overview", doc_type="overview")
def company_overview(ticker: str):
    settings = load_settings()
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={settings.alphavantage_api_key}'
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data  # dict

# --- Finhub financials new ---
@tool(name="finnhub_financials_new", description="Get financials as reported for a ticker (Finhub). Input: ticker symbol.")
@fuse(tool_name="finnhub_financials_new", doc_type="as_reported")
def finnhub_financials_new(query: str):
    settings = load_settings()
    client = finnhub.Client(api_key=settings.finnhub_api_key)
    info = client.general_news(query,'general', min_id=0)
    return str(info)