import asyncio
import json
import websockets
import os
import finnhub
import time
import requests
from langchain.schema import Document
from agno.tools import tool
from agno.tools.yfinance import YFinanceTools
from utils.config import load_settings




@tool(
    name="stock_info",
    description="Provides real-time data for a given stock ticker"
)
def stock_info_tool(ticker):
    try:
        return YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                company_info=True,
                company_news=True,
        ),
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="stock_prices",
    description="Provides real-time data ",
)
def finnhub_stock_info(query):
    """Provides real-time data."""
    try:
        settings = load_settings()
        time.sleep(2)
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
        time.sleep(2)
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
        time.sleep(2)
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



ALPACA_KEY = os.getenv("ALPACA_KEY", "YOUR_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "YOUR_SECRET")

async def _stream_alpaca_news(query: str, timeout: int = 8):
    settings = load_settings()
    uri = "wss://stream.data.alpaca.markets/v1beta1/news"
    results = []

    async with websockets.connect(uri) as ws:
        # Authenticate
        await ws.send(json.dumps({"action": "auth", "key": settings.alpaca_key, "secret": settings.alpaca_secret}))
        await ws.recv()

        # Subscribe to all news
        await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))

        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                if msg.get("T") == "n":
                    headline = msg.get("headline", "")
                    summary = msg.get("summary", "")
                    symbols = msg.get("symbols", [])
                    url = msg.get("url", "")

                    if query.lower() in headline.lower() or \
                       query.lower() in summary.lower() or \
                       query.upper() in symbols:
                        results.append(Document(
                            page_content=summary,
                            metadata={
                                "headline": headline,
                                "symbols": symbols,
                                "source": "Alpaca Real-Time News",
                                "url": url
                            }
                        ))
                        if len(results) >= 3:
                            break
        except asyncio.TimeoutError:
            pass

    return results

@tool(
    name="get_real_time_news",
    description="Fetch real-time news from Alpaca by keyword, ticker, or query. Returns structured documents."
)
def real_time_news(query: str):
    return asyncio.run(_stream_alpaca_news(query))


# Custom Retriever for Fusion
class RealTimeNewsRetriever:
    def __init__(self):
        pass

    def get_relevant_documents(self, query: str):
        return asyncio.run(_stream_alpaca_news(query))