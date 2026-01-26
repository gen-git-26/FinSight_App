# agent/agent.py 
import os
import dotenv
dotenv.load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from utils.config import load_settings
from tools.tools import (
    finnhub_stock_info,
    finnhub_basic_financials,
    finnhub_financials_as_reported,
    company_overview,
    finnhub_financials_new
)
from tools.answer import answer
from utils.logging import setup_logging
from tools.mcp_bridge import mcp_run
from tools.mcp_router import mcp_auto
from mcp_connection.manager import MCPServer


def build_agent():
    """Build financial agent with smart MCP routing."""
    settings = load_settings()
    if not settings.openai_model:
        raise ValueError("OpenAI model not configured")
    
    model = OpenAIChat(id=settings.openai_model)
    setup_logging()
    
    # Check MCP
    servers = MCPServer.from_env()
    mcp_status = f"MCP Servers: {', '.join(servers.keys())}" if servers else "No MCP servers"
    
    return Agent(
        model=model,
        description="Financial analyst with real-time market access",
        instructions=f"""You are a professional financial analyst.{mcp_status}

        **TOOL USAGE:**
        - mcp_auto: Primary tool for ALL queries (prices, news, options, fundamentals)
        - answer: Use for synthesis of multiple data points
        - Static tools: Fallback only if MCP fails

        **QUERY EXAMPLES:**
        - "Tesla stock price" → mcp_auto
        - "AAPL options" → mcp_auto  
        - "Meta Platforms news" → mcp_auto
        - "financial metrics for Tesla" → mcp_auto
        - "Get the options chain for SPY with expiration date 2024-06-21 for calls" → mcp_auto

        **RESPONSE STYLE:**
        - Direct and concise
        - Focus on key metrics
        - Cite data sources
        - Include risk disclaimers for investments

        The system automatically detects:
        - Company names → Tickers (Tesla → TSLA)
        - Crypto (bitcoin → BTC-USD)
        - Intent (price, news, options, etc.)
        - Dates from queries""",

        add_history_to_messages=False,
        show_tool_calls=False,
        tools=[
            mcp_auto,
            answer,
            mcp_run,
            finnhub_stock_info,
            finnhub_basic_financials,
            finnhub_financials_as_reported,
            company_overview,
            finnhub_financials_new
        ],
        markdown=True,
    )