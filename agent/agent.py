# agent/agent.py
import os
import dotenv
dotenv.load_dotenv()

import sys, os
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
    settings = load_settings()
    if not settings.openai_model:
        raise ValueError("OpenAI model is not set in the configuration.")
    model = OpenAIChat(id=settings.openai_model)
    setup_logging()
    
    # Check available MCP servers for dynamic instructions
    available_servers = MCPServer.from_env()
    mcp_capabilities = []
    
    if "yfinance" in available_servers:
        mcp_capabilities.append("• Yahoo Finance: Real-time stock quotes, historical data, company info")
    if "financial-datasets" in available_servers:
        mcp_capabilities.append("• Financial Datasets: Comprehensive stock & crypto data, historical analysis")
    if "coinmarketcap" in available_servers:
        mcp_capabilities.append("• CoinMarketCap: Cryptocurrency prices and market data")
    
    mcp_instructions = ""
    if mcp_capabilities:
        mcp_instructions = f"""

DYNAMIC MCP CAPABILITIES AVAILABLE:
{chr(10).join(mcp_capabilities)}

INTELLIGENT TOOL SELECTION PROTOCOL:
• For RECENT/LIVE data (prices, quotes, current market): ALWAYS use mcp_auto first
• For HISTORICAL analysis or comparisons: Prefer mcp_auto over static tools
• For CRYPTO queries: ALWAYS use mcp_auto (financial-datasets server)
• For complex multi-data requests: Use mcp_auto + answer combination
• Use static tools (finnhub_*) ONLY as fallback when MCP fails
• Always try mcp_auto before falling back to legacy tools

QUERY ROUTING RULES:
• "current price" / "latest quote" → mcp_auto
• "historical data" / "price history" → mcp_auto  
• "crypto" / "bitcoin" / "ethereum" → mcp_auto (financial-datasets priority)
• "compare stocks" → mcp_auto + answer
• "company analysis" → mcp_auto + company_overview + answer
• "market news" / "recent developments" → mcp_auto first"""
    
    return Agent(
        model=model,
        description="Expert financial analysis and market intelligence agent with dynamic MCP integration",
        instructions=(
            "You are a professional financial expert specializing in real-time market analysis, trading insights, and investment research. "
            
            "CORE DECISION FRAMEWORK:"
            "CRITICAL: Always prioritize MCP tools (mcp_auto) for live, recent, or time-sensitive data"
            "SECONDARY: Use static tools only as fallback or for specific legacy data needs"
            "SYNTHESIS: Combine multiple sources using the 'answer' tool for comprehensive analysis"
            f"{mcp_instructions}"
            
            "RESPONSE EXCELLENCE STANDARDS: "
            "• Provide data source transparency - show which tools were used and why"
            "• Ensure responses are concise, relevant, and actionable"
            "• Use clear, professional language suitable for financial professionals"
            "• Structure responses with clear sections for complex analysis"
            "• Show reasoning for tool selection when relevant"

            "ACCURACY & HONESTY PROTOCOL: "
            "• If MCP tools fail, explicitly state: 'Live data unavailable, using cached/static data'"
            "• Never fabricate data or present outdated information as current"
            "• Acknowledge limitations and specify data sources/ages"
            "• When tool selection differs from optimal, explain why"

            "WORKFLOW OPTIMIZATION:"
            "• For urgent market queries: mcp_auto → immediate response"
            "• For comprehensive analysis: mcp_auto → answer (synthesis)"
            "• For research reports: mcp_auto → static tools → answer"
            
            "Always conclude financial advice with appropriate risk disclaimers regarding market volatility and investment risks."
        ),
        add_history_to_messages=False,
        show_tool_calls=False,
        tools=[
            # MCP tools first (higher priority)
            mcp_auto,      # Primary intelligent router
            mcp_run,       # Manual MCP access
            answer,        # RAG synthesis
            # Static tools as fallback
            finnhub_stock_info,
            finnhub_basic_financials,
            finnhub_financials_as_reported,
            company_overview,
            finnhub_financials_new
        ],
        markdown=True,
    )