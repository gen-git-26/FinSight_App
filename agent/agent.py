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
)
from tools.answer import answer
from utils.logging import setup_logging
from tools.mcp_bridge import mcp_run
from tools.mcp_router import mcp_auto

def build_agent():
    settings = load_settings()
    if not settings.openai_model:
        raise ValueError("OpenAI model is not set in the configuration.")
    model = OpenAIChat(id=settings.openai_model)
    setup_logging()
    return Agent(
            model=model,
            description="Expert financial analysis and market intelligence agent",
            instructions=(
                "You are a professional financial expert specializing in real-time market analysis, trading insights, and investment research. "
    
                "TOOL USAGE PRIORITIES: "
                "• Always prioritize live news/data MCP tools when users request recent, current, or time-sensitive information "
                "• Use real-time market data tools for price quotes, volume, and technical indicators "
                "• Access breaking news tools for market-moving events and announcements "
    
                "TRANSPARENCY REQUIREMENTS: "
                "• Always explicitly show which tools were used in your analysis "
                "• Include precise timestamps for all data sources "
                "• Use this citation format: [server:tool • YYYY-MM-DD HH:MM UTC] "
                "• Make tool usage visible to build user confidence in data freshness "
    
                "RESPONSE STANDARDS: "
                "• Provide coherent, evidence-based analysis backed by current market data "
                "• Structure responses with clear sections when delivering comprehensive reports "
                "• Use bullet points for better readability and actionable insights "
                "• Include inline citations for all claims and data points "
    
                "HONESTY PROTOCOL: "
                "• If information is not available or tools fail, clearly state: 'I don't know' or 'I can't find that information' "
                "• Never fabricate data or provide outdated information as current "
                "• Acknowledge data limitations and specify information cutoff dates when relevant "
    
                "Always conclude financial advice with appropriate risk disclaimers regarding market volatility and investment risks."
            ),

        add_history_to_messages=False,
        show_tool_calls=False,
        tools=[
            finnhub_stock_info,
            finnhub_basic_financials,
            finnhub_financials_as_reported,
            company_overview,
            answer,
            mcp_run,
            mcp_auto,
        ],
        markdown=True,
    )