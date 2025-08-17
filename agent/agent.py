# agent.py
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

def build_agent():
    settings = load_settings()
    if not settings.openai_model:
        raise ValueError("OpenAI model is not set in the configuration.")
    model = OpenAIChat(id=settings.openai_model)
    setup_logging()
    return Agent(
        model=model,
        description="Financial analysis agent with Fusion RAG on every tool response.",
        instructions=(
            "Keep answers concise and data-backed. Use tools to fetch data, then the built-in Fusion RAG "
            "will verify and enrich. Prefer bullet points; include tickers and dates when citing."
        ),

        add_history_to_messages=False,
        show_tool_calls=False,
        tools=[
            finnhub_stock_info,
            finnhub_basic_financials,
            finnhub_financials_as_reported,
            company_overview,
            answer
        ],
        markdown=True,
    )

if __name__ == "__main__":
    agent = build_agent()
    if hasattr(agent, "history"):
        agent.history = []
    print("Agent built successfully")
    res = agent.run("what is the current bitcoin price?")
    print("Agent response:")
    print(res)