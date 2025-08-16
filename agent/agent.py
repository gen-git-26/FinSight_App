import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from tools.tools import *
from utils.logging import setup_logging
from tools.tools import RealTimeNewsRetriever

setting = setup_logging()

def build_agent():
    settings = load_settings()
    news_retriever = RealTimeNewsRetriever()
    model = OpenAIChat(id=settings.openai_model)
    return Agent(
        model=model,
        description="You are a financial analysis agent",
        instructions="Use tables to display data",
        add_history_to_messages=True,
        show_tool_calls=True,
        retriever=news_retriever,
        tools=[finnhub_stock_info, finnhub_basic_financials, company_overview],
        markdown=True,
    )




