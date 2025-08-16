import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat

import dotenv
dotenv.load_dotenv()

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_settings
from tools.tools import finnhub_stock_info, finnhub_basic_financials, finnhub_financials_as_reported, company_overview


def build_agent():
    settings = load_settings()
    if not settings.openai_model:
        raise ValueError("OpenAI model is not set in the configuration.")
    model = OpenAIChat(id=settings.openai_model)
    return Agent(
        model=model,
        description="You are a financial analysis agent",
        instructions="show all steps in the response, do not skip any steps, and provide detailed explanations. " \
        "use the tools provided to gather information about stocks and financials. " \
        "your anwer should be comprehensive and include all relevant data." \
        "the user will ask you to get stock information, financials, and company overview." \
        "your response should be in markdown format.",
        max_iterations=5,
        max_tokens=5000,
        add_history_to_messages=False,
        show_tool_calls=False,
        tools=[finnhub_stock_info, finnhub_basic_financials, finnhub_financials_as_reported, company_overview],
        markdown=True,
    )



if __name__ == "__main__":
    agent = build_agent()
    # Clear agent history for testing
    if hasattr(agent, "history"):
        agent.history = []
    print("Agent built successfully:")
    response = agent.run("write a comprehensive report on the stock of tesla  (TSLA), including its financials, stock information, and company overview.")
    print(response)

