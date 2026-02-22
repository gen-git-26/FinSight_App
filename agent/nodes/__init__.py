# agent/nodes/__init__.py
"""
LangGraph agent nodes for FinSight multi-agent system.

Architecture:
- Standard Flow: router → fetcher/crypto → analyst → composer
- Trading Flow (A2A): router → fetcher → analysts_team → researchers → trader → risk_manager → fund_manager → composer

Trading Flow Details:
1. Analysts Team (4 parallel): Fundamental, Sentiment, News, Technical
2. Research Team: Bull vs Bear debate (3 rounds)
3. Trader: Makes initial trading decision
4. Risk Management Team: Risky vs Neutral vs Safe debate (3 rounds)
5. Fund Manager: Final approval authority
"""
from .router import router_node
from .fetcher import fetcher_node
from .crypto import crypto_node
from .analyst import analyst_node
from .composer import composer_node

# TradingAgents nodes (A2A)
from .analysts import (
    analysts_node,
    news_analyst,
    single_fundamental_node,
    single_technical_node,
    single_sentiment_node,
    single_news_node,
)
from .router import classify_trading_subtype
from .researchers import researchers_node
from .trader import trader_node, format_trading_response
from .risk_manager import risk_manager_node
from .fund_manager import fund_manager_node, format_final_trading_response

__all__ = [
    # Standard flow
    "router_node",
    "fetcher_node",
    "crypto_node",
    "analyst_node",
    "composer_node",

    # TradingAgents flow (A2A)
    "analysts_node",
    "news_analyst",
    "researchers_node",
    "trader_node",
    "risk_manager_node",
    "fund_manager_node",
    "format_trading_response",
    "format_final_trading_response",

    # Single analyst nodes (granular routing)
    "single_fundamental_node",
    "single_technical_node",
    "single_sentiment_node",
    "single_news_node",
    "classify_trading_subtype",
]
