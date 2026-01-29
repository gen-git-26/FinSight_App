# agent/__init__.py
"""
FinSight Multi-Agent System

LangGraph-based orchestration with A2A (Agent-to-Agent) architecture:
- Standard Flow: Router → Fetcher/Crypto → Analyst → Composer
- Trading Flow: Router → Fetcher → Analysts Team → Researchers → Trader → Risk Manager → Fund Manager → Composer
"""
from agent.graph import (
    build_graph,
    get_graph,
    run_query,
    run_query_async,
    stream_query,
    reset_graph,
)
from agent.state import AgentState, ParsedQuery

__all__ = [
    # Multi-agent system
    "build_graph",
    "get_graph",
    "run_query",
    "run_query_async",
    "stream_query",
    "reset_graph",

    # State types
    "AgentState",
    "ParsedQuery",
]
