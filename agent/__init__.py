# agent/__init__.py
"""
FinSight Multi-Agent System

Two modes of operation:
1. Single Agent (legacy): Use build_agent() for the Agno-based agent
2. Multi-Agent (new): Use run_query() for LangGraph-based multi-agent system
"""
from agent.agent import build_agent
from agent.graph import (
    build_graph,
    get_graph,
    run_query,
    run_query_async,
    stream_query,
)
from agent.state import AgentState, ParsedQuery

__all__ = [
    # Legacy single agent
    "build_agent",

    # Multi-agent system
    "build_graph",
    "get_graph",
    "run_query",
    "run_query_async",
    "stream_query",

    # State types
    "AgentState",
    "ParsedQuery",
]
