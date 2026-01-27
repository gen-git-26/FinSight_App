# agent/state.py
"""
Shared state schema for LangGraph multi-agent system.
"""
from __future__ import annotations

from typing import TypedDict, Literal, Optional, List, Dict, Any
from dataclasses import dataclass, field


# Query types for routing
QueryType = Literal["stock", "crypto", "options", "news", "fundamentals", "comparison", "general"]


@dataclass
class ParsedQuery:
    """Parsed query information."""
    ticker: Optional[str] = None
    additional_tickers: List[str] = field(default_factory=list)
    intent: str = "info"
    query_type: QueryType = "general"
    raw_query: str = ""


@dataclass
class MemoryContext:
    """Memory context for the conversation."""
    user_id: str = "default"
    session_history: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_memory: str = ""


@dataclass
class FetchedData:
    """Data fetched from sources."""
    source: str = ""
    tool_used: str = ""
    raw_data: Any = None
    parsed_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class AnalysisResult:
    """Analysis results."""
    insights: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class AgentState(TypedDict, total=False):
    """
    Shared state passed between all agents in the graph.

    This is the central data structure that flows through the LangGraph.
    Each agent can read from and write to this state.
    """
    # Input
    query: str
    user_id: str

    # Router output
    parsed_query: ParsedQuery
    next_agent: str  # Which agent to route to

    # Memory
    memory: MemoryContext

    # Fetched data (from Fetcher or Crypto agent)
    fetched_data: List[FetchedData]

    # Analysis
    analysis: AnalysisResult

    # Final output
    response: str
    sources: List[str]

    # Error handling
    error: Optional[str]

    # MCP info
    mcp_server_used: str
    tools_available: List[str]
