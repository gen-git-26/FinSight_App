# agent/graph.py
"""
LangGraph Multi-Agent Orchestrator with A2A (Agent-to-Agent) Architecture.

Two main flows:

1. STANDARD FLOW (info queries):
   router → fetcher/crypto → analyst → composer

2. TRADING FLOW (A2A - TradingAgents):
   router → fetcher → analysts_team → researchers → risk_manager → trader → trading_composer

Architecture Diagram:
                         ┌─────────────┐
                         │   START     │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │   Router    │
                         └──────┬──────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         │               ┌──────┴──────┐               │
         │               │  is_trading? │              │
         │               └──────┬──────┘               │
         │                      │                      │
    ┌────┴────┐           ┌─────┴─────┐          ┌─────┴─────┐
    │  Crypto │           │  Fetcher  │          │ Composer  │
    └────┬────┘           └─────┬─────┘          │ (general) │
         │                      │                └─────┬─────┘
         │         ┌────────────┼────────────┐         │
         │         │            │            │         │
         │    [Standard]   [Trading A2A]     │         │
         │         │            │            │         │
         │         ▼            ▼            │         │
         │    ┌────────┐  ┌──────────┐       │         │
         │    │Analyst │  │ Analysts │       │         │
         │    └───┬────┘  │  Team    │       │         │
         │        │       └────┬─────┘       │         │
         │        │            │             │         │
         │        │            ▼             │         │
         │        │       ┌──────────┐       │         │
         │        │       │Researchers│      │         │
         │        │       │Bull/Bear │       │         │
         │        │       └────┬─────┘       │         │
         │        │            │             │         │
         │        │            ▼             │         │
         │        │       ┌──────────┐       │         │
         │        │       │   Risk   │       │         │
         │        │       │ Manager  │       │         │
         │        │       └────┬─────┘       │         │
         │        │            │             │         │
         │        │            ▼             │         │
         │        │       ┌──────────┐       │         │
         │        │       │  Trader  │       │         │
         │        │       └────┬─────┘       │         │
         │        │            │             │         │
         └────────┴────────────┼─────────────┴─────────┘
                               │
                               ▼
                        ┌──────────┐
                        │ Composer │
                        └────┬─────┘
                             │
                             ▼
                        ┌──────────┐
                        │   END    │
                        └──────────┘
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    # Standard flow
    router_node,
    fetcher_node,
    crypto_node,
    analyst_node,
    composer_node,
    # Trading flow (A2A)
    analysts_node,
    researchers_node,
    risk_manager_node,
    trader_node,
    format_trading_response,
)


def route_after_router(state: AgentState) -> Literal["crypto", "fetcher", "trading_fetcher", "composer"]:
    """
    Primary routing function after router.

    Determines if this is:
    - A trading query → trading_fetcher (A2A flow)
    - A crypto query → crypto
    - A general query → composer
    - Other → fetcher (standard flow)
    """
    next_agent = state.get("next_agent", "fetcher")
    is_trading = state.get("is_trading_query", False)

    if is_trading or next_agent == "trading":
        return "trading_fetcher"
    elif next_agent == "crypto":
        return "crypto"
    elif next_agent == "composer":
        return "composer"
    else:
        return "fetcher"


def route_after_fetcher(state: AgentState) -> Literal["analyst", "analysts_team"]:
    """
    Route after fetcher based on query type.

    Trading queries go to analysts_team (TradingAgents).
    Standard queries go to simple analyst.
    """
    is_trading = state.get("is_trading_query", False)

    if is_trading:
        return "analysts_team"
    return "analyst"


def trading_composer_node(state: AgentState) -> dict:
    """
    Special composer for trading queries.

    Uses the formatted trading response from the trader node.
    """
    response = format_trading_response(state)

    return {
        "response": response,
        "sources": state.get("sources", [])
    }


def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow with A2A support.

    Returns a compiled graph that handles both standard and trading queries.
    """
    graph = StateGraph(AgentState)

    # === Add all nodes ===

    # Entry point
    graph.add_node("router", router_node)

    # Standard flow nodes
    graph.add_node("fetcher", fetcher_node)
    graph.add_node("crypto", crypto_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("composer", composer_node)

    # Trading flow nodes (A2A - TradingAgents)
    graph.add_node("trading_fetcher", fetcher_node)  # Same fetcher, different path
    graph.add_node("analysts_team", analysts_node)
    graph.add_node("researchers", researchers_node)
    graph.add_node("risk_manager", risk_manager_node)
    graph.add_node("trader", trader_node)
    graph.add_node("trading_composer", trading_composer_node)

    # === Set entry point ===
    graph.set_entry_point("router")

    # === Routing from router ===
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "crypto": "crypto",
            "fetcher": "fetcher",
            "trading_fetcher": "trading_fetcher",
            "composer": "composer"
        }
    )

    # === Standard flow edges ===
    graph.add_edge("fetcher", "analyst")
    graph.add_edge("crypto", "analyst")
    graph.add_edge("analyst", "composer")
    graph.add_edge("composer", END)

    # === Trading flow edges (A2A) ===
    graph.add_edge("trading_fetcher", "analysts_team")
    graph.add_edge("analysts_team", "researchers")
    graph.add_edge("researchers", "risk_manager")
    graph.add_edge("risk_manager", "trader")
    graph.add_edge("trader", "trading_composer")
    graph.add_edge("trading_composer", END)

    return graph.compile()


# Singleton instance
_graph = None


def get_graph() -> StateGraph:
    """Get or create the graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph():
    """Reset the graph singleton (useful for testing)."""
    global _graph
    _graph = None


def run_query(query: str, user_id: str = "default") -> str:
    """
    Run a query through the multi-agent graph.

    Automatically routes to:
    - TradingAgents flow for trading questions
    - Standard flow for info queries
    - Crypto flow for crypto queries

    Args:
        query: The user's question/request
        user_id: User identifier for memory

    Returns:
        The composed response string
    """
    graph = get_graph()

    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
    }

    print(f"\n{'='*60}")
    print(f"[Graph] Starting query: {query[:80]}...")
    print(f"{'='*60}")

    result = graph.invoke(initial_state)

    is_trading = result.get("is_trading_query", False)
    print(f"\n{'='*60}")
    print(f"[Graph] Completed (A2A Trading: {is_trading})")
    print(f"{'='*60}\n")

    return result.get("response", "No response generated")


async def run_query_async(query: str, user_id: str = "default") -> str:
    """
    Async version of run_query.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
    }

    result = await graph.ainvoke(initial_state)

    return result.get("response", "No response generated")


async def stream_query(query: str, user_id: str = "default"):
    """
    Stream the query execution, yielding intermediate states.

    Useful for showing progress in the UI.

    Yields:
        Tuple of (node_name, state_update)
    """
    graph = get_graph()

    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
    }

    async for event in graph.astream(initial_state):
        for node_name, state_update in event.items():
            yield node_name, state_update


# === CLI Interface ===

def main():
    """CLI interface for testing the graph."""
    import sys

    print("\n" + "="*60)
    print("FinSight Multi-Agent System (A2A)")
    print("="*60)
    print("\nQuery Types:")
    print("- Trading: 'Should I buy AAPL?' → TradingAgents flow")
    print("- Info: 'AAPL stock price' → Standard flow")
    print("- Crypto: 'Bitcoin price' → Crypto flow")
    print("="*60 + "\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ")

    response = run_query(query)
    print("\n" + "="*60)
    print("RESPONSE:")
    print("="*60)
    print(response)


if __name__ == "__main__":
    main()
