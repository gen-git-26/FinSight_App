# agent/graph.py
"""
LangGraph Multi-Agent Orchestrator Architecture:
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Router    │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
     ┌──────────┐   ┌──────────┐   ┌──────────┐
     │  Crypto  │   │ Fetcher  │   │ Composer │
     └────┬─────┘   └────┬─────┘   │ (general)│
          │              │         └────┬─────┘
          └──────┬───────┘              │
                 │                      │
                 ▼                      │
          ┌──────────┐                  │
          │ Analyst  │                  │
          └────┬─────┘                  │
               │                        │
               └────────────┬───────────┘
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
    router_node,
    fetcher_node,
    crypto_node,
    analyst_node,
    composer_node,
)


def route_after_router(state: AgentState) -> Literal["crypto", "fetcher", "composer"]:
    """
    Routing function - decides which agent to call after router.

    Based on the 'next_agent' field set by the router node.
    """
    next_agent = state.get("next_agent", "fetcher")

    if next_agent == "crypto":
        return "crypto"
    elif next_agent == "composer":
        return "composer"
    else:
        return "fetcher"


def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow.

    Returns a compiled graph that can be invoked with a query.
    """
    # Create the graph with our state schema
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("fetcher", fetcher_node)
    graph.add_node("crypto", crypto_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("composer", composer_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional routing after router
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "crypto": "crypto",
            "fetcher": "fetcher",
            "composer": "composer"  # For general queries
        }
    )

    # Fetcher and Crypto both go to Analyst
    graph.add_edge("fetcher", "analyst")
    graph.add_edge("crypto", "analyst")

    # Analyst goes to Composer
    graph.add_edge("analyst", "composer")

    # Composer is the end
    graph.add_edge("composer", END)

    return graph.compile()


# Singleton instance
_graph = None


def get_graph() -> StateGraph:
    """Get or create the graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, user_id: str = "default") -> str:
    """
    Run a query through the multi-agent graph.

    Args:
        query: The user's question/request
        user_id: User identifier for memory

    Returns:
        The composed response string
    """
    graph = get_graph()

    # Initial state
    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
    }

    print(f"\n{'='*60}")
    print(f"[Graph] Starting query: {query[:80]}...")
    print(f"{'='*60}")

    # Run the graph
    result = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"[Graph] Completed")
    print(f"{'='*60}\n")

    return result.get("response", "No response generated")


async def run_query_async(query: str, user_id: str = "default") -> str:
    """
    Async version of run_query.

    Uses ainvoke for async execution.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
    }

    result = await graph.ainvoke(initial_state)

    return result.get("response", "No response generated")


# === Streaming Support ===

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
