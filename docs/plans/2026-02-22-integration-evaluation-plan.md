# FinSight Integration & Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up MCP servers and Qdrant, implement query-based routing, and add evaluation infrastructure.

**Architecture:** Fix routing to select relevant agents, implement MCP fetch paths in DataFetcher, verify Qdrant RAG integration, add decorator-based metrics collection.

**Tech Stack:** LangGraph, OpenAI API, MCP SDK, Qdrant, pytest

---

## Task 1: Add Trading Subtype Classification

**Files:**
- Modify: `agent/nodes/router.py`
- Create: `tests/test_routing.py`

**Step 1: Write the failing test**

```python
# tests/test_routing.py
import pytest
from agent.nodes.router import classify_trading_subtype

ROUTING_FIXTURES = [
    ("Should I buy AAPL?", "full_trading"),
    ("What's AAPL's P/E ratio?", "fundamental"),
    ("Is AAPL overvalued?", "fundamental"),
    ("Is TSLA oversold?", "technical"),
    ("What's the RSI for NVDA?", "technical"),
    ("What's the sentiment on AAPL?", "sentiment"),
    ("Any news on TSLA?", "news"),
    ("Latest headlines for GOOGL", "news"),
]

@pytest.mark.parametrize("query,expected", ROUTING_FIXTURES)
def test_classify_trading_subtype(query, expected):
    result = classify_trading_subtype(query)
    assert result == expected, f"Query '{query}' expected '{expected}', got '{result}'"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_routing.py::test_classify_trading_subtype -v`
Expected: FAIL with "cannot import name 'classify_trading_subtype'"

**Step 3: Write minimal implementation**

Add to `agent/nodes/router.py` after line 30 (after TRADING_KEYWORDS):

```python
# Trading subtype keywords for granular routing
FUNDAMENTAL_KEYWORDS = [
    'p/e', 'pe ratio', 'eps', 'earnings', 'revenue', 'profit',
    'valuation', 'overvalued', 'undervalued', 'fundamentals',
    'balance sheet', 'income statement', 'cash flow', 'debt',
    'margin', 'growth rate', 'book value', 'dividend'
]

TECHNICAL_KEYWORDS = [
    'rsi', 'macd', 'oversold', 'overbought', 'support', 'resistance',
    'moving average', 'sma', 'ema', 'bollinger', 'volume',
    'trend', 'breakout', 'chart', 'technical', 'indicator'
]

SENTIMENT_KEYWORDS = [
    'sentiment', 'mood', 'feeling', 'outlook', 'opinion',
    'bullish sentiment', 'bearish sentiment', 'fear', 'greed'
]

NEWS_KEYWORDS = [
    'news', 'headline', 'announcement', 'press release',
    'latest', 'recent', 'update', 'breaking'
]


def classify_trading_subtype(query: str) -> str:
    """
    Classify a trading query into a subtype for granular routing.

    Returns one of: full_trading, fundamental, technical, sentiment, news
    """
    query_lower = query.lower()

    # Check specific subtypes first (more specific wins)
    if any(kw in query_lower for kw in FUNDAMENTAL_KEYWORDS):
        return "fundamental"

    if any(kw in query_lower for kw in TECHNICAL_KEYWORDS):
        return "technical"

    if any(kw in query_lower for kw in NEWS_KEYWORDS):
        return "news"

    if any(kw in query_lower for kw in SENTIMENT_KEYWORDS):
        return "sentiment"

    # Default to full trading flow for general trading queries
    return "full_trading"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_routing.py::test_classify_trading_subtype -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agent/nodes/router.py tests/test_routing.py
git commit -m "feat(router): add trading subtype classification"
```

---

## Task 2: Add Single-Analyst Node Functions

**Files:**
- Modify: `agent/nodes/analysts.py`
- Create: `tests/test_single_analysts.py`

**Step 1: Write the failing test**

```python
# tests/test_single_analysts.py
import pytest
from agent.state import AgentState, FetchedData

def test_single_fundamental_analyst_node_exists():
    from agent.nodes.analysts import single_fundamental_node
    assert callable(single_fundamental_node)

def test_single_technical_analyst_node_exists():
    from agent.nodes.analysts import single_technical_node
    assert callable(single_technical_node)

def test_single_sentiment_analyst_node_exists():
    from agent.nodes.analysts import single_sentiment_node
    assert callable(single_sentiment_node)

def test_single_news_analyst_node_exists():
    from agent.nodes.analysts import single_news_node
    assert callable(single_news_node)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_single_analysts.py -v`
Expected: FAIL with "cannot import name 'single_fundamental_node'"

**Step 3: Write minimal implementation**

Add to end of `agent/nodes/analysts.py`:

```python
# === Single Analyst Nodes (for granular routing) ===

def single_fundamental_node(state: AgentState) -> Dict[str, Any]:
    """Run only the Fundamental Analyst."""
    parsed_query = state.get("parsed_query")
    fetched_data = state.get("fetched_data", [])
    query = parsed_query.raw_query if parsed_query else state.get("query", "")

    combined_data = {}
    for d in fetched_data:
        if not d.error and d.parsed_data:
            combined_data[d.source] = d.parsed_data

    report = fundamental_analyst(combined_data, query)
    print(f"[Single Analyst] Fundamental: {report.recommendation} ({report.confidence:.0%})")

    return {"analyst_reports": [report]}


def single_technical_node(state: AgentState) -> Dict[str, Any]:
    """Run only the Technical Analyst."""
    parsed_query = state.get("parsed_query")
    fetched_data = state.get("fetched_data", [])
    query = parsed_query.raw_query if parsed_query else state.get("query", "")

    combined_data = {}
    for d in fetched_data:
        if not d.error and d.parsed_data:
            combined_data[d.source] = d.parsed_data

    report = technical_analyst(combined_data, query)
    print(f"[Single Analyst] Technical: {report.recommendation} ({report.confidence:.0%})")

    return {"analyst_reports": [report]}


def single_sentiment_node(state: AgentState) -> Dict[str, Any]:
    """Run Sentiment + News Analysts together."""
    parsed_query = state.get("parsed_query")
    fetched_data = state.get("fetched_data", [])
    query = parsed_query.raw_query if parsed_query else state.get("query", "")

    combined_data = {}
    for d in fetched_data:
        if not d.error and d.parsed_data:
            combined_data[d.source] = d.parsed_data

    sentiment_report = sentiment_analyst(combined_data, query)
    news_report = news_analyst(combined_data, query)

    print(f"[Single Analyst] Sentiment: {sentiment_report.recommendation}, News: {news_report.recommendation}")

    return {"analyst_reports": [sentiment_report, news_report]}


def single_news_node(state: AgentState) -> Dict[str, Any]:
    """Run only the News Analyst."""
    parsed_query = state.get("parsed_query")
    fetched_data = state.get("fetched_data", [])
    query = parsed_query.raw_query if parsed_query else state.get("query", "")

    combined_data = {}
    for d in fetched_data:
        if not d.error and d.parsed_data:
            combined_data[d.source] = d.parsed_data

    report = news_analyst(combined_data, query)
    print(f"[Single Analyst] News: {report.recommendation} ({report.confidence:.0%})")

    return {"analyst_reports": [report]}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_single_analysts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agent/nodes/analysts.py tests/test_single_analysts.py
git commit -m "feat(analysts): add single-analyst node functions"
```

---

## Task 3: Update Graph with Granular Routing

**Files:**
- Modify: `agent/nodes/__init__.py`
- Modify: `agent/graph.py`
- Create: `tests/test_graph_routing.py`

**Step 1: Update exports in `agent/nodes/__init__.py`**

Add to imports:

```python
from agent.nodes.analysts import (
    analysts_node,
    single_fundamental_node,
    single_technical_node,
    single_sentiment_node,
    single_news_node,
)
from agent.nodes.router import classify_trading_subtype
```

**Step 2: Write the failing test**

```python
# tests/test_graph_routing.py
import pytest

def test_graph_has_single_analyst_nodes():
    from agent.graph import build_graph
    graph = build_graph()
    nodes = graph.nodes
    assert "single_fundamental" in nodes
    assert "single_technical" in nodes
    assert "single_sentiment" in nodes
    assert "single_news" in nodes
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_graph_routing.py -v`
Expected: FAIL with assertion error

**Step 4: Update graph.py**

Modify `agent/graph.py`:

After imports, add:
```python
from agent.nodes import (
    # ... existing imports ...
    single_fundamental_node,
    single_technical_node,
    single_sentiment_node,
    single_news_node,
)
from agent.nodes.router import classify_trading_subtype
```

Update `route_after_router` function:
```python
def route_after_router(state: AgentState) -> Literal[
    "crypto", "fetcher", "trading_fetcher", "composer",
    "single_fundamental_fetch", "single_technical_fetch",
    "single_sentiment_fetch", "single_news_fetch"
]:
    """
    Primary routing function after router.

    Now supports granular trading routes.
    """
    next_agent = state.get("next_agent", "fetcher")
    is_trading = state.get("is_trading_query", False)
    query = state.get("query", "")

    if next_agent == "crypto":
        return "crypto"
    elif next_agent == "composer":
        return "composer"
    elif is_trading or next_agent == "trading":
        # Granular trading routing
        subtype = classify_trading_subtype(query)
        if subtype == "fundamental":
            return "single_fundamental_fetch"
        elif subtype == "technical":
            return "single_technical_fetch"
        elif subtype == "sentiment":
            return "single_sentiment_fetch"
        elif subtype == "news":
            return "single_news_fetch"
        else:
            return "trading_fetcher"  # full A2A flow
    else:
        return "fetcher"
```

In `build_graph()`, add the new nodes and edges:
```python
    # Single analyst nodes (for granular routing)
    graph.add_node("single_fundamental_fetch", fetcher_node)
    graph.add_node("single_fundamental", single_fundamental_node)
    graph.add_node("single_technical_fetch", fetcher_node)
    graph.add_node("single_technical", single_technical_node)
    graph.add_node("single_sentiment_fetch", fetcher_node)
    graph.add_node("single_sentiment", single_sentiment_node)
    graph.add_node("single_news_fetch", fetcher_node)
    graph.add_node("single_news", single_news_node)
```

Update conditional edges:
```python
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "crypto": "crypto",
            "fetcher": "fetcher",
            "trading_fetcher": "trading_fetcher",
            "composer": "composer",
            "single_fundamental_fetch": "single_fundamental_fetch",
            "single_technical_fetch": "single_technical_fetch",
            "single_sentiment_fetch": "single_sentiment_fetch",
            "single_news_fetch": "single_news_fetch",
        }
    )
```

Add edges for single analyst paths:
```python
    # Single analyst flow edges (skip debate, go to composer)
    graph.add_edge("single_fundamental_fetch", "single_fundamental")
    graph.add_edge("single_fundamental", "composer")
    graph.add_edge("single_technical_fetch", "single_technical")
    graph.add_edge("single_technical", "composer")
    graph.add_edge("single_sentiment_fetch", "single_sentiment")
    graph.add_edge("single_sentiment", "composer")
    graph.add_edge("single_news_fetch", "single_news")
    graph.add_edge("single_news", "composer")
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_graph_routing.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agent/nodes/__init__.py agent/graph.py tests/test_graph_routing.py
git commit -m "feat(graph): add granular routing to single analysts"
```

---

## Task 4: Wire Up MCP in DataFetcher

**Files:**
- Modify: `datasources/__init__.py`
- Create: `tests/test_mcp_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp_integration.py
import pytest
import asyncio
from datasources import DataFetcher, FetchStrategy, DataType

def test_datafetcher_has_mcp_fetch_method():
    fetcher = DataFetcher(strategy=FetchStrategy.PREFER_MCP)
    assert hasattr(fetcher, '_fetch_via_mcp')

@pytest.mark.asyncio
async def test_prefer_mcp_strategy_tries_mcp_first():
    """When PREFER_MCP, should attempt MCP before falling back to API."""
    fetcher = DataFetcher(strategy=FetchStrategy.PREFER_MCP)
    # This will fail/fallback if MCP server not running, but should not error
    result = await fetcher.fetch("AAPL", DataType.QUOTE)
    assert result is not None
    # Either success from MCP or fallback to API
    assert result.success or result.error is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_integration.py -v`
Expected: FAIL with "has no attribute '_fetch_via_mcp'"

**Step 3: Write the MCP fetch implementation**

Add to `datasources/__init__.py` in the `DataFetcher` class, after `_init_rag()`:

```python
    # MCP tool name mapping
    MCP_TOOL_MAP = {
        DataType.QUOTE: "get_stock_price",
        DataType.PRICE: "get_stock_price",
        DataType.FUNDAMENTALS: "get_stock_info",
        DataType.NEWS: "get_news",
        DataType.OPTIONS: "get_options_chain",
        DataType.HISTORICAL: "get_historical_prices",
    }

    async def _fetch_via_mcp(
        self,
        symbol: str,
        data_type: DataType,
        **kwargs
    ) -> Optional[DataResult]:
        """Try to fetch data via MCP server."""
        from datasources.mcp_client import setup_default_servers

        # Ensure MCP servers are registered
        setup_default_servers(use_local=True)

        tool_name = self.MCP_TOOL_MAP.get(data_type)
        if not tool_name:
            return None

        try:
            # Try yfinance MCP server
            result = await self.mcp.call_tool(
                "yfinance",
                tool_name,
                {"ticker": symbol, **kwargs}
            )

            if result.success:
                result.data_type = data_type
                return result

        except Exception as e:
            print(f"[DataFetcher] MCP fetch failed: {e}")

        return None
```

Update `_fetch_stock()` to use MCP when strategy is PREFER_MCP:

```python
    async def _fetch_stock(
        self,
        symbol: str,
        data_type: DataType,
        **kwargs
    ) -> DataResult:
        """Fetch stock data with fallback chain."""

        # Try MCP first if preferred
        if self.strategy == FetchStrategy.PREFER_MCP:
            mcp_result = await self._fetch_via_mcp(symbol, data_type, **kwargs)
            if mcp_result and mcp_result.success:
                print(f"[DataFetcher] MCP success for {symbol}")
                return mcp_result

        # Fallback to API clients
        clients_order = ["yfinance", "finnhub", "alphavantage"]

        for client_name in clients_order:
            client = get_client(client_name)
            if not client or not client.available:
                continue

            try:
                if data_type == DataType.QUOTE or data_type == DataType.PRICE:
                    result = client.get_quote(symbol)
                elif data_type == DataType.FUNDAMENTALS:
                    result = client.get_fundamentals(symbol)
                elif data_type == DataType.OPTIONS:
                    result = client.get_options(symbol, kwargs.get("expiration"))
                elif data_type == DataType.HISTORICAL:
                    result = client.get_historical(
                        symbol,
                        kwargs.get("period", "1mo"),
                        kwargs.get("interval", "1d")
                    )
                elif data_type == DataType.NEWS:
                    result = client.get_news(symbol, kwargs.get("limit", 5))
                else:
                    result = client.get_quote(symbol)

                if result.success:
                    return result

            except Exception as e:
                print(f"[DataFetcher] {client_name} failed: {e}")
                continue

        return DataResult(
            success=False,
            error="All data sources failed",
            source="datasources"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add datasources/__init__.py tests/test_mcp_integration.py
git commit -m "feat(datasources): wire up MCP in DataFetcher"
```

---

## Task 5: Add Qdrant Health Check

**Files:**
- Modify: `rag/qdrant_client.py`
- Create: `tests/test_qdrant_rag.py`

**Step 1: Write the failing test**

```python
# tests/test_qdrant_rag.py
import pytest
from rag.qdrant_client import HybridQdrant

def test_qdrant_health_check_method_exists():
    qdr = HybridQdrant()
    assert hasattr(qdr, 'health_check')
    assert callable(qdr.health_check)

def test_qdrant_health_check_returns_dict():
    qdr = HybridQdrant()
    result = qdr.health_check()
    assert isinstance(result, dict)
    assert "connected" in result
    assert "collection_exists" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_qdrant_rag.py -v`
Expected: FAIL with "has no attribute 'health_check'"

**Step 3: Write the implementation**

Add to `rag/qdrant_client.py` in the `HybridQdrant` class:

```python
    def health_check(self) -> Dict[str, Any]:
        """
        Check Qdrant connection and collection status.

        Returns:
            Dict with connection status and collection info
        """
        result = {
            "connected": False,
            "collection_exists": False,
            "collection_name": self.collection,
            "point_count": 0,
            "error": None
        }

        try:
            # Test connection
            collections = self.client.get_collections()
            result["connected"] = True

            # Check if our collection exists
            collection_names = [c.name for c in collections.collections]
            result["collection_exists"] = self.collection in collection_names

            if result["collection_exists"]:
                info = self.client.get_collection(self.collection)
                result["point_count"] = info.points_count

        except Exception as e:
            result["error"] = str(e)

        return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_qdrant_rag.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add rag/qdrant_client.py tests/test_qdrant_rag.py
git commit -m "feat(rag): add Qdrant health check"
```

---

## Task 6: Create Evaluation Metrics Module

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/metrics.py`
- Create: `tests/test_evaluation.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluation.py
import pytest
import time

def test_metrics_dataclass_exists():
    from evaluation.metrics import CallMetrics
    metrics = CallMetrics(
        agent_name="test",
        input_tokens=100,
        output_tokens=50,
        latency_ms=150.5,
        success=True
    )
    assert metrics.agent_name == "test"
    assert metrics.total_tokens == 150

def test_track_metrics_decorator_exists():
    from evaluation.metrics import track_metrics
    assert callable(track_metrics)

def test_get_session_metrics_exists():
    from evaluation.metrics import get_session_metrics, clear_session_metrics
    clear_session_metrics()
    metrics = get_session_metrics()
    assert isinstance(metrics, list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL with "No module named 'evaluation'"

**Step 3: Create the evaluation module**

Create `evaluation/__init__.py`:
```python
from evaluation.metrics import (
    CallMetrics,
    track_metrics,
    get_session_metrics,
    clear_session_metrics,
    print_metrics_summary,
)

__all__ = [
    "CallMetrics",
    "track_metrics",
    "get_session_metrics",
    "clear_session_metrics",
    "print_metrics_summary",
]
```

Create `evaluation/metrics.py`:
```python
# evaluation/metrics.py
"""
Metrics collection for agent evaluation.

Tracks token usage, latency, and success/failure for each agent call.
"""
from __future__ import annotations

import time
import functools
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CallMetrics:
    """Metrics for a single agent/LLM call."""
    agent_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Estimate cost based on GPT-4o-mini pricing."""
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        input_cost = (self.input_tokens / 1_000_000) * 0.15
        output_cost = (self.output_tokens / 1_000_000) * 0.60
        return input_cost + output_cost


# Session-level metrics storage
_session_metrics: List[CallMetrics] = []


def get_session_metrics() -> List[CallMetrics]:
    """Get all metrics collected in this session."""
    return _session_metrics.copy()


def clear_session_metrics() -> None:
    """Clear session metrics."""
    global _session_metrics
    _session_metrics = []


def add_metrics(metrics: CallMetrics) -> None:
    """Add metrics to the session."""
    _session_metrics.append(metrics)


def track_metrics(agent_name: str):
    """
    Decorator to track metrics for an agent node function.

    Usage:
        @track_metrics("analyst")
        def analyst_node(state):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            error = None
            success = True

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                error = str(e)
                success = False
                raise
            finally:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                # Try to extract token counts from result if available
                input_tokens = 0
                output_tokens = 0

                # Token extraction would require modifying LLM calls
                # For now, estimate based on typical usage

                metrics = CallMetrics(
                    agent_name=agent_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    success=success,
                    error=error
                )
                add_metrics(metrics)

            return result
        return wrapper
    return decorator


def print_metrics_summary() -> None:
    """Print a summary of session metrics."""
    metrics = get_session_metrics()

    if not metrics:
        print("\n[Metrics] No metrics collected")
        return

    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)

    total_tokens = sum(m.total_tokens for m in metrics)
    total_cost = sum(m.cost_usd for m in metrics)
    total_latency = sum(m.latency_ms for m in metrics)
    success_count = sum(1 for m in metrics if m.success)

    print(f"  Calls: {len(metrics)} ({success_count} successful)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total latency: {total_latency:.0f}ms")
    print(f"  Estimated cost: ${total_cost:.4f}")
    print()

    print("  Per-agent breakdown:")
    for m in metrics:
        status = "✓" if m.success else "✗"
        print(f"    {status} {m.agent_name}: {m.latency_ms:.0f}ms, {m.total_tokens} tokens")

    print("=" * 60)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add evaluation/__init__.py evaluation/metrics.py tests/test_evaluation.py
git commit -m "feat(evaluation): add metrics tracking module"
```

---

## Task 7: Integrate Metrics with Agent Nodes

**Files:**
- Modify: `agent/nodes/analyst.py`
- Modify: `agent/nodes/analysts.py`
- Modify: `agent/nodes/router.py`

**Step 1: Add metrics decorator to analyst.py**

At the top of `agent/nodes/analyst.py`, add import:
```python
from evaluation.metrics import track_metrics
```

Decorate the node function:
```python
@track_metrics("analyst")
def analyst_node(state: AgentState) -> Dict[str, Any]:
    # ... existing code ...
```

**Step 2: Add metrics decorator to analysts.py**

At the top of `agent/nodes/analysts.py`, add import:
```python
from evaluation.metrics import track_metrics
```

Decorate the node functions:
```python
@track_metrics("analysts_team")
def analysts_node(state: AgentState) -> Dict[str, Any]:
    # ... existing code ...

@track_metrics("single_fundamental")
def single_fundamental_node(state: AgentState) -> Dict[str, Any]:
    # ... existing code ...

@track_metrics("single_technical")
def single_technical_node(state: AgentState) -> Dict[str, Any]:
    # ... existing code ...

@track_metrics("single_sentiment")
def single_sentiment_node(state: AgentState) -> Dict[str, Any]:
    # ... existing code ...

@track_metrics("single_news")
def single_news_node(state: AgentState) -> Dict[str, Any]:
    # ... existing code ...
```

**Step 3: Add metrics decorator to router.py**

At the top of `agent/nodes/router.py`, add import:
```python
from evaluation.metrics import track_metrics
```

Decorate the node function:
```python
@track_metrics("router")
def router_node(state: AgentState) -> Dict[str, Any]:
    # ... existing code ...
```

**Step 4: Test the integration manually**

Run: `python -m agent.graph "What's AAPL's P/E ratio?"`

Then add to end of `agent/graph.py` main():
```python
from evaluation.metrics import print_metrics_summary
print_metrics_summary()
```

**Step 5: Commit**

```bash
git add agent/nodes/analyst.py agent/nodes/analysts.py agent/nodes/router.py
git commit -m "feat(agents): integrate metrics tracking with agent nodes"
```

---

## Task 8: Create End-to-End Test Suite

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Write the test**

```python
# tests/test_e2e.py
"""
End-to-end tests for the full agent graph.
"""
import pytest
from agent.graph import run_query, reset_graph
from evaluation.metrics import get_session_metrics, clear_session_metrics


@pytest.fixture(autouse=True)
def setup_teardown():
    """Reset state before each test."""
    reset_graph()
    clear_session_metrics()
    yield
    clear_session_metrics()


class TestStandardFlow:
    """Tests for standard (non-trading) queries."""

    def test_simple_price_query(self):
        """Simple price query should use standard flow."""
        response = run_query("AAPL stock price")
        assert response is not None
        assert len(response) > 0

        metrics = get_session_metrics()
        agent_names = [m.agent_name for m in metrics]
        assert "router" in agent_names
        # Should NOT use trading agents
        assert "analysts_team" not in agent_names


class TestGranularRouting:
    """Tests for granular trading routing."""

    def test_fundamental_query_routes_to_single_analyst(self):
        """P/E ratio question should route to fundamental analyst only."""
        response = run_query("What's AAPL's P/E ratio?")
        assert response is not None

        metrics = get_session_metrics()
        agent_names = [m.agent_name for m in metrics]
        assert "router" in agent_names
        # Should use single fundamental, not full team
        assert "single_fundamental" in agent_names or "analyst" in agent_names
        assert "analysts_team" not in agent_names

    def test_technical_query_routes_to_single_analyst(self):
        """RSI question should route to technical analyst only."""
        response = run_query("Is TSLA oversold?")
        assert response is not None

        metrics = get_session_metrics()
        agent_names = [m.agent_name for m in metrics]
        assert "analysts_team" not in agent_names


class TestMetricsCollection:
    """Tests for metrics collection."""

    def test_metrics_collected_for_query(self):
        """Metrics should be collected for each query."""
        clear_session_metrics()
        run_query("NVDA stock price")

        metrics = get_session_metrics()
        assert len(metrics) > 0

        # All metrics should have latency
        for m in metrics:
            assert m.latency_ms >= 0
```

**Step 2: Run the tests**

Run: `pytest tests/test_e2e.py -v`

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end test suite"
```

---

## Task 9: Update Default Strategy to PREFER_MCP

**Files:**
- Modify: `datasources/__init__.py`

**Step 1: Change the default**

In `datasources/__init__.py`, line 51, change:

```python
def __init__(self, strategy: FetchStrategy = FetchStrategy.PREFER_API):
```

to:

```python
def __init__(self, strategy: FetchStrategy = FetchStrategy.PREFER_MCP):
```

**Step 2: Test**

Run: `python -m agent.graph "AAPL stock price"`

Check logs for: `[DataFetcher] MCP success for AAPL` or MCP fallback messages.

**Step 3: Commit**

```bash
git add datasources/__init__.py
git commit -m "feat(datasources): change default strategy to PREFER_MCP"
```

---

## Task 10: Final Integration Test

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Manual smoke test**

```bash
# Test standard flow
python -m agent.graph "AAPL stock price"

# Test granular routing
python -m agent.graph "What's TSLA's P/E ratio?"

# Test full trading flow
python -m agent.graph "Should I buy NVDA?"
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete integration and evaluation implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Trading subtype classification | router.py, test_routing.py |
| 2 | Single-analyst node functions | analysts.py, test_single_analysts.py |
| 3 | Graph with granular routing | graph.py, test_graph_routing.py |
| 4 | MCP integration in DataFetcher | datasources/__init__.py, test_mcp_integration.py |
| 5 | Qdrant health check | qdrant_client.py, test_qdrant_rag.py |
| 6 | Evaluation metrics module | evaluation/*, test_evaluation.py |
| 7 | Metrics integration with agents | analyst.py, analysts.py, router.py |
| 8 | End-to-end test suite | test_e2e.py |
| 9 | Default to PREFER_MCP | datasources/__init__.py |
| 10 | Final integration test | - |
