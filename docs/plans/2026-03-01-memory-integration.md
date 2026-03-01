# Memory Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect MemoryManager, RunCache, and RAG retrieval to the live agent graph so router enriches LLM prompts with user context, fetcher deduplicates A2A API calls, fund_manager persists trading decisions, and analysts see relevant historical data.

**Architecture:** Direct injection — each node imports and calls MemoryManager. A `run_id` UUID is generated at router entry and flows through AgentState to scope RunCache for the entire A2A pipeline. All memory calls degrade gracefully when infrastructure is absent.

**Tech Stack:** `infrastructure.memory_manager.MemoryManager`, `infrastructure.run_cache.RunCache`, `infrastructure.redis_stm.RedisSTM`, LangGraph async nodes, pytest with `unittest.mock`

---

## Task 1: Update AgentState and track_metrics decorator

**Files:**
- Modify: `agent/state.py`
- Modify: `evaluation/metrics.py`
- Test: `tests/test_memory_integration.py`

The `track_metrics` decorator currently only wraps sync functions. `router_node` uses it and must become async — so the decorator needs an async branch first.

### Step 1: Write the failing test

Create `tests/test_memory_integration.py`:

```python
# tests/test_memory_integration.py
import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dotenv
dotenv.load_dotenv()


# === Task 1 tests ===

def test_track_metrics_supports_async():
    """track_metrics decorator must work on async functions."""
    from evaluation.metrics import track_metrics, clear_session_metrics, get_session_metrics

    @track_metrics("test_async_node")
    async def my_async_node(state):
        return {"result": "ok"}

    clear_session_metrics()
    result = asyncio.run(my_async_node({}))
    assert result == {"result": "ok"}
    metrics = get_session_metrics()
    assert len(metrics) == 1
    assert metrics[0].agent_name == "test_async_node"
    assert metrics[0].success is True


def test_agent_state_has_run_id_field():
    """AgentState TypedDict must accept run_id and memory_context."""
    from agent.state import AgentState
    state: AgentState = {
        "query": "test",
        "run_id": "abc-123",
        "memory_context": None,
    }
    assert state["run_id"] == "abc-123"
    assert state["memory_context"] is None
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_memory_integration.py::test_track_metrics_supports_async tests/test_memory_integration.py::test_agent_state_has_run_id_field -v
```

Expected: `FAILED` — `track_metrics` returns a coroutine without awaiting it; `AgentState` has no `run_id`.

### Step 3: Update `evaluation/metrics.py`

Add async branch to `track_metrics`. Find the `track_metrics` function (line ~60) and replace the entire `decorator` inner function with:

```python
def decorator(func: Callable):
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            error = None
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                success = False
                raise
            finally:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                add_metrics(CallMetrics(
                    agent_name=agent_name,
                    latency_ms=latency_ms,
                    success=success,
                    error=error,
                ))
        return async_wrapper
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # existing sync wrapper — keep unchanged
```

Also add `import asyncio` at the top of `evaluation/metrics.py`.

### Step 4: Update `agent/state.py`

Add two fields to `AgentState`:

```python
class AgentState(TypedDict, total=False):
    ...existing fields...
    # Memory integration
    run_id: str           # UUID scoping RunCache for this query run
    memory_context: Any   # MemoryContext from MemoryManager.get_context()
```

### Step 5: Run tests to verify they pass

```bash
pytest tests/test_memory_integration.py::test_track_metrics_supports_async tests/test_memory_integration.py::test_agent_state_has_run_id_field -v
```

Expected: Both PASS.

### Step 6: Verify no regressions

```bash
pytest tests/test_graph_routing.py tests/test_routing.py tests/test_api.py -v
```

Expected: All 14 tests PASS.

### Step 7: Commit

```bash
git add agent/state.py evaluation/metrics.py tests/test_memory_integration.py
git commit -m "feat: add run_id/memory_context to AgentState; track_metrics supports async"
```

---

## Task 2: router_node — async + MemoryManager.get_context()

**Files:**
- Modify: `agent/nodes/router.py`
- Test: `tests/test_memory_integration.py`

### Step 1: Write the failing test

Append to `tests/test_memory_integration.py`:

```python
# === Task 2 tests ===

def test_router_node_emits_run_id_and_memory_context():
    """router_node must return run_id (UUID) and memory_context in state."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from infrastructure.memory_types import MemoryContext

    fake_context = MemoryContext()

    with patch("agent.nodes.router.get_memory_manager") as mock_mgr_factory, \
         patch("agent.nodes.router.httpx.post") as mock_post:

        # Mock MemoryManager
        mock_mgr = AsyncMock()
        mock_mgr.get_context.return_value = fake_context
        mock_mgr_factory.return_value = mock_mgr

        # Mock OpenAI response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{
                    "message": {
                        "content": '{"ticker": "AAPL", "additional_tickers": [], "intent": "price", "query_type": "stock", "next_agent": "fetcher", "is_trading_query": false}'
                    }
                }]
            }
        )
        mock_post.return_value.raise_for_status = MagicMock()

        from agent.nodes.router import router_node
        result = asyncio.run(router_node({"query": "AAPL price", "user_id": "test"}))

    assert "run_id" in result
    assert isinstance(result["run_id"], str)
    assert len(result["run_id"]) == 36  # UUID format
    assert "memory_context" in result
    assert result["memory_context"] is fake_context
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_memory_integration.py::test_router_node_emits_run_id_and_memory_context -v
```

Expected: FAIL — `router_node` is sync, has no `run_id` or `memory_context` in return.

### Step 3: Update `agent/nodes/router.py`

Add these imports at the top:

```python
import uuid
from infrastructure.memory_manager import get_memory_manager
```

Convert `router_node` to async and replace the direct `RedisSTM` usage with `MemoryManager`:

```python
@track_metrics("router")
async def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Router node - analyzes the query and decides routing.
    """
    query = state.get("query", "")
    user_id = state.get("user_id", "default")

    print(f"\n{'='*60}")
    print(f"[Router] Processing: {query}")

    cfg = load_settings()

    # Generate run_id to scope RunCache for this entire query run
    run_id = str(uuid.uuid4())

    # Get enriched memory context via MemoryManager
    manager = get_memory_manager()
    try:
        context = await manager.get_context(
            query=query,
            session_id=user_id,
            user_id=user_id,
            run_id=run_id,
        )
        memory_context = context.to_prompt_context()
    except Exception as e:
        print(f"[Router] Memory fetch failed: {e}")
        context = None
        memory_context = ""
```

Remove the old `stm = get_stm()` block entirely. In the return statement, add the two new fields:

```python
    return {
        "parsed_query": parsed_query,
        "next_agent": next_agent,
        "is_trading_query": is_trading_query,
        "run_id": run_id,
        "memory_context": context,
        "memory": {
            "user_id": user_id,
            "retrieved_memory": memory_context
        }
    }
```

Apply the same `run_id` and `memory_context` fields to the fallback `except` return block too:

```python
    except Exception as e:
        ...
        return {
            "parsed_query": ...,
            "next_agent": ...,
            "is_trading_query": ...,
            "run_id": run_id,
            "memory_context": context,
            "error": str(e)
        }
```

Also remove the unused `from infrastructure.redis_stm import get_stm` import.

### Step 4: Run test to verify it passes

```bash
pytest tests/test_memory_integration.py::test_router_node_emits_run_id_and_memory_context -v
```

Expected: PASS.

### Step 5: Verify no regressions

```bash
pytest tests/test_graph_routing.py tests/test_routing.py tests/test_api.py -v
```

Expected: All 14 tests PASS.

### Step 6: Commit

```bash
git add agent/nodes/router.py tests/test_memory_integration.py
git commit -m "feat: router_node async — get_context() enriches LLM prompt with memory"
```

---

## Task 3: fetcher_node — RunCache deduplication

**Files:**
- Modify: `agent/nodes/fetcher.py`
- Test: `tests/test_memory_integration.py`

### Step 1: Write the failing test

Append to `tests/test_memory_integration.py`:

```python
# === Task 3 tests ===

def test_fetcher_node_uses_run_cache():
    """Second fetch for same ticker+data_type returns cached result without API call."""
    from unittest.mock import patch, MagicMock, AsyncMock
    from agent.state import AgentState, ParsedQuery
    from infrastructure.memory_types import MemoryContext

    # State with a run_id
    state: AgentState = {
        "query": "AAPL price",
        "user_id": "test",
        "run_id": "run-test-123",
        "parsed_query": ParsedQuery(
            ticker="AAPL",
            intent="price",
            query_type="stock",
            raw_query="AAPL price"
        ),
    }

    fake_cached = MagicMock()
    fake_cached.error = None
    fake_cached.source = "run_cache"
    fake_cached.parsed_data = {"price": 190.0}

    with patch("agent.nodes.fetcher.get_memory_manager") as mock_mgr_factory:
        mock_mgr = MagicMock()
        # First call: cache miss; second call: cache hit
        mock_mgr.get_cached_tool.side_effect = [None, fake_cached]
        mock_mgr.cache_tool_result = AsyncMock(return_value=True)
        mock_mgr_factory.return_value = mock_mgr

        with patch("agent.nodes.fetcher.get_fetcher") as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.source = "yfinance"
            mock_result.data = MagicMock(__dict__={"price": 190.0})
            mock_result.raw = None
            mock_result.data_type = MagicMock(value="quote")
            mock_fetcher.fetch = AsyncMock(return_value=mock_result)
            mock_get_fetcher.return_value = mock_fetcher

            from agent.nodes.fetcher import fetcher_node
            # First call — cache miss, should hit API
            asyncio.run(fetcher_node(state))
            first_call_api_count = mock_fetcher.fetch.call_count

            # Second call — cache hit, should NOT hit API again
            asyncio.run(fetcher_node(state))
            second_call_api_count = mock_fetcher.fetch.call_count

    assert first_call_api_count == 1  # API called once
    assert second_call_api_count == 1  # API NOT called again (cache hit)
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_memory_integration.py::test_fetcher_node_uses_run_cache -v
```

Expected: FAIL — `fetcher_node` is sync, has no RunCache logic.

### Step 3: Update `agent/nodes/fetcher.py`

Add import at the top:

```python
from infrastructure.memory_manager import get_memory_manager
```

Replace `def fetcher_node(state: AgentState)` with `async def fetcher_node(state: AgentState)`.

Replace the internal `fetch_all()` + `asyncio.run()` block with a cache-aware version:

```python
    manager = get_memory_manager()
    run_id = state.get("run_id")

    async def fetch_with_cache(ticker: str) -> FetchedData:
        # Check RunCache first
        if run_id:
            cached = manager.get_cached_tool(run_id, data_type.value, ticker)
            if cached:
                print(f"[Fetcher] Cache hit: {ticker}/{data_type.value}")
                return cached

        # Cache miss — fetch from API
        result = await _fetch_ticker_async(fetcher, ticker, data_type)

        # Write to cache on success
        if run_id and not result.error:
            await manager.cache_tool_result(
                run_id, data_type.value, result, ticker=ticker
            )
        return result

    tasks = [fetch_with_cache(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

Remove the old `try/except` block that called `asyncio.run(fetch_all())`.

Do the same for `trading_fetcher_node` (also async, same pattern but calls `fetcher.fetch_comprehensive`):

```python
async def trading_fetcher_node(state: AgentState) -> Dict[str, Any]:
    ...
    run_id = state.get("run_id")
    manager = get_memory_manager()

    # Check cache for comprehensive data
    if run_id:
        cached = manager.get_cached_tool(run_id, "comprehensive", ticker)
        if cached:
            print(f"[Trading Fetcher] Cache hit: {ticker}/comprehensive")
            return cached

    results = await fetcher.fetch_comprehensive(ticker)

    fetched_data = [_convert_result_to_fetched_data(r, ticker) for r in results.values()]
    output = {
        "fetched_data": fetched_data,
        "sources": list(set(r.source for r in fetched_data if not r.error)),
        "ticker": ticker
    }

    # Write to cache
    if run_id:
        await manager.cache_tool_result(run_id, "comprehensive", output, ticker=ticker)

    return output
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_memory_integration.py::test_fetcher_node_uses_run_cache -v
```

Expected: PASS.

### Step 5: Verify no regressions

```bash
pytest tests/test_graph_routing.py tests/test_routing.py tests/test_api.py -v
```

Expected: All 14 tests PASS.

### Step 6: Commit

```bash
git add agent/nodes/fetcher.py tests/test_memory_integration.py
git commit -m "feat: fetcher_node async — RunCache deduplicates A2A data fetches"
```

---

## Task 4: fund_manager_node — store_decision()

**Files:**
- Modify: `agent/nodes/fund_manager.py`
- Test: `tests/test_memory_integration.py`

### Step 1: Write the failing test

Append to `tests/test_memory_integration.py`:

```python
# === Task 4 tests ===

def test_fund_manager_calls_store_decision():
    """fund_manager_node must call store_decision() with ticker and action after deciding."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from agent.nodes.fund_manager import FundManagerDecision
    from agent.state import AgentState, ParsedQuery

    state: AgentState = {
        "query": "Should I buy AAPL?",
        "user_id": "test-user",
        "parsed_query": ParsedQuery(ticker="AAPL", intent="trading", query_type="trading", raw_query="Should I buy AAPL?"),
        "trading_decision": MagicMock(action="buy", conviction=0.8, rationale="Strong momentum",
                                       position_size="5%", stop_loss="8%", take_profit="15%", key_points=[]),
        "risk_assessment": MagicMock(risk_level="medium", risk_score=0.4, approved=True,
                                      concerns=[], position_recommendation="5%", stop_loss_suggestion="8%"),
        "research_report": MagicMock(conviction_score=0.75, consensus="Bullish"),
    }

    with patch("agent.nodes.fund_manager.get_memory_manager") as mock_mgr_factory, \
         patch("agent.nodes.fund_manager.httpx.post") as mock_post:

        mock_mgr = AsyncMock()
        mock_mgr.store_decision.return_value = True
        mock_mgr.store_message.return_value = True
        mock_mgr_factory.return_value = mock_mgr

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{
                    "message": {
                        "content": '{"status": "approved", "final_action": "buy", "final_position_size": "5%", "final_stop_loss": "8%", "final_take_profit": "15%", "modifications": [], "rejection_reason": "", "execution_notes": "", "confidence": 0.8}'
                    }
                }]
            }
        )
        mock_post.return_value.raise_for_status = MagicMock()

        from agent.nodes.fund_manager import fund_manager_node
        result = asyncio.run(fund_manager_node(state))

    # store_decision must have been called with AAPL and buy action
    mock_mgr.store_decision.assert_called_once()
    call_kwargs = mock_mgr.store_decision.call_args
    assert call_kwargs.kwargs["ticker"] == "AAPL"
    assert call_kwargs.kwargs["decision"]["action"] == "buy"

    # store_message must have been called to write to STM
    mock_mgr.store_message.assert_called_once()
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_memory_integration.py::test_fund_manager_calls_store_decision -v
```

Expected: FAIL — `fund_manager_node` is sync, never calls `store_decision`.

### Step 3: Update `agent/nodes/fund_manager.py`

Add imports at the top:

```python
from infrastructure.memory_manager import get_memory_manager
```

Convert `fund_manager_node` to `async def`. After the `decision` is built (after the `except` block that creates the fallback decision), add:

```python
    # Persist decision to memory
    user_id = state.get("user_id", "default")
    manager = get_memory_manager()
    try:
        await manager.store_decision(
            user_id=user_id,
            ticker=ticker,
            query=state.get("query", ""),
            decision={
                "action": decision.final_action,
                "status": decision.status,
                "confidence": decision.confidence,
                "position_size": decision.final_position_size,
                "stop_loss": decision.final_stop_loss,
                "take_profit": decision.final_take_profit,
            },
            sentiment=research_report.conviction_score if research_report else None,
        )
        await manager.store_message(
            session_id=user_id,
            user_id=user_id,
            role="assistant",
            content=f"{decision.final_action.upper()} {ticker} — {decision.status} (confidence: {decision.confidence:.0%})",
        )
    except Exception as e:
        print(f"[Fund Manager] Memory store failed (non-fatal): {e}")
```

This block goes before the `status_emoji` / print block at the end of the function.

### Step 4: Run test to verify it passes

```bash
pytest tests/test_memory_integration.py::test_fund_manager_calls_store_decision -v
```

Expected: PASS.

### Step 5: Verify no regressions

```bash
pytest tests/test_graph_routing.py tests/test_routing.py tests/test_api.py -v
```

Expected: All 14 tests PASS.

### Step 6: Commit

```bash
git add agent/nodes/fund_manager.py tests/test_memory_integration.py
git commit -m "feat: fund_manager_node async — store_decision() persists trading decisions"
```

---

## Task 5: analysts_node — RAG chunks in prompts

**Files:**
- Modify: `agent/nodes/analysts.py`
- Test: `tests/test_memory_integration.py`

### Step 1: Write the failing test

Append to `tests/test_memory_integration.py`:

```python
# === Task 5 tests ===

def test_analysts_node_injects_rag_chunks():
    """analysts_node must include rag_chunks text in the LLM system prompt."""
    from unittest.mock import patch, MagicMock
    from infrastructure.memory_types import MemoryContext
    from agent.state import AgentState, ParsedQuery
    from agent.nodes.analysts import FetchedData

    rag_chunk_text = "AAPL reported record revenue of $120B in Q4 2025."
    memory_context = MemoryContext()
    memory_context.rag_chunks = [{"text": rag_chunk_text, "symbol": "AAPL", "score": 0.95}]

    state: AgentState = {
        "query": "Should I buy AAPL?",
        "user_id": "test",
        "memory_context": memory_context,
        "parsed_query": ParsedQuery(ticker="AAPL", intent="trading", query_type="trading", raw_query="Should I buy AAPL?"),
        "fetched_data": [],
    }

    captured_prompts = []

    def fake_post(url, **kwargs):
        messages = kwargs.get("json", {}).get("messages", [])
        for m in messages:
            if m.get("role") == "system":
                captured_prompts.append(m["content"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"findings": [], "metrics": {}, "recommendation": "neutral", "confidence": 0.5}'}}]
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("agent.nodes.analysts.httpx.post", side_effect=fake_post):
        from agent.nodes.analysts import analysts_node
        analysts_node(state)

    # At least one analyst prompt should contain the RAG chunk text
    assert any(rag_chunk_text in p for p in captured_prompts), \
        f"Expected RAG chunk in prompt. Got prompts: {[p[:100] for p in captured_prompts]}"
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_memory_integration.py::test_analysts_node_injects_rag_chunks -v
```

Expected: FAIL — RAG chunks are not currently injected into analyst prompts.

### Step 3: Update `agent/nodes/analysts.py`

Modify `_run_analyst` to accept an optional `rag_context` parameter:

```python
def _run_analyst(
    analyst_type: str,
    prompt: str,
    data: Dict[str, Any],
    query: str,
    rag_context: str = "",
) -> AnalystReport:
    """Run a single analyst agent."""
    cfg = load_settings()
    system_prompt = prompt + rag_context  # append RAG context to system prompt

    try:
        response = httpx.post(
            ...
            json={
                ...
                "messages": [
                    {"role": "system", "content": system_prompt},  # was: prompt
                    ...
                ],
                ...
            },
            ...
        )
```

In `analysts_node`, extract RAG context from state and pass it to each analyst:

```python
@track_metrics("analysts_team")
def analysts_node(state: AgentState) -> Dict[str, Any]:
    ...
    # Build RAG context string from memory_context in state
    rag_context = ""
    memory_context = state.get("memory_context")
    if memory_context and hasattr(memory_context, "rag_chunks") and memory_context.rag_chunks:
        chunks = "\n---\n".join([
            c.get("text", "")[:400] for c in memory_context.rag_chunks[:3]
        ])
        rag_context = f"\n\nRelevant historical context:\n{chunks}\n"

    # Run all analysts — pass rag_context to each
    fundamental_report = fundamental_analyst(combined_data, query, rag_context=rag_context)
    sentiment_report = sentiment_analyst(combined_data, query, rag_context=rag_context)
    news_report = news_analyst(combined_data, query, rag_context=rag_context)
    technical_report = technical_analyst(combined_data, query, rag_context=rag_context)
```

Update each individual analyst function to accept and forward `rag_context`:

```python
def fundamental_analyst(data: Dict[str, Any], query: str, rag_context: str = "") -> AnalystReport:
    return _run_analyst("fundamental", FUNDAMENTAL_ANALYST_PROMPT, data, query, rag_context)

def sentiment_analyst(data: Dict[str, Any], query: str, rag_context: str = "") -> AnalystReport:
    return _run_analyst("sentiment", SENTIMENT_ANALYST_PROMPT, data, query, rag_context)

def technical_analyst(data: Dict[str, Any], query: str, rag_context: str = "") -> AnalystReport:
    return _run_analyst("technical", TECHNICAL_ANALYST_PROMPT, data, query, rag_context)

def news_analyst(data: Dict[str, Any], query: str, rag_context: str = "") -> AnalystReport:
    return _run_analyst("news", NEWS_ANALYST_PROMPT, data, query, rag_context)
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_memory_integration.py::test_analysts_node_injects_rag_chunks -v
```

Expected: PASS.

### Step 5: Run the full memory integration test suite

```bash
pytest tests/test_memory_integration.py -v
```

Expected: All 6 tests PASS.

### Step 6: Verify no regressions

```bash
pytest tests/test_graph_routing.py tests/test_routing.py tests/test_api.py tests/test_single_analysts.py -v
```

Expected: All tests PASS.

### Step 7: Commit

```bash
git add agent/nodes/analysts.py tests/test_memory_integration.py
git commit -m "feat: analysts_node injects RAG chunks from memory_context into LLM prompts"
```

---

## Checklist

- [ ] Task 1: `AgentState` + async `track_metrics` — 2 tests pass
- [ ] Task 2: `router_node` async + `get_context()` — test passes, run_id in state
- [ ] Task 3: `fetcher_node` async + RunCache — cache hit test passes
- [ ] Task 4: `fund_manager_node` async + `store_decision()` — test passes
- [ ] Task 5: `analysts_node` RAG injection — test passes
- [ ] All 6 memory integration tests pass
- [ ] No regressions in existing test suite
