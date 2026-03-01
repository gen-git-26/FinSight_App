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


# === Task 2 tests ===

def test_router_node_emits_run_id_and_memory_context():
    """router_node must return run_id (UUID) and memory_context in state."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from infrastructure.memory_types import MemoryContext

    fake_context = MemoryContext()

    with patch("agent.nodes.router.get_memory_manager") as mock_mgr_factory, \
         patch("agent.nodes.router.httpx.post") as mock_post:

        mock_mgr = AsyncMock()
        mock_mgr.get_context.return_value = fake_context
        mock_mgr_factory.return_value = mock_mgr

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


# === Task 3 tests ===

def test_fetcher_node_uses_run_cache():
    """Second fetch for same ticker+data_type returns cached result without API call."""
    from unittest.mock import patch, MagicMock, AsyncMock
    from agent.state import AgentState, ParsedQuery

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
        mock_mgr.get_cached_tool.side_effect = [None, fake_cached]
        mock_mgr.cache_tool_result = AsyncMock(return_value=True)
        mock_mgr_factory.return_value = mock_mgr

        with patch("agent.nodes.fetcher.get_fetcher") as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.source = "yfinance"
            from types import SimpleNamespace
            mock_result.data = SimpleNamespace(price=190.0)
            mock_result.raw = None
            mock_result.data_type = MagicMock(value="quote")
            mock_result.error = None
            mock_fetcher.fetch = AsyncMock(return_value=mock_result)
            mock_get_fetcher.return_value = mock_fetcher

            from agent.nodes.fetcher import fetcher_node
            asyncio.run(fetcher_node(state))
            first_call_api_count = mock_fetcher.fetch.call_count

            asyncio.run(fetcher_node(state))
            second_call_api_count = mock_fetcher.fetch.call_count

    assert first_call_api_count == 1   # API called once on cache miss
    assert second_call_api_count == 1  # API NOT called again on cache hit


# === Task 4 tests ===

def test_fund_manager_calls_store_decision():
    """fund_manager_node must call store_decision() with ticker and action after deciding."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from agent.state import AgentState, ParsedQuery

    state: AgentState = {
        "query": "Should I buy AAPL?",
        "user_id": "test-user",
        "parsed_query": ParsedQuery(
            ticker="AAPL", intent="trading",
            query_type="trading", raw_query="Should I buy AAPL?"
        ),
        "trading_decision": MagicMock(
            action="buy", conviction=0.8, rationale="Strong momentum",
            position_size="5%", stop_loss="8%", take_profit="15%", key_points=[]
        ),
        "risk_assessment": MagicMock(
            risk_level="medium", risk_score=0.4, approved=True,
            concerns=[], position_recommendation="5%", stop_loss_suggestion="8%"
        ),
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
        asyncio.run(fund_manager_node(state))

    mock_mgr.store_decision.assert_called_once()
    call_kwargs = mock_mgr.store_decision.call_args
    assert call_kwargs.kwargs["ticker"] == "AAPL"
    assert call_kwargs.kwargs["decision"]["action"] == "buy"
    mock_mgr.store_message.assert_called_once()


# === Task 5 tests ===

def test_analysts_node_injects_rag_chunks():
    """analysts_node must include rag_chunks text in the LLM system prompt."""
    from unittest.mock import patch, MagicMock
    from infrastructure.memory_types import MemoryContext
    from agent.state import AgentState, ParsedQuery

    rag_chunk_text = "AAPL reported record revenue of $120B in Q4 2025."
    memory_context = MemoryContext()
    memory_context.rag_chunks = [{"text": rag_chunk_text, "symbol": "AAPL", "score": 0.95}]

    state: AgentState = {
        "query": "Should I buy AAPL?",
        "user_id": "test",
        "memory_context": memory_context,
        "parsed_query": ParsedQuery(
            ticker="AAPL", intent="trading",
            query_type="trading", raw_query="Should I buy AAPL?"
        ),
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

    assert any(rag_chunk_text in p for p in captured_prompts), \
        f"Expected RAG chunk in prompt. Got {len(captured_prompts)} prompts."
