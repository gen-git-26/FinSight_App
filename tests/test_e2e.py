# tests/test_e2e.py
"""End-to-end tests for FinSight multi-agent system."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


class TestRouteAfterRouter:
    """Test the routing logic in route_after_router."""

    def test_crypto_query_routes_to_crypto(self):
        from agent.graph import route_after_router
        state = {
            "next_agent": "crypto",
            "is_trading_query": False,
            "query": "Bitcoin price"
        }
        assert route_after_router(state) == "crypto"

    def test_general_query_routes_to_composer(self):
        from agent.graph import route_after_router
        state = {
            "next_agent": "composer",
            "is_trading_query": False,
            "query": "What is a P/E ratio?"
        }
        assert route_after_router(state) == "composer"

    def test_stock_query_routes_to_fetcher(self):
        from agent.graph import route_after_router
        state = {
            "next_agent": "fetcher",
            "is_trading_query": False,
            "query": "AAPL stock price"
        }
        assert route_after_router(state) == "fetcher"

    def test_fundamental_trading_query_routes_to_single_analyst(self):
        from agent.graph import route_after_router
        state = {
            "next_agent": "trading",
            "is_trading_query": True,
            "query": "What is AAPL's P/E ratio?"
        }
        assert route_after_router(state) == "single_fundamental_fetch"

    def test_technical_trading_query_routes_to_single_analyst(self):
        from agent.graph import route_after_router
        state = {
            "next_agent": "trading",
            "is_trading_query": True,
            "query": "Is TSLA oversold? What's the RSI?"
        }
        assert route_after_router(state) == "single_technical_fetch"

    def test_news_trading_query_routes_to_single_analyst(self):
        from agent.graph import route_after_router
        state = {
            "next_agent": "trading",
            "is_trading_query": True,
            "query": "Any news on MSFT?"
        }
        assert route_after_router(state) == "single_news_fetch"

    def test_sentiment_trading_query_routes_to_single_analyst(self):
        from agent.graph import route_after_router
        state = {
            "next_agent": "trading",
            "is_trading_query": True,
            "query": "What's the sentiment on GOOGL?"
        }
        assert route_after_router(state) == "single_sentiment_fetch"

    def test_full_trading_query_routes_to_trading_fetcher(self):
        from agent.graph import route_after_router
        state = {
            "next_agent": "trading",
            "is_trading_query": True,
            "query": "Should I buy NVDA?"
        }
        assert route_after_router(state) == "trading_fetcher"


class TestMetricsCollection:
    """Test that metrics are collected during graph execution."""

    def test_metrics_cleared_on_new_session(self):
        from evaluation.metrics import clear_session_metrics, get_session_metrics
        clear_session_metrics()
        assert len(get_session_metrics()) == 0

    def test_track_metrics_decorator_adds_to_session(self):
        from evaluation.metrics import (
            track_metrics, get_session_metrics, clear_session_metrics
        )
        clear_session_metrics()

        @track_metrics("test_agent")
        def dummy_function():
            return {"result": "ok"}

        dummy_function()
        metrics = get_session_metrics()
        assert len(metrics) == 1
        assert metrics[0].agent_name == "test_agent"
        assert metrics[0].success is True
        assert metrics[0].latency_ms > 0

    def test_track_metrics_captures_errors(self):
        from evaluation.metrics import (
            track_metrics, get_session_metrics, clear_session_metrics
        )
        clear_session_metrics()

        @track_metrics("failing_agent")
        def failing_function():
            raise ValueError("Test error")

        try:
            failing_function()
        except ValueError:
            pass

        metrics = get_session_metrics()
        assert len(metrics) == 1
        assert metrics[0].agent_name == "failing_agent"
        assert metrics[0].success is False
        assert "Test error" in metrics[0].error

    def test_multiple_agents_tracked_independently(self):
        from evaluation.metrics import (
            track_metrics, get_session_metrics, clear_session_metrics
        )
        clear_session_metrics()

        @track_metrics("agent_1")
        def agent_one():
            return {}

        @track_metrics("agent_2")
        def agent_two():
            return {}

        agent_one()
        agent_two()

        metrics = get_session_metrics()
        assert len(metrics) == 2
        agent_names = [m.agent_name for m in metrics]
        assert "agent_1" in agent_names
        assert "agent_2" in agent_names


class TestGraphStructure:
    """Test the graph structure supports expected flows."""

    def test_standard_flow_path_exists(self):
        """Verify: router → fetcher → analyst → composer → END"""
        from agent.graph import build_graph
        graph = build_graph()
        nodes = graph.nodes

        assert "router" in nodes
        assert "fetcher" in nodes
        assert "analyst" in nodes
        assert "composer" in nodes

    def test_trading_flow_path_exists(self):
        """Verify: trading_fetcher → analysts_team → researchers → trader → risk_manager → fund_manager"""
        from agent.graph import build_graph
        graph = build_graph()
        nodes = graph.nodes

        assert "trading_fetcher" in nodes
        assert "analysts_team" in nodes
        assert "researchers" in nodes
        assert "trader" in nodes
        assert "risk_manager" in nodes
        assert "fund_manager" in nodes
        assert "trading_composer" in nodes

    def test_single_analyst_flow_paths_exist(self):
        """Verify single analyst paths: single_X_fetch → single_X → composer"""
        from agent.graph import build_graph
        graph = build_graph()
        nodes = graph.nodes

        for analyst_type in ["fundamental", "technical", "sentiment", "news"]:
            assert f"single_{analyst_type}_fetch" in nodes
            assert f"single_{analyst_type}" in nodes


class TestDataFetcherIntegration:
    """Test DataFetcher integration with the system."""

    def test_datafetcher_default_strategy_is_prefer_mcp(self):
        from datasources import DataFetcher, FetchStrategy
        fetcher = DataFetcher()
        assert fetcher.strategy == FetchStrategy.PREFER_MCP

    def test_datafetcher_can_be_configured_with_api_strategy(self):
        from datasources import DataFetcher, FetchStrategy
        fetcher = DataFetcher(strategy=FetchStrategy.PREFER_API)
        assert fetcher.strategy == FetchStrategy.PREFER_API


class TestQdrantIntegration:
    """Test Qdrant RAG integration."""

    def test_qdrant_health_check_returns_expected_keys(self):
        from rag.qdrant_client import HybridQdrant
        qdrant = HybridQdrant()
        health = qdrant.health_check()

        assert "connected" in health
        assert "collection_exists" in health
        assert "collection_name" in health
        assert "point_count" in health
        assert "error" in health
