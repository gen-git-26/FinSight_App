# tests/test_graph_routing.py
"""Tests for granular routing in the agent graph."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_graph_has_single_analyst_nodes():
    from agent.graph import build_graph
    graph = build_graph()
    nodes = graph.nodes
    assert "single_fundamental" in nodes
    assert "single_technical" in nodes
    assert "single_sentiment" in nodes
    assert "single_news" in nodes


def test_graph_has_single_analyst_fetch_nodes():
    from agent.graph import build_graph
    graph = build_graph()
    nodes = graph.nodes
    assert "single_fundamental_fetch" in nodes
    assert "single_technical_fetch" in nodes
    assert "single_sentiment_fetch" in nodes
    assert "single_news_fetch" in nodes
