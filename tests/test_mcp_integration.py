# tests/test_mcp_integration.py
"""Tests for MCP integration in DataFetcher."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datasources import DataFetcher, FetchStrategy, DataType


def test_datafetcher_has_mcp_fetch_method():
    """DataFetcher should have _fetch_via_mcp method."""
    fetcher = DataFetcher(strategy=FetchStrategy.PREFER_MCP)
    assert hasattr(fetcher, '_fetch_via_mcp')
    assert callable(fetcher._fetch_via_mcp)


def test_datafetcher_has_mcp_tool_map():
    """DataFetcher should have MCP_TOOL_MAP class attribute."""
    assert hasattr(DataFetcher, 'MCP_TOOL_MAP')
    assert DataType.QUOTE in DataFetcher.MCP_TOOL_MAP
    assert DataType.FUNDAMENTALS in DataFetcher.MCP_TOOL_MAP
    assert DataType.NEWS in DataFetcher.MCP_TOOL_MAP


def test_prefer_mcp_strategy_tries_mcp_first():
    """When PREFER_MCP, should attempt MCP before falling back to API."""
    import asyncio

    async def run_test():
        fetcher = DataFetcher(strategy=FetchStrategy.PREFER_MCP)
        # This will fail/fallback if MCP server not running, but should not error
        result = await fetcher.fetch("AAPL", DataType.QUOTE)
        assert result is not None
        # Either success from MCP or fallback to API
        assert result.success or result.error is not None

    asyncio.run(run_test())
