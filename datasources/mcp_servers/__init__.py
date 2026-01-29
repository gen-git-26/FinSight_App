# datasources/mcp_servers/__init__.py
"""
Local MCP Servers for FinSight.

These servers can be run locally without npm dependencies.

Usage:
    # Run Yahoo Finance server
    python -m datasources.mcp_servers.yfinance_server

    # Or register and connect via MCPClient
    from datasources.mcp_client import register_mcp_server

    register_mcp_server(
        name="yfinance-local",
        command="python",
        args=["-m", "datasources.mcp_servers.yfinance_server"]
    )
"""
import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
