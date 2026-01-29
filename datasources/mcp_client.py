# datasources/mcp_client.py
"""
MCP (Model Context Protocol) client for financial data servers.

Uses the official MCP SDK to communicate with servers via JSON-RPC over stdio.
"""
from __future__ import annotations

import os
import json
import asyncio
import subprocess
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from datasources.models import DataResult, DataSourceType, DataType


@dataclass
class MCPServerConfig:
    """MCP server configuration."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


class MCPClient:
    """
    Client for connecting to MCP servers using the official MCP SDK.

    MCP servers expose tools that can be called via JSON-RPC over stdio.
    """

    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_cache: Dict[str, List[Dict]] = {}
        self._load_servers_from_env()

    def _load_servers_from_env(self):
        """Load MCP server configs from environment."""
        mcp_servers_json = os.getenv("MCP_SERVERS", "{}")
        try:
            servers_config = json.loads(mcp_servers_json)
            for name, config in servers_config.items():
                self.servers[name] = MCPServerConfig(
                    name=name,
                    command=config.get("command", ""),
                    args=config.get("args", []),
                    env=config.get("env", {})
                )
        except json.JSONDecodeError:
            pass

    def register_server(
        self,
        name: str,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None
    ):
        """Register a new MCP server."""
        self.servers[name] = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env or {}
        )
        print(f"[MCP] Registered server: {name}")

    def get_available_servers(self) -> List[str]:
        """Get list of registered server names."""
        return list(self.servers.keys())

    @asynccontextmanager
    async def connect(self, server_name: str):
        """
        Connect to an MCP server.

        Usage:
            async with mcp_client.connect("yfinance") as session:
                result = await session.call_tool("get_stock_info", {"ticker": "AAPL"})
        """
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not registered")

        config = self.servers[server_name]

        # Build server parameters
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env={**os.environ, **config.env}
        )

        # Connect to server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                yield session

    async def list_tools(self, server_name: str) -> List[Dict]:
        """List available tools from an MCP server."""
        if server_name in self.tools_cache:
            return self.tools_cache[server_name]

        try:
            async with self.connect(server_name) as session:
                result = await session.list_tools()
                tools = [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema or {}
                    }
                    for tool in result.tools
                ]
                self.tools_cache[server_name] = tools
                return tools
        except Exception as e:
            print(f"[MCP] Failed to list tools from {server_name}: {e}")
            return []

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> DataResult:
        """Call a tool on an MCP server."""
        if server_name not in self.servers:
            return DataResult(
                success=False,
                error=f"Server {server_name} not registered",
                source=server_name,
                source_type=DataSourceType.MCP
            )

        try:
            async with self.connect(server_name) as session:
                result = await session.call_tool(tool_name, arguments)

                # Parse the result
                if result.content:
                    # MCP returns content as a list of content items
                    content = result.content[0] if result.content else None
                    if content and hasattr(content, 'text'):
                        try:
                            data = json.loads(content.text)
                        except json.JSONDecodeError:
                            data = {"raw": content.text}

                        return DataResult(
                            success=True,
                            data=data,
                            source=server_name,
                            source_type=DataSourceType.MCP,
                            raw=content.text
                        )

                return DataResult(
                    success=False,
                    error="Empty response from server",
                    source=server_name,
                    source_type=DataSourceType.MCP
                )

        except FileNotFoundError as e:
            # Server command not found - fallback to API
            print(f"[MCP] Server {server_name} not found, falling back to API")
            return await self._fallback_to_api(server_name, tool_name, arguments)

        except Exception as e:
            print(f"[MCP] Error calling {tool_name} on {server_name}: {e}")
            return DataResult(
                success=False,
                error=str(e),
                source=server_name,
                source_type=DataSourceType.MCP
            )

    async def _fallback_to_api(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> DataResult:
        """Fallback to API clients when MCP server not available."""
        from datasources.api_clients import get_client

        ticker = arguments.get("ticker") or arguments.get("symbol")
        if not ticker:
            return DataResult(
                success=False,
                error="No ticker/symbol provided",
                source=server_name,
                source_type=DataSourceType.MCP
            )

        # Use yfinance as fallback
        client = get_client("yfinance")
        if not client:
            return DataResult(
                success=False,
                error="No API client available",
                source=server_name,
                source_type=DataSourceType.MCP
            )

        tool_lower = tool_name.lower()

        if "price" in tool_lower or "quote" in tool_lower or "info" in tool_lower:
            result = client.get_quote(ticker)
        elif "option" in tool_lower:
            result = client.get_options(ticker, arguments.get("expiration"))
        elif "historical" in tool_lower or "history" in tool_lower:
            result = client.get_historical(ticker, arguments.get("period", "1mo"))
        elif "fundamental" in tool_lower or "financial" in tool_lower:
            result = client.get_fundamentals(ticker)
        elif "news" in tool_lower:
            result = client.get_news(ticker)
        else:
            result = client.get_quote(ticker)

        result.source = f"{server_name} (fallback: {result.source})"
        result.source_type = DataSourceType.MCP

        return result


# === Singleton Instance ===

_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Get the singleton MCP client instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client


def register_mcp_server(
    name: str,
    command: str,
    args: List[str] = None,
    env: Dict[str, str] = None
) -> bool:
    """Helper to register a new MCP server."""
    client = get_mcp_client()
    client.register_server(name, command, args, env)
    return True


# === Pre-configured Servers ===

def setup_default_servers(use_local: bool = True):
    """
    Setup default MCP servers for financial data.

    Args:
        use_local: If True, use local Python-based servers. If False, use npm packages.
    """
    client = get_mcp_client()

    if use_local:
        # Local Yahoo Finance MCP Server (Python-based, no npm needed)
        client.register_server(
            name="yfinance",
            command="python",
            args=["-m", "datasources.mcp_servers.yfinance_server"],
            env={}
        )

        # Local Financial Datasets MCP Server (Python-based)
        api_key = os.getenv("FINANCIAL_DATASETS_API_KEY", "")
        if api_key:
            client.register_server(
                name="financial-datasets",
                command="python",
                args=["-m", "datasources.mcp_servers.financial_datasets_server"],
                env={"FINANCIAL_DATASETS_API_KEY": api_key}
            )
    else:
        # Yahoo Finance MCP (npm package)
        client.register_server(
            name="yfinance",
            command="npx",
            args=["-y", "yahoo-finance-mcp"],
            env={}
        )

        # Financial Datasets MCP (npm package)
        api_key = os.getenv("FINANCIAL_DATASETS_API_KEY", "")
        if api_key:
            client.register_server(
                name="financial-datasets",
                command="npx",
                args=["-y", "@financial-datasets/mcp"],
                env={"FINANCIAL_DATASETS_API_KEY": api_key}
            )

    print(f"[MCP] Registered {len(client.servers)} servers (local={use_local})")
