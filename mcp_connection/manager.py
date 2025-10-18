# mcp_connection/manager.py
from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

@dataclass
class MCPServer:
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]

    @staticmethod
    def from_env() -> Dict[str, "MCPServer"]:
        """
        Read MCP_SERVERS env as JSON list of:
        {
          "name": "yfinance",
          "command": "/path/to/python",
          "args": ["-u", "/path/to/server.py"],
          "env": {"PYTHONUNBUFFERED": "1"}
        }
        """
        cfg = os.getenv("MCP_SERVERS", "[]")
        data = json.loads(cfg)
        out: Dict[str, MCPServer] = {}
        for item in data:
            name = item.get("name")
            if not name:
                continue
            out[name] = MCPServer(
                name=name,
                command=item.get("command", ""),
                args=item.get("args", []),
                env=item.get("env", {}),
            )
        return out

class MCPManager:
    """
    Minimal manager to start a single MCP server process and call tools.
    We use stdio client per call for simplicity.
    """

    def __init__(self):
        self._servers = MCPServer.from_env()

    async def _call(self, server_name: str, tool: str, args: Dict[str, Any]) -> Any:
        if server_name not in self._servers:
            raise RuntimeError(f"Server '{server_name}' not configured")
        cfg = self._servers[server_name]
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env or None,
        )
        async with stdio_client(params) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            try:
                result = await session.call_tool(tool, args or {})
                # result is MCP ToolResponse -> normalize
                if hasattr(result, "content"):
                    # Try to find a text/json block
                    for block in result.content:
                        if getattr(block, "type", "") == "text":
                            return block.text
                        if getattr(block, "type", "") == "json":
                            return json.dumps(block.data)
                return str(result)
            finally:
                await session.close()

    def call_sync(self, server_name: str, tool: str, args: Dict[str, Any]) -> Any:
        return asyncio.run(self._call(server_name, tool, args))

    async def list_tools(self, server_name: str) -> Dict[str, Any]:
        if server_name not in self._servers:
            raise RuntimeError(f"Server '{server_name}' not configured")
        cfg = self._servers[server_name]
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env or None,
        )
        async with stdio_client(params) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            try:
                items = await session.list_tools()
                tools_list: List[Dict[str, Any]] = []
                for t in items:
                    if hasattr(t, "model_dump"):
                        tools_list.append(t.model_dump())
                    elif isinstance(t, dict):
                        tools_list.append(t)
                return {"tools": tools_list}
            finally:
                await session.close()

    def list_tools_sync(self, server_name: str) -> Dict[str, Any]:
        return asyncio.run(self.list_tools(server_name))
