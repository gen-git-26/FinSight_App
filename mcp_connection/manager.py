# mcp_connection/manager.py
from __future__ import annotations
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

CommandT = Union[str, list[str]]

@dataclass
class MCPServer:
    name: str
    command: CommandT
    env: Optional[Dict[str, str]] = None

    @staticmethod
    def from_env() -> Dict[str, "MCPServer"]:
        """
        Load from MCP_SERVERS (JSON).
        Example:
        MCP_SERVERS=[
          {"name":"yfinance","command":"python -m yahoo_finance_mcp.server"},
          {"name":"financial-datasets",
           "command":"python vendors/financial-datasets-mcp/server.py",
           "env":{"FINANCIAL_DATASETS_API_KEY":"<your key>"}}
        ]
        """
        raw = os.getenv("MCP_SERVERS", "").strip()
        try:
            items = json.loads(raw) if raw else []
        except Exception:
            items = []

        servers: Dict[str, MCPServer] = {}
        for it in items or []:
            name = str(it.get("name") or "").strip()
            cmd: CommandT = it.get("command") or ""
            if isinstance(cmd, list):
                # guard against accidental list("npx") => ['n','p','x']
                if len(cmd) == 1 and len(cmd[0]) == 1:
                    cmd = "".join(cmd)
            if not name or not cmd:
                continue
            servers[name] = MCPServer(name=name, command=cmd, env=it.get("env") or None)
        return servers

class MCPManager:
    def __init__(self, servers: Optional[Dict[str, MCPServer]] = None) -> None:
        self.servers = servers or MCPServer.from_env()

    # keep this because some code referenced MCPManager.from_env()
    @classmethod
    def from_env(cls) -> Dict[str, MCPServer]:
        return MCPServer.from_env()

    async def call(self, server_name: str, tool: str, arguments: Dict[str, Any]) -> str:
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server '{server_name}'. Known: {', '.join(self.servers) or 'none'}")
        srv = self.servers[server_name]
        # StdioServerParameters(command=...) expects **string** in current mcp lib
        if isinstance(srv.command, list):
            cmd = " ".join(srv.command)
        else:
            cmd = srv.command
        params = StdioServerParameters(command=cmd, env=srv.env or {})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)
                # flatten result to text/json
                parts = []
                content = getattr(result, "content", None) or []
                for c in content:
                    t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                    if t:
                        parts.append(t)
                        continue
                    j = getattr(c, "json", None) or (c.get("json") if isinstance(c, dict) else None)
                    if j is not None:
                        parts.append(json.dumps(j, ensure_ascii=False))
                        continue
                    parts.append(str(c))
                return "\n".join(parts) if parts else (json.dumps(result, ensure_ascii=False) if result else "")

    # sync helpers for tool wrappers
    @staticmethod
    def call_sync(server_name: str, tool: str, arguments: Dict[str, Any]) -> str:
        mgr = MCPManager()
        return asyncio.run(mgr.call(server_name, tool, arguments))
