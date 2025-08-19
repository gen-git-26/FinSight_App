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
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None

    @staticmethod
    def from_env() -> Dict[str, "MCPServer"]:
        """
        Load servers from MCP_SERVERS (JSON).

        Examples supported:
        1) With args:
           [{"name":"yfinance",
             "command":"/path/to/python",
             "args":["-u","/abs/path/vendors/yahoo-finance-mcp/server.py"],
             "env":{"PYTHONUNBUFFERED":"1"},
             "cwd":"/abs/path/vendors/yahoo-finance-mcp"}]

        2) Single string command (auto-split):
           [{"name":"financial-datasets",
             "command":"/path/to/python -u /abs/path/vendors/financial-datasets-mcp/server.py"}]
        """
        raw = os.getenv("MCP_SERVERS", "").strip()
        if not raw:
            return {}

        try:
            items = json.loads(raw)
        except Exception:
            items = []

        servers: Dict[str, MCPServer] = {}
        for it in items or []:
            name = str(it.get("name") or "").strip()
            if not name:
                continue

            cmd_raw = it.get("command") or ""
            args: Optional[List[str]] = it.get("args")  # may be None
            cwd = it.get("cwd") or None
            env = dict(it.get("env") or {})

            # Default: make sure output is unbuffered unless explicitly set
            env.setdefault("PYTHONUNBUFFERED", "1")

            # Normalize command/args:
            # If args is missing and command is a string with spaces -> split
            if isinstance(cmd_raw, str):
                if (args is None or not isinstance(args, list)) and cmd_raw.strip():
                    parts = shlex.split(cmd_raw)
                    command = parts[0] if parts else ""
                    args = parts[1:] if len(parts) > 1 else []
                else:
                    command = cmd_raw
                    if args is None:
                        args = []
            else:
                # Shouldn't happen in our config, but keep safe fallback
                command = str(cmd_raw)
                if args is None:
                    args = []

            if not command:
                continue

            servers[name] = MCPServer(
                name=name,
                command=command,
                args=args,
                env=env or None,
                cwd=cwd,
            )

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
            raise ValueError(
                f"Unknown MCP server '{server_name}'. Known: {', '.join(self.servers) or 'none'}"
            )

        srv = self.servers[server_name]

        # Build Stdio params with proper fields
        params = StdioServerParameters(
            command=srv.command,
            args=srv.args or [],
            env=srv.env or {},
            cwd=srv.cwd,
        )

        # Debug spawn line 
        print(
            "[MCP spawn]",
            params.command,
            params.args,
            params.cwd,
            {k: v for k, v in (params.env or {}).items()
             if k in ("PYTHONUNBUFFERED", "FINANCIAL_DATASETS_API_KEY")}
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)

                # Flatten result to text/json
                parts: List[str] = []
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
