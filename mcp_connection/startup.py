# mcp_connection/startup.py
from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
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

        Supported examples:
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
            args: Optional[List[str]] = it.get("args")
            cwd = it.get("cwd") or None
            env = dict(it.get("env") or {})

            # Ensure unbuffered python output unless explicitly set
            env.setdefault("PYTHONUNBUFFERED", "1")

            # Normalize command and args
            if isinstance(cmd_raw, str):
                if (args is None or not isinstance(args, list)) and cmd_raw.strip():
                    parts = shlex.split(cmd_raw)
                    command = parts[0] if parts else ""
                    args = parts[1:] if len(parts) > 1 else []
                else:
                    command = cmd_raw
                    if args is None:
                        args = []
            elif isinstance(cmd_raw, list):
                # Rare case where command is already a vector
                parts = [str(p) for p in cmd_raw if str(p).strip()]
                command = parts[0] if parts else ""
                if args is None:
                    args = parts[1:]
            else:
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

    # -------- Shared helpers (ENV/CWD) --------
    @staticmethod
    def build_child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Inherit full process environment and overlay any server-specific variables.
        No hard-coded keys here.
        """
        env = os.environ.copy()
        if extra:
            env.update({k: v for k, v in extra.items() if v is not None})
        return env

    @staticmethod
    def resolve_cwd(explicit: Optional[str]) -> str:
        """
        Use explicit cwd if provided, otherwise default to the current project directory.
        """
        return explicit or str(Path.cwd())

    # -------- Factories --------
    @classmethod
    def from_env(cls) -> Dict[str, MCPServer]:
        return MCPServer.from_env()

    # -------- MCP Calls --------
    async def call(self, server_name: str, tool: str, arguments: Dict[str, Any]) -> str:
        if server_name not in self.servers:
            raise ValueError(
                f"Unknown MCP server '{server_name}'. Known: {', '.join(self.servers) or 'none'}"
            )

        srv = self.servers[server_name]

        params = StdioServerParameters(
            command=srv.command,
            args=srv.args or [],
            env=self.build_child_env(srv.env),
            cwd=self.resolve_cwd(srv.cwd),
        )

        # Debug spawn line
        print(
            "[MCP spawn]",
            params.command,
            params.args,
            params.cwd,
            {k: v for k, v in (params.env or {}).items()
             if k in ("PYTHONUNBUFFERED", "FINANCIAL_DATASETS_API_KEY", "OPENAI_API_KEY")}
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)

                # Flatten result to text or JSON
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

    @staticmethod
    def call_sync(server_name: str, tool: str, arguments: Dict[str, Any]) -> str:
        mgr = MCPManager()
        return asyncio.run(mgr.call(server_name, tool, arguments))

    # ---------- Official ListTools over the MCP protocol ----------
    async def list_tools(self, server_name: str) -> Dict[str, Any]:
        if server_name not in self.servers:
            raise ValueError(
                f"Unknown MCP server '{server_name}'. Known: {', '.join(self.servers) or 'none'}"
            )

        srv = self.servers[server_name]

        params = StdioServerParameters(
            command=srv.command,
            args=srv.args or [],
            env=self.build_child_env(srv.env),
            cwd=self.resolve_cwd(srv.cwd),
        )

        # Debug spawn line for list tools
        print(
            "[MCP spawn list_tools]",
            params.command,
            params.args,
            params.cwd,
            {k: v for k, v in (params.env or {}).items()
             if k in ("PYTHONUNBUFFERED", "FINANCIAL_DATASETS_API_KEY", "OPENAI_API_KEY")}
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()

                try:
                    listed = await session.list_tools()
                except Exception:
                    listed = init

                # Normalize to {"tools": [...]}
                if hasattr(listed, "tools"):
                    items = getattr(listed, "tools") or []
                elif isinstance(listed, dict) and "tools" in listed:
                    items = listed.get("tools") or []
                else:
                    items = listed if isinstance(listed, list) else []

                tools_list: List[Dict[str, Any]] = []
                for t in items:
                    if hasattr(t, "model_dump"):
                        tools_list.append(t.model_dump())
                    elif isinstance(t, dict):
                        tools_list.append(t)
                    else:
                        # Fallback to string repr if needed
                        tools_list.append({"name": str(t)})

                return {"tools": tools_list}

    def list_tools_sync(self, server_name: str) -> Dict[str, Any]:
        return asyncio.run(self.list_tools(server_name))
