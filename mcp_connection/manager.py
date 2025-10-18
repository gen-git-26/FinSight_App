# mcp_connection/manager.py
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

DEFAULT_INIT_TIMEOUT = float(os.getenv("MCP_INIT_TIMEOUT_SEC", "3.0"))
DEFAULT_CALL_TIMEOUT = float(os.getenv("MCP_CALL_TIMEOUT_SEC", "5.0"))
DEFAULT_LIST_TIMEOUT = float(os.getenv("MCP_LIST_TIMEOUT_SEC", "4.0"))

@dataclass
class MCPServer:
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]

    @staticmethod
    def from_env() -> Dict[str, "MCPServer"]:
        """
        Expect MCP_SERVERS env as JSON list of:
        {"name":"yfinance","command":"/path/to/python","args":["-u","/path/to/server.py"],"env":{"PYTHONUNBUFFERED":"1"}}
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
    Stdio-based, on-demand MCP client with hard timeouts.
    Each call spins the server and tears it down cleanly.
    """

    def __init__(self):
        self._servers = MCPServer.from_env()

    async def _session(self, server_name: str) -> ClientSession:
        if server_name not in self._servers:
            raise RuntimeError(f"Server '{server_name}' not configured")
        cfg = self._servers[server_name]
        params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None)
        read, write = await stdio_client(params).__aenter__()  # manual to guarantee __aexit__ later
        session = ClientSession(read, write)
        try:
            await asyncio.wait_for(session.initialize(), timeout=DEFAULT_INIT_TIMEOUT)
        except Exception:
            # ensure we close transport on init failure
            try:
                await session.close()
            except Exception:
                pass
            await stdio_client(params).__aexit__(None, None, None)
            raise
        # attach cleaner for caller
        session._stdio_params = params  # type: ignore[attr-defined]
        return session

    async def _close(self, session: ClientSession) -> None:
        try:
            await session.close()
        except Exception:
            pass
        try:
            params = getattr(session, "_stdio_params", None)
            if params is not None:
                await stdio_client(params).__aexit__(None, None, None)
        except Exception:
            pass

    async def _call(self, server_name: str, tool: str, args: Dict[str, Any]) -> Any:
        session = await self._session(server_name)
        try:
            result = await asyncio.wait_for(session.call_tool(tool, args or {}), timeout=DEFAULT_CALL_TIMEOUT)
            # Normalize ToolResponse content
            if hasattr(result, "content"):
                for block in getattr(result, "content", []):
                    typ = getattr(block, "type", "")
                    if typ == "text":
                        return block.text
                    if typ == "json":
                        try:
                            return json.dumps(block.data)
                        except Exception:
                            return block.data
            return str(result)
        finally:
            await self._close(session)

    def call_sync(self, server_name: str, tool: str, args: Dict[str, Any]) -> Any:
        return asyncio.run(self._call(server_name, tool, args))

    async def list_tools(self, server_name: str) -> Dict[str, Any]:
        session = await self._session(server_name)
        try:
            items = await asyncio.wait_for(session.list_tools(), timeout=DEFAULT_LIST_TIMEOUT)
            tools_list: List[Dict[str, Any]] = []
            for t in items:
                if hasattr(t, "model_dump"):
                    tools_list.append(t.model_dump())
                elif isinstance(t, dict):
                    tools_list.append(t)
                else:
                    # best effort normalization
                    name = getattr(t, "name", None) or str(t)
                    tools_list.append({"name": name})
            return {"tools": tools_list}
        finally:
            await self._close(session)

    def list_tools_sync(self, server_name: str) -> Dict[str, Any]:
        return asyncio.run(self.list_tools(server_name))

    # Compatibility no-op for setup scripts that call get_manager().stop_all_servers()
    def stop_all_servers(self) -> None:
        return
