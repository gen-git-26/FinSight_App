# mcp_connection/manager.py
from __future__ import annotations
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "The `mcp` Python package is required. Install with: pip install mcp"
    ) from e

@dataclass
class MCPServer:
    name: str
    command: List[str]
    env: Optional[Dict[str, str]] = None

    @staticmethod
    def from_env() -> Dict[str, "MCPServer"]:
        """Load servers from env var MCP_SERVERS (JSON list of {name, command, env?}).
        Example:
          MCP_SERVERS='[{"name":"yfinance","command":["python","-m","yahoo_finance_mcp.server"]}]'
        """
        raw = os.getenv("MCP_SERVERS", "[]").strip()
        try:
            items = json.loads(raw) if raw else []
        except Exception:
            items = []
        servers: Dict[str, MCPServer] = {}
        for it in items:
            name = str(it.get("name", "")).strip()
            cmd = it.get("command") or []
            if not name or not cmd:
                continue
            servers[name] = MCPServer(name=name, command=list(cmd), env=it.get("env") or None)
        return servers

async def _collect_text(tool_result: Any) -> str:
    """Flatten MCP result content → string (text + json blocks)."""
    if tool_result is None:
        return ""
    try:
        contents = getattr(tool_result, "content", None) or tool_result.get("content", [])
    except Exception:
        contents = []
    parts: List[str] = []
    for c in contents or []:
        t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
        if isinstance(t, str) and t.strip():
            parts.append(t)
            continue
        j = getattr(c, "json", None) or (c.get("json") if isinstance(c, dict) else None)
        if j is not None:
            try:
                parts.append(json.dumps(j, ensure_ascii=False))
                continue
            except Exception:
                parts.append(str(j))
                continue
        parts.append(str(c))
    if not parts:
        try:
            return json.dumps(tool_result, ensure_ascii=False)
        except Exception:
            return str(tool_result)
    return "\n".join(parts)

async def _list_tool_names(server: MCPServer) -> List[str]:
    params = StdioServerParameters(command=server.command, env=server.env or {})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names: List[str] = []
            for t in tools or []:
                name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                if name:
                    names.append(name)
            return names

async def call_mcp_tool_once(server: MCPServer, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Start server (stdio), init session, call tool, return flattened text."""
    params = StdioServerParameters(command=server.command, env=server.env or {})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                tools = await session.list_tools()
                names = [getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) for t in (tools or [])]
                if tool_name not in set(filter(None, names)):
                    raise ValueError(f"Tool '{tool_name}' not found on MCP server '{server.name}'")
            except Exception:
                pass
            result = await session.call_tool(tool_name, arguments)
            return await _collect_text(result)

class MCPManager:
    def __init__(self, servers: Optional[Dict[str, MCPServer]] = None) -> None:
        self.servers = servers or MCPServer.from_env()

    def list_servers(self) -> List[str]:
        return list(self.servers.keys())

    async def list_tool_names(self, server_name: str) -> List[str]:
        if server_name not in self.servers:
            return []
        return await _list_tool_names(self.servers[server_name])

    async def call(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server '{server_name}'. Known: {', '.join(self.servers.keys()) or 'none'}")
        return await call_mcp_tool_once(self.servers[server_name], tool_name, arguments)

# Convenience sync wrappers
def call_sync(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    mgr = MCPManager()
    return asyncio.run(mgr.call(server_name, tool_name, arguments))

def list_tool_names_sync(server_name: str) -> List[str]:
    mgr = MCPManager()
    return asyncio.run(mgr.list_tool_names(server_name))