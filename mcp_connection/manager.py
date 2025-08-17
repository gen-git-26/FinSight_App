# mcp/manager.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

@dataclass
class MCPServer:
    name: str
    command: List[str]
    env: Optional[Dict[str, str]] = None

    @staticmethod
    def from_env() -> Dict[str, "MCPServer"]:
        raw = os.getenv("MCP_SERVERS", "[]").strip()
        try:
            cfg = json.loads(raw) if raw else []
        except Exception:
            cfg = []
        servers: Dict[str, MCPServer] = {}
        for item in cfg:
            name = str(item.get("name", "")).strip()
            cmd  = item.get("command") or []
            if name and cmd:
                servers[name] = MCPServer(name=name, command=list(cmd), env=item.get("env") or None)
        return servers

async def _collect_text(result: Any) -> str:
    try:
        content = getattr(result, "content", None) or result.get("content", [])
    except Exception:
        content = []
    parts: List[str] = []
    for c in content or []:
        t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
        if isinstance(t, str) and t.strip():
            parts.append(t)
            continue
        j = getattr(c, "json", None) or (c.get("json") if isinstance(c, dict) else None)
        if j is not None:
            try:
                parts.append(json.dumps(j, ensure_ascii=False))
            except Exception:
                parts.append(str(j))
            continue
        parts.append(str(c))
    return "\n".join(parts) if parts else ""

class MCPManager:
    def __init__(self, servers: Optional[Dict[str, MCPServer]] = None) -> None:
        self.servers = servers or MCPServer.from_env()

    async def list_tools(self, server_name: str) -> List[str]:
        srv = self.servers.get(server_name)
        if not srv:
            return []
        params = StdioServerParameters(command=srv.command, env=srv.env or {})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None) for t in tools or []]

    async def call(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        srv = self.servers.get(server_name)
        if not srv:
            raise ValueError(f"Unknown MCP server '{server_name}'. Known: {', '.join(self.servers) or 'none'}")
        params = StdioServerParameters(command=srv.command, env=srv.env or {})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return await _collect_text(result)
