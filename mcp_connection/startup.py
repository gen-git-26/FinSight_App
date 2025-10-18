# mcp_connection/startup.py
from __future__ import annotations

from typing import Dict

from .manager import MCPManager, MCPServer

_manager: MCPManager | None = None

def get_manager() -> MCPManager:
    """Singleton-style accessor for the MCP manager."""
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager

def _health_check_server(manager: MCPManager, name: str) -> bool:
    """
    Health-check a server by attempting to list its tools via stdio client.
    This spawns the server on-demand and then cleanly tears it down.
    """
    try:
        data = manager.list_tools_sync(name)
        return isinstance(data, dict) and isinstance(data.get("tools"), list)
    except Exception:
        return False

def startup_mcp_servers() -> Dict[str, bool]:
    """
    DO NOT spawn persistent processes here.
    For stdio-based MCP servers, the client should spawn them on-demand.
    We just run a health-check per configured server.
    """
    manager = get_manager()
    servers = MCPServer.from_env()
    results: Dict[str, bool] = {}
    for name in servers.keys():
        results[name] = _health_check_server(manager, name)
    return results

def stop_all_servers() -> None:
    """
    No-op for stdio flow — nothing persistent was started here.
    Exists for compatibility with setup scripts that call it.
    """
    return
