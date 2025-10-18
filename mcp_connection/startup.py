# mcp_connection/startup.py
from __future__ import annotations
from typing import Dict, Any
from mcp_connection.manager import MCPManager, MCPServer

_manager: MCPManager | None = None

def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager

def _health_check_server(manager: MCPManager, name: str) -> bool:
    """
    Health-check a server by attempting to list its tools via stdio client.
    Accept both return shapes: dict {"tools": [...]} or a raw list.
    """
    try:
        data: Any = manager.list_tools_sync(name)
        if isinstance(data, dict):
            tools = data.get("tools", [])
            return isinstance(tools, list)
        if isinstance(data, list):
            return True  # raw list of tools is also OK
        # Any non-exception response means the stdio roundtrip worked.
        return True
    except Exception:
        return False

def startup_mcp_servers() -> Dict[str, bool]:
    """
    Do NOT spawn persistent processes here.
    Just run a health check per configured server; stdio client spawns on demand.
    """
    manager = get_manager()
    servers = MCPServer.from_env()
    results: Dict[str, bool] = {}
    for name in servers.keys():
        results[name] = _health_check_server(manager, name)
    return results

def stop_all_servers() -> None:
    """
    No persistent processes were spawned here; nothing to stop.
    Exists for compatibility with setup scripts that call it.
    """
    return
