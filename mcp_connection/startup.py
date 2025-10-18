# mcp_connection/startup.py
from __future__ import annotations
from typing import Dict, Any
from .manager import MCPManager, MCPServer

_manager: MCPManager | None = None

def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager

def _health_check_server(manager: MCPManager, name: str) -> bool:
    """Fast, non-blocking health probe using manager timeouts."""
    try:
        data: Any = manager.list_tools_sync(name)
        if isinstance(data, dict):
            tools = data.get("tools", [])
            return isinstance(tools, list)
        return True
    except Exception:
        return False

def startup_mcp_servers() -> Dict[str, bool]:
    """Do not spawn persistent processes. Probe each configured server once."""
    manager = get_manager()
    servers = MCPServer.from_env()
    results: Dict[str, bool] = {}
    for name in servers.keys():
        results[name] = _health_check_server(manager, name)
    return results

def stop_all_servers() -> None:
    """No persistent processes were started here."""
    return
