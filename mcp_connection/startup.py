# mcp_connection/startup.py
from __future__ import annotations

import subprocess
import os
from typing import Dict

from .manager import MCPServer, MCPManager

_manager: MCPManager | None = None
_processes: Dict[str, subprocess.Popen] = {}

def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager

def _start_server(name: str, server: MCPServer) -> bool:
    """
    Start a persistent MCP server process with Popen if you want them alive during the app.
    Note: Our MCPManager.call_sync already spins a stdio client per call. This is optional.
    """
    try:
        env = os.environ.copy()
        env.update(server.env or {})
        proc = subprocess.Popen(
            [server.command] + list(server.args),
            env=env
        )
        _processes[name] = proc
        return True
    except Exception:
        return False

def startup_mcp_servers() -> Dict[str, bool]:
    """
    Start all servers specified in MCP_SERVERS.
    Returns dict of name -> started_successfully.
    """
    servers = MCPServer.from_env()
    results: Dict[str, bool] = {}
    for name, srv in servers.items():
        # Optional: keep them alive. If you prefer "per-call" stdio only, mark True without starting.
        ok = _start_server(name, srv)
        results[name] = bool(ok)
    return results

def stop_all_servers() -> None:
    for name, proc in list(_processes.items()):
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        _processes.pop(name, None)
