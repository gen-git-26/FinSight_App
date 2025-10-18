# mcp_connection/startup.py
from __future__ import annotations

import multiprocessing as mp
from typing import Dict, Any

from .manager import MCPManager, MCPServer

_manager: MCPManager | None = None

def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager

def _probe_server(name: str) -> bool:
    """
    Run inside a separate process to avoid hanging the main process.
    """
    try:
        mgr = MCPManager()
        data: Any = mgr.list_tools_sync(name)
        if isinstance(data, dict):
            tools = data.get("tools", [])
            return isinstance(tools, list)
        if isinstance(data, list):
            return True
        return True  # any non-exception response is OK
    except Exception:
        return False

def _health_check_with_timeout(name: str, timeout_sec: float = 3.0) -> bool:
    """
    Spawn a child process to probe the server with a hard timeout.
    If it hangs, we terminate the child and return False (but we do NOT block).
    """
    q: mp.Queue = mp.Queue(maxsize=1)
    def _runner(q: mp.Queue):
        ok = _probe_server(name)
        try:
            q.put(bool(ok))
        except Exception:
            pass

    p = mp.Process(target=_runner, args=(q,), daemon=True)
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        return False

    try:
        return bool(q.get_nowait())
    except Exception:
        return False

def startup_mcp_servers() -> Dict[str, bool]:
    """
    Do NOT spawn persistent processes here.
    Just run a health check per configured server using a hard timeout.
    """
    servers = MCPServer.from_env()
    results: Dict[str, bool] = {}
    for name in servers.keys():
        results[name] = _health_check_with_timeout(name, timeout_sec=3.0)
    return results

def stop_all_servers() -> None:
    """
    No persistent processes were spawned here; nothing to stop.
    Exists for compatibility with setup scripts that call it.
    """
    return
