# setup_and_run.py

"""
MCP Server Startup Manager
Automatically starts and manages MCP servers when the application loads.
"""

import os
import sys
import shlex
import subprocess
import threading
import time
import logging
from typing import Dict, Optional
from pathlib import Path

from .manager import MCPServer, MCPManager

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manages the lifecycle of MCP servers."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.startup_threads: Dict[str, threading.Thread] = {}
        self.startup_timeout = 30  # seconds

    def _build_cmd(self, server: MCPServer) -> Optional[list]:
        """
        Build a robust command vector from server.command and server.args.
        Supports:
          - command as string with embedded args
          - command as string without args
          - command as list (vector form)
          - args as list or empty
        """
        base: list
        if isinstance(server.command, list):
            base = [str(p) for p in server.command if str(p).strip()]
        else:
            s = (server.command or "").strip()
            base = shlex.split(s) if s else []

        extra = server.args or []
        cmd = base + extra
        return cmd if cmd else None

    def _start_server_process(self, name: str, server: MCPServer) -> bool:
        """Start a single MCP server process."""
        try:
            logger.info(f"Starting MCP server: {name}")

            cmd = self._build_cmd(server)
            if not cmd:
                logger.error(f"Empty command for MCP server '{name}'")
                return False

            # Prepare environment and working directory consistently with MCPManager
            env = MCPManager.build_child_env(server.env)
            cwd = MCPManager.resolve_cwd(server.cwd)

            logger.debug(f"Exec for '{name}': cmd={cmd}, cwd={cwd}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=cwd,
            )

            self.processes[name] = process
            logger.info(f"MCP server '{name}' started with PID {process.pid}")

            # Monitor process health
            def monitor():
                # Give it time to start
                time.sleep(2)
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(f"MCP server '{name}' failed to start")
                    # Print minimal but useful diagnostics
                    env_delta = {k: env[k] for k in ("PYTHONUNBUFFERED", "OPENAI_API_KEY", "FINANCIAL_DATASETS_API_KEY") if k in env}
                    logger.error(f"Command: {cmd}")
                    logger.error(f"CWD: {cwd}")
                    logger.error(f"ENV keys: {list(env.keys())[:5]}... total={len(env)} delta={env_delta}")
                    logger.error(f"STDOUT:\n{stdout}")
                    logger.error(f"STDERR:\n{stderr}")
                else:
                    logger.info(f"MCP server '{name}' is healthy")

            t = threading.Thread(target=monitor, daemon=True)
            t.start()
            self.startup_threads[name] = t
            return True

        except Exception as e:
            logger.error(f"Failed to start MCP server '{name}': {e}")
            return False

    def start_all_servers(self) -> Dict[str, bool]:
        """Start all configured MCP servers."""
        servers = MCPServer.from_env()
        results: Dict[str, bool] = {}

        if not servers:
            logger.warning("No MCP servers configured in environment")
            return results

        logger.info(f"Starting {len(servers)} MCP servers...")

        for name, server in servers.items():
            success = self._start_server_process(name, server)
            results[name] = success
            if success:
                logger.info(f"Server '{name}' startup initiated")
            else:
                logger.error(f"Server '{name}' startup failed")

        # Wait a moment for all servers to initialize
        time.sleep(3)
        return results

    def stop_all_servers(self):
        """Stop all running MCP servers."""
        logger.info("Stopping all MCP servers...")

        for name, process in list(self.processes.items()):
            try:
                if process.poll() is None:
                    logger.info(f"Stopping server: {name}")
                    process.terminate()
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                        logger.info(f"Server '{name}' stopped gracefully")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing server: {name}")
                        process.kill()
                        process.wait()
                        logger.info(f"Server '{name}' force stopped")
                else:
                    logger.info(f"Server '{name}' already stopped")
            except Exception as e:
                logger.error(f"Error stopping server '{name}': {e}")

            self.processes.pop(name, None)

        logger.info("All MCP servers stopped")

    def get_server_status(self) -> Dict[str, str]:
        """Get the status of all servers."""
        status: Dict[str, str] = {}

        for name, process in self.processes.items():
            if process.poll() is None:
                status[name] = "Running"
            else:
                status[name] = f"Stopped (exit code: {process.poll()})"

        # Mark configured but not started
        configured = MCPServer.from_env()
        for name in configured:
            if name not in status:
                status[name] = "Not Started"

        return status

    def restart_server(self, name: str) -> bool:
        """Restart a specific server."""
        logger.info(f"Restarting server: {name}")

        # Stop if running
        if name in self.processes:
            process = self.processes[name]
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            self.processes.pop(name, None)

        # Start again
        servers = MCPServer.from_env()
        if name in servers:
            return self._start_server_process(name, servers[name])

        logger.error(f"Server '{name}' not found in configuration")
        return False


# Global manager instance
_manager: Optional[MCPServerManager] = None


def get_manager() -> MCPServerManager:
    """Get the global server manager instance."""
    global _manager
    if _manager is None:
        _manager = MCPServerManager()
    return _manager


def startup_mcp_servers() -> Dict[str, bool]:
    """Convenience function to start all MCP servers."""
    return get_manager().start_all_servers()


def shutdown_mcp_servers():
    """Convenience function to stop all MCP servers."""
    if _manager:
        _manager.stop_all_servers()


# Cleanup on exit
import atexit
atexit.register(shutdown_mcp_servers)
