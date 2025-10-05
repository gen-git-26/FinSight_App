# mcp_connection/startup.py
"""
MCP Server Startup Manager
Automatically starts and manages MCP servers when the application loads.
"""

import os
import sys
import subprocess
import threading
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .manager import MCPServer

logger = logging.getLogger(__name__)

class MCPServerManager:
    """Manages the lifecycle of MCP servers."""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.startup_threads: Dict[str, threading.Thread] = {}
        self.startup_timeout = 30  # seconds
        
    def _start_server_process(self, name: str, server: MCPServer) -> bool:
        """Start a single MCP server process."""
        try:
            logger.info(f"Starting MCP server: {name}")
            
            # Prepare command
            if isinstance(server.command, list):
                cmd = server.command
            else:
                cmd = server.command.split()
            
            # Prepare environment
            env = os.environ.copy()
            if server.env:
                env.update(server.env)
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=Path.cwd()
            )
            
            self.processes[name] = process
            logger.info(f"MCP server '{name}' started with PID {process.pid}")
            
            # Monitor process health
            def monitor():
                time.sleep(2)  # Give it time to start
                if process.poll() is not None:
                    # Process died
                    stdout, stderr = process.communicate()
                    logger.error(f"MCP server '{name}' failed to start")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                else:
                    logger.info(f"MCP server '{name}' is healthy")
            
            threading.Thread(target=monitor, daemon=True).start()
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server '{name}': {e}")
            return False
    
    def start_all_servers(self) -> Dict[str, bool]:
        """Start all configured MCP servers."""
        servers = MCPServer.from_env()
        results = {}
        
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
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # Still running
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
        
        self.processes.clear()
        logger.info("All MCP servers stopped")
    
    def get_server_status(self) -> Dict[str, str]:
        """Get the status of all servers."""
        status = {}
        
        for name, process in self.processes.items():
            if process.poll() is None:
                status[name] = "Running"
            else:
                status[name] = f"Stopped (exit code: {process.poll()})"
        
        # Check for configured but not started servers
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
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            del self.processes[name]
        
        # Start again
        servers = MCPServer.from_env()
        if name in servers:
            return self._start_server_process(name, servers[name])
        else:
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