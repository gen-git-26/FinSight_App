# tools/mcp_bridge.py
from __future__ import annotations
import json
from typing import Any, Dict

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager

@tool(
    name="mcp_run",
    description="Run any MCP tool. Inputs: server (e.g., 'financial-datasets'|'yfinance'), tool name, args_json."
)
@fuse(tool_name="mcp", doc_type="mcp")
def mcp_run(server: str, tool: str, args_json: str = "{}") -> str:
    """Execute MCP tool with proper error handling."""
    try:
        # Parse arguments
        if isinstance(args_json, str):
            args: Dict[str, Any] = json.loads(args_json) if args_json else {}
        elif isinstance(args_json, dict):
            args = args_json
        else:
            args = {}
        
        # Create manager and execute call
        manager = MCPManager()
        result = manager.call_sync(server, tool, args)
        
        return result
        
    except json.JSONDecodeError as e:
        return f"MCP call failed - Invalid JSON arguments: {e}"
    except Exception as e:
        return f"MCP call failed: {str(e)}"