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
    try:
        args: Dict[str, Any] = json.loads(args_json) if args_json else {}
    except Exception as e:
        raise ValueError(f"args_json must be JSON: {e}")
    return MCPManager.call_sync(server, tool, args)
