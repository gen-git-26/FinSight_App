# tools/mcp_bridge.py
# tools/mcp_bridge.py
from __future__ import annotations
import json
from typing import Any, Dict

from agno.tools import tool
from mcp_connection.manager import MCPManager

mgr = MCPManager()

@tool(
    name="mcp_run",
    description=(
        "Run any MCP tool. Inputs: server (e.g., 'yfinance'|'coinmarketcap'), "
        "tool (e.g., 'get_stock_info'), args_json (JSON string). Returns raw text."
    ),
)
async def mcp_run(server: str, tool: str, args_json: str = "{}") -> str:
    try:
        args: Dict[str, Any] = json.loads(args_json) if args_json else {}
    except Exception as e:
        return f"args_json must be valid JSON: {e}"
    return await mgr.call(server, tool, args)
