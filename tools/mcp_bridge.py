# tools/mcp_bridge.py
from __future__ import annotations
import json
from typing import Any, Dict

from agno.tools import tool
from rag.fusion import fuse
from mcp.manager import call_sync

@tool(
    name="mcp_run",
    description=(
        "Call any MCP tool dynamically. Inputs: server (e.g. 'yfinance' or 'coinmarketcap'), "
        "tool (e.g. 'get_stock_info'), args_json (stringified JSON of arguments)."
    ),
)
@fuse(tool_name="mcp", doc_type="mcp")
def mcp_run(server: str, tool: str, args_json: str = "{}") -> str:
    """Generic bridge: invokes an MCP tool and returns its raw string output.
    The Fusion decorator ingests the result into Qdrant automatically.
    """
    try:
        args: Dict[str, Any] = json.loads(args_json) if args_json else {}
    except Exception as e:
        raise ValueError(f"args_json must be valid JSON: {e}")
    return call_sync(server, tool, args)