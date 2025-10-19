# tools/answer.py 
from __future__ import annotations

import json
from typing import Dict, Any

from agno.tools import tool
from mcp_connection.manager import MCPServer
from tools.mcp_router import route_and_call


def _format_output(payload: Dict[str, Any]) -> tuple[str, str, bool]:
    """
    Simple formatter - just clean display.
    
    Returns:
        (text, display_type, is_dataframe)
    """
    if payload.get('error'):
        return f"⚠️ {payload['error']}", 'error', False
    
    parsed = payload.get('parsed')
    raw = payload.get('raw', '')
    
    # Case 1: List of dicts → Table
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return json.dumps(parsed), 'table', True
    
    # Case 2: Dict → Key-value display
    if isinstance(parsed, dict):
        lines = []
        for key, value in list(parsed.items())[:30]:
            if not key.startswith('_'):
                lines.append(f"**{key}**: {value}")
        return '\n'.join(lines), 'dict', False
    
    # Case 3: Raw text
    return raw[:5000], 'text', False


def answer_core(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    """
    Simple answer - calls router and formats.
    """
    print(f"[answer] Query: {query[:100]}")
    
    servers = MCPServer.from_env()
    if not servers:
        return {
            'answer': '⚠️ No MCP servers configured',
            'snippets': [],
            'display_type': 'error',
            'is_dataframe': False,
            'meta': {}
        }
    
    # Call router
    result = route_and_call(query)
    
    # Format
    answer_text, display_type, is_dataframe = _format_output(result)
    
    return {
        'answer': answer_text,
        'snippets': [],
        'display_type': display_type,
        'is_dataframe': is_dataframe,
        'meta': {
            'server': result.get('route', {}).get('server'),
            'tool': result.get('route', {}).get('tool')
        }
    }


@tool(name="answer", description="Get financial data via MCP")
def answer(query: str, ticker: str = "", style: str = "") -> Dict[str, Any]:
    return answer_core(query, ticker, style)