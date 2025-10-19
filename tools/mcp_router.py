# tools/mcp_router.py 
from __future__ import annotations

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from agno.tools import tool
from rag.fusion import fuse
from mcp_connection.manager import MCPManager, MCPServer
from tools.query_parser import parse_query_simple, ParsedQuery


# ============================================================================
# HELPERS
# ============================================================================

def _extract_date_from_query(query: str) -> Optional[str]:
    """
    Extract explicit date from query.
    Formats: YYYY-MM-DD, MM-DD-YYYY, MM/DD/YYYY
    """
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
        r'(\d{2}-\d{2}-\d{4})',  # MM-DD-YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            date_str = match.group(1)
            # Normalize to YYYY-MM-DD
            try:
                if '/' in date_str:
                    parts = date_str.split('/')
                    return f"{parts[2]}-{parts[0]}-{parts[1]}"
                elif '-' in date_str and len(date_str.split('-')[0]) == 2:
                    parts = date_str.split('-')
                    return f"{parts[2]}-{parts[0]}-{parts[1]}"
                else:
                    return date_str
            except:
                pass
    
    return None


def _calculate_next_monthly_expiration() -> str:
    """
    Calculate next monthly options expiration (3rd Friday).
    Always returns a FUTURE date.
    """
    today = datetime.now()
    
    # Try current month first
    year = today.year
    month = today.month
    
    # Find 3rd Friday of current month
    first_day = datetime(year, month, 1)
    # Find first Friday (weekday 4)
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    third_friday = first_friday + timedelta(weeks=2)
    
    # If 3rd Friday already passed, get next month
    if third_friday < today:
        month += 1
        if month > 12:
            month = 1
            year += 1
        
        first_day = datetime(year, month, 1)
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(weeks=2)
    
    result = third_friday.strftime('%Y-%m-%d')
    print(f"[mcp_router] Calculated next expiration: {result}")
    return result


def _is_crypto(ticker: Optional[str]) -> bool:
    """Check if ticker is crypto."""
    return ticker and ticker.endswith('-USD')


def _match_tool_dynamic(intent: str, available_tools: List[str], is_crypto: bool) -> Optional[str]:
    """
    Match intent to tool using flexible patterns.
    Works with ANY MCP server dynamically.
    """
    tools_lower = {t.lower(): t for t in available_tools}
    
    # Define flexible patterns
    patterns_by_intent = {
        'price': [
            r'current.*crypto.*price' if is_crypto else r'current.*stock.*price',
            r'stock.*info' if not is_crypto else r'crypto.*price',
            r'get.*price',
            r'.*quote.*',
        ],
        'historical': [
            r'historical.*crypto' if is_crypto else r'historical.*stock',
            r'historical.*price',
            r'.*history.*'
        ],
        'news': [r'.*news.*', r'.*article.*'],
        'fundamentals': [
            r'financial.*statement',
            r'.*statement.*',
            r'.*fundamental.*'
        ],
        'options': [r'option.*chain', r'.*option.*', r'expiration'],
        'info': [r'.*info.*', r'.*overview.*'],
    }
    
    # Try exact patterns first
    for pattern in patterns_by_intent.get(intent, []):
        for tool_lower, tool_name in tools_lower.items():
            if re.search(pattern, tool_lower):
                return tool_name
    
    # Fallback: simple keyword match
    keywords = {
        'price': ['price', 'quote', 'current'],
        'historical': ['historical', 'history'],
        'news': ['news'],
        'options': ['option'],
        'fundamentals': ['statement', 'financial'],
        'info': ['info'],
    }
    
    for keyword in keywords.get(intent, []):
        for tool_lower, tool_name in tools_lower.items():
            if keyword in tool_lower:
                return tool_name
    
    # Last resort
    return available_tools[0] if available_tools else None


def _build_args_dynamic(
    tool_name: str,
    parsed: ParsedQuery,
    schema: Dict
) -> Dict[str, Any]:
    """
    Build arguments dynamically from schema.
    NO hardcoding - reads schema properties!
    """
    props = schema.get('properties', {})
    args = {}
    
    # 1. Ticker/Symbol
    if parsed.ticker:
        if 'ticker' in props:
            args['ticker'] = parsed.ticker
        elif 'symbol' in props:
            args['symbol'] = parsed.ticker
    
    # 2. Options handling
    if 'option' in tool_name.lower():
        # Option type
        if 'option_type' in props:
            q = parsed.raw_query.lower()
            if 'call' in q:
                args['option_type'] = 'calls'
            elif 'put' in q:
                args['option_type'] = 'puts'
            else:
                args['option_type'] = 'chain'
        
        # Expiration date - FIXED LOGIC
        if 'expiration_date' in props:
            # Priority 1: Extract date from query
            explicit_date = _extract_date_from_query(parsed.raw_query)
            if explicit_date:
                args['expiration_date'] = explicit_date
                print(f"[mcp_router] Using explicit date from query: {explicit_date}")
            else:
                # Priority 2: Calculate next monthly expiration
                args['expiration_date'] = _calculate_next_monthly_expiration()
    
    # 3. Time periods (historical)
    if 'period' in props:
        q = parsed.raw_query.lower()
        if '6 month' in q or '6mo' in q:
            args['period'] = '6mo'
        elif '1 year' in q or '1y' in q:
            args['period'] = '1y'
        elif '1 month' in q or '1mo' in q:
            args['period'] = '1mo'
        else:
            args['period'] = '1y'
    
    if 'interval' in props:
        args['interval'] = '1d'
    
    # 4. Financial statement type
    if 'financial_type' in props:
        q = parsed.raw_query.lower()
        if 'balance' in q:
            args['financial_type'] = 'balance_sheet'
        elif 'cash flow' in q or 'cashflow' in q:
            args['financial_type'] = 'cashflow'
        else:
            args['financial_type'] = 'income_stmt'
    
    return args


def _parse_mcp_response(raw: str) -> tuple[Optional[Any], str]:
    """Parse MCP response to extract JSON."""
    if not isinstance(raw, str):
        return None, str(raw)
    
    # Try direct JSON parse
    try:
        parsed = json.loads(raw)
        return parsed, raw
    except:
        pass
    
    # Try to extract JSON from string
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed, match.group(0)
            except:
                pass
    
    return None, raw


# ============================================================================
# MAIN ROUTING
# ============================================================================

def route_and_call(query: str) -> Dict[str, Any]:
    """
    Smart router with LLM parsing and dynamic tool matching.
    """
    print(f"\n{'='*60}")
    print(f"[Router] Query: {query}")
    
    try:
        # Step 1: Parse with LLM (simple and fast)
        parsed = parse_query_simple(query)
        print(f"[Router] Parsed - Ticker: {parsed.ticker}, Intent: {parsed.intent}")
        
        # Step 2: Check servers
        servers = MCPServer.from_env()
        if not servers:
            return {
                'error': 'No MCP servers configured',
                'route': {},
                'parsed': None,
                'raw': ''
            }
        
        # Step 3: Pick server
        is_crypto = _is_crypto(parsed.ticker)
        if is_crypto and 'financial-datasets' in servers:
            server_name = 'financial-datasets'
        elif 'yfinance' in servers:
            server_name = 'yfinance'
        else:
            server_name = next(iter(servers))
        
        print(f"[Router] Server: {server_name} (crypto={is_crypto})")
        
        # Step 4: Get tools dynamically
        manager = MCPManager()
        tools_data = manager.list_tools_sync(server_name)
        
        available_tools = []
        tools_map = {}
        
        for t in tools_data.get('tools', []):
            name = t.get('name', '')
            if name and name.lower() not in ['meta', 'health', 'status']:
                available_tools.append(name)
                tools_map[name] = t
        
        if not available_tools:
            return {
                'error': f'No tools in {server_name}',
                'route': {'server': server_name},
                'parsed': None,
                'raw': ''
            }
        
        print(f"[Router] Available tools: {available_tools}")
        
        # Step 5: Match tool
        tool_name = _match_tool_dynamic(parsed.intent, available_tools, is_crypto)
        if not tool_name:
            return {
                'error': f'No tool matched for intent: {parsed.intent}',
                'route': {'server': server_name},
                'parsed': None,
                'raw': ''
            }
        
        print(f"[Router] Matched tool: {tool_name}")
        
        # Step 6: Build args
        tool_schema = tools_map[tool_name].get('inputSchema', {})
        args = _build_args_dynamic(tool_name, parsed, tool_schema)
        
        # Check required fields
        required = tool_schema.get('required', [])
        if any(r in ['ticker', 'symbol'] for r in required) and not parsed.ticker:
            return {
                'error': 'Could not extract ticker. Please specify (e.g., AAPL, TSLA, BTC-USD)',
                'route': {'server': server_name, 'tool': tool_name},
                'parsed': None,
                'raw': ''
            }
        
        print(f"[Router] Args: {args}")
        
        # Step 7: Call tool
        result = manager.call_sync(server_name, tool_name, args)
        
        # Step 8: Parse result
        parsed_result, cleaned_raw = _parse_mcp_response(result)
        
        print(f"[Router] Success - {len(str(result))} chars")
        print(f"{'='*60}\n")
        
        return {
            'route': {
                'server': server_name,
                'tool': tool_name,
                'primary_ticker': parsed.ticker,
                'intent': parsed.intent
            },
            'parsed': parsed_result,
            'raw': cleaned_raw,
            'error': None
        }
    
    except Exception as e:
        print(f"[Router] ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'error': str(e),
            'route': {},
            'parsed': None,
            'raw': ''
        }


# ============================================================================
# AGNO TOOL
# ============================================================================

@tool(
    name="mcp_auto",
    description="Smart MCP router with LLM parsing and dynamic tool matching"
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    """Agent entry point."""
    result = route_and_call(query)
    
    if result.get('error'):
        return f"Error: {result['error']}"
    
    route = result.get('route', {})
    raw = result.get('raw', '')
    
    return f"[{route.get('server')}/{route.get('tool')}]\n\n{raw}"