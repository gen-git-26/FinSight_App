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
    """Extract explicit date from query."""
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
        r'(\d{2}-\d{2}-\d{4})',  # MM-DD-YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            date_str = match.group(1)
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
    """Calculate next monthly options expiration (3rd Friday)."""
    today = datetime.now()
    year, month = today.year, today.month
    
    first_day = datetime(year, month, 1)
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    third_friday = first_friday + timedelta(weeks=2)
    
    if third_friday < today:
        month += 1
        if month > 12:
            month, year = 1, year + 1
        
        first_day = datetime(year, month, 1)
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(weeks=2)
    
    result = third_friday.strftime('%Y-%m-%d')
    print(f"[mcp_router] Calculated expiration: {result}")
    return result


def _is_crypto(ticker: Optional[str]) -> bool:
    """Check if ticker is crypto."""
    return ticker and ticker.endswith('-USD')


def _match_tool_dynamic(intent: str, available_tools: List[str], is_crypto: bool) -> Optional[str]:
    """Match intent to tool using patterns."""
    tools_lower = {t.lower(): t for t in available_tools}
    
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
        ],
        'news': [r'.*news.*'],
        'fundamentals': [r'financial.*statement', r'.*statement.*'],
        'options': [r'option.*chain', r'.*option.*'],
        'info': [r'.*info.*'],
    }
    
    for pattern in patterns_by_intent.get(intent, []):
        for tool_lower, tool_name in tools_lower.items():
            if re.search(pattern, tool_lower):
                return tool_name
    
    # Fallback: keyword match
    keywords = {
        'price': ['price', 'quote'],
        'historical': ['historical'],
        'news': ['news'],
        'options': ['option'],
        'fundamentals': ['statement', 'financial'],
        'info': ['info'],
    }
    
    for keyword in keywords.get(intent, []):
        for tool_lower, tool_name in tools_lower.items():
            if keyword in tool_lower:
                return tool_name
    
    return available_tools[0] if available_tools else None


def _build_args_dynamic(tool_name: str, parsed: ParsedQuery, schema: Dict) -> Dict[str, Any]:
    """Build arguments from schema."""
    props = schema.get('properties', {})
    args = {}
    
    # Ticker/Symbol
    if parsed.ticker:
        if 'ticker' in props:
            args['ticker'] = parsed.ticker
        elif 'symbol' in props:
            args['symbol'] = parsed.ticker
    
    # Options
    if 'option' in tool_name.lower():
        if 'option_type' in props:
            q = parsed.raw_query.lower()
            if 'call' in q:
                args['option_type'] = 'calls'
            elif 'put' in q:
                args['option_type'] = 'puts'
            else:
                args['option_type'] = 'chain'
        
        if 'expiration_date' in props:
            explicit_date = _extract_date_from_query(parsed.raw_query)
            args['expiration_date'] = explicit_date or _calculate_next_monthly_expiration()
    
    # Time periods
    if 'period' in props:
        q = parsed.raw_query.lower()
        if '6 month' in q:
            args['period'] = '6mo'
        elif '1 year' in q:
            args['period'] = '1y'
        else:
            args['period'] = '1y'
    
    if 'interval' in props:
        args['interval'] = '1d'
    
    # Financial type
    if 'financial_type' in props:
        q = parsed.raw_query.lower()
        if 'balance' in q:
            args['financial_type'] = 'balance_sheet'
        elif 'cash flow' in q:
            args['financial_type'] = 'cashflow'
        else:
            args['financial_type'] = 'income_stmt'
    
    return args


def _parse_mcp_response(raw: str) -> tuple[Optional[Any], str]:
    """Parse MCP response."""
    if not isinstance(raw, str):
        return None, str(raw)
    
    try:
        parsed = json.loads(raw)
        return parsed, raw
    except:
        pass
    
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed, match.group(0)
            except:
                pass
    
    return None, raw


def _call_single_ticker(
    parsed: ParsedQuery,
    ticker: str,
    manager: MCPManager,
    server_name: str,
    tools_map: Dict
) -> Dict[str, Any]:
    """
    Call MCP for a single ticker.
    Returns result dict.
    """
    # Update parsed query with current ticker
    parsed_copy = ParsedQuery(
        ticker=ticker,
        additional_tickers=[],
        intent=parsed.intent,
        raw_query=parsed.raw_query
    )
    
    is_crypto = _is_crypto(ticker)
    available_tools = list(tools_map.keys())
    
    # Match tool
    tool_name = _match_tool_dynamic(parsed.intent, available_tools, is_crypto)
    if not tool_name:
        return {'error': f'No tool for {ticker}'}
    
    # Build args
    tool_schema = tools_map[tool_name].get('inputSchema', {})
    args = _build_args_dynamic(tool_name, parsed_copy, tool_schema)
    
    print(f"[mcp_router] Calling {tool_name} for {ticker}: {args}")
    
    # Call MCP
    try:
        result = manager.call_sync(server_name, tool_name, args)
        parsed_result, cleaned_raw = _parse_mcp_response(result)
        
        return {
            'ticker': ticker,
            'tool': tool_name,
            'parsed': parsed_result,
            'raw': cleaned_raw,
            'error': None
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'error': str(e)
        }


# ============================================================================
# MAIN ROUTING
# ============================================================================

def route_and_call(query: str) -> Dict[str, Any]:
    """
    Smart router with multi-ticker support.
    If multiple tickers detected, calls for each and combines results.
    """
    print(f"\n{'='*60}")
    print(f"[Router] Query: {query}")
    
    try:
        # Step 1: Parse query
        parsed = parse_query_simple(query)
        print(f"[Router] Parsed - Ticker: {parsed.ticker}, Additional: {parsed.additional_tickers}, Intent: {parsed.intent}")
        
        # Step 2: Check servers
        servers = MCPServer.from_env()
        if not servers:
            return {'error': 'No MCP servers configured', 'route': {}, 'parsed': None, 'raw': ''}
        
        # Step 3: Pick server
        is_crypto = _is_crypto(parsed.ticker)
        if is_crypto and 'financial-datasets' in servers:
            server_name = 'financial-datasets'
        elif 'yfinance' in servers:
            server_name = 'yfinance'
        else:
            server_name = next(iter(servers))
        
        print(f"[Router] Server: {server_name} (crypto={is_crypto})")
        
        # Step 4: Get tools
        manager = MCPManager()
        tools_data = manager.list_tools_sync(server_name)
        
        tools_map = {}
        for t in tools_data.get('tools', []):
            name = t.get('name', '')
            if name and name.lower() not in ['meta', 'health', 'status']:
                tools_map[name] = t
        
        if not tools_map:
            return {'error': f'No tools in {server_name}', 'route': {}, 'parsed': None, 'raw': ''}
        
        print(f"[Router] Available tools: {list(tools_map.keys())}")
        
        # Step 5: Handle multi-ticker comparison
        all_tickers = [parsed.ticker] if parsed.ticker else []
        all_tickers.extend(parsed.additional_tickers)
        
        if not all_tickers:
            return {'error': 'No ticker found', 'route': {}, 'parsed': None, 'raw': ''}
        
        # Call for each ticker
        results = []
        for ticker in all_tickers:
            result = _call_single_ticker(parsed, ticker, manager, server_name, tools_map)
            results.append(result)
        
        # Combine results
        if len(results) == 1:
            # Single ticker - return as before
            single = results[0]
            if single.get('error'):
                return {'error': single['error'], 'route': {}, 'parsed': None, 'raw': ''}
            
            print(f"[Router] Success - {len(str(single['raw']))} chars")
            print(f"{'='*60}\n")
            
            return {
                'route': {
                    'server': server_name,
                    'tool': single['tool'],
                    'primary_ticker': parsed.ticker,
                    'intent': parsed.intent
                },
                'parsed': single['parsed'],
                'raw': single['raw'],
                'error': None
            }
        
        else:
            # Multiple tickers - combine
            print(f"[Router] Multi-ticker comparison: {len(results)} results")
            
            combined_raw = ""
            combined_parsed = {}
            
            for i, res in enumerate(results):
                ticker = res.get('ticker', f'Unknown{i}')
                
                if res.get('error'):
                    combined_raw += f"\n\n### {ticker}\nError: {res['error']}"
                else:
                    combined_raw += f"\n\n### {ticker}\n{res.get('raw', '')}"
                    combined_parsed[ticker] = res.get('parsed')
            
            print(f"[Router] Success - combined {len(combined_raw)} chars")
            print(f"{'='*60}\n")
            
            return {
                'route': {
                    'server': server_name,
                    'tool': results[0].get('tool', 'multiple'),
                    'primary_ticker': parsed.ticker,
                    'additional_tickers': parsed.additional_tickers,
                    'intent': parsed.intent
                },
                'parsed': combined_parsed,
                'raw': combined_raw.strip(),
                'error': None
            }
    
    except Exception as e:
        print(f"[Router] ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return {'error': str(e), 'route': {}, 'parsed': None, 'raw': ''}


# ============================================================================
# AGNO TOOL
# ============================================================================

@tool(
    name="mcp_auto",
    description="Smart MCP router with multi-ticker comparison support"
)
@fuse(tool_name="mcp-auto", doc_type="mcp")
def mcp_auto(query: str) -> str:
    """Agent entry point."""
    result = route_and_call(query)
    
    if result.get('error'):
        return f"Error: {result['error']}"
    
    route = result.get('route', {})
    raw = result.get('raw', '')
    
    # Add header with context
    tickers = [route.get('primary_ticker')]
    if route.get('additional_tickers'):
        tickers.extend(route['additional_tickers'])
    
    header = f"[{route.get('server')}/{route.get('tool')} | Tickers: {', '.join(filter(None, tickers))}]"
    
    return f"{header}\n\n{raw}"