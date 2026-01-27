# agent/nodes/fetcher.py
"""
Fetcher Agent - Fetches stock data from MCP servers and APIs.
"""
from __future__ import annotations

import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from agent.state import AgentState, FetchedData, ParsedQuery
from mcp_connection.manager import MCPManager, MCPServer
from utils.config import load_settings


def _extract_date_from_query(query: str) -> Optional[str]:
    """Extract explicit date from query."""
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{2}/\d{2}/\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return None


def _calculate_next_expiration() -> str:
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

    return third_friday.strftime('%Y-%m-%d')


def _match_tool(intent: str, tools: List[str]) -> Optional[str]:
    """Match intent to available tool."""
    tools_lower = {t.lower(): t for t in tools}

    patterns = {
        'price': ['price', 'quote', 'info'],
        'options': ['option'],
        'fundamentals': ['financial', 'statement'],
        'historical': ['historical'],
        'news': ['news'],
    }

    for keyword in patterns.get(intent, ['info']):
        for tool_lower, tool_name in tools_lower.items():
            if keyword in tool_lower:
                return tool_name

    return tools[0] if tools else None


def _build_args(tool_name: str, parsed: ParsedQuery, schema: Dict) -> Dict[str, Any]:
    """Build arguments for tool call."""
    props = schema.get('properties', {})
    args = {}

    # Ticker/Symbol
    if parsed.ticker:
        if 'ticker' in props:
            args['ticker'] = parsed.ticker
        elif 'symbol' in props:
            args['symbol'] = parsed.ticker

    # Options specific
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
            date = _extract_date_from_query(parsed.raw_query)
            args['expiration_date'] = date or _calculate_next_expiration()

    # Financial type
    if 'financial_type' in props:
        q = parsed.raw_query.lower()
        if 'balance' in q:
            args['financial_type'] = 'balance_sheet'
        elif 'cash' in q:
            args['financial_type'] = 'cashflow'
        else:
            args['financial_type'] = 'income_stmt'

    # Period
    if 'period' in props:
        args['period'] = '1y'
    if 'interval' in props:
        args['interval'] = '1d'

    return args


def _fetch_single_ticker(
    ticker: str,
    parsed: ParsedQuery,
    manager: MCPManager,
    server_name: str,
    tools_map: Dict
) -> FetchedData:
    """Fetch data for a single ticker."""
    print(f"[Fetcher] Fetching {ticker}...")

    tool_name = _match_tool(parsed.intent, list(tools_map.keys()))
    if not tool_name:
        return FetchedData(
            source=server_name,
            error=f"No suitable tool for intent: {parsed.intent}"
        )

    # Build args
    temp_parsed = ParsedQuery(
        ticker=ticker,
        intent=parsed.intent,
        query_type=parsed.query_type,
        raw_query=parsed.raw_query
    )
    schema = tools_map[tool_name].get('inputSchema', {})
    args = _build_args(tool_name, temp_parsed, schema)

    print(f"[Fetcher] Calling {tool_name} with {args}")

    try:
        result = manager.call_sync(server_name, tool_name, args)

        # Parse result
        parsed_data = {}
        if isinstance(result, str):
            try:
                parsed_data = json.loads(result)
            except:
                parsed_data = {"raw": result}
        elif isinstance(result, dict):
            parsed_data = result

        return FetchedData(
            source=server_name,
            tool_used=tool_name,
            raw_data=result,
            parsed_data=parsed_data
        )

    except Exception as e:
        return FetchedData(
            source=server_name,
            tool_used=tool_name,
            error=str(e)
        )


def fetcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetcher node - fetches stock data from MCP servers.

    Handles: stocks, options, fundamentals, news
    """
    parsed = state.get("parsed_query")
    if not parsed:
        return {"error": "No parsed query", "fetched_data": []}

    print(f"\n[Fetcher] Intent: {parsed.intent}, Ticker: {parsed.ticker}")

    # Get MCP servers
    servers = MCPServer.from_env()
    if not servers:
        # Fallback to direct API
        return _fallback_fetch(state)

    # Select server (yfinance for stocks)
    server_name = 'yfinance' if 'yfinance' in servers else next(iter(servers))
    print(f"[Fetcher] Using server: {server_name}")

    # Get available tools
    manager = MCPManager()
    tools_data = manager.list_tools_sync(server_name)

    tools_map = {}
    for t in tools_data.get('tools', []):
        name = t.get('name', '')
        if name and name.lower() not in ['meta', 'health', 'status']:
            tools_map[name] = t

    if not tools_map:
        return {"error": f"No tools in {server_name}", "fetched_data": []}

    print(f"[Fetcher] Available tools: {list(tools_map.keys())}")

    # Collect all tickers
    tickers = []
    if parsed.ticker:
        tickers.append(parsed.ticker)
    tickers.extend(parsed.additional_tickers or [])

    if not tickers:
        return {"error": "No tickers found", "fetched_data": []}

    # Fetch for each ticker
    results = []
    for ticker in tickers:
        data = _fetch_single_ticker(ticker, parsed, manager, server_name, tools_map)
        results.append(data)

    print(f"[Fetcher] Fetched {len(results)} results")

    return {
        "fetched_data": results,
        "mcp_server_used": server_name,
        "tools_available": list(tools_map.keys())
    }


def _fallback_fetch(state: AgentState) -> Dict[str, Any]:
    """Fallback to direct API calls if MCP not available."""
    import finnhub
    from utils.config import load_settings

    parsed = state.get("parsed_query")
    cfg = load_settings()

    results = []

    if parsed and parsed.ticker:
        try:
            client = finnhub.Client(api_key=cfg.finnhub_api_key)

            if parsed.intent == 'price':
                data = client.quote(parsed.ticker)
            elif parsed.intent == 'fundamentals':
                data = client.company_basic_financials(parsed.ticker, 'all')
            else:
                data = client.quote(parsed.ticker)

            results.append(FetchedData(
                source="finnhub",
                tool_used="fallback",
                raw_data=data,
                parsed_data=data if isinstance(data, dict) else {"data": data}
            ))

        except Exception as e:
            results.append(FetchedData(
                source="finnhub",
                error=str(e)
            ))

    return {
        "fetched_data": results,
        "mcp_server_used": "finnhub-fallback"
    }
