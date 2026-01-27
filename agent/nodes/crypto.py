# agent/nodes/crypto.py
"""
Crypto Agent - Dedicated agent for cryptocurrency queries.

Supports:
- Custom MCP servers for crypto data
- Multiple crypto data sources
- Fallback to public APIs
"""
from __future__ import annotations

import os
import json
import httpx
from typing import Dict, Any, List, Optional

from agent.state import AgentState, FetchedData, ParsedQuery
from mcp_connection.manager import MCPManager, MCPServer
from utils.config import load_settings


# Crypto symbol mappings
CRYPTO_SYMBOLS = {
    'bitcoin': 'BTC-USD',
    'btc': 'BTC-USD',
    'ethereum': 'ETH-USD',
    'eth': 'ETH-USD',
    'solana': 'SOL-USD',
    'sol': 'SOL-USD',
    'dogecoin': 'DOGE-USD',
    'doge': 'DOGE-USD',
    'ripple': 'XRP-USD',
    'xrp': 'XRP-USD',
    'cardano': 'ADA-USD',
    'ada': 'ADA-USD',
    'polkadot': 'DOT-USD',
    'dot': 'DOT-USD',
    'avalanche': 'AVAX-USD',
    'avax': 'AVAX-USD',
    'chainlink': 'LINK-USD',
    'link': 'LINK-USD',
    'polygon': 'MATIC-USD',
    'matic': 'MATIC-USD',
}


def _normalize_crypto_ticker(ticker: Optional[str], query: str) -> str:
    """Normalize crypto ticker to standard format."""
    if ticker and ticker.endswith('-USD'):
        return ticker

    # Check if ticker is in mapping
    if ticker:
        ticker_lower = ticker.lower()
        if ticker_lower in CRYPTO_SYMBOLS:
            return CRYPTO_SYMBOLS[ticker_lower]
        # Add -USD if not present
        return f"{ticker.upper()}-USD"

    # Try to extract from query
    query_lower = query.lower()
    for name, symbol in CRYPTO_SYMBOLS.items():
        if name in query_lower:
            return symbol

    return "BTC-USD"  # Default


def _get_crypto_mcp_server() -> Optional[str]:
    """
    Get the crypto MCP server name.

    Priority:
    1. CRYPTO_MCP_SERVER env var (for custom MCP)
    2. financial-datasets (built-in)
    3. yfinance (fallback)
    """
    # Check for custom crypto MCP server
    custom_server = os.getenv('CRYPTO_MCP_SERVER')
    if custom_server:
        return custom_server

    servers = MCPServer.from_env()
    if not servers:
        return None

    # Priority order for crypto
    for server_name in ['crypto', 'financial-datasets', 'coinstats', 'yfinance']:
        if server_name in servers:
            return server_name

    return next(iter(servers)) if servers else None


def _fetch_from_mcp(
    ticker: str,
    parsed: ParsedQuery,
    server_name: str
) -> FetchedData:
    """Fetch crypto data from MCP server."""
    manager = MCPManager()

    # Get available tools
    tools_data = manager.list_tools_sync(server_name)
    tools = tools_data.get('tools', [])

    # Find crypto-related tool
    tool_name = None
    tool_schema = {}

    for t in tools:
        name = t.get('name', '').lower()
        if any(kw in name for kw in ['crypto', 'price', 'quote', 'coin']):
            tool_name = t.get('name')
            tool_schema = t.get('inputSchema', {})
            break

    if not tool_name:
        # Use first available tool
        if tools:
            tool_name = tools[0].get('name')
            tool_schema = tools[0].get('inputSchema', {})

    if not tool_name:
        return FetchedData(
            source=server_name,
            error="No suitable crypto tool found"
        )

    # Build args
    props = tool_schema.get('properties', {})
    args = {}

    if 'ticker' in props:
        args['ticker'] = ticker
    elif 'symbol' in props:
        args['symbol'] = ticker
    elif 'coin' in props:
        args['coin'] = ticker.replace('-USD', '').lower()

    print(f"[Crypto] Calling {tool_name} with {args}")

    try:
        result = manager.call_sync(server_name, tool_name, args)

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


def _fetch_from_coingecko(ticker: str) -> FetchedData:
    """Fallback: Fetch from CoinGecko public API."""
    coin_id = ticker.replace('-USD', '').lower()

    # Map common symbols to CoinGecko IDs
    coingecko_ids = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'sol': 'solana',
        'doge': 'dogecoin',
        'xrp': 'ripple',
        'ada': 'cardano',
        'dot': 'polkadot',
        'avax': 'avalanche-2',
        'link': 'chainlink',
        'matic': 'matic-network',
    }

    coin_id = coingecko_ids.get(coin_id, coin_id)

    try:
        response = httpx.get(
            f"https://api.coingecko.com/api/v3/simple/price",
            params={
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        if coin_id in data:
            coin_data = data[coin_id]
            return FetchedData(
                source="coingecko",
                tool_used="simple_price",
                raw_data=data,
                parsed_data={
                    "symbol": ticker,
                    "price": coin_data.get('usd'),
                    "change_24h": coin_data.get('usd_24h_change'),
                    "market_cap": coin_data.get('usd_market_cap'),
                    "volume_24h": coin_data.get('usd_24h_vol'),
                }
            )
        else:
            return FetchedData(
                source="coingecko",
                error=f"Coin {coin_id} not found"
            )

    except Exception as e:
        return FetchedData(
            source="coingecko",
            error=str(e)
        )


def _fetch_from_coinstats(ticker: str) -> FetchedData:
    """Fetch from CoinStats API (if API key available)."""
    cfg = load_settings()

    if not cfg.coinstats_api_key:
        return FetchedData(source="coinstats", error="No API key")

    coin_id = ticker.replace('-USD', '').lower()

    try:
        response = httpx.get(
            f"https://openapiv1.coinstats.app/coins/{coin_id}",
            headers={
                "X-API-KEY": cfg.coinstats_api_key,
                "Accept": "application/json"
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        return FetchedData(
            source="coinstats",
            tool_used="coins",
            raw_data=data,
            parsed_data=data
        )

    except Exception as e:
        return FetchedData(
            source="coinstats",
            error=str(e)
        )


def crypto_node(state: AgentState) -> Dict[str, Any]:
    """
    Crypto Agent node - handles all cryptocurrency queries.

    Data source priority:
    1. Custom MCP server (CRYPTO_MCP_SERVER env var)
    2. Built-in MCP servers (financial-datasets, yfinance)
    3. CoinStats API
    4. CoinGecko public API (fallback)
    """
    parsed = state.get("parsed_query")
    if not parsed:
        return {"error": "No parsed query", "fetched_data": []}

    print(f"\n[Crypto] Processing crypto query")
    print(f"[Crypto] Ticker: {parsed.ticker}, Query: {parsed.raw_query}")

    # Normalize ticker
    ticker = _normalize_crypto_ticker(parsed.ticker, parsed.raw_query)
    print(f"[Crypto] Normalized ticker: {ticker}")

    results = []
    success = False

    # Try MCP server first
    mcp_server = _get_crypto_mcp_server()
    if mcp_server:
        print(f"[Crypto] Trying MCP server: {mcp_server}")
        data = _fetch_from_mcp(ticker, parsed, mcp_server)
        if not data.error:
            results.append(data)
            success = True
        else:
            print(f"[Crypto] MCP failed: {data.error}")

    # Try CoinStats if MCP failed
    if not success:
        print(f"[Crypto] Trying CoinStats...")
        data = _fetch_from_coinstats(ticker)
        if not data.error:
            results.append(data)
            success = True
        else:
            print(f"[Crypto] CoinStats failed: {data.error}")

    # Fallback to CoinGecko
    if not success:
        print(f"[Crypto] Falling back to CoinGecko...")
        data = _fetch_from_coingecko(ticker)
        results.append(data)
        if not data.error:
            success = True

    print(f"[Crypto] Fetched {len(results)} results, success={success}")

    return {
        "fetched_data": results,
        "mcp_server_used": mcp_server or "public-api",
        "sources": [r.source for r in results if not r.error]
    }


# === MCP Server Registration Helper ===

def register_crypto_mcp(
    server_name: str,
    command: str,
    args: List[str] = None,
    env: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Helper to register a new crypto MCP server.

    Example usage:
        register_crypto_mcp(
            server_name="my-crypto-mcp",
            command="npx",
            args=["-y", "@my/crypto-mcp-server"],
            env={"API_KEY": "xxx"}
        )

    This updates the MCP_SERVERS environment variable.
    """
    import json

    current_servers = os.getenv('MCP_SERVERS', '{}')
    try:
        servers = json.loads(current_servers)
    except:
        servers = {}

    servers[server_name] = {
        "command": command,
        "args": args or [],
        "env": env or {}
    }

    os.environ['MCP_SERVERS'] = json.dumps(servers)
    os.environ['CRYPTO_MCP_SERVER'] = server_name

    print(f"[Crypto] Registered MCP server: {server_name}")

    return {"status": "registered", "server": server_name}
