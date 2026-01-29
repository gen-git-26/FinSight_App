# tests/test_mcp_servers.py
"""
Tests for MCP Servers.

Tests:
- yfinance MCP Server
- financial-datasets MCP Server
"""
import os
import sys
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datasources.mcp_client import get_mcp_client, setup_default_servers


def print_result(name: str, result):
    """Pretty print test result."""
    if result.success:
        print(f"  ✓ {name}")
        if isinstance(result.data, dict):
            for k, v in list(result.data.items())[:5]:
                if v is not None:
                    val = str(v)[:50] if isinstance(v, str) else v
                    print(f"    {k}: {val}")
    else:
        print(f"  ✗ {name}: {result.error}")


async def test_yfinance_mcp():
    """Test yfinance MCP Server."""
    print("\n" + "=" * 50)
    print("YFINANCE MCP SERVER")
    print("=" * 50)

    client = get_mcp_client()

    # List tools
    tools = await client.list_tools("yfinance")
    print(f"\n  Tools available: {len(tools)}")
    for t in tools:
        print(f"    - {t['name']}")

    # Test get_stock_price
    print("\n[Test] get_stock_price('AAPL')")
    result = await client.call_tool("yfinance", "get_stock_price", {"ticker": "AAPL"})
    print_result("Stock Price", result)

    # Test get_stock_info
    print("\n[Test] get_stock_info('TSLA')")
    result = await client.call_tool("yfinance", "get_stock_info", {"ticker": "TSLA"})
    print_result("Stock Info", result)

    # Test get_news
    print("\n[Test] get_news('NVDA')")
    result = await client.call_tool("yfinance", "get_news", {"ticker": "NVDA", "limit": 3})
    print_result("News", result)

    # Test get_options_chain
    print("\n[Test] get_options_chain('SPY')")
    result = await client.call_tool("yfinance", "get_options_chain", {"ticker": "SPY"})
    print_result("Options", result)

    return True


async def test_financial_datasets_mcp():
    """Test financial-datasets MCP Server."""
    print("\n" + "=" * 50)
    print("FINANCIAL-DATASETS MCP SERVER")
    print("=" * 50)

    api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
    if not api_key:
        print("  ⚠ FINANCIAL_DATASETS_API_KEY not configured")
        return False

    client = get_mcp_client()

    # List tools
    tools = await client.list_tools("financial-datasets")
    print(f"\n  Tools available: {len(tools)}")
    for t in tools:
        print(f"    - {t['name']}")

    # Test get_current_stock_price
    print("\n[Test] get_current_stock_price('MSFT')")
    result = await client.call_tool("financial-datasets", "get_current_stock_price", {"ticker": "MSFT"})
    print_result("Current Price", result)

    # Test get_income_statements
    print("\n[Test] get_income_statements('AAPL')")
    result = await client.call_tool("financial-datasets", "get_income_statements", {
        "ticker": "AAPL",
        "period": "annual",
        "limit": 1
    })
    print_result("Income Statement", result)

    # Test get_company_news
    print("\n[Test] get_company_news('GOOGL')")
    result = await client.call_tool("financial-datasets", "get_company_news", {
        "ticker": "GOOGL",
        "limit": 3
    })
    print_result("Company News", result)

    # Test get_current_crypto_price
    print("\n[Test] get_current_crypto_price('BTC')")
    result = await client.call_tool("financial-datasets", "get_current_crypto_price", {"ticker": "BTC"})
    print_result("Crypto Price", result)

    return True


async def main():
    """Run all MCP tests."""
    print("\n" + "=" * 50)
    print("MCP SERVERS TEST SUITE")
    print("=" * 50)

    # Setup servers
    setup_default_servers(use_local=True)

    results = {
        "yfinance": await test_yfinance_mcp(),
        "financial-datasets": await test_financial_datasets_mcp(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL/SKIP"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
