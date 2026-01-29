# tests/test_api_clients.py
"""
Tests for API clients (excluding yfinance - already in MCP).

Tests:
- Finnhub API
- Alpha Vantage API
- CoinGecko API
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datasources.api_clients import (
    FinnhubClient,
    AlphaVantageClient,
    CoinGeckoClient,
    get_client
)


def print_result(name: str, result):
    """Pretty print test result."""
    if result.success:
        print(f"  ✓ {name}")
        if hasattr(result.data, '__dict__'):
            for k, v in result.data.__dict__.items():
                if v is not None and k != 'source':
                    print(f"    {k}: {v}")
        elif isinstance(result.data, dict):
            for k, v in list(result.data.items())[:5]:
                print(f"    {k}: {v}")
        elif isinstance(result.data, list):
            print(f"    items: {len(result.data)}")
    else:
        print(f"  ✗ {name}: {result.error}")


def test_finnhub():
    """Test Finnhub API client."""
    print("\n" + "=" * 50)
    print("FINNHUB API")
    print("=" * 50)

    client = FinnhubClient()

    if not client.available:
        print("  ⚠ Finnhub API key not configured")
        return False

    print(f"  API Key: {os.getenv('FINNHUB_API_KEY', '')[:10]}...")

    # Test quote
    print("\n[Test] get_quote('AAPL')")
    result = client.get_quote("AAPL")
    print_result("Quote", result)

    # Test fundamentals
    print("\n[Test] get_fundamentals('MSFT')")
    result = client.get_fundamentals("MSFT")
    print_result("Fundamentals", result)

    # Test news
    print("\n[Test] get_news('TSLA')")
    result = client.get_news("TSLA", limit=3)
    print_result("News", result)

    return True


def test_alphavantage():
    """Test Alpha Vantage API client."""
    print("\n" + "=" * 50)
    print("ALPHA VANTAGE API")
    print("=" * 50)

    client = AlphaVantageClient()

    if not client.available:
        print("  ⚠ Alpha Vantage API key not configured")
        return False

    print(f"  API Key: {os.getenv('ALPHAVANTAGE_API_KEY', '')[:10]}...")

    # Test quote
    print("\n[Test] get_quote('IBM')")
    result = client.get_quote("IBM")
    print_result("Quote", result)

    # Test fundamentals
    print("\n[Test] get_fundamentals('GOOGL')")
    result = client.get_fundamentals("GOOGL")
    print_result("Fundamentals", result)

    return True


def test_coingecko():
    """Test CoinGecko API client."""
    print("\n" + "=" * 50)
    print("COINGECKO API (Public)")
    print("=" * 50)

    client = CoinGeckoClient()

    # Test Bitcoin
    print("\n[Test] get_quote('BTC')")
    result = client.get_quote("BTC")
    print_result("Bitcoin", result)

    # Test Ethereum
    print("\n[Test] get_quote('ETH')")
    result = client.get_quote("ETH")
    print_result("Ethereum", result)

    # Test Solana
    print("\n[Test] get_quote('SOL')")
    result = client.get_quote("SOL")
    print_result("Solana", result)

    # Test historical
    print("\n[Test] get_historical('BTC', period='7')")
    result = client.get_historical("BTC", period="7")
    print_result("Historical", result)

    return True


def main():
    """Run all API tests."""
    print("\n" + "=" * 50)
    print("API CLIENTS TEST SUITE")
    print("=" * 50)

    results = {
        "Finnhub": test_finnhub(),
        "Alpha Vantage": test_alphavantage(),
        "CoinGecko": test_coingecko(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL/SKIP"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
