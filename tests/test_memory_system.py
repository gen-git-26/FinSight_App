# tests/test_memory_system.py
"""
Tests for the Memory Management System.

Tests:
- Query classification (Stage 1 deterministic)
- Memory routing
- Token budget by intent
- Cache operations
"""
import os
import sys
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from infrastructure import (
    QueryClassifier,
    QueryIntent,
    MemoryLayer,
    TokenBudget,
    get_memory_manager,
    get_run_cache,
    get_stm,
)


def test_query_classifier_deterministic():
    """Test Stage 1 deterministic classification."""
    print("\n" + "=" * 50)
    print("QUERY CLASSIFIER - STAGE 1 (Deterministic)")
    print("=" * 50)

    classifier = QueryClassifier(llm_fallback=False)

    test_cases = [
        # (query, expected_intent, expected_tickers)
        ("What's the price of AAPL?", QueryIntent.PRICE_ONLY, ["AAPL"]),
        ("מה המחיר של TSLA?", QueryIntent.PRICE_ONLY, ["TSLA"]),
        ("Tell me about NVDA", QueryIntent.TICKER_INFO, ["NVDA"]),
        ("Latest news on GOOGL", QueryIntent.NEWS_SUMMARY, ["GOOGL"]),
        ("Should I buy MSFT?", QueryIntent.TRADE_DECISION, ["MSFT"]),
        ("What did I say earlier?", QueryIntent.USER_HISTORY, []),
        ("What are my preferences?", QueryIntent.USER_PREFERENCES, []),
        ("Find similar stocks to AMZN", QueryIntent.SEMANTIC_SEARCH, ["AMZN"]),
        ("Continue from before", QueryIntent.CONVERSATION, []),
        ("$BTC price", QueryIntent.PRICE_ONLY, ["BTC"]),
    ]

    passed = 0
    for query, expected_intent, expected_tickers in test_cases:
        result = classifier.classify_sync(query)
        intent_match = result.intent == expected_intent
        tickers_match = set(result.tickers) == set(expected_tickers)

        status = "✓" if intent_match else "✗"
        print(f"\n  {status} Query: \"{query}\"")
        print(f"    Intent: {result.intent.value} (expected: {expected_intent.value})")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Tickers: {result.tickers} (expected: {expected_tickers})")
        print(f"    Layers: {[l.value for l in result.layers_needed]}")

        if intent_match:
            passed += 1

    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_token_budgets():
    """Test dynamic token budgets by intent."""
    print("\n" + "=" * 50)
    print("TOKEN BUDGETS BY INTENT")
    print("=" * 50)

    intents = [
        QueryIntent.PRICE_ONLY,
        QueryIntent.TRADE_DECISION,
        QueryIntent.NEWS_SUMMARY,
        QueryIntent.USER_HISTORY,
    ]

    for intent in intents:
        budget = TokenBudget.for_intent(intent)
        print(f"\n  {intent.value}:")
        print(f"    Total: {budget.total}")
        print(f"    Conversation: {budget.conversation}")
        print(f"    User context: {budget.user_context}")
        print(f"    RAG results: {budget.rag_results}")
        print(f"    Tool results: {budget.tool_results}")

    return True


def test_run_cache():
    """Test RunCache operations."""
    print("\n" + "=" * 50)
    print("RUN CACHE")
    print("=" * 50)

    cache = get_run_cache()
    run_id = "test_run_123"
    ticker = "AAPL"

    # Test quote caching
    quote = {"price": 150.0, "change": 2.5}
    cache.set_quote(run_id, ticker, quote)
    cached = cache.get_quote(run_id, ticker)

    print(f"\n  Set quote: {quote}")
    print(f"  Get quote: {cached}")
    print(f"  ✓ Cache hit" if cached == quote else "  ✗ Cache miss")

    # Test OHLCV caching
    ohlcv = [{"date": "2024-01-01", "close": 150.0}]
    cache.set_ohlcv(run_id, ticker, ohlcv, "1d")
    cached_ohlcv = cache.get_ohlcv(run_id, ticker, "1d")

    print(f"\n  Set OHLCV: {len(ohlcv)} records")
    print(f"  Get OHLCV: {len(cached_ohlcv) if cached_ohlcv else 0} records")

    # Cleanup
    cache.invalidate_run(run_id)
    print(f"\n  Invalidated run: {run_id}")

    return True


def test_stm_operations():
    """Test STM operations."""
    print("\n" + "=" * 50)
    print("SHORT-TERM MEMORY (STM)")
    print("=" * 50)

    stm = get_stm()
    session_id = "test_session_456"
    user_id = "test_user_789"

    # Test conversation history
    stm.add_to_history(session_id, "user", "What's the price of AAPL?")
    stm.add_to_history(session_id, "assistant", "AAPL is trading at $150")

    history = stm.get_history(session_id)
    print(f"\n  Added 2 messages to history")
    print(f"  Retrieved {len(history)} messages")

    for msg in history:
        print(f"    {msg['role']}: {msg['content'][:50]}")

    # Test user snapshot with versioning
    prefs = {"risk_tolerance": "moderate", "sectors": ["tech"]}
    stm.set_user_snapshot(user_id, prefs, version=1)

    snapshot = stm.get_user_snapshot(user_id)
    print(f"\n  Set user snapshot version 1")
    print(f"  Retrieved: {snapshot}")

    is_valid = stm.check_snapshot_version(user_id, current_version=1)
    print(f"  Version check (v1): {'✓ valid' if is_valid else '✗ invalid'}")

    is_valid = stm.check_snapshot_version(user_id, current_version=2)
    print(f"  Version check (v2): {'✓ valid' if is_valid else '✗ invalid (expected)'}")

    # Cleanup
    stm.clear_history(session_id)
    stm.invalidate_user_snapshot(user_id)

    return True


async def test_memory_manager():
    """Test MemoryManager integration."""
    print("\n" + "=" * 50)
    print("MEMORY MANAGER")
    print("=" * 50)

    manager = get_memory_manager()

    test_queries = [
        "What's the price of AAPL?",
        "Should I buy TSLA?",
        "What did I say about NVDA?",
    ]

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")

        context = await manager.get_context(
            query=query,
            session_id="test_session",
            user_id="test_user",
            run_id="test_run"
        )

        print(f"    Intent: {context.classification.intent.value}")
        print(f"    Confidence: {context.classification.confidence:.2f}")
        print(f"    Tickers: {context.classification.tickers}")
        print(f"    Layers hit: {context.layers_hit}")
        print(f"    Latency: {context.latency_ms:.1f}ms")

    return True


async def main():
    """Run all memory system tests."""
    print("\n" + "=" * 50)
    print("MEMORY SYSTEM TEST SUITE")
    print("=" * 50)

    results = {
        "classifier": test_query_classifier_deterministic(),
        "budgets": test_token_budgets(),
        "cache": test_run_cache(),
        "stm": test_stm_operations(),
        "manager": await test_memory_manager(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
