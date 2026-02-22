import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.nodes.router import classify_trading_subtype

ROUTING_FIXTURES = [
    ("Should I buy AAPL?", "full_trading"),
    ("What's AAPL's P/E ratio?", "fundamental"),
    ("Is AAPL overvalued?", "fundamental"),
    ("Is TSLA oversold?", "technical"),
    ("What's the RSI for NVDA?", "technical"),
    ("What's the sentiment on AAPL?", "sentiment"),
    ("Any news on TSLA?", "news"),
    ("Latest headlines for GOOGL", "news"),
]

@pytest.mark.parametrize("query,expected", ROUTING_FIXTURES)
def test_classify_trading_subtype(query, expected):
    result = classify_trading_subtype(query)
    assert result == expected, f"Query '{query}' expected '{expected}', got '{result}'"
