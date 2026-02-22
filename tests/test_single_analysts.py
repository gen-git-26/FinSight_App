import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.state import AgentState, FetchedData

def test_single_fundamental_analyst_node_exists():
    from agent.nodes.analysts import single_fundamental_node
    assert callable(single_fundamental_node)

def test_single_technical_analyst_node_exists():
    from agent.nodes.analysts import single_technical_node
    assert callable(single_technical_node)

def test_single_sentiment_analyst_node_exists():
    from agent.nodes.analysts import single_sentiment_node
    assert callable(single_sentiment_node)

def test_single_news_analyst_node_exists():
    from agent.nodes.analysts import single_news_node
    assert callable(single_news_node)
