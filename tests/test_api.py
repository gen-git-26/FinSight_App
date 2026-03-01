# tests/test_api.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


def test_health():
    from api import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_response():
    from api import app
    fake_state = {
        "response": "AAPL is trading at $190.",
        "sources": ["yfinance"],
        "is_trading_query": False,
    }
    with patch("api.get_graph") as mock_get_graph:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = fake_state
        mock_get_graph.return_value = mock_graph

        client = TestClient(app)
        response = client.post("/api/query", json={"query": "AAPL price"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "AAPL is trading at $190."
    assert body["sources"] == ["yfinance"]
    assert body["flow"] == "standard"


def test_query_trading_flow():
    from api import app
    fake_state = {
        "response": "Recommendation: BUY AAPL.",
        "sources": ["yfinance", "finnhub"],
        "is_trading_query": True,
    }
    with patch("api.get_graph") as mock_get_graph:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = fake_state
        mock_get_graph.return_value = mock_graph

        client = TestClient(app)
        response = client.post(
            "/api/query",
            json={"query": "Should I buy AAPL?", "user_id": "test-user"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["flow"] == "trading"
    assert "BUY" in body["response"]


def test_query_agent_error_returns_500():
    from api import app
    with patch("api.get_graph") as mock_get_graph:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = RuntimeError("OpenAI timeout")
        mock_get_graph.return_value = mock_graph

        client = TestClient(app)
        response = client.post("/api/query", json={"query": "AAPL price"})

    assert response.status_code == 500
    assert "OpenAI timeout" in response.json()["detail"]
