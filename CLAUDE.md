# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinSight is a Multi-Agent Financial Analysis System built with LangGraph. It provides real-time market data, options analysis, fundamental research, and trading recommendations through a Streamlit interface.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install MCP servers (required for data fetching)
pip install -e vendors/yahoo-finance-mcp
pip install -e vendors/financial-datasets-mcp

# Run the application
streamlit run app.py

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_api_clients.py

# CLI testing (bypasses Streamlit)
python -m agent.graph "Should I buy AAPL?"
```

## Architecture

### Two Query Processing Flows

**Standard Flow** (info queries like "Tesla stock price"):
```
router → fetcher/crypto → analyst → composer → response
```

**Trading Flow (A2A)** (trading queries like "Should I buy AAPL?"):
```
router → fetcher → analysts_team (4 parallel) → researchers (bull/bear debate, 3 rounds)
       → trader (BUY/SELL/HOLD) → risk_manager (3 personas, 3 rounds) → fund_manager → composer
```

### Key Directories

- `agent/` - LangGraph multi-agent orchestration
  - `graph.py` - Graph builder, routing logic, entry points (`run_query`, `run_query_async`)
  - `state.py` - `AgentState` TypedDict shared across all nodes
  - `nodes/` - Agent implementations (router, fetcher, crypto, analyst, composer, trading agents)

- `datasources/` - Data fetching layer
  - `mcp_client.py` - MCP server communication
  - `api_clients.py` - Direct API clients (Finnhub, Alpha Vantage)
  - `mcp_servers/` - MCP server implementations (yfinance, financial-datasets)

- `infrastructure/` - System services
  - `redis_stm.py` - Short-term memory (Redis)
  - `postgres_ltm.py` - Long-term memory (PostgreSQL)
  - `memory_manager.py` - Memory orchestration
  - `query_classifier.py` - Query classification

- `rag/` - Retrieval-Augmented Generation
  - `qdrant_client.py` - Vector DB wrapper
  - `embeddings.py` - Embedding generation
  - `fusion.py` - RRF (Reciprocal Rank Fusion)

### Query Routing

Router (`agent/nodes/router.py`) uses hybrid routing:
1. LLM-based detection (primary)
2. Keyword fallback (secondary)

Routing keywords:
- **Trading**: "buy", "sell", "should I", "invest", "forecast" → A2A flow
- **Crypto**: bitcoin, ethereum, etc. → Crypto agent
- **Info**: price/data requests → Standard flow

### State Management

All agents share `AgentState` (TypedDict in `agent/state.py`). Key fields:
- `query`, `user_id` - Input
- `parsed_query`, `next_agent`, `is_trading_query` - Router output
- `fetched_data` - Data from Fetcher/Crypto
- `analyst_reports`, `research_report`, `trading_decision`, `risk_assessment`, `fund_manager_decision` - Trading flow
- `response`, `sources` - Final output

### Data Sources

- **yfinance** - Stock data via MCP
- **Finnhub** - Financial news & fundamentals
- **Alpha Vantage** - Time series & technical analysis
- **CoinStats API** - Crypto data

## Environment Variables

Required in `.env`:
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_MODEL` - Model selection (default: gpt-4o-mini)

Optional:
- `QDRANT_URL`, `QDRANT_API_KEY` - Vector DB for RAG
- `REDIS_HOST`, `REDIS_PORT` - Short-term memory
- `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY`, `COINSTATS_API_KEY` - Data source APIs
- `FINANCIAL_DATASETS_API_KEY` - Financial datasets MCP

## Python Version

Requires Python 3.12+
