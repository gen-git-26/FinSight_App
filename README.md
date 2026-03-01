# FinSight

**Multi-Agent Financial Analysis System**

FinSight is a production-grade financial intelligence platform that orchestrates multiple specialized AI agents to deliver real-time market data, options analysis, fundamental research, and trade recommendations. Built on LangGraph with a Streamlit interface.

---

## Overview

Most financial AI tools are single-agent wrappers around a data API. FinSight is different: it uses a graph-based multi-agent architecture where each agent has a defined role and queries are routed through the appropriate pipeline depending on their nature.

A price inquiry and a trade decision are handled by fundamentally different execution paths — the system knows the difference and applies the right depth of analysis for each.

---

## Key Capabilities

- **Dual query pipeline** — standard info queries and full trading analysis run on separate, optimized paths
- **Agent-to-Agent (A2A) trading flow** — four parallel analysts, bull/bear researcher debate, independent risk assessment
- **Multi-layer memory** — Redis (session), PostgreSQL (long-term), Qdrant (semantic retrieval), and in-run tool cache
- **Resilient data fetching** — automatic fallback chain across yfinance, Finnhub, and Alpha Vantage
- **MCP-native data layer** — local FastMCP servers for yfinance and Financial Datasets API
- **RAG integration** — every successful data fetch is ingested into a hybrid dense/sparse vector store

---

## Query Flows

### Standard Flow

Used for informational queries: price checks, news, fundamental data, crypto quotes.

```
router → fetcher / crypto → analyst → composer → response
```

### Trading Flow (A2A)

Triggered by trading intent: buy/sell decisions, investment recommendations, forecasts.

```
router → fetcher
       → analysts_team (4 agents, parallel)
       → researchers (bull/bear debate, 3 rounds)
       → trader (BUY / SELL / HOLD)
       → risk_manager (3 personas, 3 rounds)
       → fund_manager
       → composer → response
```

**Routing logic** — the router uses LLM-based intent detection with keyword fallback:

| Signal | Destination |
|--------|-------------|
| "buy", "sell", "should I", "invest", "forecast" | Trading flow |
| BTC, ETH, bitcoin, ethereum, crypto tickers | Crypto agent |
| price, data, compare, news | Standard flow |

---

## Architecture

### Project Structure

```
FinSight_App/
├── app.py                          # Streamlit application entry point
├── agent/
│   ├── graph.py                    # LangGraph graph builder, run_query / run_query_async
│   ├── state.py                    # AgentState TypedDict (shared across all nodes)
│   └── nodes/
│       ├── router.py               # Intent detection and flow routing
│       ├── fetcher.py              # Stock data fetching with fallback chain
│       ├── crypto.py               # Cryptocurrency data agent
│       ├── analyst.py              # Single-pass analysis (standard flow)
│       ├── analysts.py             # Parallel 4-analyst team (trading flow)
│       ├── researchers.py          # Bull/bear debate, multi-round
│       ├── trader.py               # Trade decision synthesis
│       ├── risk_manager.py         # Multi-persona risk assessment
│       ├── fund_manager.py         # Final allocation decision
│       └── composer.py             # Response formatting
├── datasources/
│   ├── __init__.py                 # DataFetcher unified interface
│   ├── models.py                   # StockQuote, Fundamentals, OptionsData, NewsItem, CryptoQuote
│   ├── api_clients.py              # YFinanceClient, FinnhubClient, AlphaVantageClient, CoinGeckoClient
│   ├── mcp_client.py               # MCP protocol client (JSON-RPC over stdio)
│   └── mcp_servers/
│       ├── yfinance_server.py      # Local yfinance MCP server (7 tools, FastMCP)
│       └── financial_datasets_server.py  # Financial Datasets API MCP server (11 tools)
├── infrastructure/
│   ├── memory_manager.py           # Unified memory interface with smart routing
│   ├── memory_types.py             # QueryIntent, MemoryLayer, TokenBudget, MemoryContext
│   ├── query_classifier.py         # 2-stage classifier: regex/keywords + LLM fallback
│   ├── redis_stm.py                # Short-term memory: session history, user snapshot
│   ├── run_cache.py                # Run-level tool cache (Redis DB 1)
│   ├── postgres_ltm.py             # Long-term memory: user profiles, trading decisions
│   └── postgres_summaries.py       # Pre-computed summaries (updated at write-time)
├── rag/
│   ├── qdrant_client.py            # HybridQdrant: dense (3072d) + sparse (BM25) + RRF fusion
│   ├── embeddings.py               # OpenAI text-embedding-3-large, 7-day file cache
│   └── fusion.py                   # ingest_raw(), retrieve(), rerank_and_summarize()
└── utils/
    ├── config.py                   # load_settings() from .env
    └── cache.py                    # FileTTLCache for embeddings
```

### Data Fetching — Fallback Chain

```
Stock:   MCP (yfinance) → yfinance direct → Finnhub → Alpha Vantage
Crypto:  MCP (yfinance) → yfinance (BTC-USD format) → CoinGecko
```

Fetch strategies are configurable via `FetchStrategy` enum: `PREFER_MCP` (default), `FIRST_SUCCESS`, `ALL_SOURCES`.

After every successful fetch, data is automatically ingested into Qdrant for semantic retrieval.

### Memory System

```
Query → 2-Stage Classifier → Route to minimal layers → Race pattern fetch → Token budget
```

| Layer | Backend | Purpose | Latency |
|-------|---------|---------|---------|
| RunCache | Redis DB 1 | Deduplicates A2A tool fetches within a single run | < 10ms |
| STM | Redis DB 0 | Session history and user snapshot with versioning | < 10ms |
| LTM | PostgreSQL | User profiles and trading decision history | ~30ms |
| RAG | Qdrant | Semantic retrieval of historical context | ~50ms |

Token budgets are dynamically allocated per query intent — a `PRICE_ONLY` query gets 600 tokens from RunCache only; a `TRADE_DECISION` query gets 5000 tokens across all layers.

---

## Installation

### Prerequisites

- Python 3.12 or higher
- OpenAI API key (required)

### 1. Clone the Repository

```bash
git clone https://github.com/gen-git-26/FinSight_App
cd FinSight_App
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Local MCP Servers

```bash
pip install -e datasources/mcp_servers
```

---

## Configuration

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Memory — Short-term (Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_CACHE_DB=1

# Memory — Long-term (PostgreSQL)
DATABASE_URL=postgresql://user:pass@localhost:5432/finsight

# RAG (Qdrant)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION=finsight

# Data Sources (optional — extends fallback chain)
FINNHUB_API_KEY=...
ALPHAVANTAGE_API_KEY=...
FINANCIAL_DATASETS_API_KEY=...
```

Only `OPENAI_API_KEY` is strictly required. Memory layers and additional data sources degrade gracefully when their configuration is absent.

---

## Usage

### Run the Application

```bash
streamlit run app.py --server.port 8502 --server.headless true
```

### CLI Mode (bypasses Streamlit)

```bash
python -m agent.graph "Should I buy AAPL?"
```

### Example Queries

```
"Tesla stock price"
"Should I buy NVDA right now?"
"Compare the quarterly income statements of Amazon and Google"
"AAPL options expiring 2025-06-20"
"Bitcoin price"
"What are the key financial metrics for Meta Platforms?"
"Bull or bear case for MSFT this quarter?"
```

---

## Testing

```bash
# Full test suite
pytest tests/

# Individual test files
pytest tests/test_api_clients.py
pytest tests/test_mcp_servers.py
pytest tests/test_memory_system.py

# Memory system direct run
python tests/test_memory_system.py
```

---


## License

MIT
