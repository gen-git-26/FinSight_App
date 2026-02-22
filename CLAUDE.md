# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinSight is a Multi-Agent Financial Analysis System built with LangGraph. It provides real-time market data, options analysis, fundamental research, and trading recommendations through a Streamlit interface.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install local MCP servers
pip install -e datasources/mcp_servers

# Run the application
streamlit run app.py --server.port 8502 --server.headless true

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_api_clients.py
pytest tests/test_mcp_servers.py
pytest tests/test_memory_system.py

# CLI testing (bypasses Streamlit)
python -m agent.graph "Should I buy AAPL?"

# Run memory system tests directly
python tests/test_memory_system.py
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
  - `nodes/` - Agent implementations:
    - Standard: `router.py`, `fetcher.py`, `crypto.py`, `analyst.py`, `composer.py`
    - A2A Trading: `analysts.py`, `researchers.py`, `trader.py`, `risk_manager.py`, `fund_manager.py`

- `datasources/` - Data fetching layer
  - `__init__.py` - `DataFetcher` unified interface with automatic fallback chain
  - `models.py` - Data models: `StockQuote`, `Fundamentals`, `OptionsData`, `NewsItem`, `CryptoQuote`
  - `api_clients.py` - `YFinanceClient`, `FinnhubClient`, `AlphaVantageClient`, `CoinGeckoClient`
  - `mcp_client.py` - MCP protocol client (JSON-RPC over stdio)
  - `mcp_servers/yfinance_server.py` - Local yfinance MCP server (7 tools, FastMCP)
  - `mcp_servers/financial_datasets_server.py` - Financial Datasets API MCP server (11 tools)

- `infrastructure/` - Memory management system
  - `memory_manager.py` - **Unified interface** with smart routing and race pattern
  - `memory_types.py` - `QueryIntent`, `MemoryLayer`, `TokenBudget`, `MemoryContext` enums/dataclasses
  - `query_classifier.py` - **2-stage classifier**: regex/keywords (Stage 1) + LLM fallback (Stage 2)
  - `redis_stm.py` - Short-term memory: session history, user snapshot with versioning
  - `run_cache.py` - **Run-level tool cache**: deduplicates A2A data fetches (Redis DB 1)
  - `postgres_ltm.py` - Long-term memory: user profiles, trading decisions (PostgreSQL)
  - `postgres_summaries.py` - Pre-computed user/ticker summaries (updated at write-time)

- `rag/` - Retrieval-Augmented Generation
  - `qdrant_client.py` - `HybridQdrant`: dense (3072d) + sparse (BM25) vectors + RRF fusion
  - `embeddings.py` - OpenAI `text-embedding-3-large` with 7-day file cache
  - `fusion.py` - `ingest_raw()`, `retrieve()`, `rerank_and_summarize()`

- `utils/` - Utilities
  - `config.py` - `load_settings()` from `.env`
  - `cache.py` - `FileTTLCache` for embeddings

### Query Routing

Router (`agent/nodes/router.py`) uses hybrid routing:
1. LLM-based detection (primary)
2. Keyword fallback (secondary)

Routing keywords:
- **Trading**: "buy", "sell", "should I", "invest", "forecast" → A2A flow
- **Crypto**: bitcoin, ethereum, BTC, ETH, etc. → Crypto agent
- **Info**: price/data requests → Standard flow

### State Management

All agents share `AgentState` (TypedDict in `agent/state.py`). Key fields:
- `query`, `user_id` - Input
- `parsed_query`, `next_agent`, `is_trading_query` - Router output
- `fetched_data` - Data from Fetcher/Crypto
- `analyst_reports`, `research_report`, `trading_decision`, `risk_assessment`, `fund_manager_decision` - Trading flow
- `response`, `sources` - Final output

### Data Sources — Fallback Chain

```
Stock query:  yfinance → Finnhub → Alpha Vantage
Crypto query: yfinance (BTC-USD format) → CoinGecko
```

**Fetch strategies** (`FetchStrategy` enum in `datasources/__init__.py`):
- `PREFER_MCP` (default) - Try MCP first, fallback to API
- `FIRST_SUCCESS` - Stop at first success
- `ALL_SOURCES` - Fetch from all (for comparison)

After every successful fetch, data is automatically ingested into Qdrant (RAG).

### Memory System Architecture

```
Query → 2-Stage Classifier → Route to minimal layers → Race pattern fetch → Token budget
```

**Memory layers** (in order of latency):
| Layer | Storage | Use case | Latency |
|-------|---------|----------|---------|
| `RUN_CACHE` | Redis DB 1 | Tool results within A2A run | <10ms |
| `STM` | Redis DB 0 | Session history, user snapshot | <10ms |
| `LTM` | PostgreSQL | User profiles, decision history | ~30ms |
| `RAG` | Qdrant | Semantic search, historical context | ~50ms |

**Token budgets by intent** (dynamic, set in `TokenBudget.for_intent()`):
- `PRICE_ONLY` → 600 tokens, RunCache only
- `TRADE_DECISION` → 5000 tokens, all layers
- `USER_HISTORY` → 3000 tokens, STM + LTM

**Key design decisions:**
- `RunCache` separate from STM — tool results vs conversation memory
- Postgres summaries updated at write-time (not read-time) for fast retrieval
- User snapshot in Redis with version number → invalidated when Postgres updates

## Integration Status

### ✅ Connected & Working
- Full LangGraph agent graph (both flows)
- All API clients with fallback chain
- RAG ingestion (DataFetcher → Qdrant on every successful fetch)
- Redis STM (router reads, composer writes)
- Streamlit UI

### ❌ Built but NOT Yet Connected to Agents
- `MemoryManager` — exists, not called by agent nodes
- `QueryClassifier` — exists, not used by router
- `RunCache` — exists, not used in A2A flow
- `PostgreSQL LTM` — exists, requires `DATABASE_URL` in `.env`
- `PostgresSummaries` — exists, not called after trading decisions
- `RAG retrieval` — ingestion works, retrieval not used in agent prompts
- MCP servers — code exists, not added to DataFetcher fallback chain

### Next Integration Steps (Priority Order)
1. **P1**: Add `DATABASE_URL` to `.env` and initialize Postgres on startup
2. **P1**: Call `memory_manager.get_context()` in `router_node` and pass to state
3. **P1**: Call `memory_manager.store_decision()` in `fund_manager_node`
4. **P2**: Check `RunCache` in `fetcher_node` before API calls (A2A deduplication)
5. **P2**: Pass RAG context chunks into analyst LLM prompts
6. **P2**: Replace router classification with `QueryClassifier`

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Optional — Memory:
```
DATABASE_URL=postgresql://user:pass@localhost:5432/finsight   # Needed for LTM
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0                    # STM database
REDIS_CACHE_DB=1              # RunCache database
```

Optional — RAG:
```
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION=finsight
```

Optional — Data Sources:
```
FINNHUB_API_KEY=...
ALPHAVANTAGE_API_KEY=...
FINANCIAL_DATASETS_API_KEY=...
COINSTATS_API_KEY=...
```

## Python Version

Requires Python 3.12+

## Reference Documents

- `PROJECT.md` — Full architecture diagram, connectivity matrix, and integration roadmap
