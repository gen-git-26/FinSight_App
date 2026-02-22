# FinSight — Project Summary

> Multi-Agent Financial Analysis System | LangGraph + A2A Architecture

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          app.py (Streamlit UI)                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ run_query()
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH AGENT GRAPH                        │
│                                                                     │
│    ┌─────────┐     ┌─────────┐     ┌──────────┐    ┌──────────┐   │
│    │ ROUTER  │────►│ FETCHER │────►│ ANALYST  │───►│ COMPOSER │   │
│    │         │     │         │     │          │    │          │   │
│    └────┬────┘     └─────────┘     └──────────┘    └──────────┘   │
│         │                                                           │
│         │ (if trading query)                                        │
│         ▼                                                           │
│    ┌──────────────────────────────────────────────────────────┐    │
│    │                  A2A TRADING FLOW                        │    │
│    │                                                          │    │
│    │  ┌──────────┐  4× parallel   ┌──────────────────────┐  │    │
│    │  │ FETCHER  │──────────────►│ ANALYSTS TEAM         │  │    │
│    │  │(trading) │               │ Fundamental           │  │    │
│    │  └──────────┘               │ Sentiment             │  │    │
│    │                             │ Technical             │  │    │
│    │                             │ News                  │  │    │
│    │                             └──────────┬───────────┘  │    │
│    │                                        │               │    │
│    │                             ┌──────────▼───────────┐  │    │
│    │                             │ RESEARCHERS           │  │    │
│    │                             │ Bull ◄─3 rounds─► Bear│  │    │
│    │                             └──────────┬───────────┘  │    │
│    │                                        │               │    │
│    │                             ┌──────────▼───────────┐  │    │
│    │                             │ TRADER                │  │    │
│    │                             │ BUY / SELL / HOLD    │  │    │
│    │                             └──────────┬───────────┘  │    │
│    │                                        │               │    │
│    │                             ┌──────────▼───────────┐  │    │
│    │                             │ RISK MANAGER          │  │    │
│    │                             │ Risky/Neutral/Safe    │  │    │
│    │                             │ 3-persona × 3 rounds  │  │    │
│    │                             └──────────┬───────────┘  │    │
│    │                                        │               │    │
│    │                             ┌──────────▼───────────┐  │    │
│    │                             │ FUND MANAGER          │  │    │
│    │                             │ APPROVE/MODIFY/REJECT │  │    │
│    │                             └──────────┬───────────┘  │    │
│    │                                        │               │    │
│    └────────────────────────────────────────┼───────────────┘    │
│                                             │                      │
│                                    ┌────────▼────────┐            │
│                                    │    COMPOSER      │            │
│                                    └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
┌─────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│  DATA LAYER │   │  MEMORY / INFRA   │   │    RAG LAYER         │
│             │   │                   │   │                      │
│  yfinance   │   │  Redis STM        │   │  Qdrant              │
│  Finnhub    │   │  Redis RunCache   │   │  OpenAI Embeddings   │
│  AlphaVant. │   │  Postgres LTM     │   │  Hybrid Search (RRF) │
│  CoinGecko  │   │  PG Summaries     │   │  fusion.py           │
│  MCP servers│   │  QueryClassifier  │   │                      │
└─────────────┘   └───────────────────┘   └──────────────────────┘
```

---

## 2. Two Query Flows

### Standard Flow — Info Queries
*"What's the price of AAPL?" / "Tesla news"*
```
ROUTER → FETCHER → ANALYST → COMPOSER
```

### Trading Flow (A2A) — Decision Queries
*"Should I buy TSLA?" / "Analyze NVDA for me"*
```
ROUTER → FETCHER → ANALYSTS(×4) → RESEARCHERS → TRADER → RISK MANAGER → FUND MANAGER → COMPOSER
```

**Router detection keywords:**
- Trading: `buy`, `sell`, `should i`, `invest`, `forecast`, `worth buying`
- Crypto: `bitcoin`, `ethereum`, `BTC`, `ETH`, etc.
- Info: price, news, fundamentals → Standard flow

---

## 3. Component Breakdown

### 3.1 Agent Graph (`agent/`)

| File | Role |
|------|------|
| `graph.py` | LangGraph builder, routing logic, `run_query()` entry |
| `state.py` | `AgentState` TypedDict — shared across all nodes |
| `nodes/router.py` | Classifies query, extracts tickers, routes |
| `nodes/fetcher.py` | Fetches data via DataFetcher (stocks, options, news) |
| `nodes/crypto.py` | Handles crypto queries (BTC-USD normalization) |
| `nodes/analyst.py` | Single LLM analyst (standard flow) |
| `nodes/composer.py` | Final response formatting, saves to STM |
| `nodes/analysts.py` | 4 parallel specialists (A2A) |
| `nodes/researchers.py` | Bull vs Bear debate, 3 rounds (A2A) |
| `nodes/trader.py` | BUY/SELL/HOLD decision with conviction score (A2A) |
| `nodes/risk_manager.py` | 3-persona risk debate (A2A) |
| `nodes/fund_manager.py` | Final approval — APPROVE/MODIFY/REJECT (A2A) |

### 3.2 Data Layer (`datasources/`)

| File | Role |
|------|------|
| `__init__.py` | `DataFetcher` — unified interface with automatic fallback |
| `models.py` | Data types: `StockQuote`, `Fundamentals`, `OptionsData`, `NewsItem`, `CryptoQuote` |
| `api_clients.py` | `YFinanceClient`, `FinnhubClient`, `AlphaVantageClient`, `CoinGeckoClient` |
| `mcp_client.py` | MCP protocol client (JSON-RPC over stdio) |
| `mcp_servers/yfinance_server.py` | Local yfinance MCP server — 7 tools |
| `mcp_servers/financial_datasets_server.py` | Financial Datasets API MCP server — 11 tools |

**Fallback chain:**
```
fetch("AAPL") → yfinance → Finnhub → Alpha Vantage
fetch("BTC")  → yfinance (BTC-USD) → CoinGecko
```

### 3.3 Memory / Infrastructure (`infrastructure/`)

| File | Role |
|------|------|
| `memory_manager.py` | Unified interface — routing, race pattern, token budgets |
| `memory_types.py` | `QueryIntent`, `MemoryLayer`, `TokenBudget`, `MemoryContext` |
| `query_classifier.py` | 2-stage: regex/keywords (Stage 1) + LLM fallback (Stage 2) |
| `redis_stm.py` | Session history, user preference snapshot (with versioning) |
| `run_cache.py` | Tool result deduplication within a run (A2A optimization) |
| `postgres_ltm.py` | User profiles, decision history (PostgreSQL) |
| `postgres_summaries.py` | Pre-computed user/ticker summaries (write-time updates) |

**Memory routing by intent:**
```
PRICE_ONLY      → RunCache only          → ~10ms, 600 tokens
TRADE_DECISION  → RunCache + LTM + RAG  → ~200ms, 5000 tokens
USER_HISTORY    → STM + LTM             → ~50ms, 3000 tokens
USER_PREFS      → STM snapshot → LTM    → ~30ms, 2000 tokens
SEMANTIC_SEARCH → RAG only              → ~50ms, 3500 tokens
```

### 3.4 RAG Layer (`rag/`)

| File | Role |
|------|------|
| `qdrant_client.py` | `HybridQdrant` — dense (3072d) + sparse (BM25) + RRF fusion |
| `embeddings.py` | `text-embedding-3-large` with 7-day file cache |
| `fusion.py` | `ingest_raw()`, `retrieve()`, `rerank_and_summarize()`, `@fuse` decorator |

**Payload indices (pre-filter before similarity):**
```
symbol → KEYWORD (e.g., "AAPL")
type   → KEYWORD (e.g., "news", "fundamentals")
date   → TEXT
user   → KEYWORD
```

---

## 4. Connectivity Status

### ✅ Fully Connected

| Component | Connected To |
|-----------|-------------|
| `app.py` | `agent.graph.run_query()` |
| `router_node` | All nodes (routing logic) |
| `fetcher_node` | `DataFetcher` → all API clients |
| `crypto_node` | `DataFetcher` → yfinance + CoinGecko |
| `analyst_node` | OpenAI API |
| `composer_node` | OpenAI API + `RedisSTM` (saves messages) |
| `analysts_node` (A2A) | OpenAI API |
| `researchers_node` (A2A) | OpenAI API |
| `trader_node` (A2A) | OpenAI API |
| `risk_manager_node` (A2A) | OpenAI API |
| `fund_manager_node` (A2A) | OpenAI API |
| `DataFetcher` | RAG ingestion (writes to Qdrant on success) |

### ⚠️ Partially Connected

| Component | Connected | Missing |
|-----------|-----------|---------|
| `RedisSTM` | Router (reads), Composer (writes) | Other nodes don't use session context |
| `RAG / Qdrant` | DataFetcher ingests data | Agents never retrieve from RAG |
| `MemoryManager` | Infrastructure built | Not called by any agent node |
| `MCP Servers` | Code exists, tested | Not added to DataFetcher fallback chain |

### ❌ Not Connected (Built but Dormant)

| Component | Status | What's Missing |
|-----------|--------|----------------|
| `QueryClassifier` | Built, tested | Not called by any node |
| `RunCache` | Built, tested | Not called by any node |
| `PostgreSQL LTM` | Built | No `DATABASE_URL` in `.env`, never initialized |
| `PostgresSummaries` | Built | Never called after trading decisions |
| `MemoryManager` | Built, tested | Not called by agent nodes |
| LTM user context | Built | Agents don't pass user preferences to LLM prompts |

---

## 5. What Still Needs Connecting

### Priority 1 — Critical Path

**A. PostgreSQL setup**
- Add `DATABASE_URL` to `.env`
- Initialize tables on app startup
- `postgres_summaries.save_decision_with_summaries()` must be called after `fund_manager_node`

**B. Memory Manager → Agent Nodes**
- `fetcher_node` should call `cache.get_quote(run_id, ticker)` before fetching
- `router_node` should call `memory_manager.get_context()` and pass `MemoryContext` to state
- `fund_manager_node` should call `memory_manager.store_decision()` after approval

**C. Memory Context → LLM Prompts**
- `analyst_node` / `analysts_node` prompts should include user risk profile
- `risk_manager_node` should receive user's historical risk decisions
- `trader_node` should know user's previous positions on the same ticker

---

### Priority 2 — Quality

**D. RAG Retrieval in Analysts**
- `analysts_node` should call `rag.fusion.retrieve(query, symbol=ticker)`
- Pass 3–5 relevant chunks to each analyst's LLM prompt
- Immediately improves analysis quality with historical context

**E. QueryClassifier → Router**
- Replace router's inline LLM classification with `get_classifier().classify(query)`
- Use `classification.layers_needed` to drive memory fetch decisions
- Removes duplicate classification logic

**F. RunCache in A2A Flow**
- `fetcher_node` (trading) should set `run_cache.set_quote(run_id, ticker, data)`
- All 4 `analysts_node` calls should check `run_cache.get_*` before fetching
- Prevents 4× duplicate API calls for same ticker in A2A flow

---

### Priority 3 — Enhancement

**G. MCP Servers in DataFetcher**
- Add `mcp_yfinance` and `mcp_financial_datasets` to the fallback chain
- Or run MCP + API in parallel, return first success

**H. Token Budget in Composer**
- `composer_node` should receive `MemoryContext.budget` from state
- Trim conversation history and RAG chunks to budget limits

**I. Cache Invalidation Hooks**
- After `postgres_summaries.update_user_summary()`, call `stm.invalidate_user_snapshot(user_id)`
- Ensures Redis snapshot is refreshed on next query

---

## 6. File Connection Map

```
app.py
└── agent/graph.py
    ├── agent/state.py (AgentState)
    ├── agent/nodes/router.py
    │   └── infrastructure/redis_stm.py ✅
    ├── agent/nodes/fetcher.py
    │   ├── datasources/ ✅
    │   └── rag/fusion.py (ingest only) ✅
    ├── agent/nodes/crypto.py
    │   └── datasources/ ✅
    ├── agent/nodes/analyst.py
    │   └── utils/config.py ✅
    ├── agent/nodes/composer.py
    │   ├── utils/config.py ✅
    │   └── infrastructure/redis_stm.py ✅
    ├── agent/nodes/analysts.py → utils/config.py ✅
    ├── agent/nodes/researchers.py → utils/config.py ✅
    ├── agent/nodes/trader.py → utils/config.py ✅
    ├── agent/nodes/risk_manager.py → utils/config.py ✅
    └── agent/nodes/fund_manager.py → utils/config.py ✅

                     DISCONNECTED
─────────────────────────────────────────────────────
infrastructure/memory_manager.py  ←─ not called by agents
infrastructure/query_classifier.py ←─ not called by agents
infrastructure/run_cache.py        ←─ not called by agents
infrastructure/postgres_ltm.py     ←─ not initialized
infrastructure/postgres_summaries.py ←─ not called
datasources/mcp_client.py          ←─ not in fallback chain
rag/qdrant_client.py (retrieval)   ←─ ingestion only
rag/fusion.py (retrieve)           ←─ not called by agents
```

---

## 7. Environment Variables Required

```bash
# REQUIRED
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

# MEMORY (needed to activate LTM)
DATABASE_URL=postgresql://user:pass@localhost:5432/finsight
REDIS_HOST=localhost
REDIS_PORT=6379

# RAG (needed for semantic search)
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION=finsight

# DATA SOURCES
FINNHUB_API_KEY=...
ALPHAVANTAGE_API_KEY=...
FINANCIAL_DATASETS_API_KEY=...
COINSTATS_API_KEY=...

# OPTIONAL
OPENAI_EMBED_MODEL=text-embedding-3-large
REDIS_CACHE_DB=1
```

---

## 8. Component Build Status

| Layer | Component | Built | Connected | Priority to Connect |
|-------|-----------|:-----:|:---------:|:-------------------:|
| **Agent** | Router | ✅ | ✅ | — |
| **Agent** | Fetcher | ✅ | ✅ | — |
| **Agent** | Crypto | ✅ | ✅ | — |
| **Agent** | Analyst | ✅ | ✅ | — |
| **Agent** | Composer | ✅ | ✅ | — |
| **Agent** | Analysts (A2A) | ✅ | ✅ | — |
| **Agent** | Researchers (A2A) | ✅ | ✅ | — |
| **Agent** | Trader (A2A) | ✅ | ✅ | — |
| **Agent** | Risk Manager (A2A) | ✅ | ✅ | — |
| **Agent** | Fund Manager (A2A) | ✅ | ✅ | — |
| **Data** | yfinance | ✅ | ✅ | — |
| **Data** | Finnhub | ✅ | ✅ | — |
| **Data** | Alpha Vantage | ✅ | ✅ | — |
| **Data** | CoinGecko | ✅ | ✅ | — |
| **Data** | MCP yfinance | ✅ | ❌ | P3 |
| **Data** | MCP financial-datasets | ✅ | ❌ | P3 |
| **Memory** | Redis STM | ✅ | ⚠️ | P1 |
| **Memory** | Run Cache | ✅ | ❌ | P2 |
| **Memory** | Postgres LTM | ✅ | ❌ | P1 |
| **Memory** | Postgres Summaries | ✅ | ❌ | P1 |
| **Memory** | Memory Manager | ✅ | ❌ | P1 |
| **Memory** | Query Classifier | ✅ | ❌ | P2 |
| **RAG** | Qdrant (ingest) | ✅ | ✅ | — |
| **RAG** | Qdrant (retrieve) | ✅ | ❌ | P2 |
| **RAG** | Embeddings | ✅ | ✅ | — |
| **RAG** | Fusion / Rerank | ✅ | ❌ | P2 |
| **UI** | Streamlit | ✅ | ✅ | — |

**Legend:** ✅ Done / ⚠️ Partial / ❌ Not connected | P1=Critical / P2=Quality / P3=Enhancement
