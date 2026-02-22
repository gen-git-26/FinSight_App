# FinSight Integration & Evaluation Design

## Overview

Wire up MCP servers and Qdrant, fix trading flow routing, and add evaluation infrastructure to measure token usage, latency, and answer accuracy.

## Priority Order

1. Fix trading flow routing
2. Wire up MCP in DataFetcher
3. Verify Qdrant RAG integration
4. Add evaluation layer
5. Build test suite

---

## 1. Fix Trading Flow Routing

### Problem

Every trading query runs through ALL 6 agents (10+ LLM calls):
```
fetcher → analysts_team (4) → researchers (3 rounds) → trader → risk_manager (3 rounds) → fund_manager
```

### Solution: Query-Based Routing

Route to only the relevant agent(s) based on query type.

| Query Type | Example | Agents Used |
|------------|---------|-------------|
| `full_trading` | "Should I buy AAPL?" | All 6 agents (full A2A) |
| `fundamental` | "What's AAPL's P/E ratio?" | Fundamental Analyst only |
| `technical` | "Is AAPL oversold?" | Technical Analyst only |
| `sentiment` | "What's the sentiment on AAPL?" | Sentiment + News Analysts |
| `news` | "Any news on TSLA?" | News Analyst only |
| `info` | "AAPL stock price" | Standard flow (no trading agents) |

### Changes Required

1. **`agent/nodes/router.py`** - Add query classification for trading subtypes
2. **`agent/graph.py`** - Add conditional edges to route to individual analysts
3. **New single-analyst paths** - Skip debate, go straight to composer

### Routing Logic

```
Router detects "trading" query
  ↓
Classify subtype (full/fundamental/technical/sentiment/news)
  ↓
If full_trading → existing A2A flow
If specific analyst → fetch → that analyst → composer (skip debate)
```

---

## 2. Wire Up MCP in DataFetcher

### Problem

- `DataFetcher.__init__()` creates `self.mcp = get_mcp_client()`
- `_fetch_stock()` ignores it, only calls API clients directly
- `FetchStrategy.PREFER_MCP` exists but isn't implemented

### Solution

Implement the MCP path in `_fetch_stock()`:

```
_fetch_stock() flow:
  ↓
If strategy == PREFER_MCP:
  → Try MCP server (yfinance MCP)
  → If fails, fallback to API clients
  ↓
If strategy == PREFER_API:
  → Try API clients first
  → If all fail, try MCP
```

### Changes Required

1. **`datasources/__init__.py`** - Implement `PREFER_MCP` path in `_fetch_stock()`
2. **MCP tool mapping** - Map `DataType` to MCP tool names:
   - `DataType.QUOTE` → `get_stock_price`
   - `DataType.FUNDAMENTALS` → `get_stock_info`
   - `DataType.NEWS` → `get_news`
   - `DataType.OPTIONS` → `get_options_chain`
3. **Default strategy** - Change default from `PREFER_API` to `PREFER_MCP`

### Error Handling

- MCP server not running → fallback to API
- MCP call timeout (>5s) → fallback to API
- Invalid response → fallback to API

---

## 3. Verify Qdrant RAG Integration

### Current State

- `rag/qdrant_client.py` exists with Qdrant wrapper
- `DataFetcher._ingest_to_rag()` calls `rag.fusion.ingest_raw()`
- Qdrant credentials in `.env` (QDRANT_URL, QDRANT_API_KEY)

### Verification Needed

1. Connection works - Can we connect to Qdrant instance?
2. Ingest works - Does fetched data get stored?
3. Retrieval works - Can we query vectors and get results?
4. RAG in agents - Are agents using retrieved context?

### Changes Required

1. **`rag/qdrant_client.py`** - Add connection health check method
2. **`rag/fusion.py`** - Verify `ingest_raw()` correctly embeds and stores
3. **Router/Analyst** - Optionally retrieve relevant past data before analysis

### Data Flow

```
Fetch data → Store in Qdrant (with embeddings)
     ↓
Next query → Retrieve similar past data → Enrich analyst context
```

---

## 4. Add Evaluation Layer

### Goal

Measure token usage, latency, memory, and answer accuracy across all agent calls.

### Module Structure

```
evaluation/
├── __init__.py
├── metrics.py      # Token, latency, memory tracking
├── collector.py    # Captures metrics during agent runs
├── reporter.py     # Aggregates and displays results
└── accuracy.py     # Compares responses to expected outputs
```

### Metrics Captured Per Agent Call

| Metric | How |
|--------|-----|
| Input tokens | Parse from OpenAI response `usage.prompt_tokens` |
| Output tokens | Parse from OpenAI response `usage.completion_tokens` |
| Cost | Calculate from model pricing |
| Latency | `time.perf_counter()` before/after LLM call |
| Agent name | Which node in the graph |
| Success/Error | Did the call complete? |

### Integration Method: Decorator Pattern

```python
@track_metrics
def analyst_node(state):
    ...
```

### Output Formats

- **Dev feedback:** Print summary after each query
- **Batch evaluation:** JSON/CSV export for comparison
- **Production monitoring:** Log to file or external service

---

## 5. Build Test Suite

### Structure

```
tests/
├── test_mcp_servers.py      # Already exists
├── test_mcp_integration.py  # NEW - DataFetcher + MCP
├── test_qdrant_rag.py       # NEW - RAG ingest/retrieval
├── test_routing.py          # NEW - Query classification & routing
├── test_evaluation.py       # NEW - Metrics capture
└── test_e2e.py              # NEW - Full query flows
```

### Test Categories

| Test File | What It Tests |
|-----------|---------------|
| `test_mcp_integration.py` | DataFetcher calls MCP, fallback works |
| `test_qdrant_rag.py` | Store to Qdrant, retrieve, embeddings correct |
| `test_routing.py` | Query → correct agent(s) selected |
| `test_evaluation.py` | Metrics decorator captures tokens/latency |
| `test_e2e.py` | Full queries through graph, response quality |

### Test Fixtures

```python
# Routing tests
ROUTING_FIXTURES = [
    ("Should I buy AAPL?", "full_trading"),
    ("What's AAPL's P/E ratio?", "fundamental"),
    ("Is TSLA oversold?", "technical"),
    ("NVDA stock price", "info"),
]

# Accuracy tests
ACCURACY_FIXTURES = [
    ("AAPL stock price", lambda r: "price" in r.lower()),
    ("What's Tesla's market cap?", lambda r: "$" in r or "billion" in r.lower()),
]
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_routing.py

# With coverage
pytest tests/ --cov=agent --cov=datasources
```

---

## Dependencies

### Already Available
- Qdrant instance (running)
- MCP servers (tested, working)
- LangGraph orchestration
- OpenAI API

### Using Fallbacks
- Redis (in-memory fallback)
- PostgreSQL (in-memory fallback)

---

## Success Criteria

1. Trading queries route to correct subset of agents
2. MCP servers are called by DataFetcher (visible in logs)
3. Data appears in Qdrant after fetch
4. Metrics (tokens, latency) display after each query
5. All tests pass
