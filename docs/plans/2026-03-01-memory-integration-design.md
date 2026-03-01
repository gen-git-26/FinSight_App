# Memory Integration Design

**Date:** 2026-03-01
**Status:** Approved
**Scope:** Connect built-but-disconnected memory components to the agent graph

---

## Goal

Wire up the four built-but-disconnected memory integration points so the agent graph uses Redis STM, RunCache, and Qdrant RAG in live queries. PostgreSQL LTM is out of scope (no DATABASE_URL configured) — MemoryManager degrades gracefully when it's absent.

---

## Integration Points

| Priority | Component | Node | What it does |
|----------|-----------|------|-------------|
| P1-A | MemoryManager.get_context() | router_node | Enriches LLM prompt with user history, preferences, RAG |
| P1-B | MemoryManager.store_decision() | fund_manager_node | Persists trading decisions to STM (+ Postgres when available) |
| P2-A | RunCache | fetcher_node | Deduplicates A2A data fetches across 4 parallel analysts |
| P2-B | RAG chunks | analysts.py | Injects rag_chunks from memory_context into analyst prompts |

---

## Architecture

Four files change. No new files. No new graph nodes.

```
AgentState (agent/state.py)
  + run_id: str          ← generated in router, scopes RunCache for entire A2A run
  + memory_context: Any  ← MemoryContext from get_context(), read by analysts

router_node     async def   → get_context() → stores run_id + memory_context in state
fetcher_node    async def   → RunCache check before API call, write after success
fund_manager    async def   → store_decision() + store_message() after decision
analysts.py                 → reads memory_context.rag_chunks → injects into LLM prompts
```

### Degradation Behavior

Everything is additive. If infrastructure is absent:

| Missing | Effect |
|---------|--------|
| Redis not running | `get_context()` returns empty `MemoryContext`, RunCache is a no-op |
| Postgres not configured | `store_decision()` logs warning, returns `False`, no crash |
| Qdrant not configured | `rag_chunks` is `[]`, analyst prompts unchanged |

Agent behavior is identical to today when all memory infrastructure is absent.

---

## Section 1: AgentState Changes

**File:** `agent/state.py`

Add two new optional fields:

```python
class AgentState(TypedDict, total=False):
    ...existing fields...
    run_id: str           # UUID scoping RunCache for this query run
    memory_context: Any   # MemoryContext dataclass from MemoryManager
```

---

## Section 2: router_node

**File:** `agent/nodes/router.py`

### Changes
1. Convert `router_node` to `async def`
2. Generate `run_id = str(uuid.uuid4())` at entry
3. Replace direct `RedisSTM` call with `MemoryManager.get_context()`
4. Use `context.to_prompt_context()` instead of raw history string

### Before (today)
```python
stm = get_stm()
session_history = stm.get_history(user_id, limit=5)
memory_context = "\n".join([
    f"{m['role']}: {m['content'][:100]}"
    for m in session_history
]) if session_history else ""
```

### After
```python
import uuid
from infrastructure.memory_manager import get_memory_manager

run_id = str(uuid.uuid4())
manager = get_memory_manager()
context = await manager.get_context(
    query=query,
    session_id=user_id,
    user_id=user_id,
    run_id=run_id,
)
memory_context = context.to_prompt_context()
```

### Added to return dict
```python
return {
    ...existing fields...,
    "run_id": run_id,
    "memory_context": context,
}
```

---

## Section 3: fetcher_node

**File:** `agent/nodes/fetcher.py`

### Changes
1. Convert `fetcher_node` to `async def` (already uses asyncio internally)
2. Before each ticker fetch: check RunCache
3. After successful fetch: write result to RunCache

### Pattern (applied to each ticker)
```python
from infrastructure.memory_manager import get_memory_manager

manager = get_memory_manager()
run_id = state.get("run_id")

# Cache check
if run_id:
    cached = manager.get_cached_tool(run_id, data_type.value, ticker)
    if cached:
        return cached  # skip API call

# Fetch from API (existing logic unchanged)
result = await fetcher.fetch(ticker, data_type)

# Cache write
if run_id and result.success:
    await manager.cache_tool_result(run_id, data_type.value, result, ticker=ticker)
```

### Impact
In the A2A flow, 4 parallel analysts all trigger `fetcher_node` for the same ticker. Today: 4 independent API calls. After: 1 API call + 3 cache hits from Redis (<10ms each).

---

## Section 4: fund_manager_node

**File:** `agent/nodes/fund_manager.py`

### Changes
1. Convert `fund_manager_node` to `async def`
2. After decision: call `store_decision()` to persist
3. After decision: call `store_message()` to write conversation turn to STM

### store_decision call
```python
from infrastructure.memory_manager import get_memory_manager

manager = get_memory_manager()
user_id = state.get("user_id", "default")

await manager.store_decision(
    user_id=user_id,
    ticker=ticker,
    query=state.get("query", ""),
    decision={
        "action": decision.final_action,
        "status": decision.status,
        "confidence": decision.confidence,
        "position_size": decision.final_position_size,
        "stop_loss": decision.final_stop_loss,
        "take_profit": decision.final_take_profit,
    },
    sentiment=research_report.conviction_score if research_report else None,
)
```

### store_message call
```python
await manager.store_message(
    session_id=user_id,
    user_id=user_id,
    role="assistant",
    content=f"{decision.final_action.upper()} {ticker} — {decision.status} (confidence: {decision.confidence:.0%})",
)
```

### Without Postgres
`store_decision()` logs a warning and returns `False`. `store_message()` (Redis STM) still succeeds. No crash.

---

## Section 5: RAG chunks in analyst prompts

**File:** `agent/nodes/analysts.py`

### Pattern (applied to each of the 4 analysts)
```python
memory_context = state.get("memory_context")
rag_context = ""
if memory_context and hasattr(memory_context, "rag_chunks") and memory_context.rag_chunks:
    chunks = "\n---\n".join([
        c.get("text", "")[:400] for c in memory_context.rag_chunks[:3]
    ])
    rag_context = f"\n\nRelevant historical context:\n{chunks}\n"
```

Injected at end of system prompt:
```python
{"role": "system", "content": analyst_system_prompt + rag_context}
```

**Token impact:** Max 3 chunks × 400 chars ≈ ~400 tokens per analyst. Within `TRADE_DECISION` budget of 5000 tokens.

**If Qdrant absent:** `rag_chunks == []` → `rag_context == ""` → prompt unchanged.

---

## Error Handling

All memory operations are wrapped in `try/except` inside `MemoryManager` — nodes do not need their own error handling around memory calls. A memory failure is logged and silently skipped.

---

## Testing

**New file:** `tests/test_memory_integration.py`

Three test cases:

1. **router_node emits run_id and memory_context** — mock `MemoryManager.get_context()`, assert state contains `run_id` (UUID string) and `memory_context`

2. **fund_manager_node calls store_decision** — mock `MemoryManager.store_decision()`, assert it's called with correct `ticker` and `action` after a trading decision

3. **fetcher_node uses RunCache** — mock `RunCache`, assert second fetch for same ticker returns cached result without calling the API

---

## Out of Scope

- PostgreSQL LTM (`DATABASE_URL` not configured)
- `QueryClassifier` replacing router's LLM classification (separate track)
- MCP server integration in DataFetcher (separate track)
- `PostgresSummaries` (requires Postgres)
