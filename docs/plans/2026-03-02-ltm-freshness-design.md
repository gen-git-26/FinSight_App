# LTM Freshness & Validity Design

**Date:** 2026-03-02
**Status:** Approved
**Scope:** PostgreSQL LTM schema, Qdrant RAG payload, MemoryPolicy behavioral contract
**Out of scope:** LangGraph node wiring, routing code, test suite (see implementation plan)

---

## Context and motivation

The LTM stores user-centric data: past trading decisions, conversation history, user preferences, and behavioral patterns. It does **not** cache raw market data — prices, fundamentals, and news are always fetched live by the DataFetcher.

The current schema has no staleness enforcement. A trading decision from six months ago is retrieved with the same weight as one from yesterday. This is a correctness risk in a finance context: an LLM receiving unlabelled historical data can treat it as current truth.

**The governing principle:** memory is governed data with explicit freshness semantics, not a passive log.

---

## Section 1 — Validity Classes

A `ValidityClass` maps a data type to a validity window. The window is anchored to financial reality, not to arbitrary TTLs.

| Validity Class        | Window            | Rationale |
|-----------------------|-------------------|-----------|
| `price_snapshot`      | 15–60 minutes     | Intra-day equity price; stale in minutes during market hours |
| `end_of_day_price`    | 48 hours          | "Yesterday close" style context; useful for 1–2 days |
| `breaking_news`       | 48–72 hours       | Earnings surprises, macro events — market impact fades fast |
| `news_sentiment`      | 7 days            | Aggregated analyst/media sentiment; degrades slowly |
| `session_memory`      | 3 days            | Raw conversational turns; short window prevents noise accumulation |
| `session_summary`     | 30 days           | LLM-summarised session insights; more durable than raw turns |
| `trading_decision`    | horizon-dependent | Day trade: 3–7d · Swing: 30d · Long-term: 90–180d |
| `fundamental_data`    | 90 days           | Quarterly earnings cycle; stale after next filing |
| `behavioral_pattern`  | 180 days          | Slowly evolving user habits |
| `user_preference`     | permanent (`NULL`)| Valid until user explicitly changes it |

**Note on `market_hours_sensitive`:** Crypto is 24/7; equities have a closing price. `price_snapshot` TTL should be computed against market hours when available. For MVP, use calendar time.

**Note on `trading_decision` horizon:** the horizon must be stored in the `decision` JSONB field. If no horizon is recorded, default to 30 days (swing).

---

## Section 2 — Schema

### 2.1 New columns (added to all LTM tables)

| Column | Type | Description |
|--------|------|-------------|
| `validity_class` | `VARCHAR(50)` | Which rule applies (from table above) |
| `valid_for_context_until` | `TIMESTAMP` | Computed at insert by DB trigger; `NULL` = permanent |
| `as_of` | `TIMESTAMP` | When the fact was true — not necessarily `created_at`. Example: ingesting yesterday's earnings transcript today sets `as_of` to yesterday. |
| `source` | `VARCHAR(255)` | Tool name, provider, or document id that produced the record |

`created_at` is retained as the ingestion timestamp (audit log). `as_of` is the semantic timestamp (fact timestamp). They differ whenever data is ingested with a lag.

### 2.2 Constraints

```sql
-- Validity window must end after the fact was true
ALTER TABLE trading_decisions
  ADD CONSTRAINT chk_valid_after_as_of
  CHECK (valid_for_context_until IS NULL OR valid_for_context_until >= as_of);
```

Same constraint applied to `conversation_memory` and `user_patterns`.

### 2.3 DB trigger — compute `valid_for_context_until`

Do not compute expiry in application code only. A PostgreSQL function enforces it at the DB layer.

```sql
CREATE OR REPLACE FUNCTION set_validity_window()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.valid_for_context_until IS NOT NULL THEN
    RETURN NEW;  -- caller set it explicitly, respect it
  END IF;

  NEW.valid_for_context_until := CASE NEW.validity_class
    WHEN 'price_snapshot'     THEN NEW.as_of + INTERVAL '1 hour'
    WHEN 'end_of_day_price'   THEN NEW.as_of + INTERVAL '48 hours'
    WHEN 'breaking_news'      THEN NEW.as_of + INTERVAL '72 hours'
    WHEN 'news_sentiment'     THEN NEW.as_of + INTERVAL '7 days'
    WHEN 'session_memory'     THEN NEW.as_of + INTERVAL '3 days'
    WHEN 'session_summary'    THEN NEW.as_of + INTERVAL '30 days'
    WHEN 'fundamental_data'   THEN NEW.as_of + INTERVAL '90 days'
    WHEN 'behavioral_pattern' THEN NEW.as_of + INTERVAL '180 days'
    WHEN 'user_preference'    THEN NULL  -- permanent
    ELSE NEW.as_of + INTERVAL '30 days' -- safe default (swing horizon)
  END;

  -- trading_decision: check horizon stored in decision JSONB
  IF NEW.validity_class = 'trading_decision' THEN
    NEW.valid_for_context_until := NEW.as_of + (
      CASE NEW.decision->>'horizon'
        WHEN 'day'       THEN INTERVAL '7 days'
        WHEN 'long_term' THEN INTERVAL '180 days'
        ELSE                  INTERVAL '30 days'  -- swing default
      END
    );
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_set_validity_window
  BEFORE INSERT ON trading_decisions
  FOR EACH ROW EXECUTE FUNCTION set_validity_window();

-- Repeat for conversation_memory and user_patterns
```

### 2.4 Indexes

Validity filter runs on every retrieval. Add compound indexes:

```sql
-- Primary retrieval path: user + class + freshness
CREATE INDEX idx_td_user_class_valid
  ON trading_decisions (user_id, validity_class, valid_for_context_until DESC);

CREATE INDEX idx_cm_user_session_valid
  ON conversation_memory (user_id, session_id, valid_for_context_until DESC);

CREATE INDEX idx_up_user_class_valid
  ON user_patterns (user_id, validity_class, valid_for_context_until DESC);
```

### 2.5 Qdrant payload

Each chunk ingested into Qdrant receives:

```json
{
  "validity_class": "fundamental_data",
  "valid_until": 1748822400,
  "as_of": 1740960000,
  "source": "yfinance_server",
  "ticker": "AAPL"
}
```

`valid_until` and `as_of` are **epoch integers (seconds)**, not ISO strings. This avoids timezone parsing and makes numeric comparison safe and fast.

Retrieval filter: `valid_until > int(time.time())`.

---

## Section 3 — Enforcement at Read Time

### 3.1 SQL condition

All context-retrieval queries include:

```sql
WHERE (valid_for_context_until > NOW() OR valid_for_context_until IS NULL)
```

### 3.2 Recency weighting within the validity window

Within a valid window, prefer newer records. Retrieval functions order by `as_of DESC` and apply a soft cap on the number of rows injected into the prompt (default: 5 items per class).

`user_preference` (permanent) is retrieved separately and never mixed into recency ranking — it is injected as a stable profile block, not as a scored memory item.

### 3.3 Prompt stamping — standardised format

All memory-sourced facts injected into an LLM prompt must carry:

```
[validity_class] (as_of: YYYY-MM-DD, age: Nd): <content>
Context only — verify with live data before treating as current.
```

Example:
```
trading_decision (as_of: 2026-02-12, age: 18d): BUY AAPL — fund manager approved.
Context only — verify with live data before treating as current.
```

This label is mandatory for all classes except `user_preference` and `behavioral_pattern`, which carry a softer label:
```
user_preference (stable): Risk tolerance = moderate. Prefers growth over income.
```

---

## Section 4 — MemoryPolicy Decision Table (Behavioral Contract)

The validity schema explains how rows expire. The MemoryPolicy table explains **when memory is allowed to influence an answer**. This is the behavioral contract — it is enforceable and testable independently of the TTL mechanism.

**The fundamental rule:**

> **Market truth always comes from tools, not memory.**
>
> If the intent is `current_price`, `daily_move`, `latest_news`, or anything containing "now / today / latest," then `price_data` and `news_sentiment` from memory are **context-only** — they cannot serve as the authoritative answer. Any fact not fetched live in the current run must carry an `as_of` stamp.

### Policy table

| Intent | Examples | Allowed validity classes | Live tools required | Enforcement notes |
|--------|----------|--------------------------|--------------------|--------------------|
| `current_price` | "AAPL price?" "price now" | `user_preference`, `behavioral_pattern`, `trading_decision` (context only) | **Yes** | `price_snapshot` from memory shown as "previous snapshot only" — never as truth |
| `daily_move` | "How did TSLA do today?" | `user_preference`, `behavioral_pattern`, `news_sentiment` (context only) | **Yes** | Fetch live quote + day range; memory adds user framing only |
| `latest_news` | "Latest news on NVDA" | `news_sentiment` (if valid), `user_preference` | **Yes** | Always fetch fresh news; stored sentiment is secondary and stamped |
| `company_fundamentals` | "Revenue growth, margins" | `fundamental_data` (if valid), `trading_decision` (context), `user_preference` | Usually | If "latest earnings" → fetch live; cached fundamentals valid within 90d but must be stamped `as_of` |
| `strategy_recommendation` | "Should I buy?" "entry plan" | `trading_decision`, `behavioral_pattern`, `user_preference`, `fundamental_data`, `news_sentiment` | Often | Market conditions require live tools; memory informs preferences and prior rationale — not current facts |
| `explain_last_decision` | "Why did we say buy?" | `trading_decision`, `session_memory` | **No** | Pure memory recall; stamp date + "conditions may have changed" |
| `recap_conversation` | "What did we discuss?" | `session_memory`, `session_summary` | **No** | Conversational recall only; no live tools needed |
| `preference_lookup` | "What's my risk tolerance?" | `user_preference`, `behavioral_pattern` | **No** | True LTM territory; no market data involved |
| `update_preference` | "Avoid crypto from now on" | `user_preference` | **No** | Write path; triggers summary refresh |
| `research_compare` | "AAPL vs MSFT" | `fundamental_data`, `news_sentiment`, `user_preference`, `behavioral_pattern` | Often | If "as of today" → require live; cached fundamentals require `as_of` stamp |
| `portfolio_sensitive` | "What to do with my holdings?" | `user_preference`, `behavioral_pattern`, `trading_decision` | **Yes** | Holdings and prices must be live; memory provides constraints and history only |

### MemoryPolicy node output (contract)

The MemoryPolicy step outputs a `MemoryPolicy` object that retrieval nodes consume:

```python
@dataclass
class MemoryPolicy:
    intent: str                        # classified intent
    allowed_classes: list[str]         # validity classes retrieval may use
    require_live_tools: bool           # must DataFetcher run before composing?
    max_items_per_class: int           # cap on injected memory items (default: 5)
    context_only_classes: list[str]    # classes allowed but never authoritative
    recency_weight: bool               # order by as_of DESC within window
```

LangGraph wiring details (node I/O, StateGraph routing, test cases) are specified in the implementation plan.

---

## Section 5 — Cleanup Policy

| Table | Action | Trigger |
|-------|--------|---------|
| `trading_decisions` | Never deleted — permanent audit log | — |
| `conversation_memory` | Hard delete rows where `valid_for_context_until < NOW() - INTERVAL '30 days'` | Nightly job or on-connect |
| `user_patterns` | Soft-expire via `valid_for_context_until`; hard delete after 1 year | Nightly job |
| Qdrant | Delete points where `valid_until < epoch_now` | Qdrant TTL index or nightly sweep |

---

## Open questions (deferred to implementation)

1. **market_hours_sensitive:** For MVP use calendar time. Post-MVP: detect equity vs crypto and adjust `price_snapshot` TTL accordingly.
2. **session_summary generation:** Who writes `session_summary` rows — a dedicated summariser node, or the composer? Deferred to implementation plan.
3. **Postgres cleanup job:** Cron inside the container, or a LangGraph scheduled node? Deferred.
