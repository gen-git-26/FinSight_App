# LTM Freshness & Validity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire Postgres into Docker Compose, enforce freshness semantics on all memory writes/reads, and add a MemoryPolicy gate that governs which memory classes are allowed per query intent.

**Architecture:** New `ValidityClass` enum drives a DB trigger that computes `valid_for_context_until` at insert time. All read paths filter on that column. A `MemoryPolicy` dataclass (keyed by `QueryIntent`) is produced by the router and consumed by memory fetchers to enforce which classes may influence an answer and whether live tools are required.

**Tech Stack:** PostgreSQL 15 (psycopg2), Redis (existing), Qdrant (existing), LangGraph, Python 3.12

---

## Task 1: Docker Compose — Postgres service

**Files:**
- Modify: `docker-compose.yml`
- No test needed (infrastructure change)

**Step 1: Add postgres service to docker-compose.yml**

Replace the entire file with:

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: finsight
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: finsight_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  finsight:
    build: .
    ports:
      - "8000:8000"
      - "8502:8502"
    env_file:
      - .env
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

volumes:
  postgres_data:
```

**Step 2: Add Postgres vars to .env (local dev)**

Append to `.env`:
```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=finsight
POSTGRES_USER=postgres
POSTGRES_PASSWORD=finsight_dev
```

**Step 3: Verify Docker Compose is valid**

```bash
docker compose config --quiet && echo "OK"
```
Expected: `OK`

**Step 4: Start Postgres and verify it's reachable**

```bash
docker compose up postgres -d
sleep 3
docker compose exec postgres pg_isready -U postgres
```
Expected: `localhost:5432 - accepting connections`

**Step 5: Commit**

```bash
git add docker-compose.yml .env
git commit -m "feat: add postgres service to docker-compose with named volume"
```

---

## Task 2: ValidityClass enum (`infrastructure/validity.py`)

**Files:**
- Create: `infrastructure/validity.py`
- Create: `tests/test_ltm_freshness.py`

**Step 1: Write the failing test**

Create `tests/test_ltm_freshness.py`:

```python
# tests/test_ltm_freshness.py
"""Tests for LTM freshness and validity enforcement."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timezone


# === Task 2: ValidityClass enum ===

def test_validity_class_window_seconds_price_snapshot():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.PRICE_SNAPSHOT.window_seconds == 3600

def test_validity_class_window_seconds_end_of_day():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.END_OF_DAY_PRICE.window_seconds == 48 * 3600

def test_validity_class_window_seconds_fundamental():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.FUNDAMENTAL_DATA.window_seconds == 90 * 86400

def test_validity_class_user_preference_is_permanent():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.USER_PREFERENCE.window_seconds is None

def test_compute_valid_until_returns_epoch_int_for_finite_class():
    from infrastructure.validity import ValidityClass, compute_valid_until
    as_of = int(datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc).timestamp())
    result = compute_valid_until(ValidityClass.PRICE_SNAPSHOT, as_of)
    assert result == as_of + 3600

def test_compute_valid_until_returns_none_for_permanent():
    from infrastructure.validity import ValidityClass, compute_valid_until
    as_of = int(datetime(2026, 3, 2, tzinfo=timezone.utc).timestamp())
    assert compute_valid_until(ValidityClass.USER_PREFERENCE, as_of) is None

def test_validity_class_trading_decision_default_is_swing():
    from infrastructure.validity import ValidityClass
    assert ValidityClass.TRADING_DECISION.window_seconds == 30 * 86400

def test_horizon_window_seconds_day():
    from infrastructure.validity import horizon_window_seconds
    assert horizon_window_seconds("day") == 7 * 86400

def test_horizon_window_seconds_long_term():
    from infrastructure.validity import horizon_window_seconds
    assert horizon_window_seconds("long_term") == 180 * 86400

def test_horizon_window_seconds_unknown_defaults_to_swing():
    from infrastructure.validity import horizon_window_seconds
    assert horizon_window_seconds("unknown_value") == 30 * 86400
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ltm_freshness.py -v 2>&1 | head -30
```
Expected: `ImportError: No module named 'infrastructure.validity'`

**Step 3: Create `infrastructure/validity.py`**

```python
# infrastructure/validity.py
"""
ValidityClass — maps data types to freshness windows.

Windows are anchored to financial cycles, not arbitrary TTLs.
All times are in seconds. None means permanent (user_preference).
"""
from __future__ import annotations
from enum import Enum
from typing import Optional


class ValidityClass(str, Enum):
    """Data type → validity window mapping."""
    PRICE_SNAPSHOT     = "price_snapshot"      # 1 hour  (intra-day equity)
    END_OF_DAY_PRICE   = "end_of_day_price"    # 48 hours (yesterday close)
    BREAKING_NEWS      = "breaking_news"       # 72 hours (market-moving events)
    NEWS_SENTIMENT     = "news_sentiment"      # 7 days   (aggregated sentiment)
    SESSION_MEMORY     = "session_memory"      # 3 days   (raw conversation turns)
    SESSION_SUMMARY    = "session_summary"     # 30 days  (summarised insights)
    TRADING_DECISION   = "trading_decision"    # 30 days  (swing default; see horizon)
    FUNDAMENTAL_DATA   = "fundamental_data"    # 90 days  (quarterly cycle)
    BEHAVIORAL_PATTERN = "behavioral_pattern"  # 180 days (slowly evolving habits)
    USER_PREFERENCE    = "user_preference"     # permanent (until explicitly changed)

    @property
    def window_seconds(self) -> Optional[int]:
        """Return validity window in seconds, or None for permanent."""
        windows = {
            ValidityClass.PRICE_SNAPSHOT:     1 * 3600,
            ValidityClass.END_OF_DAY_PRICE:   48 * 3600,
            ValidityClass.BREAKING_NEWS:      72 * 3600,
            ValidityClass.NEWS_SENTIMENT:     7 * 86400,
            ValidityClass.SESSION_MEMORY:     3 * 86400,
            ValidityClass.SESSION_SUMMARY:    30 * 86400,
            ValidityClass.TRADING_DECISION:   30 * 86400,
            ValidityClass.FUNDAMENTAL_DATA:   90 * 86400,
            ValidityClass.BEHAVIORAL_PATTERN: 180 * 86400,
            ValidityClass.USER_PREFERENCE:    None,
        }
        return windows[self]


# Horizon → window mapping for trading decisions
_HORIZON_WINDOWS = {
    "day":       7 * 86400,
    "swing":     30 * 86400,
    "long_term": 180 * 86400,
}


def horizon_window_seconds(horizon: str) -> int:
    """Return validity window for a trading horizon. Defaults to swing (30d)."""
    return _HORIZON_WINDOWS.get(horizon, 30 * 86400)


def compute_valid_until(
    validity_class: ValidityClass,
    as_of_epoch: int,
    horizon: Optional[str] = None
) -> Optional[int]:
    """
    Compute valid_until as epoch integer.

    Args:
        validity_class: The ValidityClass for this data type.
        as_of_epoch: When the fact was true (epoch seconds).
        horizon: For TRADING_DECISION only — 'day', 'swing', 'long_term'.

    Returns:
        Epoch int (valid_until), or None if permanent.
    """
    if validity_class == ValidityClass.USER_PREFERENCE:
        return None

    if validity_class == ValidityClass.TRADING_DECISION and horizon:
        return as_of_epoch + horizon_window_seconds(horizon)

    window = validity_class.window_seconds
    if window is None:
        return None
    return as_of_epoch + window
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ltm_freshness.py::test_validity_class_window_seconds_price_snapshot tests/test_ltm_freshness.py::test_validity_class_window_seconds_fundamental tests/test_ltm_freshness.py::test_validity_class_user_preference_is_permanent tests/test_ltm_freshness.py::test_compute_valid_until_returns_epoch_int_for_finite_class tests/test_ltm_freshness.py::test_compute_valid_until_returns_none_for_permanent tests/test_ltm_freshness.py::test_horizon_window_seconds_day tests/test_ltm_freshness.py::test_horizon_window_seconds_long_term tests/test_ltm_freshness.py::test_horizon_window_seconds_unknown_defaults_to_swing -v
```
Expected: 8 passed

**Step 5: Commit**

```bash
git add infrastructure/validity.py tests/test_ltm_freshness.py
git commit -m "feat: ValidityClass enum with per-type freshness windows"
```

---

## Task 3: Schema migration — postgres_ltm.py

**Files:**
- Modify: `infrastructure/postgres_ltm.py`

**Step 1: Write the failing tests**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 3: INIT_SQL schema migration ===

def test_init_sql_contains_validity_columns():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "validity_class" in INIT_SQL
    assert "valid_for_context_until" in INIT_SQL
    assert "as_of" in INIT_SQL
    assert "source" in INIT_SQL

def test_init_sql_contains_trigger():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "set_validity_window" in INIT_SQL
    assert "BEFORE INSERT" in INIT_SQL

def test_init_sql_contains_check_constraint():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "chk_valid_after_as_of" in INIT_SQL

def test_init_sql_contains_compound_index():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "idx_td_user_class_valid" in INIT_SQL
    assert "idx_cm_user_session_valid" in INIT_SQL

def test_init_sql_contains_unique_constraint_user_patterns():
    from infrastructure.postgres_ltm import INIT_SQL
    assert "UNIQUE (user_id, pattern_type)" in INIT_SQL
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ltm_freshness.py::test_init_sql_contains_validity_columns tests/test_ltm_freshness.py::test_init_sql_contains_trigger -v
```
Expected: FAILED

**Step 3: Replace INIT_SQL in `infrastructure/postgres_ltm.py`**

Find the `INIT_SQL` constant (lines 57–110) and replace with:

```python
INIT_SQL = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    profile JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading decisions history (permanent audit log — never deleted)
CREATE TABLE IF NOT EXISTS trading_decisions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    ticker VARCHAR(20) NOT NULL,
    query TEXT,
    decision JSONB NOT NULL,
    analyst_reports JSONB,
    research_report JSONB,
    risk_assessment JSONB,
    fund_manager_decision JSONB,
    validity_class VARCHAR(50) DEFAULT 'trading_decision',
    valid_for_context_until TIMESTAMP,
    as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_valid_after_as_of_td
        CHECK (valid_for_context_until IS NULL OR valid_for_context_until >= as_of)
);

-- Conversation memory
CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    validity_class VARCHAR(50) DEFAULT 'session_memory',
    valid_for_context_until TIMESTAMP,
    as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_valid_after_as_of_cm
        CHECK (valid_for_context_until IS NULL OR valid_for_context_until >= as_of)
);

-- User preferences / learned patterns
CREATE TABLE IF NOT EXISTS user_patterns (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    validity_class VARCHAR(50) DEFAULT 'behavioral_pattern',
    valid_for_context_until TIMESTAMP,
    as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, pattern_type),
    CONSTRAINT chk_valid_after_as_of_up
        CHECK (valid_for_context_until IS NULL OR valid_for_context_until >= as_of)
);

-- Validity window trigger function
CREATE OR REPLACE FUNCTION set_validity_window()
RETURNS TRIGGER AS $$
DECLARE
    horizon TEXT;
BEGIN
    -- Respect explicit value if caller set it
    IF NEW.valid_for_context_until IS NOT NULL THEN
        RETURN NEW;
    END IF;

    -- trading_decision: check horizon from decision JSONB
    IF NEW.validity_class = 'trading_decision' THEN
        horizon := NEW.decision->>'horizon';
        NEW.valid_for_context_until := NEW.as_of + (
            CASE horizon
                WHEN 'day'       THEN INTERVAL '7 days'
                WHEN 'long_term' THEN INTERVAL '180 days'
                ELSE                  INTERVAL '30 days'
            END
        );
        RETURN NEW;
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
        WHEN 'user_preference'    THEN NULL
        ELSE                           NEW.as_of + INTERVAL '30 days'
    END;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_td_validity
    BEFORE INSERT ON trading_decisions
    FOR EACH ROW EXECUTE FUNCTION set_validity_window();

CREATE OR REPLACE TRIGGER trg_cm_validity
    BEFORE INSERT ON conversation_memory
    FOR EACH ROW EXECUTE FUNCTION set_validity_window();

CREATE OR REPLACE TRIGGER trg_up_validity
    BEFORE INSERT ON user_patterns
    FOR EACH ROW EXECUTE FUNCTION set_validity_window();

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trading_decisions_user ON trading_decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_decisions_ticker ON trading_decisions(ticker);
CREATE INDEX IF NOT EXISTS idx_conversation_memory_user ON conversation_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_user_patterns_user ON user_patterns(user_id);

-- Compound indexes for validity-filtered retrieval
CREATE INDEX IF NOT EXISTS idx_td_user_class_valid
    ON trading_decisions (user_id, validity_class, valid_for_context_until DESC);
CREATE INDEX IF NOT EXISTS idx_cm_user_session_valid
    ON conversation_memory (user_id, session_id, valid_for_context_until DESC);
CREATE INDEX IF NOT EXISTS idx_up_user_class_valid
    ON user_patterns (user_id, validity_class, valid_for_context_until DESC);
"""
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ltm_freshness.py::test_init_sql_contains_validity_columns tests/test_ltm_freshness.py::test_init_sql_contains_trigger tests/test_ltm_freshness.py::test_init_sql_contains_check_constraint tests/test_ltm_freshness.py::test_init_sql_contains_compound_index tests/test_ltm_freshness.py::test_init_sql_contains_unique_constraint_user_patterns -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add infrastructure/postgres_ltm.py tests/test_ltm_freshness.py
git commit -m "feat: schema migration — validity columns, trigger, indexes in postgres_ltm"
```

---

## Task 4: Schema migration — postgres_summaries.py

**Files:**
- Modify: `infrastructure/postgres_summaries.py`

**Step 1: Write the failing tests**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 4: Summary tables ===

def test_summary_sql_contains_validity_columns():
    from infrastructure.postgres_summaries import SUMMARY_TABLES_SQL
    assert "validity_class" in SUMMARY_TABLES_SQL
    assert "valid_for_context_until" in SUMMARY_TABLES_SQL
    assert "as_of" in SUMMARY_TABLES_SQL
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_ltm_freshness.py::test_summary_sql_contains_validity_columns -v
```
Expected: FAILED

**Step 3: Add validity columns to SUMMARY_TABLES_SQL in `infrastructure/postgres_summaries.py`**

In `SUMMARY_TABLES_SQL`, update `user_profile_summary` to add after `updated_at`:
```sql
    as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validity_class VARCHAR(50) DEFAULT 'user_preference',
    valid_for_context_until TIMESTAMP
```

Update `user_ticker_summary` similarly after `updated_at`:
```sql
    as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validity_class VARCHAR(50) DEFAULT 'trading_decision',
    valid_for_context_until TIMESTAMP
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_ltm_freshness.py::test_summary_sql_contains_validity_columns -v
```
Expected: PASSED

**Step 5: Commit**

```bash
git add infrastructure/postgres_summaries.py tests/test_ltm_freshness.py
git commit -m "feat: add validity columns to summary tables schema"
```

---

## Task 5: LTM write paths — accept as_of and validity_class

**Files:**
- Modify: `infrastructure/postgres_ltm.py`

**Step 1: Write the failing tests**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 5: LTM write paths ===

def test_save_trading_decision_includes_validity_fields():
    """save_trading_decision SQL must include validity columns."""
    from unittest.mock import MagicMock, patch
    from infrastructure.postgres_ltm import PostgresLTM

    ltm = PostgresLTM()
    executed_sql = []

    mock_cursor = MagicMock()
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.execute = lambda sql, params: executed_sql.append(sql)

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor = MagicMock(return_value=mock_cursor)

    with patch.object(ltm, 'get_connection', return_value=mock_conn):
        ltm.save_trading_decision(
            user_id="u1", ticker="AAPL", query="buy?",
            decision={"action": "BUY", "horizon": "swing"},
            validity_class="trading_decision",
            as_of=1740960000,
            source="fund_manager_node"
        )

    assert executed_sql, "No SQL was executed"
    assert "validity_class" in executed_sql[0]
    assert "as_of" in executed_sql[0]
    assert "source" in executed_sql[0]

def test_save_message_includes_validity_fields():
    """save_message SQL must include validity columns."""
    from unittest.mock import MagicMock, patch
    from infrastructure.postgres_ltm import PostgresLTM

    ltm = PostgresLTM()
    executed_sql = []

    mock_cursor = MagicMock()
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.execute = lambda sql, params: executed_sql.append(sql)

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor = MagicMock(return_value=mock_cursor)

    with patch.object(ltm, 'get_connection', return_value=mock_conn):
        ltm.save_message(
            user_id="u1", role="user", content="hello",
            validity_class="session_memory",
            as_of=1740960000,
            source="composer_node"
        )

    assert executed_sql, "No SQL was executed"
    assert "validity_class" in executed_sql[0]
    assert "as_of" in executed_sql[0]
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ltm_freshness.py::test_save_trading_decision_includes_validity_fields tests/test_ltm_freshness.py::test_save_message_includes_validity_fields -v
```
Expected: FAILED (TypeError — unexpected keyword argument)

**Step 3: Update `save_trading_decision` in `infrastructure/postgres_ltm.py`**

Find `save_trading_decision` and update its signature and SQL:

```python
def save_trading_decision(
    self,
    user_id: str,
    ticker: str,
    query: str,
    decision: Dict[str, Any],
    analyst_reports: Optional[List] = None,
    research_report: Optional[Dict] = None,
    risk_assessment: Optional[Dict] = None,
    fund_manager_decision: Optional[Dict] = None,
    session_id: Optional[str] = None,
    validity_class: str = "trading_decision",
    as_of: Optional[int] = None,
    source: Optional[str] = None,
) -> bool:
    """Save a trading decision to history."""
    import time
    as_of_ts = datetime.fromtimestamp(as_of) if as_of else datetime.utcnow()

    with self.get_connection() as conn:
        if conn is None:
            return False

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trading_decisions
                (user_id, session_id, ticker, query, decision,
                 analyst_reports, research_report, risk_assessment,
                 fund_manager_decision, validity_class, as_of, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, session_id, ticker, query,
                Json(decision),
                Json(analyst_reports) if analyst_reports else None,
                Json(research_report) if research_report else None,
                Json(risk_assessment) if risk_assessment else None,
                Json(fund_manager_decision) if fund_manager_decision else None,
                validity_class, as_of_ts, source
            ))
            return True
```

Find `save_message` and update:

```python
def save_message(
    self,
    user_id: str,
    role: str,
    content: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    validity_class: str = "session_memory",
    as_of: Optional[int] = None,
    source: Optional[str] = None,
) -> bool:
    """Save a conversation message."""
    as_of_ts = datetime.fromtimestamp(as_of) if as_of else datetime.utcnow()

    with self.get_connection() as conn:
        if conn is None:
            return False

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_memory
                (user_id, session_id, role, content, metadata,
                 validity_class, as_of, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, session_id, role, content, Json(metadata or {}),
                validity_class, as_of_ts, source
            ))
            return True
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ltm_freshness.py::test_save_trading_decision_includes_validity_fields tests/test_ltm_freshness.py::test_save_message_includes_validity_fields -v
```
Expected: 2 passed

**Step 5: Commit**

```bash
git add infrastructure/postgres_ltm.py tests/test_ltm_freshness.py
git commit -m "feat: LTM write paths accept validity_class, as_of, source"
```

---

## Task 6: LTM read paths — staleness filter

**Files:**
- Modify: `infrastructure/postgres_ltm.py`
- Modify: `infrastructure/postgres_summaries.py`

**Step 1: Write the failing tests**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 6: LTM read paths — staleness filter ===

def test_get_trading_history_filters_expired_rows():
    """get_trading_history must include valid_for_context_until filter."""
    from unittest.mock import MagicMock, patch
    from infrastructure.postgres_ltm import PostgresLTM

    ltm = PostgresLTM()
    executed_sql = []

    mock_cursor = MagicMock()
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.execute = lambda sql, params: executed_sql.append(sql)
    mock_cursor.fetchall = MagicMock(return_value=[])

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor = MagicMock(return_value=mock_cursor)

    with patch.object(ltm, 'get_connection', return_value=mock_conn):
        ltm.get_trading_history(user_id="u1", context_only=True)

    assert executed_sql, "No SQL was executed"
    assert "valid_for_context_until" in executed_sql[0]

def test_get_conversation_history_filters_expired_rows():
    """get_conversation_history must include valid_for_context_until filter."""
    from unittest.mock import MagicMock, patch
    from infrastructure.postgres_ltm import PostgresLTM

    ltm = PostgresLTM()
    executed_sql = []

    mock_cursor = MagicMock()
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.execute = lambda sql, params: executed_sql.append(sql)
    mock_cursor.fetchall = MagicMock(return_value=[])

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor = MagicMock(return_value=mock_cursor)

    with patch.object(ltm, 'get_connection', return_value=mock_conn):
        ltm.get_conversation_history(user_id="u1", context_only=True)

    assert executed_sql, "No SQL was executed"
    assert "valid_for_context_until" in executed_sql[0]
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ltm_freshness.py::test_get_trading_history_filters_expired_rows tests/test_ltm_freshness.py::test_get_conversation_history_filters_expired_rows -v
```
Expected: FAILED (TypeError — unexpected keyword argument `context_only`)

**Step 3: Update read methods in `infrastructure/postgres_ltm.py`**

Update `get_trading_history` signature and SQL:

```python
def get_trading_history(
    self,
    user_id: str,
    ticker: Optional[str] = None,
    limit: int = 50,
    context_only: bool = True,
) -> List[Dict]:
    """Get trading decision history.

    Args:
        context_only: If True (default), filter out expired records.
                      Set False only for audit/admin queries.
    """
    freshness_clause = (
        "AND (valid_for_context_until > NOW() OR valid_for_context_until IS NULL)"
        if context_only else ""
    )
    with self.get_connection() as conn:
        if conn is None:
            return []

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if ticker:
                cur.execute(f"""
                    SELECT * FROM trading_decisions
                    WHERE user_id = %s AND ticker = %s {freshness_clause}
                    ORDER BY as_of DESC
                    LIMIT %s
                """, (user_id, ticker, limit))
            else:
                cur.execute(f"""
                    SELECT * FROM trading_decisions
                    WHERE user_id = %s {freshness_clause}
                    ORDER BY as_of DESC
                    LIMIT %s
                """, (user_id, limit))

            return [dict(row) for row in cur.fetchall()]
```

Update `get_conversation_history` similarly:

```python
def get_conversation_history(
    self,
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 100,
    context_only: bool = True,
) -> List[Dict]:
    """Get conversation history."""
    freshness_clause = (
        "AND (valid_for_context_until > NOW() OR valid_for_context_until IS NULL)"
        if context_only else ""
    )
    with self.get_connection() as conn:
        if conn is None:
            return []

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if session_id:
                cur.execute(f"""
                    SELECT * FROM conversation_memory
                    WHERE user_id = %s AND session_id = %s {freshness_clause}
                    ORDER BY as_of ASC
                    LIMIT %s
                """, (user_id, session_id, limit))
            else:
                cur.execute(f"""
                    SELECT * FROM conversation_memory
                    WHERE user_id = %s {freshness_clause}
                    ORDER BY as_of DESC
                    LIMIT %s
                """, (user_id, limit))

            return [dict(row) for row in cur.fetchall()]
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ltm_freshness.py::test_get_trading_history_filters_expired_rows tests/test_ltm_freshness.py::test_get_conversation_history_filters_expired_rows -v
```
Expected: 2 passed

**Step 5: Commit**

```bash
git add infrastructure/postgres_ltm.py tests/test_ltm_freshness.py
git commit -m "feat: LTM read paths filter expired rows via valid_for_context_until"
```

---

## Task 7: Qdrant ingest — epoch validity payload + retrieval filter

**Files:**
- Modify: `rag/fusion.py`
- Modify: `infrastructure/memory_manager.py`

**Step 1: Write the failing tests**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 7: Qdrant ingest validity payload ===

def test_ingest_raw_payload_contains_valid_until_as_epoch():
    """ingest_raw must add valid_until (epoch int) and as_of to payload."""
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch

    captured_items = []

    async def fake_upsert(items):
        captured_items.extend(items)

    mock_qdr = MagicMock()
    mock_qdr.ensure_collections = MagicMock()
    mock_qdr.upsert_snippets = fake_upsert

    with patch("rag.fusion.HybridQdrant", return_value=mock_qdr), \
         patch("rag.fusion.embed_texts", new=AsyncMock(return_value=[[0.1] * 10])):
        from rag.fusion import ingest_raw
        asyncio.run(ingest_raw(
            tool="yfinance",
            raw="AAPL price is 180",
            symbol="AAPL",
            doc_type="price_snapshot",
            as_of=1740960000,
        ))

    assert captured_items, "No items were upserted"
    item = captured_items[0]
    assert "valid_until" in item, "valid_until missing from payload"
    assert isinstance(item["valid_until"], int), "valid_until must be epoch int"
    assert "as_of" in item
    assert item["as_of"] == 1740960000
    assert item["validity_class"] == "price_snapshot"
    # price_snapshot window = 3600s
    assert item["valid_until"] == 1740960000 + 3600
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_ltm_freshness.py::test_ingest_raw_payload_contains_valid_until_as_epoch -v
```
Expected: FAILED

**Step 3: Update `ingest_raw` in `rag/fusion.py`**

Change the function signature and payload building:

```python
async def ingest_raw(
    *,
    tool: str,
    raw: Any,
    symbol: str = "",
    doc_type: str = "",
    date: str = "",
    as_of: Optional[int] = None,
) -> List[str]:
    """Ingest raw API output as small, queryable snippets and return generated IDs."""
    import time
    from infrastructure.validity import ValidityClass, compute_valid_until

    qdr = HybridQdrant()
    qdr.ensure_collections()
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    chunks = _chunk_text(text)
    items = []
    ids = []

    # Map doc_type to ValidityClass (default: news_sentiment for unknown types)
    _DOC_TYPE_TO_VALIDITY = {
        "price_snapshot": ValidityClass.PRICE_SNAPSHOT,
        "end_of_day_price": ValidityClass.END_OF_DAY_PRICE,
        "breaking_news": ValidityClass.BREAKING_NEWS,
        "news_sentiment": ValidityClass.NEWS_SENTIMENT,
        "fundamental_data": ValidityClass.FUNDAMENTAL_DATA,
        "session_memory": ValidityClass.SESSION_MEMORY,
    }
    validity_class = _DOC_TYPE_TO_VALIDITY.get(doc_type, ValidityClass.NEWS_SENTIMENT)
    as_of_epoch = as_of or int(time.time())
    valid_until = compute_valid_until(validity_class, as_of_epoch)

    for i, ch in enumerate(chunks):
        pid = f"{tool}-{symbol or 'NA'}-{i}"
        ids.append(pid)
        items.append({
            "id": pid,
            "text": ch,
            "symbol": symbol,
            "type": doc_type,
            "date": date,
            "source": tool,
            "validity_class": validity_class.value,
            "as_of": as_of_epoch,
            "valid_until": valid_until,
        })
    if items:
        await qdr.upsert_snippets(items)
    return ids
```

**Step 4: Add `valid_until` filter to `_fetch_rag` in `infrastructure/memory_manager.py`**

In `_fetch_rag`, find where `must_conditions` is built (around line 284) and add:

```python
import time as _time
# Filter out expired chunks
must_conditions.append(
    rest.FieldCondition(
        key="valid_until",
        range=rest.Range(gte=int(_time.time()))
    )
)
```

Add this block after the ticker filter (before `limit = 5 ...`).

**Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_ltm_freshness.py::test_ingest_raw_payload_contains_valid_until_as_epoch -v
```
Expected: PASSED

**Step 6: Commit**

```bash
git add rag/fusion.py infrastructure/memory_manager.py tests/test_ltm_freshness.py
git commit -m "feat: Qdrant ingest adds validity payload; retrieval filters expired chunks"
```

---

## Task 8: MemoryPolicy class

**Files:**
- Create: `infrastructure/memory_policy.py`

**Step 1: Write the failing tests**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 8: MemoryPolicy ===

def test_memory_policy_current_price_requires_live_tools():
    from infrastructure.memory_policy import get_policy
    from infrastructure.memory_types import QueryIntent
    policy = get_policy(QueryIntent.PRICE_ONLY)
    assert policy.require_live_tools is True

def test_memory_policy_current_price_excludes_price_snapshot():
    from infrastructure.memory_policy import get_policy
    from infrastructure.memory_types import QueryIntent
    policy = get_policy(QueryIntent.PRICE_ONLY)
    assert "price_snapshot" not in policy.allowed_classes

def test_memory_policy_preference_lookup_no_live_tools():
    from infrastructure.memory_policy import get_policy
    from infrastructure.memory_types import QueryIntent
    policy = get_policy(QueryIntent.USER_PREFERENCES)
    assert policy.require_live_tools is False

def test_memory_policy_trade_decision_allows_all_stable_classes():
    from infrastructure.memory_policy import get_policy
    from infrastructure.memory_types import QueryIntent
    policy = get_policy(QueryIntent.TRADE_DECISION)
    assert "trading_decision" in policy.allowed_classes
    assert "user_preference" in policy.allowed_classes
    assert "fundamental_data" in policy.allowed_classes

def test_memory_policy_trade_decision_price_snapshot_is_context_only():
    from infrastructure.memory_policy import get_policy
    from infrastructure.memory_types import QueryIntent
    policy = get_policy(QueryIntent.TRADE_DECISION)
    assert "price_snapshot" in policy.context_only_classes

def test_memory_policy_explain_last_decision_no_live_tools():
    from infrastructure.memory_policy import get_policy
    from infrastructure.memory_types import QueryIntent
    policy = get_policy(QueryIntent.USER_HISTORY)
    assert policy.require_live_tools is False
    assert "trading_decision" in policy.allowed_classes
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ltm_freshness.py::test_memory_policy_current_price_requires_live_tools tests/test_ltm_freshness.py::test_memory_policy_preference_lookup_no_live_tools -v
```
Expected: FAILED

**Step 3: Create `infrastructure/memory_policy.py`**

```python
# infrastructure/memory_policy.py
"""
MemoryPolicy — behavioral contract for memory retrieval per query intent.

Maps QueryIntent → which validity classes may influence the answer,
whether live tools are required, and which classes are context-only
(may be shown but never treated as authoritative).

The fundamental rule: market truth always comes from tools, not memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from infrastructure.memory_types import QueryIntent


@dataclass
class MemoryPolicy:
    """Policy governing memory retrieval for a given query intent."""
    intent: str
    allowed_classes: List[str]         # validity classes retrieval may use
    require_live_tools: bool           # must DataFetcher run before composing?
    context_only_classes: List[str]    # allowed but never authoritative
    max_items_per_class: int = 5       # cap on injected memory items
    recency_weight: bool = True        # order by as_of DESC within window


# Intent → MemoryPolicy mapping
# Based on the design doc: docs/plans/2026-03-02-ltm-freshness-design.md
_POLICY_TABLE: dict[QueryIntent, MemoryPolicy] = {
    QueryIntent.PRICE_ONLY: MemoryPolicy(
        intent="current_price",
        allowed_classes=["user_preference", "behavioral_pattern", "trading_decision"],
        require_live_tools=True,
        context_only_classes=["trading_decision"],  # price_snapshot excluded entirely
    ),
    QueryIntent.NEWS_SUMMARY: MemoryPolicy(
        intent="latest_news",
        allowed_classes=["news_sentiment", "user_preference"],
        require_live_tools=True,
        context_only_classes=["news_sentiment"],
    ),
    QueryIntent.TRADE_DECISION: MemoryPolicy(
        intent="strategy_recommendation",
        allowed_classes=[
            "trading_decision", "behavioral_pattern", "user_preference",
            "fundamental_data", "news_sentiment",
        ],
        require_live_tools=True,
        context_only_classes=["price_snapshot", "news_sentiment"],
    ),
    QueryIntent.USER_HISTORY: MemoryPolicy(
        intent="explain_last_decision",
        allowed_classes=["trading_decision", "session_memory", "session_summary"],
        require_live_tools=False,
        context_only_classes=[],
    ),
    QueryIntent.USER_PREFERENCES: MemoryPolicy(
        intent="preference_lookup",
        allowed_classes=["user_preference", "behavioral_pattern"],
        require_live_tools=False,
        context_only_classes=[],
    ),
    QueryIntent.TICKER_INFO: MemoryPolicy(
        intent="company_fundamentals",
        allowed_classes=["fundamental_data", "trading_decision", "user_preference"],
        require_live_tools=True,
        context_only_classes=["fundamental_data"],
    ),
    QueryIntent.SEMANTIC_SEARCH: MemoryPolicy(
        intent="research_compare",
        allowed_classes=["fundamental_data", "news_sentiment", "user_preference", "behavioral_pattern"],
        require_live_tools=True,
        context_only_classes=["fundamental_data", "news_sentiment"],
    ),
    QueryIntent.CONVERSATION: MemoryPolicy(
        intent="recap_conversation",
        allowed_classes=["session_memory", "session_summary"],
        require_live_tools=False,
        context_only_classes=[],
    ),
    QueryIntent.UNKNOWN: MemoryPolicy(
        intent="unknown",
        allowed_classes=["user_preference", "session_memory"],
        require_live_tools=True,
        context_only_classes=["session_memory"],
    ),
}


def get_policy(intent: QueryIntent) -> MemoryPolicy:
    """Return the MemoryPolicy for a given query intent."""
    return _POLICY_TABLE.get(intent, _POLICY_TABLE[QueryIntent.UNKNOWN])
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ltm_freshness.py::test_memory_policy_current_price_requires_live_tools tests/test_ltm_freshness.py::test_memory_policy_current_price_excludes_price_snapshot tests/test_ltm_freshness.py::test_memory_policy_preference_lookup_no_live_tools tests/test_ltm_freshness.py::test_memory_policy_trade_decision_allows_all_stable_classes tests/test_ltm_freshness.py::test_memory_policy_trade_decision_price_snapshot_is_context_only tests/test_ltm_freshness.py::test_memory_policy_explain_last_decision_no_live_tools -v
```
Expected: 6 passed

**Step 5: Commit**

```bash
git add infrastructure/memory_policy.py tests/test_ltm_freshness.py
git commit -m "feat: MemoryPolicy behavioral contract — intent to allowed validity classes"
```

---

## Task 9: AgentState + router — emit memory_policy

**Files:**
- Modify: `agent/state.py`
- Modify: `agent/nodes/router.py`

**Step 1: Write the failing tests**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 9: router emits memory_policy ===

def test_agent_state_accepts_memory_policy_field():
    from agent.state import AgentState
    from infrastructure.memory_policy import MemoryPolicy
    state: AgentState = {
        "query": "test",
        "memory_policy": None,
    }
    assert state.get("memory_policy") is None  # field exists, value is None

def test_router_emits_memory_policy():
    import asyncio
    from unittest.mock import patch, AsyncMock, MagicMock
    from infrastructure.memory_types import MemoryContext, ClassificationResult, QueryIntent, MemoryLayer

    fake_classification = ClassificationResult(
        intent=QueryIntent.PRICE_ONLY,
        confidence=0.95,
        tickers=["AAPL"],
        layers_needed=[MemoryLayer.RUN_CACHE]
    )
    fake_context = MemoryContext()
    fake_context.classification = fake_classification

    mock_manager = MagicMock()
    mock_manager.get_context = AsyncMock(return_value=fake_context)

    with patch("agent.nodes.router.get_memory_manager", return_value=mock_manager), \
         patch("agent.nodes.router.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"ticker":"AAPL","additional_tickers":[],"intent":"price","query_type":"stock","next_agent":"fetcher","is_trading_query":false}'}}]
        }
        mock_post.return_value.raise_for_status = MagicMock()

        from agent.nodes.router import router_node
        result = asyncio.run(router_node({"query": "AAPL price?", "user_id": "u1"}))

    assert "memory_policy" in result
    assert result["memory_policy"] is not None
    assert result["memory_policy"].require_live_tools is True
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ltm_freshness.py::test_agent_state_accepts_memory_policy_field tests/test_ltm_freshness.py::test_router_emits_memory_policy -v
```
Expected: FAILED

**Step 3: Add `memory_policy` field to `agent/state.py`**

In `AgentState`, after the `memory_context` line, add:

```python
memory_policy: Any    # MemoryPolicy from infrastructure.memory_policy
```

**Step 4: Update `agent/nodes/router.py` to emit memory_policy**

Add import at top of router.py:

```python
from infrastructure.memory_policy import get_policy
```

In `router_node`, after the `context = await manager.get_context(...)` block, add:

```python
# Derive MemoryPolicy from classifier output
memory_policy = None
if context and context.classification:
    memory_policy = get_policy(context.classification.intent)
```

In the return dict (both the success and fallback paths), add:

```python
"memory_policy": memory_policy,
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_ltm_freshness.py::test_agent_state_accepts_memory_policy_field tests/test_ltm_freshness.py::test_router_emits_memory_policy -v
```
Expected: 2 passed

**Step 6: Commit**

```bash
git add agent/state.py agent/nodes/router.py tests/test_ltm_freshness.py
git commit -m "feat: router emits memory_policy based on classifier intent"
```

---

## Task 10: Prompt stamping

**Files:**
- Modify: `infrastructure/memory_types.py`

**Step 1: Write the failing tests**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 10: Prompt stamping ===

def test_stamp_memory_fact_includes_as_of_and_age():
    from infrastructure.memory_types import stamp_memory_fact
    result = stamp_memory_fact(
        validity_class="trading_decision",
        as_of_epoch=1740960000,
        content="BUY AAPL — fund manager approved",
        context_only=True
    )
    assert "as_of:" in result
    assert "trading_decision" in result
    assert "Context only" in result
    assert "BUY AAPL" in result

def test_stamp_memory_fact_user_preference_no_context_only_label():
    from infrastructure.memory_types import stamp_memory_fact
    result = stamp_memory_fact(
        validity_class="user_preference",
        as_of_epoch=1740960000,
        content="Risk tolerance = moderate",
        context_only=False
    )
    assert "user_preference" in result
    assert "Context only" not in result

def test_to_prompt_context_stamps_trading_decisions():
    from infrastructure.memory_types import MemoryContext
    ctx = MemoryContext()
    ctx.ticker_history = [{
        "last_decision": "BUY",
        "ticker": "AAPL",
        "last_analysis_date": "2026-02-12T00:00:00",
        "validity_class": "trading_decision",
        "as_of": 1739318400,
    }]
    result = ctx.to_prompt_context()
    assert "as_of" in result or "trading_decision" in result
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ltm_freshness.py::test_stamp_memory_fact_includes_as_of_and_age tests/test_ltm_freshness.py::test_stamp_memory_fact_user_preference_no_context_only_label -v
```
Expected: FAILED

**Step 3: Add `stamp_memory_fact` to `infrastructure/memory_types.py`**

Add this function after the `MemoryContext` class:

```python
def stamp_memory_fact(
    validity_class: str,
    as_of_epoch: int,
    content: str,
    context_only: bool = True,
) -> str:
    """
    Format a memory-sourced fact with freshness label for LLM prompt injection.

    Format: "[class] (as_of: YYYY-MM-DD, age: Nd): <content>\\nContext only — verify with live data."
    """
    from datetime import datetime, timezone
    as_of_dt = datetime.fromtimestamp(as_of_epoch, tz=timezone.utc)
    age_days = (datetime.now(tz=timezone.utc) - as_of_dt).days
    as_of_str = as_of_dt.strftime("%Y-%m-%d")

    stamp = f"{validity_class} (as_of: {as_of_str}, age: {age_days}d): {content}"
    if context_only:
        stamp += "\nContext only — verify with live data before treating as current."
    return stamp
```

**Step 4: Update `to_prompt_context` in `MemoryContext` to stamp historical decisions**

Replace the `if self.ticker_history:` block in `to_prompt_context`:

```python
if self.ticker_history:
    stamped = []
    for rec in self.ticker_history:
        as_of = rec.get("as_of")
        vc = rec.get("validity_class", "trading_decision")
        content = (
            f"{rec.get('ticker', '?')}: last decision {rec.get('last_decision', '?')}"
            f" on {rec.get('last_analysis_date', '?')}"
        )
        if isinstance(as_of, (int, float)):
            stamped.append(stamp_memory_fact(vc, int(as_of), content, context_only=True))
        else:
            stamped.append(content)
    parts.append("Prior decisions:\n" + "\n".join(stamped))
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_ltm_freshness.py::test_stamp_memory_fact_includes_as_of_and_age tests/test_ltm_freshness.py::test_stamp_memory_fact_user_preference_no_context_only_label tests/test_ltm_freshness.py::test_to_prompt_context_stamps_trading_decisions -v
```
Expected: 3 passed

**Step 6: Commit**

```bash
git add infrastructure/memory_types.py tests/test_ltm_freshness.py
git commit -m "feat: stamp_memory_fact formats memory with as_of/age/context-only label"
```

---

## Task 11: Startup initialization

**Files:**
- Modify: `api.py`
- Modify: `agent/graph.py`

**Step 1: Add startup init to `api.py`**

Read `api.py` first, then find the FastAPI app instantiation and add a startup event:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Postgres tables on startup (idempotent)
    try:
        from infrastructure.postgres_summaries import get_summaries
        get_summaries()  # singleton: calls initialize() on first use
        print("[Startup] Postgres initialized")
    except Exception as e:
        print(f"[Startup] Postgres not available: {e}")
    yield

app = FastAPI(lifespan=lifespan, ...)  # add lifespan= to existing FastAPI()
```

**Step 2: Add startup init to `agent/graph.py`**

At module level, after all imports, add:

```python
def _init_db() -> None:
    """Initialize Postgres on first graph compilation (idempotent, degrades gracefully)."""
    try:
        from infrastructure.postgres_summaries import get_summaries
        get_summaries()
    except Exception as e:
        print(f"[Graph] Postgres init skipped: {e}")

_init_db()
```

**Step 3: Verify app imports without error**

```bash
python -c "import agent.graph; print('OK')"
```
Expected: `[Graph] Postgres init skipped: ...` (or OK if Postgres is running) then `OK`

**Step 4: Commit**

```bash
git add api.py agent/graph.py
git commit -m "feat: initialize Postgres tables on startup in api.py and graph.py"
```

---

## Task 12: QueryClassifier integration in router (P2)

**Files:**
- Modify: `agent/nodes/router.py`

**Context:** The QueryClassifier is already used inside `MemoryManager.get_context()` for memory routing. The router additionally makes its own LLM call for agent routing (which node to go to). P2 wires classifier output → agent routing, eliminating the redundant LLM call for high-confidence classifications.

**Step 1: Write the failing test**

Append to `tests/test_ltm_freshness.py`:

```python
# === Task 12: QueryClassifier in router ===

def test_router_skips_llm_when_classifier_is_high_confidence():
    """If classifier confidence >= 0.85, router must NOT call the LLM."""
    import asyncio
    from unittest.mock import patch, AsyncMock, MagicMock
    from infrastructure.memory_types import (
        MemoryContext, ClassificationResult, QueryIntent, MemoryLayer
    )

    fake_classification = ClassificationResult(
        intent=QueryIntent.TRADE_DECISION,
        confidence=0.92,
        tickers=["TSLA"],
        layers_needed=[MemoryLayer.LTM, MemoryLayer.RAG]
    )
    fake_context = MemoryContext()
    fake_context.classification = fake_classification

    mock_manager = MagicMock()
    mock_manager.get_context = AsyncMock(return_value=fake_context)

    with patch("agent.nodes.router.get_memory_manager", return_value=mock_manager), \
         patch("agent.nodes.router.httpx.post") as mock_llm:

        from agent.nodes.router import router_node
        result = asyncio.run(router_node({"query": "Should I buy Tesla?", "user_id": "u1"}))

    # LLM must not have been called
    mock_llm.assert_not_called()
    assert result["is_trading_query"] is True
    assert result["next_agent"] == "trading"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_ltm_freshness.py::test_router_skips_llm_when_classifier_is_high_confidence -v
```
Expected: FAILED (LLM is still called)

**Step 3: Add classifier-driven routing to `agent/nodes/router.py`**

Add this mapping and helper near the top of the file (after imports):

```python
from infrastructure.memory_types import QueryIntent

# Map classifier intent → agent routing (when confidence is high enough)
_INTENT_TO_AGENT: dict[QueryIntent, tuple[str, str, bool]] = {
    # intent: (next_agent, query_type, is_trading_query)
    QueryIntent.PRICE_ONLY:       ("fetcher",  "stock",   False),
    QueryIntent.TICKER_INFO:      ("fetcher",  "stock",   False),
    QueryIntent.NEWS_SUMMARY:     ("fetcher",  "news",    False),
    QueryIntent.TRADE_DECISION:   ("trading",  "trading", True),
    QueryIntent.USER_HISTORY:     ("composer", "general", False),
    QueryIntent.USER_PREFERENCES: ("composer", "general", False),
    QueryIntent.SEMANTIC_SEARCH:  ("fetcher",  "general", False),
    QueryIntent.CONVERSATION:     ("composer", "general", False),
}

_CLASSIFIER_CONFIDENCE_THRESHOLD = 0.85
```

In `router_node`, after the `memory_context_str = context.to_prompt_context()` block, add:

```python
# Fast path: use classifier result if confidence is high (skips LLM call)
if (
    context
    and context.classification
    and context.classification.confidence >= _CLASSIFIER_CONFIDENCE_THRESHOLD
    and context.classification.intent in _INTENT_TO_AGENT
):
    intent = context.classification.intent
    next_agent, query_type, is_trading_query = _INTENT_TO_AGENT[intent]
    tickers = context.classification.tickers
    ticker = tickers[0] if tickers else None

    # Crypto override
    if any(t in (tickers or []) for t in ["BTC-USD", "ETH-USD", "BTC", "ETH"]):
        next_agent = "crypto"
        query_type = "crypto"
        is_trading_query = False

    parsed_query = ParsedQuery(
        ticker=ticker,
        additional_tickers=tickers[1:] if tickers else [],
        intent=intent.value,
        query_type=query_type,
        raw_query=query
    )
    print(f"[Router] Classifier fast-path → {next_agent} (confidence: {context.classification.confidence:.2f})")
    memory_policy = get_policy(intent)
    return {
        "parsed_query": parsed_query,
        "next_agent": next_agent,
        "is_trading_query": is_trading_query,
        "run_id": run_id,
        "memory_context": context,
        "memory_policy": memory_policy,
        "memory": {"user_id": user_id, "retrieved_memory": memory_context_str},
    }
```

This block goes between the memory fetch and the LLM call (`try: response = httpx.post(...)`). The `try/except` LLM block remains as the fallback for low-confidence classifications.

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_ltm_freshness.py::test_router_skips_llm_when_classifier_is_high_confidence -v
```
Expected: PASSED

**Step 5: Run full test suite to check for regressions**

```bash
python -m pytest tests/test_ltm_freshness.py tests/test_memory_integration.py -v
```
Expected: All tests from `test_ltm_freshness.py` pass. `test_memory_integration.py` tests pass (pre-existing failures in other files are unrelated).

**Step 6: Commit**

```bash
git add agent/nodes/router.py tests/test_ltm_freshness.py
git commit -m "feat: QueryClassifier fast-path in router — skips LLM for high-confidence intents"
```

---

## Final verification

```bash
python -m pytest tests/test_ltm_freshness.py -v
python -c "import agent.graph; print('Graph import OK')"
docker compose up postgres -d && sleep 3 && python -c "
import dotenv; dotenv.load_dotenv()
from infrastructure.postgres_summaries import get_summaries
s = get_summaries()
print('Postgres OK:', s._initialized)
"
```

Expected: All tests pass, Postgres initializes cleanly.
