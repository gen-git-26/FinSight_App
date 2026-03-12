# Streamlit Cloud Deployment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy FinSight to Streamlit Community Cloud so external users can access all features (Standard + Trading A2A flows) via a public URL, protected by a simple password.

**Architecture:** Add a secrets-sync block + password gate to `app.py`. The critical ordering constraint is: (1) secrets sync, (2) import `skeleton.py` which calls `st.set_page_config` at module level, (3) password gate (`st.text_input`). Streamlit requires `set_page_config` to be the first rendering command. No other code changes — MCP fallback is already handled by existing try/except in `_fetch_via_mcp`, and `psycopg2-binary` needs no system packages.

**Tech Stack:** Streamlit Community Cloud (free), `st.secrets`, `os.environ`

**Spec:** `docs/superpowers/specs/2026-03-12-streamlit-cloud-deployment-design.md`

---

## Chunk 1: Password Gate + Secrets Sync in `app.py`

### Files
- Modify: `app.py`
- Test: `tests/test_app_startup.py` (new)

### Critical ordering constraint

`ui/skeleton.py` calls `st.set_page_config(...)` at **module level** (line 33). Streamlit requires `set_page_config` to be the first Streamlit rendering call. Therefore:

```
1. st.secrets sync   ← st.secrets access is NOT a rendering command, safe to run first
2. from ui.skeleton import main   ← triggers set_page_config (first rendering command)
3. password gate: st.text_input / st.stop   ← rendering commands, must come after set_page_config
4. dotenv.load_dotenv()
5. from agent.graph import run_query   ← triggers Postgres/Redis init via _ensure_env_loaded
6. main(query_fn=run_query)
```

Note: `_ensure_env_loaded()` in `utils/config.py` calls `load_dotenv(override=True)`. On Streamlit Cloud there is no `.env` file, so `override=True` against a missing file is harmless — secrets already loaded into `os.environ` are not overwritten. This is safe.

---

### Task 1: Write failing test for app.py structure

- [ ] **Step 1: Write the test file**

Create `tests/test_app_startup.py`:

```python
"""
Tests for app.py startup: secrets sync and password gate.

We test three things:
1. The secrets sync logic (tested in isolation — importing app.py would invoke Streamlit rendering)
2. The structure of app.py source (confirms the sync block was added and ordering is correct)
3. Edge cases for the sync logic
"""
import os
import ast
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Structure tests: verify app.py source contains required blocks in correct order
# ---------------------------------------------------------------------------

def _app_source() -> str:
    return (Path(__file__).parent.parent / "app.py").read_text()


def test_app_contains_secrets_sync():
    """app.py must contain the st.secrets sync block."""
    src = _app_source()
    assert "st.secrets.items()" in src, "secrets sync block missing from app.py"
    assert "os.environ.setdefault" in src, "setdefault call missing from app.py"


def test_app_secrets_sync_inside_try_except():
    """secrets sync must be wrapped in try/except to avoid crash when no secrets.toml exists."""
    src = _app_source()
    # try block must come before st.secrets.items()
    assert src.index("try:") < src.index("st.secrets.items()"), (
        "st.secrets.items() must be inside a try block"
    )


def test_app_skeleton_import_before_password_gate():
    """skeleton import (which triggers set_page_config) must precede st.text_input."""
    src = _app_source()
    assert "from ui.skeleton import main" in src, "skeleton import missing"
    assert "st.text_input" in src, "password gate missing"
    assert src.index("from ui.skeleton import main") < src.index("st.text_input"), (
        "skeleton must be imported (triggering set_page_config) before st.text_input is called"
    )


def test_app_agent_import_after_skeleton_import():
    """agent.graph import must come after skeleton import (secrets must be in os.environ first)."""
    src = _app_source()
    assert "from agent.graph import run_query" in src
    assert src.index("from ui.skeleton import main") < src.index("from agent.graph import run_query"), (
        "agent.graph import must come after skeleton import"
    )


# ---------------------------------------------------------------------------
# Logic tests: secrets sync behaviour (tested in isolation)
# ---------------------------------------------------------------------------

def _run_sync(secrets_dict, existing_env=None):
    """Helper: run the secrets sync logic against a fake st.secrets."""
    import streamlit as st
    env_patch = existing_env or {}
    with patch.dict(os.environ, env_patch, clear=False):
        # Remove keys not in existing_env
        for k in secrets_dict:
            if k not in (existing_env or {}):
                os.environ.pop(k, None)
        with patch.object(type(st.secrets), "items", return_value=secrets_dict.items()):
            try:
                for _k, _v in st.secrets.items():
                    if isinstance(_v, str):
                        os.environ.setdefault(_k, _v)
            except Exception:
                pass
        return {k: os.environ.get(k) for k in secrets_dict}


def test_secrets_sync_copies_strings_to_environ():
    result = _run_sync({"OPENAI_API_KEY": "sk-test", "REDIS_HOST": "myhost"})
    assert result["OPENAI_API_KEY"] == "sk-test"
    assert result["REDIS_HOST"] == "myhost"


def test_secrets_sync_does_not_override_existing():
    result = _run_sync(
        {"OPENAI_API_KEY": "sk-from-secrets"},
        existing_env={"OPENAI_API_KEY": "sk-existing"}
    )
    assert result["OPENAI_API_KEY"] == "sk-existing"


def test_secrets_sync_skips_non_string_values():
    """Nested TOML sections (dicts) must not be written to os.environ."""
    import streamlit as st
    os.environ.pop("FLAT_KEY", None)
    os.environ.pop("SECTION", None)
    fake = {"SECTION": {"key": "val"}, "FLAT_KEY": "flat_val"}
    with patch.object(type(st.secrets), "items", return_value=fake.items()):
        try:
            for _k, _v in st.secrets.items():
                if isinstance(_v, str):
                    os.environ.setdefault(_k, _v)
        except Exception:
            pass
    assert os.environ.get("FLAT_KEY") == "flat_val"
    assert "SECTION" not in os.environ


def test_secrets_sync_silent_on_missing_secrets_toml():
    """When no secrets.toml exists locally, the sync block must not raise."""
    import streamlit as st
    with patch.object(type(st.secrets), "items", side_effect=FileNotFoundError):
        try:
            for _k, _v in st.secrets.items():
                if isinstance(_v, str):
                    os.environ.setdefault(_k, _v)
        except Exception:
            pass  # must NOT propagate — test passes only if we reach this point
```

- [ ] **Step 2: Run tests — structure tests should FAIL (app.py not yet modified)**

```bash
cd /workspaces/FinSight_App && source .venv/bin/activate && pytest tests/test_app_startup.py -v
```

Expected:
- `test_app_contains_secrets_sync` → **FAIL**
- `test_app_skeleton_import_before_password_gate` → **FAIL**
- Logic tests (`test_secrets_sync_*`) → PASS (testing isolated logic)

---

### Task 2: Implement the changes in `app.py`

- [ ] **Step 3: Replace `app.py` with the new version**

```python
# app.py
"""
FinSight - Multi-Agent Financial Analysis System

Powered by LangGraph with A2A (Agent-to-Agent) architecture:
- Standard Flow:  Router -> Fetcher/Crypto -> Analyst -> Composer
- Trading Flow:   Router -> Fetcher -> Analysts Team -> Researchers
                  -> Trader -> Risk Manager -> Fund Manager -> Composer
"""
from __future__ import annotations

import os
import streamlit as st

# Sync Streamlit Cloud secrets → os.environ so infrastructure configs
# (Redis, Postgres, OpenAI) can read them via os.getenv().
# Wrapped in try/except: locally, no secrets.toml exists — dotenv.load_dotenv() handles that.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

# skeleton import triggers st.set_page_config (must be first Streamlit rendering call)
from ui.skeleton import main  # noqa: E402

# Password gate — remove the 4 lines below to open public access
_pw = os.environ.get("FINSIGHT_PASSWORD", "")
if _pw:
    _entered = st.text_input("Password", type="password")
    if _entered != _pw:
        st.stop()

import dotenv  # noqa: E402
dotenv.load_dotenv()

from agent.graph import run_query  # noqa: E402

main(query_fn=run_query)
```

- [ ] **Step 4: Run tests — all should PASS**

```bash
cd /workspaces/FinSight_App && source .venv/bin/activate && pytest tests/test_app_startup.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
cd /workspaces/FinSight_App && source .venv/bin/activate && pytest tests/ -v --ignore=tests/test_mcp_servers.py 2>&1 | tail -25
```

Expected: same pass/fail ratio as baseline (pre-existing failures in `test_memory_system.py::test_memory_manager` are acceptable — async def without pytest-asyncio, unrelated to this change).

- [ ] **Step 6: Commit**

```bash
cd /workspaces/FinSight_App
git add app.py tests/test_app_startup.py
git commit -m "feat: add Streamlit Cloud secrets sync and password gate to app.py"
```

---

## Chunk 2: Streamlit Cloud Setup (Manual Steps)

These steps are performed in the browser — no code changes.

### Task 3: Connect repo and configure secrets

- [ ] **Step 1: Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub**

- [ ] **Step 2: Click "New app"**
  - Repository: `gen-git-26/FinSight_App`
  - Branch: `main`
  - Main file path: `app.py`

- [ ] **Step 3: Click "Advanced settings" before deploying**

- [ ] **Step 4: Enter all Secrets in the text area (TOML format)**

> **Important:** Replace every `"..."` with the real value before saving. Leaving placeholders will cause auth failures.

```toml
OPENAI_API_KEY = "sk-..."           # required
OPENAI_MODEL = "gpt-4o-mini"
FINSIGHT_PASSWORD = "choose-a-password"   # required — users will enter this

REDIS_HOST = "redis-19151.c74.us-east-1-4.ec2.cloud.redislabs.com"
REDIS_PORT = "19151"
REDIS_PASSWORD = "..."              # required — from RedisLabs dashboard
REDIS_DB = "0"
REDIS_CACHE_DB = "1"

POSTGRES_HOST = "ep-noisy-unit-aln3jis9-pooler.c-3.eu-central-1.aws.neon.tech"
POSTGRES_PORT = "5432"
POSTGRES_DB = "neondb"
POSTGRES_USER = "neondb_owner"
POSTGRES_PASSWORD = "..."           # required — from Neon dashboard

QDRANT_URL = "https://..."          # required — from Qdrant dashboard
QDRANT_API_KEY = "..."              # required

FINNHUB_API_KEY = "..."             # optional — API fallback
ALPHAVANTAGE_API_KEY = "..."        # optional — API fallback
FINANCIAL_DATASETS_API_KEY = "..."  # optional — MCP server (not used on Streamlit Cloud)
```

- [ ] **Step 5: Click "Deploy!"**

Streamlit Cloud installs `requirements.txt` and launches (~2–4 minutes first deploy).

- [ ] **Step 6: Verify the deployed app**
  - Password screen appears ✓
  - Enter password → app loads ✓
  - Test standard flow: `"What is Apple's stock price?"` → returns price data ✓
  - Test trading flow: `"Should I buy AAPL?"` → returns BUY/SELL/HOLD recommendation ✓

---

## Reversal Instructions

To remove the password gate and restore original `app.py`:

```python
# Restore app.py to original:
from __future__ import annotations

import dotenv
dotenv.load_dotenv()

from agent.graph import run_query
from ui.skeleton import main

main(query_fn=run_query)
```

```bash
git add app.py
git commit -m "revert: remove Streamlit Cloud secrets sync and password gate"
git push
```

Streamlit Cloud auto-deploys on push — no further action needed.
