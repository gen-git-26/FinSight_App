# Design: Streamlit Cloud Deployment

**Date:** 2026-03-12
**Status:** Approved (v2 — post spec review)

## Goal

Deploy FinSight to Streamlit Community Cloud so external users can access the full application (Standard + Trading A2A flows) via a public URL, without installing anything.

## Constraints

- Free hosting (Streamlit Community Cloud)
- All features must work: price queries, crypto, full trading flow (4 analysts, researchers, risk manager, fund manager)
- Simple password gate (removable later)
- All external services remain unchanged (Redis, PostgreSQL/Neon, Qdrant)

## Architecture

No structural changes to the agent graph or memory system. Three targeted changes only.

---

## Code Changes

### 1. `app.py` — Secrets sync + Password gate

**Two responsibilities in one block, added before the agent import:**

```python
# app.py — add at the top, before `from agent.graph import run_query`
import os
import streamlit as st

# 1. Sync st.secrets → os.environ so all infra configs (Redis, Postgres, OpenAI)
#    can read them via os.getenv(). Must happen before agent.graph is imported
#    because that import initialises Postgres tables and the memory manager.
#    Wrapped in try/except: locally, no secrets.toml exists — dotenv.load_dotenv()
#    handles that case instead.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # local dev: .env loaded by dotenv.load_dotenv() below

# 2. Password gate (remove these 4 lines to open access)
_pw = os.environ.get("FINSIGHT_PASSWORD", "")
if _pw:
    _entered = st.text_input("Password", type="password")
    if _entered != _pw:
        st.stop()
```

**Import order constraint:** All `st.*` imports and `from ui.skeleton import main` must remain below this block. `set_page_config` inside `skeleton.py` runs at import time — this is fine because Streamlit processes it before rendering, but the password check must be after all imports and before `main(...)`.

**Reversal:** Delete the two blocks above (secrets sync + password gate). `dotenv.load_dotenv()` already handles local `.env` files in development.

---

### 2. `datasources/__init__.py` — MCP graceful fallback

**No code change needed.** The fallback is already handled.

`setup_default_servers(use_local=True)` (line 86) only registers a `MCPServerConfig` into a dict — it does not spawn subprocesses. The actual subprocess is spawned inside `MCPClient.connect()` → `stdio_client()`, which is called by `call_tool()`. That call is already inside the existing `try/except Exception` block (lines 92–106). When Streamlit Cloud blocks the subprocess, the exception is caught, `_fetch_via_mcp` returns `None`, and the `PREFER_MCP` strategy falls back to API clients automatically.

**Nothing to change. No reversal needed.**

---

### 3. `packages.txt` — NOT needed

`psycopg2-binary` is a self-contained wheel that bundles its own `libpq`. It does not require `libpq-dev`. No `packages.txt` file is needed for the current `requirements.txt`.

**If in future `psycopg2` (non-binary) is used instead**, then `libpq-dev` would be required. For now: no `packages.txt`.

---

## Secrets Configuration

All env vars entered in **Streamlit Cloud → App Settings → Secrets** (TOML format).
The secrets sync code in `app.py` copies them to `os.environ` at startup.

```toml
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-4o-mini"
FINSIGHT_PASSWORD = "..."

REDIS_HOST = "redis-19151.c74.us-east-1-4.ec2.cloud.redislabs.com"
REDIS_PORT = "19151"
REDIS_PASSWORD = "..."
REDIS_DB = "0"
REDIS_CACHE_DB = "1"

POSTGRES_HOST = "ep-noisy-unit-aln3jis9-pooler.c-3.eu-central-1.aws.neon.tech"
POSTGRES_PORT = "5432"
POSTGRES_DB = "neondb"
POSTGRES_USER = "neondb_owner"
POSTGRES_PASSWORD = "..."

QDRANT_URL = "..."
QDRANT_API_KEY = "..."

FINNHUB_API_KEY = "..."
ALPHAVANTAGE_API_KEY = "..."
```

---

## What Does NOT Change

- `agent/` — all nodes, graph, state unchanged
- `infrastructure/` — memory manager, Redis, Postgres, Qdrant unchanged
- `.streamlit/config.toml` — theme loaded automatically by Streamlit Cloud
- `requirements.txt` — no additions needed
- Both query flows (Standard + Trading A2A) work as-is

---

## Files Changed

| File | Change | Reversal |
|------|--------|---------|
| `app.py` | +12 lines (secrets sync + password gate) | Delete the 2 blocks |
| `datasources/__init__.py` | No change needed — fallback already works | N/A |
| `packages.txt` | Not created | N/A |

---

## Fallback Plan

If Streamlit Cloud proves insufficient → deploy via Render.com using the existing `Dockerfile` + supervisord. MCP servers will work (full Docker, no subprocess restriction). Free tier sleeps after 15 min inactivity (~30s cold start).
