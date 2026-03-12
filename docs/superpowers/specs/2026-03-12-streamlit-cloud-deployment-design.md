# Design: Streamlit Cloud Deployment

**Date:** 2026-03-12
**Status:** Approved

## Goal

Deploy FinSight to Streamlit Community Cloud so external users can access the full application (Standard + Trading A2A flows) via a public URL, without installing anything.

## Constraints

- Free hosting (Streamlit Community Cloud)
- All features must work: price queries, crypto, full trading flow (4 analysts, researchers, risk manager, fund manager)
- Simple password gate (removable later)
- All external services remain unchanged (Redis, PostgreSQL/Neon, Qdrant)

## Architecture

No structural changes to the agent graph or memory system. Three targeted changes only:

### 1. Password Gate — `app.py`

Add 5 lines before `main(query_fn=run_query)`:

```python
import os, streamlit as st
_pw = st.secrets.get("FINSIGHT_PASSWORD", os.getenv("FINSIGHT_PASSWORD", ""))
if _pw:
    entered = st.text_input("Password", type="password")
    if entered != _pw:
        st.stop()
```

**Reversal:** Delete these 5 lines.

### 2. MCP Graceful Fallback — `datasources/__init__.py`

Current default: `FetchStrategy.PREFER_MCP` (tries MCP subprocess first).
On Streamlit Cloud, `create_subprocess_exec` is blocked — MCP init will fail.

Change: wrap MCP server setup in try/except; on failure, downgrade strategy to `FIRST_SUCCESS` (direct API calls). The fallback chain already exists: yfinance → Finnhub → AlphaVantage.

**No functional difference for users** — all data arrives via API clients.

**Reversal:** Remove try/except, restore unconditional `PREFER_MCP`.

### 3. System Dependencies — `packages.txt` (new file)

Streamlit Cloud uses `packages.txt` for apt packages before pip install.
`psycopg2-binary` requires `libpq-dev`:

```
libpq-dev
```

**Reversal:** Delete `packages.txt`.

## Secrets Configuration

All env vars entered in Streamlit Cloud → Settings → Secrets (TOML format):

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

## What Does NOT Change

- `agent/` — all nodes, graph, state unchanged
- `infrastructure/` — memory manager, Redis, Postgres, Qdrant unchanged
- `.streamlit/config.toml` — theme loaded automatically by Streamlit Cloud
- `requirements.txt` — no additions needed (all deps already listed)
- Both query flows (Standard + Trading A2A) work as-is

## Fallback Plan

If Streamlit Cloud proves insufficient (performance, subprocess restrictions):
→ Deploy via Render.com using the existing `Dockerfile` + supervisord.
MCP servers will work on Render (full Docker environment). Free tier has 15-min sleep on inactivity.

## Files Changed

| File | Change | Reversible |
|------|--------|-----------|
| `app.py` | +5 lines password gate | Delete 5 lines |
| `datasources/__init__.py` | MCP try/except fallback | Remove try/except |
| `packages.txt` | New file, 1 line | Delete file |

## Out of Scope

- CI/CD pipeline (Streamlit Cloud auto-deploys on git push)
- Custom domain
- Rate limiting per user
- Render deployment (only if Streamlit Cloud fails)
