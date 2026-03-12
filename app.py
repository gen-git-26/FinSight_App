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
