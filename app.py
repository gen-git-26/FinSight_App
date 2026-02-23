# app.py
"""
FinSight - Multi-Agent Financial Analysis System

Powered by LangGraph with A2A (Agent-to-Agent) architecture:
- Standard Flow:  Router -> Fetcher/Crypto -> Analyst -> Composer
- Trading Flow:   Router -> Fetcher -> Analysts Team -> Researchers
                  -> Trader -> Risk Manager -> Fund Manager -> Composer
"""
from __future__ import annotations

import dotenv
dotenv.load_dotenv()

from agent.graph import run_query
from ui.skeleton import main

main(query_fn=run_query)
