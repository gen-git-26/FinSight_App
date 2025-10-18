# tools/query_parser.py - FIXED VERSION
from __future__ import annotations

import json
import httpx
import asyncio
from typing import Optional, List
from dataclasses import dataclass
from utils.config import load_settings
from tools.async_utils import run_async_safe
from tools.time_parser import parse_time_range_to_days


@dataclass
class ParsedQuery:
    """Structured representation of a financial query."""
    primary_ticker: Optional[str]
    secondary_tickers: List[str]
    intent: str  # "price_quote", "news", "analysis", "options", etc.
    data_type: str  # "realtime", "historical", "fundamental"
    time_range: Optional[str]  # "30 days", "1 year", etc.
    specific_date: Optional[str]  # YYYY-MM-DD
    options_type: Optional[str]  # "calls", "puts"
    expiration_days: Optional[int]
    company_names: List[str]
    raw_intent: str  # original query


async def _parse_query_with_llm_async(query: str) -> ParsedQuery:
    """
    Use OpenAI GPT to parse the user query structurally.
    This is the actual async implementation.
    Then enhance with time_parser for accurate time conversions.
    """
    cfg = load_settings()
    
    system_prompt = """You are a financial query parser. Extract structured information from user queries.
    
Return ONLY valid JSON (no markdown, no code blocks, just raw JSON) with exactly these fields:
{
  "primary_ticker": "AAPL or null if not found",
  "secondary_tickers": ["GOOGL"],
  "intent": "one of: price_quote, news, analysis, comparison, options, fundamentals, dividend, insider, recommendation, historical",
  "data_type": "one of: realtime, historical, fundamental",
  "time_range": "30 days or null",
  "specific_date": "YYYY-MM-DD or null",
  "options_type": "calls or puts or chain or null",
  "expiration_days": 30 or null,
  "company_names": ["Apple Inc"]
}

Rules:
- primary_ticker: MAIN stock/crypto symbol (UPPERCASE)
- intent: What does user want? Recognize: options, fundamentals, price, news, analysis, historical, etc.
- data_type: Is it real-time, historical, or fundamental data?
- time_range: Any mention of period? ("last 6 months", "30 days", "1 year")
- expiration_days: For options, extract days until expiration
- options_type: "calls" or "puts"
- company_names: Full names like "Apple Inc", "Microsoft Corporation"

Be strict about JSON format - NO markdown, NO code blocks, just raw JSON."""

    user_message = f"""Parse this financial query:

Query: {query}

Return ONLY valid JSON, no other text."""

    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": cfg.openai_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.0,  # Deterministic
        "max_tokens": 300
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            
            content = data["choices"][0]["message"]["content"].strip()
            
            # Try to extract JSON from various formats
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) > 1:
                    content = parts[1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
            
            # Parse JSON
            parsed = json.loads(content)
            
            return ParsedQuery(
                primary_ticker=parsed.get("primary_ticker"),
                secondary_tickers=parsed.get("secondary_tickers", []),
                intent=parsed.get("intent", "analysis"),
                data_type=parsed.get("data_type", "realtime"),
                time_range=parsed.get("time_range"),
                specific_date=parsed.get("specific_date"),
                options_type=parsed.get("options_type"),
                expiration_days=parsed.get("expiration_days"),
                company_names=parsed.get("company_names", []),
                raw_intent=query
            )
    except json.JSONDecodeError as e:
        print(f"[QueryParser] JSON decode error: {e}")
        print(f"[QueryParser] Content was: {content[:200]}")
        return ParsedQuery(
            primary_ticker=None,
            secondary_tickers=[],
            intent="analysis",
            data_type="realtime",
            time_range=None,
            specific_date=None,
            options_type=None,
            expiration_days=None,
            company_names=[],
            raw_intent=query
        )
    except Exception as e:
        print(f"[QueryParser] Error: {e}")
        return ParsedQuery(
            primary_ticker=None,
            secondary_tickers=[],
            intent="analysis",
            data_type="realtime",
            time_range=None,
            specific_date=None,
            options_type=None,
            expiration_days=None,
            company_names=[],
            raw_intent=query
        )


def parse_query_with_llm(query: str) -> ParsedQuery:
    """
    Safe sync wrapper around async parser.
    Can be called from Streamlit or sync context without event loop conflicts.
    """
    try:
        return run_async_safe(_parse_query_with_llm_async(query))
    except Exception as e:
        print(f"[parse_query_with_llm] Wrapper error: {e}")
        return ParsedQuery(
            primary_ticker=None,
            secondary_tickers=[],
            intent="analysis",
            data_type="realtime",
            time_range=None,
            specific_date=None,
            options_type=None,
            expiration_days=None,
            company_names=[],
            raw_intent=query
        )