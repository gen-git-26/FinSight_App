# tools/query_parser.py
from __future__ import annotations

import json
import httpx
from typing import Optional
from dataclasses import dataclass
from utils.config import load_settings


@dataclass
class ParsedQuery:
    """Simple parsed query structure."""
    ticker: Optional[str]  # Just the ticker
    intent: str  # price, news, options, fundamentals, historical
    raw_query: str  # Original query


def parse_query_simple(query: str) -> ParsedQuery:
    """
    Use LLM to extract ONLY ticker and intent - nothing more.
    Fast, simple, effective.
    """
    cfg = load_settings()
    
    system_prompt = """You are a ticker extraction expert. Extract ONLY the stock ticker and intent.

Return ONLY this JSON format (no markdown, no explanations):
{
  "ticker": "AAPL",
  "intent": "price"
}

Rules:
1. ticker: Stock symbol (AAPL, TSLA, META, etc.) or crypto with -USD (BTC-USD)
   - "Tesla" → "TSLA"
   - "Apple" → "AAPL" 
   - "Meta Platforms" → "META"
   - "bitcoin" → "BTC-USD"
   - If no ticker found → null

2. intent: ONE word only
   - "price" = current price/quote
   - "news" = news articles
   - "options" = options chain
   - "fundamentals" = financial statements
   - "historical" = historical prices
   - "info" = general company info

Examples:
- "Tesla stock price" → {"ticker": "TSLA", "intent": "price"}
- "AAPL options" → {"ticker": "AAPL", "intent": "options"}
- "Meta Platforms news" → {"ticker": "META", "intent": "news"}
- "financial metrics for Tesla" → {"ticker": "TSLA", "intent": "fundamentals"}
- "bitcoin price" → {"ticker": "BTC-USD", "intent": "price"}
"""

    user_message = f"Extract ticker and intent from: {query}"

    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg.openai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": cfg.openai_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.0,
                "max_tokens": 100  # Very short - we only need ticker + intent
            },
            timeout=10.0
        )
        
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Clean markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        parsed = json.loads(content)
        
        return ParsedQuery(
            ticker=parsed.get("ticker"),
            intent=parsed.get("intent", "info"),
            raw_query=query
        )
    
    except Exception as e:
        print(f"[query_parser] Error: {e}, using fallback")
        # Fallback: try simple regex
        import re
        ticker_match = re.search(r'\b([A-Z]{2,5}(?:-USD)?)\b', query.upper())
        ticker = ticker_match.group(1) if ticker_match else None
        
        # Simple intent detection
        q = query.lower()
        if 'option' in q:
            intent = 'options'
        elif 'news' in q:
            intent = 'news'
        elif 'price' in q or 'quote' in q:
            intent = 'price'
        elif 'financial' in q or 'statement' in q or 'fundamental' in q:
            intent = 'fundamentals'
        elif 'history' in q or 'historical' in q:
            intent = 'historical'
        else:
            intent = 'info'
        
        return ParsedQuery(ticker=ticker, intent=intent, raw_query=query)