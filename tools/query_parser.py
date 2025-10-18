# tools/query_parser.py
import json
import httpx
from typing import Dict, List, Optional
from dataclasses import dataclass
from utils.config import load_settings

@dataclass
class ParsedQuery:
    """Structured representation of a financial query."""
    primary_ticker: Optional[str]
    secondary_tickers: List[str]
    intent: str  # "price", "news", "analysis", "comparison", "options", etc.
    data_type: str  # "realtime", "historical", "fundamental"
    time_range: Optional[str]  # "30 days", "1 year", etc.
    specific_date: Optional[str]  # YYYY-MM-DD
    options_type: Optional[str]  # "calls", "puts", "chain"
    expiration_days: Optional[int]
    company_names: List[str]
    raw_intent: str  # original user query


async def parse_query_with_llm(query: str) -> ParsedQuery:
    """
    Use OpenAI GPT to parse the user query structurally.
    This is much smarter than regex and catches nuances.
    """
    cfg = load_settings()
    
    system_prompt = """You are a financial query parser. Extract structured information from user queries.
    
Return ONLY valid JSON (no markdown, no code blocks) with exactly these fields:
{
  "primary_ticker": "MSFT or null",
  "secondary_tickers": ["GOOGL"],
  "intent": "one of: price_quote, news, analysis, comparison, options, fundamentals, dividend, insider, recommendation",
  "data_type": "one of: realtime, historical, fundamental",
  "time_range": "30 days or null",
  "specific_date": "YYYY-MM-DD or null",
  "options_type": "calls or puts or chain or null",
  "expiration_days": 30 or null,
  "company_names": ["Apple Inc"]
}"""

    user_message = f"""Parse this financial query and extract structured information:

Query: {query}

Rules:
- primary_ticker: The main stock/crypto symbol (uppercase)
- secondary_tickers: Any other symbols mentioned
- intent: What does the user want to know?
- data_type: Is it real-time, historical, or fundamental?
- time_range: Any mention of time period? (e.g., "last 6 months", "30 days")
- specific_date: Any specific date mentioned?
- options_type: If asking about options, what type?
- expiration_days: Days until expiration (e.g., "30 days" = 30)
- company_names: Full company names mentioned

Be strict: only return valid JSON."""

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
            
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
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
    except Exception as e:
        print(f"[QueryParser] Error: {e}")
        # Fallback: return empty parsed query
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