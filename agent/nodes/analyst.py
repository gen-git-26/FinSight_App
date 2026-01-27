# agent/nodes/analyst.py
"""
Analyst Agent - Analyzes fetched financial data and generates insights.
"""
from __future__ import annotations

import json
import httpx
from typing import Dict, Any, List

from agent.state import AgentState, AnalysisResult, FetchedData
from utils.config import load_settings


ANALYST_PROMPT = """You are a financial analyst. Analyze the provided data and generate insights.

**Your task:**
1. Identify key metrics and their significance
2. Highlight important trends or anomalies
3. Provide context for the numbers
4. Note any concerns or positive signals

**Output format (JSON):**
{
  "insights": [
    "Key insight 1",
    "Key insight 2"
  ],
  "metrics": {
    "metric_name": "value with context"
  },
  "summary": "Brief 1-2 sentence summary"
}

**Guidelines:**
- Be concise and factual
- Focus on actionable insights
- Avoid speculation
- Include relevant comparisons when data allows
- For crypto: mention volatility, market trends
- For stocks: mention valuation, growth metrics
"""


def _analyze_with_llm(data: List[FetchedData], query: str) -> AnalysisResult:
    """Use LLM to analyze the fetched data."""
    cfg = load_settings()

    # Prepare data summary for LLM
    data_summary = []
    for d in data:
        if d.error:
            continue
        summary = {
            "source": d.source,
            "tool": d.tool_used,
            "data": d.parsed_data if d.parsed_data else str(d.raw_data)[:2000]
        }
        data_summary.append(summary)

    if not data_summary:
        return AnalysisResult(
            insights=["No valid data to analyze"],
            summary="Analysis failed - no data available"
        )

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
                    {"role": "system", "content": ANALYST_PROMPT},
                    {"role": "user", "content": f"Query: {query}\n\nData:\n{json.dumps(data_summary, indent=2, default=str)[:4000]}"}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            },
            timeout=20.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)

        return AnalysisResult(
            insights=parsed.get("insights", []),
            metrics=parsed.get("metrics", {}),
            summary=parsed.get("summary", "")
        )

    except Exception as e:
        print(f"[Analyst] LLM analysis failed: {e}")
        return _basic_analysis(data)


def _basic_analysis(data: List[FetchedData]) -> AnalysisResult:
    """Basic analysis without LLM."""
    insights = []
    metrics = {}

    for d in data:
        if d.error:
            continue

        parsed = d.parsed_data or {}

        # Extract common metrics
        if 'price' in parsed:
            metrics['current_price'] = parsed['price']
            insights.append(f"Current price: ${parsed['price']}")

        if 'c' in parsed:  # Finnhub quote format
            metrics['current_price'] = parsed['c']
            metrics['change'] = parsed.get('d', 0)
            metrics['change_percent'] = parsed.get('dp', 0)
            insights.append(f"Price: ${parsed['c']}, Change: {parsed.get('dp', 0):.2f}%")

        if 'change_24h' in parsed:
            change = parsed['change_24h']
            direction = "up" if change > 0 else "down"
            insights.append(f"24h change: {direction} {abs(change):.2f}%")
            metrics['change_24h'] = change

        if 'market_cap' in parsed:
            mc = parsed['market_cap']
            if mc and mc > 1_000_000_000:
                metrics['market_cap'] = f"${mc/1_000_000_000:.2f}B"
            elif mc:
                metrics['market_cap'] = f"${mc/1_000_000:.2f}M"

        if 'volume_24h' in parsed:
            vol = parsed['volume_24h']
            if vol and vol > 1_000_000_000:
                metrics['volume_24h'] = f"${vol/1_000_000_000:.2f}B"

    summary = f"Analyzed {len([d for d in data if not d.error])} data sources"

    return AnalysisResult(
        insights=insights if insights else ["Data retrieved successfully"],
        metrics=metrics,
        summary=summary
    )


def analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyst node - analyzes the fetched data.

    Takes fetched_data from state and produces analysis.
    """
    fetched_data = state.get("fetched_data", [])
    parsed_query = state.get("parsed_query")
    query = parsed_query.raw_query if parsed_query else state.get("query", "")

    print(f"\n[Analyst] Analyzing {len(fetched_data)} data sources")

    if not fetched_data:
        return {
            "analysis": AnalysisResult(
                insights=["No data to analyze"],
                summary="No data was fetched"
            )
        }

    # Check for errors
    errors = [d for d in fetched_data if d.error]
    valid_data = [d for d in fetched_data if not d.error]

    if not valid_data:
        return {
            "analysis": AnalysisResult(
                insights=[f"Data fetch failed: {errors[0].error}" if errors else "Unknown error"],
                summary="Analysis failed due to data fetch errors"
            ),
            "error": errors[0].error if errors else "No valid data"
        }

    # Analyze with LLM
    analysis = _analyze_with_llm(valid_data, query)

    print(f"[Analyst] Generated {len(analysis.insights)} insights")

    return {"analysis": analysis}
