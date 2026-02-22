# agent/nodes/analysts.py
"""
Analyst Agents - Based on TradingAgents framework.

Four specialized analysts that work in parallel:
1. Fundamental Analyst - Financial metrics, earnings, ratios
2. Sentiment Analyst - Market mood, social signals, investor psychology
3. News Analyst - Breaking news, macro events, company announcements
4. Technical Analyst - Price patterns, indicators, charts
"""
from __future__ import annotations

import json
import httpx
from typing import Dict, Any, List
from dataclasses import dataclass, field

from agent.state import AgentState, FetchedData
from utils.config import load_settings
from evaluation.metrics import track_metrics


@dataclass
class AnalystReport:
    """Report from a single analyst."""
    analyst_type: str
    findings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""  # bullish, bearish, neutral
    confidence: float = 0.5
    raw_data: Any = None


# === ANALYST PROMPTS ===

FUNDAMENTAL_ANALYST_PROMPT = """You are a Fundamental Analyst specializing in financial statements and company metrics.

**Your Role:**
Analyze company financials to assess intrinsic value and financial health.

**Focus Areas:**
- Revenue growth and profitability trends
- P/E ratio, P/B ratio, EV/EBITDA
- Debt levels and cash flow
- Earnings surprises and guidance
- Competitive positioning

**Output JSON:**
{
  "findings": ["Key finding 1", "Key finding 2"],
  "metrics": {"pe_ratio": 25.5, "revenue_growth": "15%"},
  "recommendation": "bullish|bearish|neutral",
  "confidence": 0.75
}
"""

SENTIMENT_ANALYST_PROMPT = """You are a Sentiment Analyst specializing in market mood and investor psychology.

**Your Role:**
Gauge market sentiment through news, social media, and investor behavior.

**Focus Areas:**
- News sentiment (positive/negative coverage)
- Social media trends and retail interest
- Institutional investor movements
- Options flow and put/call ratios
- Fear & Greed indicators

**Output JSON:**
{
  "findings": ["Key finding 1", "Key finding 2"],
  "metrics": {"sentiment_score": 0.7, "news_sentiment": "positive"},
  "recommendation": "bullish|bearish|neutral",
  "confidence": 0.65
}
"""

TECHNICAL_ANALYST_PROMPT = """You are a Technical Analyst specializing in price action and chart patterns.

**Your Role:**
Analyze price movements, trends, and technical indicators.

**Focus Areas:**
- Support and resistance levels
- Moving averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- Volume analysis
- Chart patterns (head & shoulders, triangles, etc.)
- Trend direction and momentum

**Output JSON:**
{
  "findings": ["Key finding 1", "Key finding 2"],
  "metrics": {"rsi": 65, "trend": "uptrend", "support": 150.0},
  "recommendation": "bullish|bearish|neutral",
  "confidence": 0.70
}
"""

NEWS_ANALYST_PROMPT = """You are a News Analyst specializing in financial news and macro events.

**Your Role:**
Analyze breaking news, company announcements, and macroeconomic events that impact the asset.

**Focus Areas:**
- Breaking news and headlines
- Earnings announcements and guidance
- M&A activity and partnerships
- Regulatory news and legal issues
- Macroeconomic events (Fed decisions, inflation data)
- Industry-specific developments
- Analyst upgrades/downgrades

**Output JSON:**
{
  "findings": ["Key news finding 1", "Key news finding 2"],
  "metrics": {"news_count": 5, "sentiment": "positive", "major_events": ["event1"]},
  "recommendation": "bullish|bearish|neutral",
  "confidence": 0.65
}
"""


def _run_analyst(
    analyst_type: str,
    prompt: str,
    data: Dict[str, Any],
    query: str
) -> AnalystReport:
    """Run a single analyst agent."""
    cfg = load_settings()

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
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Query: {query}\n\nData:\n{json.dumps(data, indent=2, default=str)[:3000]}"}
                ],
                "temperature": 0.3,
                "max_tokens": 400
            },
            timeout=25.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)

        return AnalystReport(
            analyst_type=analyst_type,
            findings=parsed.get("findings", []),
            metrics=parsed.get("metrics", {}),
            recommendation=parsed.get("recommendation", "neutral"),
            confidence=parsed.get("confidence", 0.5),
            raw_data=data
        )

    except Exception as e:
        print(f"[{analyst_type}] Error: {e}")
        return AnalystReport(
            analyst_type=analyst_type,
            findings=[f"Analysis failed: {str(e)}"],
            recommendation="neutral",
            confidence=0.0
        )


def fundamental_analyst(data: Dict[str, Any], query: str) -> AnalystReport:
    """Fundamental analysis agent."""
    print(f"[Fundamental Analyst] Analyzing...")
    return _run_analyst("fundamental", FUNDAMENTAL_ANALYST_PROMPT, data, query)


def sentiment_analyst(data: Dict[str, Any], query: str) -> AnalystReport:
    """Sentiment analysis agent."""
    print(f"[Sentiment Analyst] Analyzing...")
    return _run_analyst("sentiment", SENTIMENT_ANALYST_PROMPT, data, query)


def technical_analyst(data: Dict[str, Any], query: str) -> AnalystReport:
    """Technical analysis agent."""
    print(f"[Technical Analyst] Analyzing...")
    return _run_analyst("technical", TECHNICAL_ANALYST_PROMPT, data, query)


def news_analyst(data: Dict[str, Any], query: str) -> AnalystReport:
    """News analysis agent."""
    print(f"[News Analyst] Analyzing...")
    return _run_analyst("news", NEWS_ANALYST_PROMPT, data, query)


@track_metrics("analysts_team")
def analysts_node(state: AgentState) -> Dict[str, Any]:
    """
    Analysts Team node - runs all analysts in parallel.

    This replaces the simple analyst node with the TradingAgents approach.
    """
    parsed_query = state.get("parsed_query")
    fetched_data = state.get("fetched_data", [])
    query = parsed_query.raw_query if parsed_query else state.get("query", "")

    print(f"\n[Analysts Team] Starting parallel analysis")

    # Prepare data for analysts
    combined_data = {}
    for d in fetched_data:
        if not d.error and d.parsed_data:
            combined_data[d.source] = d.parsed_data

    # Run all analysts (in production, these run in parallel)
    reports = []

    # Fundamental analysis
    fundamental_report = fundamental_analyst(combined_data, query)
    reports.append(fundamental_report)
    print(f"[Fundamental] Recommendation: {fundamental_report.recommendation} ({fundamental_report.confidence:.0%})")

    # Sentiment analysis
    sentiment_report = sentiment_analyst(combined_data, query)
    reports.append(sentiment_report)
    print(f"[Sentiment] Recommendation: {sentiment_report.recommendation} ({sentiment_report.confidence:.0%})")

    # News analysis
    news_report = news_analyst(combined_data, query)
    reports.append(news_report)
    print(f"[News] Recommendation: {news_report.recommendation} ({news_report.confidence:.0%})")

    # Technical analysis
    technical_report = technical_analyst(combined_data, query)
    reports.append(technical_report)
    print(f"[Technical] Recommendation: {technical_report.recommendation} ({technical_report.confidence:.0%})")

    print(f"[Analysts Team] Completed {len(reports)} analyses (4 analysts)")

    return {
        "analyst_reports": reports
    }


# === Single Analyst Nodes (for granular routing) ===

def _prepare_analyst_data(state: AgentState) -> tuple:
    """Extract query and combine fetched data from state."""
    parsed_query = state.get("parsed_query")
    fetched_data = state.get("fetched_data", [])
    query = parsed_query.raw_query if parsed_query else state.get("query", "")

    combined_data = {}
    for d in fetched_data:
        if not d.error and d.parsed_data:
            combined_data[d.source] = d.parsed_data

    return query, combined_data


@track_metrics("single_fundamental")
def single_fundamental_node(state: AgentState) -> Dict[str, Any]:
    """Run only the Fundamental Analyst."""
    query, combined_data = _prepare_analyst_data(state)
    report = fundamental_analyst(combined_data, query)
    print(f"[Single Analyst] Fundamental: {report.recommendation} ({report.confidence:.0%})")
    return {"analyst_reports": [report]}


@track_metrics("single_technical")
def single_technical_node(state: AgentState) -> Dict[str, Any]:
    """Run only the Technical Analyst."""
    query, combined_data = _prepare_analyst_data(state)
    report = technical_analyst(combined_data, query)
    print(f"[Single Analyst] Technical: {report.recommendation} ({report.confidence:.0%})")
    return {"analyst_reports": [report]}


@track_metrics("single_sentiment")
def single_sentiment_node(state: AgentState) -> Dict[str, Any]:
    """Run Sentiment + News Analysts together."""
    query, combined_data = _prepare_analyst_data(state)
    sentiment_report = sentiment_analyst(combined_data, query)
    news_report = news_analyst(combined_data, query)
    print(f"[Single Analyst] Sentiment: {sentiment_report.recommendation} ({sentiment_report.confidence:.0%}), News: {news_report.recommendation} ({news_report.confidence:.0%})")
    return {"analyst_reports": [sentiment_report, news_report]}


@track_metrics("single_news")
def single_news_node(state: AgentState) -> Dict[str, Any]:
    """Run only the News Analyst."""
    query, combined_data = _prepare_analyst_data(state)
    report = news_analyst(combined_data, query)
    print(f"[Single Analyst] News: {report.recommendation} ({report.confidence:.0%})")
    return {"analyst_reports": [report]}
