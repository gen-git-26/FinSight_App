# agent/nodes/trader.py
"""
Trader Agent - Makes final trading decision.

Synthesizes all inputs:
- Analyst reports
- Research debate (Bull vs Bear)
- Risk assessment

Produces actionable trading recommendation.
"""
from __future__ import annotations

import json
import httpx
from typing import Dict, Any, List
from dataclasses import dataclass, field

from agent.state import AgentState
from utils.config import load_settings


@dataclass
class TradingDecision:
    """Final trading decision."""
    action: str = "hold"  # buy, sell, hold
    conviction: float = 0.5  # 0 to 1
    position_size: str = ""  # e.g., "5% of portfolio"
    entry_price: str = ""  # Target entry
    stop_loss: str = ""
    take_profit: str = ""
    time_horizon: str = ""  # short-term, medium-term, long-term
    rationale: str = ""
    key_points: List[str] = field(default_factory=list)


TRADER_PROMPT = """You are a Professional Trader making a final trading decision.

**Your Role:**
Synthesize all analysis and make a clear, actionable decision.

**You have access to:**
1. Analyst reports (fundamental, sentiment, technical)
2. Research debate (bull vs bear arguments)
3. Risk assessment

**Decision Framework:**
- BUY: Strong conviction in upside, acceptable risk
- SELL: Strong conviction in downside or need to exit
- HOLD: Insufficient conviction or unfavorable risk/reward

**Output JSON:**
{
  "action": "buy|sell|hold",
  "conviction": 0.75,
  "position_size": "3-5% of portfolio",
  "entry_price": "Current market or limit at $X",
  "stop_loss": "X% below entry or at $X",
  "take_profit": "X% above entry or at $X",
  "time_horizon": "short-term|medium-term|long-term",
  "rationale": "2-3 sentence explanation of the decision",
  "key_points": ["Point 1", "Point 2", "Point 3"]
}

Be decisive but prudent. Every decision should be justified.
"""


def trader_node(state: AgentState) -> Dict[str, Any]:
    """
    Trader node - makes the final trading decision.

    This is the culmination of the TradingAgents workflow.
    """
    analyst_reports = state.get("analyst_reports", [])
    research_report = state.get("research_report")
    risk_assessment = state.get("risk_assessment")
    parsed_query = state.get("parsed_query")
    trading_recommendation = state.get("trading_recommendation", "hold")

    ticker = parsed_query.ticker if parsed_query else "Unknown"

    print(f"\n[Trader] Making final decision for {ticker}")

    # Compile all information for the trader
    trading_context = {
        "ticker": ticker,
        "initial_recommendation": trading_recommendation,

        "analyst_reports": {
            report.analyst_type: {
                "findings": report.findings[:3],
                "recommendation": report.recommendation,
                "confidence": report.confidence
            }
            for report in analyst_reports
        },

        "research_debate": {
            "bull_arguments": research_report.bull_arguments[:3] if research_report else [],
            "bear_arguments": research_report.bear_arguments[:3] if research_report else [],
            "consensus": research_report.consensus if research_report else "",
            "conviction_score": research_report.conviction_score if research_report else 0,
        },

        "risk_assessment": {
            "risk_level": risk_assessment.risk_level if risk_assessment else "unknown",
            "risk_score": risk_assessment.risk_score if risk_assessment else 0.5,
            "concerns": risk_assessment.concerns[:3] if risk_assessment else [],
            "position_recommendation": risk_assessment.position_recommendation if risk_assessment else "",
            "stop_loss_suggestion": risk_assessment.stop_loss_suggestion if risk_assessment else "",
            "approved": risk_assessment.approved if risk_assessment else True,
        }
    }

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
                    {"role": "system", "content": TRADER_PROMPT},
                    {"role": "user", "content": f"Make trading decision:\n{json.dumps(trading_context, indent=2, default=str)}"}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            },
            timeout=25.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        decision = TradingDecision(
            action=result.get("action", "hold"),
            conviction=result.get("conviction", 0.5),
            position_size=result.get("position_size", ""),
            entry_price=result.get("entry_price", ""),
            stop_loss=result.get("stop_loss", ""),
            take_profit=result.get("take_profit", ""),
            time_horizon=result.get("time_horizon", ""),
            rationale=result.get("rationale", ""),
            key_points=result.get("key_points", [])
        )

    except Exception as e:
        print(f"[Trader] Error: {e}")
        decision = TradingDecision(
            action="hold",
            conviction=0.0,
            rationale=f"Decision failed: {str(e)}",
            key_points=["Unable to make decision due to error"]
        )

    print(f"[Trader] Decision: {decision.action.upper()} (conviction: {decision.conviction:.0%})")

    return {
        "trading_decision": decision
    }


def format_trading_response(state: AgentState) -> str:
    """Format the trading decision as a user-friendly response."""
    decision = state.get("trading_decision")
    research_report = state.get("research_report")
    risk_assessment = state.get("risk_assessment")
    parsed_query = state.get("parsed_query")

    ticker = parsed_query.ticker if parsed_query else "Unknown"

    parts = []

    # Header
    action_emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(decision.action, "⚪")
    parts.append(f"## {ticker} Trading Analysis")
    parts.append(f"\n### Decision: {action_emoji} **{decision.action.upper()}**")
    parts.append(f"*Conviction: {decision.conviction:.0%}*\n")

    # Rationale
    if decision.rationale:
        parts.append(f"**Rationale:** {decision.rationale}\n")

    # Key Points
    if decision.key_points:
        parts.append("**Key Points:**")
        for point in decision.key_points:
            parts.append(f"- {point}")
        parts.append("")

    # Trade Parameters
    if decision.action != "hold":
        parts.append("### Trade Parameters")
        if decision.position_size:
            parts.append(f"- **Position Size:** {decision.position_size}")
        if decision.entry_price:
            parts.append(f"- **Entry:** {decision.entry_price}")
        if decision.stop_loss:
            parts.append(f"- **Stop Loss:** {decision.stop_loss}")
        if decision.take_profit:
            parts.append(f"- **Take Profit:** {decision.take_profit}")
        if decision.time_horizon:
            parts.append(f"- **Time Horizon:** {decision.time_horizon}")
        parts.append("")

    # Risk Assessment
    if risk_assessment:
        risk_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "extreme": "🔴"
        }.get(risk_assessment.risk_level, "⚪")
        parts.append(f"### Risk Assessment: {risk_emoji} {risk_assessment.risk_level.upper()}")
        if risk_assessment.concerns:
            parts.append("**Concerns:**")
            for concern in risk_assessment.concerns[:3]:
                parts.append(f"- {concern}")
        parts.append("")

    # Bull vs Bear Summary
    if research_report and research_report.consensus:
        parts.append(f"### Research Consensus")
        parts.append(f"{research_report.consensus}\n")

    # Disclaimer
    parts.append("---")
    parts.append("*⚠️ This is not financial advice. Always do your own research and consider your risk tolerance.*")

    return "\n".join(parts)
