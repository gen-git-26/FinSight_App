# agent/nodes/fund_manager.py
"""
Fund Manager - Final approval authority for trades.

The Fund Manager is the last checkpoint before execution.
Reviews:
- Trading decision from Trader
- Risk assessment from Risk Team
- Overall market conditions

Can APPROVE, REJECT, or MODIFY the trade.
"""
from __future__ import annotations

import json
import httpx
from typing import Dict, Any, List
from dataclasses import dataclass, field

from agent.state import AgentState
from utils.config import load_settings


@dataclass
class FundManagerDecision:
    """Final decision from the Fund Manager."""
    status: str = "approved"  # approved, rejected, modified
    final_action: str = "hold"  # buy, sell, hold
    final_position_size: str = ""
    final_stop_loss: str = ""
    final_take_profit: str = ""
    modifications: List[str] = field(default_factory=list)
    rejection_reason: str = ""
    execution_notes: str = ""
    confidence: float = 0.5


FUND_MANAGER_PROMPT = """You are the FUND MANAGER with final approval authority.

**Your Role:**
Make the final decision on whether to execute this trade.
You can APPROVE, REJECT, or MODIFY the trade.

**You are reviewing:**
1. Trader's decision and rationale
2. Risk team's assessment
3. Overall conviction and risk levels

**Decision Framework:**

APPROVE if:
- Risk assessment is approved
- Conviction is sufficient for the risk level
- Trade parameters are sensible

MODIFY if:
- Good trade idea but needs adjustment
- Position size should be different
- Stop-loss/take-profit needs tweaking

REJECT if:
- Risk level is too high
- Conviction is too low
- Market conditions are unfavorable
- Risk team did not approve

**Input Summary:**
Trading Decision: {trading_decision}
Risk Assessment: {risk_assessment}
Conviction Score: {conviction}
Risk Level: {risk_level}
Risk Approved: {risk_approved}

**Output JSON:**
{{
  "status": "approved|rejected|modified",
  "final_action": "buy|sell|hold",
  "final_position_size": "5% of portfolio",
  "final_stop_loss": "8% below entry",
  "final_take_profit": "15% above entry",
  "modifications": ["Modification 1 if any"],
  "rejection_reason": "If rejected, explain why",
  "execution_notes": "Any special execution instructions",
  "confidence": 0.75,
  "reasoning": "Brief explanation of your decision"
}}
"""


def fund_manager_node(state: AgentState) -> Dict[str, Any]:
    """
    Fund Manager node - final approval authority.

    Reviews all prior decisions and makes the final call.
    """
    trading_decision = state.get("trading_decision")
    risk_assessment = state.get("risk_assessment")
    research_report = state.get("research_report")
    parsed_query = state.get("parsed_query")

    ticker = parsed_query.ticker if parsed_query else "Unknown"

    print(f"\n{'='*50}")
    print(f"[Fund Manager] Final review for {ticker}")
    print(f"{'='*50}")

    # Prepare summary for Fund Manager
    conviction = research_report.conviction_score if research_report else 0
    risk_level = risk_assessment.risk_level if risk_assessment else "unknown"
    risk_approved = risk_assessment.approved if risk_assessment else False

    trading_summary = {
        "action": trading_decision.action if trading_decision else "unknown",
        "conviction": trading_decision.conviction if trading_decision else 0,
        "rationale": trading_decision.rationale if trading_decision else "",
        "position_size": trading_decision.position_size if trading_decision else "",
        "stop_loss": trading_decision.stop_loss if trading_decision else "",
        "take_profit": trading_decision.take_profit if trading_decision else ""
    }

    risk_summary = {
        "risk_level": risk_level,
        "risk_score": risk_assessment.risk_score if risk_assessment else 0.5,
        "approved": risk_approved,
        "concerns": risk_assessment.concerns[:3] if risk_assessment else [],
        "position_recommendation": risk_assessment.position_recommendation if risk_assessment else "",
        "stop_loss_suggestion": risk_assessment.stop_loss_suggestion if risk_assessment else ""
    }

    prompt = FUND_MANAGER_PROMPT.format(
        trading_decision=json.dumps(trading_summary, indent=2),
        risk_assessment=json.dumps(risk_summary, indent=2),
        conviction=conviction,
        risk_level=risk_level,
        risk_approved=risk_approved
    )

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
                    {"role": "user", "content": f"Make your final decision for {ticker}."}
                ],
                "temperature": 0.2,
                "max_tokens": 400
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

        decision = FundManagerDecision(
            status=result.get("status", "approved"),
            final_action=result.get("final_action", "hold"),
            final_position_size=result.get("final_position_size", ""),
            final_stop_loss=result.get("final_stop_loss", ""),
            final_take_profit=result.get("final_take_profit", ""),
            modifications=result.get("modifications", []),
            rejection_reason=result.get("rejection_reason", ""),
            execution_notes=result.get("execution_notes", ""),
            confidence=result.get("confidence", 0.5)
        )

    except Exception as e:
        print(f"[Fund Manager] Error: {e}")
        decision = FundManagerDecision(
            status="rejected",
            final_action="hold",
            rejection_reason=f"Decision failed: {str(e)}",
            confidence=0.0
        )

    status_emoji = {
        "approved": "✅",
        "modified": "🔄",
        "rejected": "❌"
    }.get(decision.status, "❓")

    print(f"\n[Fund Manager] Decision: {status_emoji} {decision.status.upper()}")
    print(f"[Fund Manager] Final Action: {decision.final_action}")
    print(f"[Fund Manager] Confidence: {decision.confidence:.0%}")
    print(f"{'='*50}\n")

    return {
        "fund_manager_decision": decision
    }


def format_final_trading_response(state: AgentState) -> str:
    """Format the complete trading response including Fund Manager decision."""
    fm_decision = state.get("fund_manager_decision")
    trading_decision = state.get("trading_decision")
    risk_assessment = state.get("risk_assessment")
    research_report = state.get("research_report")
    parsed_query = state.get("parsed_query")

    ticker = parsed_query.ticker if parsed_query else "Unknown"

    parts = []

    # Header with Fund Manager decision
    status_emoji = {
        "approved": "✅",
        "modified": "🔄",
        "rejected": "❌"
    }.get(fm_decision.status if fm_decision else "unknown", "❓")

    action_emoji = {
        "buy": "🟢",
        "sell": "🔴",
        "hold": "🟡"
    }.get(fm_decision.final_action if fm_decision else "hold", "⚪")

    parts.append(f"## {ticker} Trading Analysis")
    parts.append(f"\n### Fund Manager Decision: {status_emoji} **{fm_decision.status.upper() if fm_decision else 'PENDING'}**")
    parts.append(f"### Final Action: {action_emoji} **{fm_decision.final_action.upper() if fm_decision else 'HOLD'}**")
    parts.append(f"*Confidence: {fm_decision.confidence:.0%}*\n" if fm_decision else "")

    # Rejection reason if rejected
    if fm_decision and fm_decision.status == "rejected":
        parts.append(f"**Rejection Reason:** {fm_decision.rejection_reason}\n")

    # Modifications if modified
    if fm_decision and fm_decision.status == "modified" and fm_decision.modifications:
        parts.append("**Modifications:**")
        for mod in fm_decision.modifications:
            parts.append(f"- {mod}")
        parts.append("")

    # Trade Parameters (if approved or modified)
    if fm_decision and fm_decision.status in ["approved", "modified"]:
        parts.append("### Trade Parameters")
        if fm_decision.final_position_size:
            parts.append(f"- **Position Size:** {fm_decision.final_position_size}")
        if fm_decision.final_stop_loss:
            parts.append(f"- **Stop Loss:** {fm_decision.final_stop_loss}")
        if fm_decision.final_take_profit:
            parts.append(f"- **Take Profit:** {fm_decision.final_take_profit}")
        if fm_decision.execution_notes:
            parts.append(f"- **Notes:** {fm_decision.execution_notes}")
        parts.append("")

    # Risk Assessment Summary
    if risk_assessment:
        risk_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "extreme": "🔴"
        }.get(risk_assessment.risk_level, "⚪")
        parts.append(f"### Risk Assessment: {risk_emoji} {risk_assessment.risk_level.upper()}")
        if risk_assessment.concerns:
            parts.append("**Key Concerns:**")
            for concern in risk_assessment.concerns[:3]:
                parts.append(f"- {concern}")
        parts.append("")

    # Research Consensus
    if research_report and research_report.consensus:
        parts.append("### Research Consensus")
        parts.append(f"{research_report.consensus}")
        parts.append(f"\n*Conviction: {research_report.conviction_score:+.2f}*\n")

    # Trader's Original Rationale
    if trading_decision and trading_decision.rationale:
        parts.append("### Trader's Rationale")
        parts.append(f"{trading_decision.rationale}\n")

    # Key Points
    if trading_decision and trading_decision.key_points:
        parts.append("### Key Points")
        for point in trading_decision.key_points[:4]:
            parts.append(f"- {point}")
        parts.append("")

    # Disclaimer
    parts.append("---")
    parts.append("*⚠️ This is not financial advice. Always do your own research and consider your risk tolerance.*")

    return "\n".join(parts)
