# agent/nodes/risk_manager.py
"""
Risk Management Team - 3-persona debate (3 rounds).

Three risk personas with different perspectives:
1. RISKY - Aggressive, opportunity-focused
2. NEUTRAL - Balanced, data-driven
3. SAFE - Conservative, risk-averse

The team debates for 3 rounds to reach a risk consensus.
"""
from __future__ import annotations

import json
import httpx
from typing import Dict, Any, List
from dataclasses import dataclass, field

from agent.state import AgentState
from utils.config import load_settings


@dataclass
class RiskDebateRound:
    """Single round of risk debate."""
    round_number: int
    risky_view: str
    neutral_view: str
    safe_view: str


@dataclass
class RiskAssessment:
    """Risk assessment from the risk team."""
    risk_level: str = "medium"  # low, medium, high, extreme
    risk_score: float = 0.5  # 0 to 1
    debate_rounds: List[RiskDebateRound] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    position_recommendation: str = ""
    stop_loss_suggestion: str = ""
    take_profit_suggestion: str = ""
    max_position_size: str = ""
    approved: bool = True
    approval_conditions: List[str] = field(default_factory=list)


RISKY_PERSONA_PROMPT = """You are the RISKY risk manager. You are AGGRESSIVE and OPPORTUNITY-FOCUSED.

**Your Mindset:**
- Higher risk = higher reward
- Fortune favors the bold
- Missing opportunities is also a risk
- Position sizing can be larger if conviction is high

**Current Round:** {round_num}/3
**Previous Discussion:**
NEUTRAL: {neutral_previous}
SAFE: {safe_previous}

**Output JSON:**
{{
  "view": "Your risk assessment argument (2-3 sentences)",
  "position_size": "10-15% of portfolio",
  "risk_tolerance": "high",
  "key_point": "One key point supporting a more aggressive approach"
}}
"""

NEUTRAL_PERSONA_PROMPT = """You are the NEUTRAL risk manager. You are BALANCED and DATA-DRIVEN.

**Your Mindset:**
- Follow the data, not emotions
- Risk should be proportional to expected return
- Consider both upside and downside
- Standard position sizing with appropriate stops

**Current Round:** {round_num}/3
**Previous Discussion:**
RISKY: {risky_previous}
SAFE: {safe_previous}

**Output JSON:**
{{
  "view": "Your balanced risk assessment (2-3 sentences)",
  "position_size": "5-8% of portfolio",
  "risk_tolerance": "medium",
  "key_point": "One key point for a balanced approach"
}}
"""

SAFE_PERSONA_PROMPT = """You are the SAFE risk manager. You are CONSERVATIVE and RISK-AVERSE.

**Your Mindset:**
- Capital preservation is paramount
- The market can stay irrational longer than you can stay solvent
- Better to miss an opportunity than lose capital
- Small positions, tight stops

**Current Round:** {round_num}/3
**Previous Discussion:**
RISKY: {risky_previous}
NEUTRAL: {neutral_previous}

**Output JSON:**
{{
  "view": "Your conservative risk assessment (2-3 sentences)",
  "position_size": "2-3% of portfolio",
  "risk_tolerance": "low",
  "key_point": "One key concern that warrants caution"
}}
"""

RISK_MODERATOR_PROMPT = """You are the Chief Risk Officer synthesizing a 3-round risk debate.

**Risk Team Debate Transcript:**
{debate_transcript}

**Trading Context:**
- Trading Decision: {trading_decision}
- Research Conviction: {conviction_score}

**Your Task:**
Synthesize the debate and produce a final risk assessment.

**Output JSON:**
{{
  "risk_level": "low|medium|high|extreme",
  "risk_score": 0.45,
  "approved": true,
  "approval_conditions": ["Condition 1 for approval", "Condition 2"],
  "concerns": ["Main concern 1", "Main concern 2"],
  "mitigations": ["Mitigation strategy 1", "Mitigation 2"],
  "position_recommendation": "5% of portfolio maximum",
  "stop_loss": "8% below entry",
  "take_profit": "15-20% above entry",
  "reasoning": "Brief explanation of the risk decision"
}}

Consider all three perspectives but weight based on the quality of arguments.
"""


def _call_risk_persona(
    prompt: str,
    context: Dict[str, Any],
    round_num: int,
    risky_prev: str = "N/A",
    neutral_prev: str = "N/A",
    safe_prev: str = "N/A"
) -> Dict[str, Any]:
    """Call a risk persona with formatted prompt."""
    cfg = load_settings()

    formatted_prompt = prompt.format(
        round_num=round_num,
        risky_previous=risky_prev,
        neutral_previous=neutral_prev,
        safe_previous=safe_prev
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
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": f"Context:\n{json.dumps(context, indent=2, default=str)[:3000]}"}
                ],
                "temperature": 0.4,
                "max_tokens": 300
            },
            timeout=25.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        print(f"[Risk Persona] Error: {e}")
        return {"view": f"Error: {str(e)}", "risk_tolerance": "medium"}


def _run_risk_debate(context: Dict[str, Any], num_rounds: int = 3) -> List[RiskDebateRound]:
    """Run the 3-persona risk debate for specified rounds."""
    rounds = []
    risky_prev = ""
    neutral_prev = ""
    safe_prev = ""

    for round_num in range(1, num_rounds + 1):
        print(f"\n[Risk Debate] Round {round_num}/{num_rounds}")

        # RISKY speaks first
        print(f"[RISKY] Presenting view...")
        risky_result = _call_risk_persona(
            RISKY_PERSONA_PROMPT, context, round_num,
            risky_prev, neutral_prev, safe_prev
        )
        risky_view = risky_result.get("view", "")

        # NEUTRAL responds
        print(f"[NEUTRAL] Presenting view...")
        neutral_result = _call_risk_persona(
            NEUTRAL_PERSONA_PROMPT, context, round_num,
            risky_view, neutral_prev, safe_prev
        )
        neutral_view = neutral_result.get("view", "")

        # SAFE responds
        print(f"[SAFE] Presenting view...")
        safe_result = _call_risk_persona(
            SAFE_PERSONA_PROMPT, context, round_num,
            risky_view, neutral_view, safe_prev
        )
        safe_view = safe_result.get("view", "")

        # Store round
        rounds.append(RiskDebateRound(
            round_number=round_num,
            risky_view=risky_view,
            neutral_view=neutral_view,
            safe_view=safe_view
        ))

        # Update for next round
        risky_prev = risky_view
        neutral_prev = neutral_view
        safe_prev = safe_view

        print(f"[RISKY R{round_num}] {risky_view[:60]}...")
        print(f"[NEUTRAL R{round_num}] {neutral_view[:60]}...")
        print(f"[SAFE R{round_num}] {safe_view[:60]}...")

    return rounds


def _moderate_risk_debate(
    rounds: List[RiskDebateRound],
    trading_decision: str,
    conviction_score: float
) -> Dict[str, Any]:
    """Chief Risk Officer synthesizes the debate."""
    cfg = load_settings()

    # Build transcript
    transcript_parts = []
    for r in rounds:
        transcript_parts.append(f"=== Round {r.round_number} ===")
        transcript_parts.append(f"RISKY: {r.risky_view}")
        transcript_parts.append(f"NEUTRAL: {r.neutral_view}")
        transcript_parts.append(f"SAFE: {r.safe_view}")
        transcript_parts.append("")

    debate_transcript = "\n".join(transcript_parts)

    prompt = RISK_MODERATOR_PROMPT.format(
        debate_transcript=debate_transcript,
        trading_decision=trading_decision,
        conviction_score=conviction_score
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
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Synthesize the risk debate and provide final assessment."}
                ],
                "temperature": 0.2,
                "max_tokens": 450
            },
            timeout=25.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        print(f"[CRO] Error: {e}")
        return {
            "risk_level": "high",
            "risk_score": 0.7,
            "approved": False,
            "concerns": [f"Risk assessment failed: {str(e)}"]
        }


def risk_manager_node(state: AgentState) -> Dict[str, Any]:
    """
    Risk Management Team node - 3-persona, 3-round debate.

    Three risk personas (Risky, Neutral, Safe) debate the risk
    of the proposed trade.
    """
    trading_decision = state.get("trading_decision")
    research_report = state.get("research_report")
    trading_recommendation = state.get("trading_recommendation", "hold")
    parsed_query = state.get("parsed_query")

    ticker = parsed_query.ticker if parsed_query else "Unknown"
    conviction = research_report.conviction_score if research_report else 0

    print(f"\n{'='*50}")
    print(f"[Risk Team] Starting 3-persona debate for {ticker}")
    print(f"[Risk Team] Trade: {trading_recommendation}, Conviction: {conviction:+.2f}")
    print(f"{'='*50}")

    # Prepare context for risk team
    context = {
        "ticker": ticker,
        "trading_recommendation": trading_recommendation,
        "conviction_score": conviction,
        "research_consensus": research_report.consensus if research_report else "",
        "key_risks": research_report.key_risks if research_report else [],
        "key_opportunities": research_report.key_opportunities if research_report else [],
        "trading_decision": {
            "action": trading_decision.action if trading_decision else "unknown",
            "conviction": trading_decision.conviction if trading_decision else 0,
            "rationale": trading_decision.rationale if trading_decision else ""
        } if trading_decision else {}
    }

    # Run 3-round debate
    debate_rounds = _run_risk_debate(context, num_rounds=3)

    # CRO synthesizes
    print(f"\n[CRO] Synthesizing risk assessment...")
    moderation = _moderate_risk_debate(
        debate_rounds,
        trading_recommendation,
        conviction
    )

    # Build risk assessment
    risk_assessment = RiskAssessment(
        risk_level=moderation.get("risk_level", "medium"),
        risk_score=moderation.get("risk_score", 0.5),
        debate_rounds=debate_rounds,
        concerns=moderation.get("concerns", []),
        mitigations=moderation.get("mitigations", []),
        position_recommendation=moderation.get("position_recommendation", ""),
        stop_loss_suggestion=moderation.get("stop_loss", ""),
        take_profit_suggestion=moderation.get("take_profit", ""),
        max_position_size=moderation.get("position_recommendation", ""),
        approved=moderation.get("approved", True),
        approval_conditions=moderation.get("approval_conditions", [])
    )

    print(f"\n[Risk Team] Assessment completed")
    print(f"[Risk Team] Risk Level: {risk_assessment.risk_level}")
    print(f"[Risk Team] Approved: {risk_assessment.approved}")
    print(f"{'='*50}\n")

    return {
        "risk_assessment": risk_assessment
    }
