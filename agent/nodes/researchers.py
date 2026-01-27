# agent/nodes/researchers.py
"""
Research Team - Bull vs Bear Debate (3 rounds).

Two researchers with opposing viewpoints debate the market:
1. Bull Researcher - Argues for upside potential
2. Bear Researcher - Argues for downside risks

The debate runs for 3 rounds, with each researcher responding to the other's arguments.
"""
from __future__ import annotations

import json
import httpx
from typing import Dict, Any, List
from dataclasses import dataclass, field

from agent.state import AgentState
from utils.config import load_settings


@dataclass
class DebateRound:
    """Single round of debate."""
    round_number: int
    bull_argument: str
    bear_argument: str


@dataclass
class ResearchReport:
    """Report from the research debate."""
    bull_arguments: List[str] = field(default_factory=list)
    bear_arguments: List[str] = field(default_factory=list)
    debate_rounds: List[DebateRound] = field(default_factory=list)
    consensus: str = ""
    conviction_score: float = 0.5  # -1 (very bearish) to 1 (very bullish)
    key_risks: List[str] = field(default_factory=list)
    key_opportunities: List[str] = field(default_factory=list)
    final_recommendation: str = "hold"


BULL_RESEARCHER_PROMPT = """You are a BULLISH Researcher in a structured debate. Your job is to argue WHY the asset could go UP.

**Current Round:** {round_num}/3
**Previous Bear Argument:** {bear_previous}

**Your Task:**
1. If this is round 1: Present your strongest bullish case
2. If rounds 2-3: Counter the bear's arguments while reinforcing your position

**Output JSON:**
{{
  "argument": "Your main argument for this round (2-3 sentences)",
  "counter_points": ["Counter to bear argument 1", "Counter 2"],
  "key_opportunity": "One key opportunity you see",
  "conviction": 0.75
}}

Be persuasive, use data from the analyst reports, and respond directly to bear arguments in rounds 2-3.
"""

BEAR_RESEARCHER_PROMPT = """You are a BEARISH Researcher in a structured debate. Your job is to argue WHY the asset could go DOWN.

**Current Round:** {round_num}/3
**Previous Bull Argument:** {bull_previous}

**Your Task:**
1. If this is round 1: Present your strongest bearish case
2. If rounds 2-3: Counter the bull's arguments while reinforcing your position

**Output JSON:**
{{
  "argument": "Your main argument for this round (2-3 sentences)",
  "counter_points": ["Counter to bull argument 1", "Counter 2"],
  "key_risk": "One key risk you see",
  "conviction": 0.65
}}

Be persuasive, use data from the analyst reports, and respond directly to bull arguments in rounds 2-3.
"""

DEBATE_MODERATOR_PROMPT = """You are a Research Moderator synthesizing a 3-round Bull vs Bear debate.

**Debate Transcript:**
{debate_transcript}

**Your Task:**
Evaluate both sides objectively and reach a balanced conclusion.

**Output JSON:**
{{
  "consensus": "A balanced 2-3 sentence summary of the debate conclusion",
  "conviction_score": 0.3,
  "strongest_bull_point": "The most compelling bullish argument from the debate",
  "strongest_bear_point": "The most compelling bearish argument from the debate",
  "key_risks": ["Risk 1", "Risk 2"],
  "key_opportunities": ["Opportunity 1", "Opportunity 2"],
  "recommendation": "buy|sell|hold",
  "reasoning": "Brief explanation of why this recommendation"
}}

Consider the strength of arguments, quality of counters, and overall conviction levels.
"""


def _call_researcher(
    prompt: str,
    analyst_data: Dict[str, Any],
    round_num: int,
    previous_argument: str = "None - this is the first round"
) -> Dict[str, Any]:
    """Call a researcher with the formatted prompt."""
    cfg = load_settings()

    formatted_prompt = prompt.format(
        round_num=round_num,
        bull_previous=previous_argument,
        bear_previous=previous_argument
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
                    {"role": "user", "content": f"Analyst Reports:\n{json.dumps(analyst_data, indent=2, default=str)[:3500]}"}
                ],
                "temperature": 0.5,
                "max_tokens": 350
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
        print(f"[Researcher] Error: {e}")
        return {
            "argument": f"Unable to form argument: {str(e)}",
            "conviction": 0.5
        }


def _run_debate(analyst_data: Dict[str, Any], num_rounds: int = 3) -> List[DebateRound]:
    """Run the Bull vs Bear debate for specified number of rounds."""
    rounds = []
    bull_previous = ""
    bear_previous = ""

    for round_num in range(1, num_rounds + 1):
        print(f"\n[Debate] Round {round_num}/{num_rounds}")

        # Bull argues first (or counters bear)
        print(f"[Bull] Presenting argument...")
        bull_result = _call_researcher(
            BULL_RESEARCHER_PROMPT,
            analyst_data,
            round_num,
            bear_previous
        )
        bull_argument = bull_result.get("argument", "")

        # Bear responds (or counters bull)
        print(f"[Bear] Presenting counter-argument...")
        bear_result = _call_researcher(
            BEAR_RESEARCHER_PROMPT,
            analyst_data,
            round_num,
            bull_argument  # Bear sees bull's current argument
        )
        bear_argument = bear_result.get("argument", "")

        # Store round
        rounds.append(DebateRound(
            round_number=round_num,
            bull_argument=bull_argument,
            bear_argument=bear_argument
        ))

        # Update previous arguments for next round
        bull_previous = bull_argument
        bear_previous = bear_argument

        print(f"[Bull R{round_num}] {bull_argument[:80]}...")
        print(f"[Bear R{round_num}] {bear_argument[:80]}...")

    return rounds


def _moderate_debate(rounds: List[DebateRound], analyst_data: Dict[str, Any]) -> Dict[str, Any]:
    """Moderator synthesizes the debate and reaches conclusion."""
    cfg = load_settings()

    # Build debate transcript
    transcript_parts = []
    for r in rounds:
        transcript_parts.append(f"=== Round {r.round_number} ===")
        transcript_parts.append(f"BULL: {r.bull_argument}")
        transcript_parts.append(f"BEAR: {r.bear_argument}")
        transcript_parts.append("")

    debate_transcript = "\n".join(transcript_parts)

    prompt = DEBATE_MODERATOR_PROMPT.format(debate_transcript=debate_transcript)

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
                    {"role": "user", "content": f"Original Analyst Data:\n{json.dumps(analyst_data, indent=2, default=str)[:2000]}"}
                ],
                "temperature": 0.3,
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

        return json.loads(content)

    except Exception as e:
        print(f"[Moderator] Error: {e}")
        return {
            "consensus": "Unable to reach consensus",
            "conviction_score": 0,
            "recommendation": "hold"
        }


def researchers_node(state: AgentState) -> Dict[str, Any]:
    """
    Research Team node - 3-round Bull vs Bear debate.

    Takes analyst reports and conducts a structured debate between
    bullish and bearish perspectives over 3 rounds.
    """
    analyst_reports = state.get("analyst_reports", [])
    parsed_query = state.get("parsed_query")
    ticker = parsed_query.ticker if parsed_query else "Unknown"

    print(f"\n{'='*50}")
    print(f"[Research Team] Starting 3-round debate for {ticker}")
    print(f"{'='*50}")

    # Prepare analyst data for researchers
    analyst_data = {}
    for report in analyst_reports:
        analyst_data[report.analyst_type] = {
            "findings": report.findings,
            "metrics": report.metrics,
            "recommendation": report.recommendation,
            "confidence": report.confidence
        }

    # Run 3-round debate
    debate_rounds = _run_debate(analyst_data, num_rounds=3)

    # Moderate and synthesize
    print(f"\n[Moderator] Synthesizing debate results...")
    moderation = _moderate_debate(debate_rounds, analyst_data)

    # Build research report
    research_report = ResearchReport(
        bull_arguments=[r.bull_argument for r in debate_rounds],
        bear_arguments=[r.bear_argument for r in debate_rounds],
        debate_rounds=debate_rounds,
        consensus=moderation.get("consensus", ""),
        conviction_score=moderation.get("conviction_score", 0),
        key_risks=moderation.get("key_risks", []),
        key_opportunities=moderation.get("key_opportunities", []),
        final_recommendation=moderation.get("recommendation", "hold")
    )

    print(f"\n[Research Team] Debate completed")
    print(f"[Research Team] Conviction: {research_report.conviction_score:+.2f}")
    print(f"[Research Team] Recommendation: {research_report.final_recommendation}")
    print(f"{'='*50}\n")

    return {
        "research_report": research_report,
        "trading_recommendation": research_report.final_recommendation
    }
