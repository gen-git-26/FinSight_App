# agent/nodes/composer.py
"""
Composer Agent - Composes the final response for the user.
"""
from __future__ import annotations

import json
import httpx
from typing import Dict, Any, List

from agent.state import AgentState, AnalysisResult
from memory.manager import persist_turn
from utils.config import load_settings


COMPOSER_PROMPT = """You are a financial assistant composing a response for a user.

**Guidelines:**
- Be concise and professional
- Present data clearly (use markdown formatting)
- Highlight the most important information first
- Include relevant disclaimers for investment-related queries
- For comparisons: use tables when appropriate
- For prices: include change percentage
- For crypto: mention volatility context

**Available data:**
- Query: The user's original question
- Analysis: Insights and metrics from the data
- Sources: Where the data came from

**Output:**
Compose a clear, helpful response in markdown format.
Do NOT include JSON or code blocks unless showing data tables.
Keep the response focused and under 300 words.
"""


def _compose_with_llm(state: AgentState) -> str:
    """Use LLM to compose the final response."""
    cfg = load_settings()

    parsed_query = state.get("parsed_query")
    query = parsed_query.raw_query if parsed_query else state.get("query", "")
    analysis = state.get("analysis")
    fetched_data = state.get("fetched_data", [])
    sources = state.get("sources", [])

    # Build context for LLM
    context = {
        "query": query,
        "query_type": parsed_query.query_type if parsed_query else "general",
        "ticker": parsed_query.ticker if parsed_query else None,
        "analysis": {
            "insights": analysis.insights if analysis else [],
            "metrics": analysis.metrics if analysis else {},
            "summary": analysis.summary if analysis else ""
        } if analysis else None,
        "sources": sources or [d.source for d in fetched_data if not d.error],
        "raw_data_preview": str(fetched_data[0].parsed_data)[:500] if fetched_data and fetched_data[0].parsed_data else None
    }

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
                    {"role": "system", "content": COMPOSER_PROMPT},
                    {"role": "user", "content": f"Compose response for:\n{json.dumps(context, indent=2, default=str)[:3000]}"}
                ],
                "temperature": 0.5,
                "max_tokens": 600
            },
            timeout=20.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"[Composer] LLM failed: {e}")
        return _basic_compose(state)


def _basic_compose(state: AgentState) -> str:
    """Basic response composition without LLM."""
    parsed_query = state.get("parsed_query")
    analysis = state.get("analysis")
    fetched_data = state.get("fetched_data", [])
    error = state.get("error")

    parts = []

    # Header
    if parsed_query and parsed_query.ticker:
        parts.append(f"## {parsed_query.ticker}")
    elif parsed_query:
        parts.append(f"## Financial Query")

    # Error handling
    if error:
        parts.append(f"\n**Error:** {error}")
        return "\n".join(parts)

    # Analysis insights
    if analysis and analysis.insights:
        parts.append("\n**Key Points:**")
        for insight in analysis.insights[:5]:
            parts.append(f"- {insight}")

    # Metrics
    if analysis and analysis.metrics:
        parts.append("\n**Metrics:**")
        for key, value in list(analysis.metrics.items())[:6]:
            parts.append(f"- **{key.replace('_', ' ').title()}:** {value}")

    # Summary
    if analysis and analysis.summary:
        parts.append(f"\n{analysis.summary}")

    # Sources
    sources = [d.source for d in fetched_data if not d.error]
    if sources:
        parts.append(f"\n*Data sources: {', '.join(set(sources))}*")

    # Disclaimer
    if parsed_query and parsed_query.query_type in ['stock', 'crypto', 'options']:
        parts.append("\n---\n*This is not financial advice. Always do your own research.*")

    return "\n".join(parts)


def _handle_general_query(state: AgentState) -> str:
    """Handle general queries that don't need data fetching."""
    cfg = load_settings()
    query = state.get("query", "")
    memory = state.get("memory", {})

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
                    {
                        "role": "system",
                        "content": "You are a helpful financial assistant. Answer questions about finance, investing, and markets. Be concise and accurate."
                    },
                    {
                        "role": "user",
                        "content": f"Memory context: {memory.get('retrieved_memory', '')[:500]}\n\nQuestion: {query}"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=20.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"I apologize, I couldn't process your question. Error: {e}"


async def _save_to_memory(query: str, response: str) -> None:
    """Save the interaction to memory."""
    try:
        await persist_turn(query, response)
    except Exception as e:
        print(f"[Composer] Memory save failed: {e}")


def composer_node(state: AgentState) -> Dict[str, Any]:
    """
    Composer node - creates the final response.

    This is the final node in the graph.
    """
    print(f"\n[Composer] Composing final response")

    parsed_query = state.get("parsed_query")
    next_agent = state.get("next_agent")

    # Handle general queries (no data fetching needed)
    if next_agent == "composer" or (parsed_query and parsed_query.query_type == "general"):
        response = _handle_general_query(state)
    else:
        # Compose response from fetched data and analysis
        response = _compose_with_llm(state)

    # Get sources
    fetched_data = state.get("fetched_data", [])
    sources = list(set([d.source for d in fetched_data if not d.error]))

    # Save to memory (async)
    query = parsed_query.raw_query if parsed_query else state.get("query", "")
    try:
        import asyncio
        asyncio.create_task(_save_to_memory(query, response))
    except:
        pass  # Memory save is best-effort

    print(f"[Composer] Response length: {len(response)} chars")

    return {
        "response": response,
        "sources": sources
    }
