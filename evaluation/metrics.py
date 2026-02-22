# evaluation/metrics.py
"""
Metrics collection for agent evaluation.

Tracks token usage, latency, and success/failure for each agent call.
"""
from __future__ import annotations

import time
import functools
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CallMetrics:
    """Metrics for a single agent/LLM call."""
    agent_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Estimate cost based on GPT-4o-mini pricing."""
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        input_cost = (self.input_tokens / 1_000_000) * 0.15
        output_cost = (self.output_tokens / 1_000_000) * 0.60
        return input_cost + output_cost


# Session-level metrics storage
_session_metrics: List[CallMetrics] = []


def get_session_metrics() -> List[CallMetrics]:
    """Get all metrics collected in this session."""
    return _session_metrics.copy()


def clear_session_metrics() -> None:
    """Clear session metrics."""
    global _session_metrics
    _session_metrics = []


def add_metrics(metrics: CallMetrics) -> None:
    """Add metrics to the session."""
    _session_metrics.append(metrics)


def track_metrics(agent_name: str):
    """
    Decorator to track metrics for an agent node function.

    Usage:
        @track_metrics("analyst")
        def analyst_node(state):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            error = None
            success = True

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                error = str(e)
                success = False
                raise
            finally:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                # Try to extract token counts from result if available
                input_tokens = 0
                output_tokens = 0

                # Token extraction would require modifying LLM calls
                # For now, estimate based on typical usage

                metrics = CallMetrics(
                    agent_name=agent_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    success=success,
                    error=error
                )
                add_metrics(metrics)

            return result
        return wrapper
    return decorator


def print_metrics_summary() -> None:
    """Print a summary of session metrics."""
    metrics = get_session_metrics()

    if not metrics:
        print("\n[Metrics] No metrics collected")
        return

    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)

    total_tokens = sum(m.total_tokens for m in metrics)
    total_cost = sum(m.cost_usd for m in metrics)
    total_latency = sum(m.latency_ms for m in metrics)
    success_count = sum(1 for m in metrics if m.success)

    print(f"  Calls: {len(metrics)} ({success_count} successful)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total latency: {total_latency:.0f}ms")
    print(f"  Estimated cost: ${total_cost:.4f}")
    print()

    print("  Per-agent breakdown:")
    for m in metrics:
        status = "+" if m.success else "x"
        print(f"    {status} {m.agent_name}: {m.latency_ms:.0f}ms, {m.total_tokens} tokens")

    print("=" * 60)
