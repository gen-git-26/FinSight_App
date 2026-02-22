# evaluation/__init__.py
"""
Evaluation module for tracking agent metrics.

Provides:
- CallMetrics: Dataclass for individual call metrics
- track_metrics: Decorator for automatic metrics capture
- Session functions: get_session_metrics, clear_session_metrics, print_metrics_summary
"""
from evaluation.metrics import (
    CallMetrics,
    track_metrics,
    get_session_metrics,
    clear_session_metrics,
    add_metrics,
    print_metrics_summary,
)

__all__ = [
    "CallMetrics",
    "track_metrics",
    "get_session_metrics",
    "clear_session_metrics",
    "add_metrics",
    "print_metrics_summary",
]
