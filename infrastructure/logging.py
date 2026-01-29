# infrastructure/logging.py
"""
Loguru-based structured logging for FinSight.

Features:
- Colored console output
- JSON file logging
- Agent-specific log filtering
- Performance tracking
"""
from __future__ import annotations

import os
import sys
from typing import Optional
from datetime import datetime

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging
    logger = logging.getLogger("finsight")


# Log format for console
CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[agent]}</cyan> | "
    "<level>{message}</level>"
)

# Log format for JSON file
JSON_FORMAT = "{message}"


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = True,
    json_file: bool = True,
    agent_filter: Optional[str] = None
) -> None:
    """
    Setup Loguru logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        console: Enable console output
        json_file: Enable JSON file logging
        agent_filter: Only show logs from specific agent
    """
    if not LOGURU_AVAILABLE:
        logging.basicConfig(level=level)
        return

    # Remove default handler
    logger.remove()

    # Console handler
    if console:
        logger.add(
            sys.stderr,
            format=CONSOLE_FORMAT,
            level=level,
            colorize=True,
            filter=lambda record: (
                agent_filter is None or
                record["extra"].get("agent", "") == agent_filter
            )
        )

    # Create log directory
    if json_file:
        os.makedirs(log_dir, exist_ok=True)

        # JSON log file (rotated daily)
        logger.add(
            f"{log_dir}/finsight_{{time:YYYY-MM-DD}}.json",
            format=JSON_FORMAT,
            level=level,
            rotation="00:00",
            retention="7 days",
            serialize=True
        )

        # Agent-specific log files
        for agent in ["router", "fetcher", "crypto", "analysts", "researchers", "risk", "trader", "fund_manager"]:
            logger.add(
                f"{log_dir}/{agent}_{{time:YYYY-MM-DD}}.log",
                format=CONSOLE_FORMAT,
                level=level,
                rotation="00:00",
                retention="3 days",
                filter=lambda record, a=agent: record["extra"].get("agent", "").lower() == a
            )


def get_logger(agent_name: str = "system"):
    """
    Get a logger bound to a specific agent.

    Usage:
        log = get_logger("router")
        log.info("Processing query")
    """
    if LOGURU_AVAILABLE:
        return logger.bind(agent=agent_name)
    else:
        return logging.getLogger(f"finsight.{agent_name}")


# === Convenience functions ===

class AgentLogger:
    """Logger wrapper for agents with timing support."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.log = get_logger(agent_name)
        self._start_times = {}

    def info(self, message: str, **kwargs):
        self.log.info(message, **kwargs)

    def debug(self, message: str, **kwargs):
        self.log.debug(message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.log.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log.error(message, **kwargs)

    def start_timer(self, operation: str):
        """Start timing an operation."""
        self._start_times[operation] = datetime.now()
        self.log.debug(f"Started: {operation}")

    def end_timer(self, operation: str) -> float:
        """End timing and log duration."""
        if operation in self._start_times:
            duration = (datetime.now() - self._start_times[operation]).total_seconds()
            self.log.info(f"Completed: {operation} ({duration:.2f}s)")
            del self._start_times[operation]
            return duration
        return 0.0

    def log_decision(self, decision: str, details: dict = None):
        """Log an agent decision."""
        self.log.info(f"Decision: {decision}", extra={"details": details or {}})

    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context."""
        self.log.error(f"Error in {context}: {str(error)}", exc_info=True)


# Pre-configured loggers for each agent
router_log = AgentLogger("router")
fetcher_log = AgentLogger("fetcher")
crypto_log = AgentLogger("crypto")
analysts_log = AgentLogger("analysts")
researchers_log = AgentLogger("researchers")
risk_log = AgentLogger("risk")
trader_log = AgentLogger("trader")
fund_manager_log = AgentLogger("fund_manager")
composer_log = AgentLogger("composer")


# Initialize logging on import
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs"),
    console=True,
    json_file=os.getenv("LOG_JSON", "true").lower() == "true"
)
