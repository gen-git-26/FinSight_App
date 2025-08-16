# utils/logging.py
# This module sets up logging for the application with colored output.
import logging
from colorlog import ColoredFormatter


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s:%(reset)s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
