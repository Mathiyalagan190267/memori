"""
Memori Labs — Unified Logging Utility
=====================================

Provides a centralized logger for all Memori modules.

Usage:
    from memori.utils.logger import logger

    logger.info("Starting conscious ingestion")
    logger.debug("Loaded memory entries", extra={"count": 42})

This uses Python's `logging` module with colored output for local
development and structured JSON for production logs.

Author: YourName (@yourhandle)
Date: 2025-10-29
"""

import logging
import sys
import os
from datetime import datetime


class MemoriFormatter(logging.Formatter):
    """Custom log formatter with color and timestamps."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] {record.levelname}: {record.getMessage()}"
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        return f"{color}{message}{reset}"


def get_logger(name: str = "memori") -> logging.Logger:
    """Initialize and return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = MemoriFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        level = os.getenv("MEMORI_LOG_LEVEL", "INFO").upper()
        logger.setLevel(level)
    return logger


# Create default logger instance
logger = get_logger()
