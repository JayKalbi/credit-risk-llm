"""
Structured Logging Configuration for Hybrid RAG Engine.

Replaces all print() statements with proper Python logging.
Supports human-readable format for development and JSON for production.

Usage:
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started", extra={"chunks": 42})
"""

import logging
import sys

_initialized_loggers: set = set()


def get_logger(name: str = "hybrid_rag") -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if name not in _initialized_loggers:
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.propagate = False
        _initialized_loggers.add(name)

    return logger


def configure_root_logger(level: str = "INFO", json_format: bool = False) -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON format for production log aggregation.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"module":"%(name)s","message":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
