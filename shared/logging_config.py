"""
Centralized logging configuration for QuIP Protocol.

This module provides:
- Custom formatter with readable timestamps
- Separate loggers for different components
- Configuration via TOML and CLI
- File and console output options
- aiohttp log suppression and redirection
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class QuipFormatter(logging.Formatter):
    """Custom formatter for QuIP Protocol logs with readable timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        # Format timestamp as MM/DD/YYYY HH:MMAM/PM UTC
        dt = datetime.fromtimestamp(record.created)
        timestamp = dt.strftime("%m/%d/%Y %I:%M%p UTC")

        # Get logger name (remove 'shared.' prefix if present)
        logger_name = record.name
        if logger_name.startswith('shared.'):
            logger_name = logger_name[7:]
        if logger_name.startswith('CPU.') or logger_name.startswith('GPU.') or logger_name.startswith('QPU.'):
            # Extract miner type from path
            parts = logger_name.split('.')
            if len(parts) >= 2:
                logger_name = parts[1]  # e.g., 'cpu_miner' -> 'cpu_miner'

        # Format: [LoggerName] MM/DD/YYYY HH:MMAM UTC - Message
        formatted = f"[{logger_name}] {timestamp} - {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(
    log_level: str = "INFO",
    node_log_file: Optional[str] = None,
    http_log_file: Optional[str] = None,
    node_name: str = "quip-node"
) -> Dict[str, logging.Logger]:
    """
    Setup centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        node_log_file: Path to node log file (None for stderr)
        http_log_file: Path to HTTP log file (None to suppress aiohttp logs)
        node_name: Node name for log file naming

    Returns:
        Dictionary of configured loggers
    """

    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create custom formatter
    formatter = QuipFormatter()

    # Setup console handler (default to stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Setup file handler for node logs if specified
    if node_log_file:
        # Ensure directory exists
        log_path = Path(node_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            node_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)

        # If file logging is enabled, also log to console at WARNING level or higher
        console_handler.setLevel(max(numeric_level, logging.WARNING))

        root_logger.addHandler(file_handler)

    root_logger.addHandler(console_handler)

    # Configure aiohttp logging
    if http_log_file:
        # Ensure directory exists
        http_log_path = Path(http_log_file)
        http_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create aiohttp logger with file handler
        aiohttp_logger = logging.getLogger('aiohttp')
        aiohttp_logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        for handler in aiohttp_logger.handlers[:]:
            aiohttp_logger.removeHandler(handler)

        http_file_handler = logging.handlers.RotatingFileHandler(
            http_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        http_file_handler.setLevel(logging.DEBUG)
        http_file_handler.setFormatter(formatter)
        aiohttp_logger.addHandler(http_file_handler)

        # Prevent aiohttp logs from propagating to root logger
        aiohttp_logger.propagate = False
    else:
        # Suppress aiohttp logs entirely
        aiohttp_logger = logging.getLogger('aiohttp')
        aiohttp_logger.setLevel(logging.CRITICAL)
        aiohttp_logger.propagate = False

    # Create component-specific loggers
    loggers = {}

    # NetworkNode logger
    network_node_logger = logging.getLogger('shared.network_node')
    network_node_logger.setLevel(numeric_level)
    loggers['network_node'] = network_node_logger

    # Node logger
    node_logger = logging.getLogger('shared.node')
    node_logger.setLevel(numeric_level)
    loggers['node'] = node_logger

    # Miner loggers
    miner_types = ['cpu_miner', 'gpu_miner', 'qpu_miner', 'sa_miner']
    for miner_type in miner_types:
        miner_logger = logging.getLogger(f'shared.{miner_type}')
        miner_logger.setLevel(numeric_level)
        loggers[miner_type] = miner_logger

    # Blockchain logger
    blockchain_logger = logging.getLogger('quantum_blockchain')
    blockchain_logger.setLevel(numeric_level)
    loggers['blockchain'] = blockchain_logger

    return loggers


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        name: Logger name (e.g., 'network_node', 'cpu_miner', etc.)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f'shared.{name}')


def update_log_level(loggers: Dict[str, logging.Logger], level: str):
    """
    Update log level for all loggers.

    Args:
        loggers: Dictionary of loggers from setup_logging()
        level: New log level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Update root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)

    # Update component loggers
    for logger in loggers.values():
        logger.setLevel(numeric_level)


def shutdown_logging():
    """Shutdown logging system and close all handlers."""
    logging.shutdown()