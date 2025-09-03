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
import multiprocessing as mp
import sys
from datetime import datetime
from logging.handlers import QueueListener, QueueHandler
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


class QuipFormatter(logging.Formatter):
    """Custom formatter for QuIP Protocol logs with readable timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        # Format timestamp as ISO 8601 extended: YYYY-MM-DDTHH:MM:SS.ffffff+00:00
        dt = datetime.fromtimestamp(record.created)
        timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

        # Get log level
        level_name = record.levelname

        # Parse logger context for component and identifier
        component, identifier = self._parse_logger_context(record)

        # Format: [filename:lineno][identifier] TIMESTAMP LEVEL - Message
        location = f"{record.filename}:{record.lineno}"
        formatted = f"[{location}][{identifier}] {timestamp} {level_name} - {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted

    def _parse_logger_context(self, record: logging.LogRecord) -> tuple[str, str]:
        """Parse logger name and extract component and identifier."""
        logger_name = record.name

        # Handle miner loggers: miner.{miner_id}
        if logger_name.startswith('miner.'):
            miner_id = logger_name.split('.', 1)[1]
            return 'miner', miner_id

        # Handle network node loggers: network_node.{node_id}
        if logger_name.startswith('network_node.'):
            node_id = logger_name.split('.', 1)[1]
            return 'network_node', node_id

        # Handle node loggers: node.{node_id}
        if logger_name.startswith('node.'):
            node_id = logger_name.split('.', 1)[1]
            return 'node', node_id

        # Handle legacy shared.* loggers for backward compatibility
        if logger_name.startswith('shared.'):
            component = logger_name[7:]  # Remove 'shared.' prefix
            return component, 'legacy'

        # Handle other module-level loggers by extracting meaningful names
        if '.' in logger_name:
            parts = logger_name.split('.')
            if len(parts) >= 2:
                # For loggers like 'quantum_blockchain_network', 'blockchain_base', etc.
                if 'blockchain' in logger_name:
                    return 'blockchain', parts[-1]
                elif 'network' in logger_name:
                    return 'network', parts[-1]
                elif 'miner' in logger_name:
                    return 'miner', parts[-1]
                else:
                    return parts[0], parts[-1]

        # Fallback for other loggers
        return 'unknown', logger_name


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
        # Create aiohttp logger
        aiohttp_logger = logging.getLogger('aiohttp')
        aiohttp_logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        for handler in aiohttp_logger.handlers[:]:
            aiohttp_logger.removeHandler(handler)

        # Support special values 'stderr' and 'stdout' to route HTTP logs to console
        target = str(http_log_file).strip().lower()
        if target in ("stderr", "stdout"):
            stream = sys.stderr if target == "stderr" else sys.stdout
            http_stream_handler = logging.StreamHandler(stream)
            http_stream_handler.setLevel(logging.DEBUG)
            http_stream_handler.setFormatter(formatter)
            aiohttp_logger.addHandler(http_stream_handler)
            # Do not propagate to root to avoid duplicate messages
            aiohttp_logger.propagate = False
        else:
            # Ensure directory exists for file logging
            http_log_path = Path(http_log_file)
            http_log_path.parent.mkdir(parents=True, exist_ok=True)

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

    # NetworkNode logger - use node_name parameter
    network_node_logger = logging.getLogger(f'network_node.{node_name}')
    network_node_logger.setLevel(numeric_level)
    loggers['network_node'] = network_node_logger

    # Node logger - use node_name parameter
    node_logger = logging.getLogger(f'node.{node_name}')
    node_logger.setLevel(numeric_level)
    loggers['node'] = node_logger

    # Configure miner parent logger to ensure all miner.* loggers inherit proper formatting
    miner_parent_logger = logging.getLogger('miner')
    miner_parent_logger.setLevel(numeric_level)
    # Ensure propagation is enabled (should be default, but let's be explicit)
    miner_parent_logger.propagate = True
    loggers['miner'] = miner_parent_logger

    # Keep individual miner type loggers for backward compatibility
    miner_types = ['cpu_miner', 'gpu_miner', 'qpu_miner', 'sa_miner']
    for miner_type in miner_types:
        miner_logger = logging.getLogger(f'miner.{miner_type}')
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


def setup_multiprocess_logging() -> Tuple[mp.Queue, QueueListener]:
    """Set up logging for multiprocessing environment.

    Returns:
        Tuple of (log_queue, listener) for multiprocessing logging.
    """
    # Create queue for inter-process communication
    log_queue = mp.Queue()

    # Handler that sends log records to queue
    queue_handler = QueueHandler(log_queue)

    # Get root logger and add queue handler
    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    # Create listener that processes queue in main process
    formatter = QuipFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    listener = QueueListener(log_queue, console_handler)
    listener.start()

    return log_queue, listener


def init_component_logger(component: str, identifier: str) -> logging.Logger:
    """
    Initialize a component logger with proper setup.

    This function creates a logger with the standard naming convention,
    ensures proper propagation, and sets up the global log variable
    for use by static functions in the module.

    Args:
        component: Component type (e.g., 'network_node', 'miner', 'node')
        identifier: Unique identifier for this instance

    Returns:
        Configured logger instance
    """
    # Create logger with standard naming convention
    logger = logging.getLogger(f'{component}.{identifier}')

    # Ensure propagation to root logger for proper formatting
    logger.propagate = True

    # Set global logger for static functions in this module
    global log
    log = logger

    return logger


def shutdown_logging():
    """Shutdown logging system and close all handlers."""
    logging.shutdown()