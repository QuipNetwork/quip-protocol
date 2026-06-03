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
from typing import Optional, Dict


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

        # Miner loggers: miner.{miner_id} (created by shared/miner_worker.py)
        if logger_name.startswith('miner.'):
            miner_id = logger_name.split('.', 1)[1]
            return 'miner', miner_id

        # Module-level loggers (e.g. 'substrate.miner_controller', 'shared.base_miner')
        if '.' in logger_name:
            parts = logger_name.split('.')
            if 'miner' in logger_name:
                return 'miner', parts[-1]
            return parts[0], parts[-1]

        # Fallback for other loggers
        return 'unknown', logger_name


def _make_rotating_file_handler(
    path: str,
    formatter: logging.Formatter,
    level: int,
    backup_count: int = 5,
) -> logging.handlers.RotatingFileHandler:
    """Create a RotatingFileHandler after ensuring the parent directory exists.

    Args:
        path: Absolute path to the log file.
        formatter: Formatter to attach to the handler.
        level: Numeric logging level for the handler.
        backup_count: Number of backup files to keep (default 5).

    Returns:
        Configured RotatingFileHandler ready to be added to a logger.
    """
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=backup_count,
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    return fh


def setup_logging(
    log_level: str = "INFO",
    node_log_file: Optional[str] = None,
    http_log_file: Optional[str] = None,
) -> Dict[str, logging.Logger]:
    """
    Setup centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        node_log_file: Path to node log file (None for stderr)
        http_log_file: Path to HTTP log file (None to suppress aiohttp logs)

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
        root_logger.addHandler(
            _make_rotating_file_handler(node_log_file, formatter, numeric_level)
        )

    root_logger.addHandler(console_handler)

    # Configure aiohttp logging
    aiohttp_logger = logging.getLogger('aiohttp')
    aiohttp_logger.propagate = False

    # Remove any existing handlers
    for handler in aiohttp_logger.handlers[:]:
        aiohttp_logger.removeHandler(handler)

    if http_log_file:
        aiohttp_logger.setLevel(logging.DEBUG)

        # Support special values 'stderr' and 'stdout' to route HTTP logs to console
        target = str(http_log_file).strip().lower()
        if target in ("stderr", "stdout"):
            stream = sys.stderr if target == "stderr" else sys.stdout
            http_stream_handler = logging.StreamHandler(stream)
            http_stream_handler.setLevel(logging.DEBUG)
            http_stream_handler.setFormatter(formatter)
            aiohttp_logger.addHandler(http_stream_handler)
        else:
            aiohttp_logger.addHandler(
                _make_rotating_file_handler(
                    http_log_file, formatter, logging.DEBUG, backup_count=3
                )
            )
    else:
        # Suppress aiohttp logs entirely
        aiohttp_logger.setLevel(logging.CRITICAL)

    # Configure miner parent logger so all miner.* children inherit formatting
    miner_parent_logger = logging.getLogger('miner')
    miner_parent_logger.setLevel(numeric_level)
    miner_parent_logger.propagate = True

    return {'miner': miner_parent_logger}


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        name: Logger name (e.g., 'miner', 'base_miner', etc.)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f'shared.{name}')


def init_component_logger(component: str, identifier: str) -> logging.Logger:
    """
    Initialize a component logger with proper setup.

    This function creates a logger with the standard naming convention,
    ensures proper propagation, and sets up the global log variable
    for use by static functions in the module.

    Args:
        component: Component type (e.g., 'miner', 'substrate')
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


def log_writer_main(log_queue, stop_event, log_file_path, level) -> None:
    """Sole owner of the file/console log handlers; drains the shared queue.

    Replaces the in-process QueueListener thread. All processes (controller
    + workers) route records here via QueueHandler, so this is the only
    writer of the log file — no double-write or rotation race. A None on the
    queue or a set stop_event ends the loop.

    Args:
        log_queue: Multiprocessing queue of LogRecord objects.
        stop_event: Multiprocessing Event; when set the loop exits after
            draining any pending records.
        log_file_path: Absolute path for the RotatingFileHandler, or None to
            skip file output (console-only mode).
        level: Numeric logging level applied to all handlers.
    """
    import queue as _queue

    fmt = QuipFormatter()
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    handlers = [console]
    if log_file_path:
        handlers.append(_make_rotating_file_handler(log_file_path, fmt, level))

    def _emit(record) -> None:
        for h in handlers:
            if record.levelno >= h.level:
                try:
                    h.handle(record)
                except Exception as exc:  # noqa: BLE001
                    # One unformattable record must not kill the sole log
                    # writer — that would silently lose ALL logging. Report
                    # to stderr and keep draining.
                    print(f"log_writer: failed to emit record: {exc}",
                          file=sys.stderr)

    while not stop_event.is_set():
        try:
            record = log_queue.get(timeout=0.2)
        except _queue.Empty:
            continue
        if record is None:
            break
        _emit(record)
    # Drain anything still queued at shutdown (stop_event may fire before the
    # None sentinel is dequeued) so the final records aren't lost.
    while True:
        try:
            record = log_queue.get_nowait()
        except _queue.Empty:
            break
        if record is None:
            break
        _emit(record)
    for h in handlers:
        h.close()
