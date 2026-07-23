"""Uniform config-override discipline for the dwave miner.

Python mirror of the Rust ``quip_miner_core::config`` helpers so every miner
type behaves the same: config (from the coordinator's ``Configure``) overrides
CLI/env with a warning, unrecognized keys warn, and secrets follow the
``*_file`` convention (config carries a path; the value is read from that file
and never travels over the wire).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Keys consumed by the shared session layer, not any single backend; never
# warned about as unknown. Mirrors miner-core's SESSION_KEYS.
SESSION_KEYS = frozenset({"num_sweeps"})

T = TypeVar("T")


def config_override(name: str, cli: T, from_config: Optional[T]) -> T:
    """Resolve one setting with ``config > CLI`` precedence.

    Warns and returns the config value only on an effective change; otherwise
    returns ``cli`` silently.
    """
    if from_config is not None and from_config != cli:
        logger.warning(
            "config overrides %s: %s -> %s (from coordinator)", name, cli, from_config
        )
        return from_config
    return cli


def warn_unknown_fields(backend: str, present: Iterable[str], known: Iterable[str]) -> None:
    """Warn once per unrecognized key. Session keys are never flagged."""
    known_all = set(known) | SESSION_KEYS
    for key in present:
        if key not in known_all:
            logger.warning("config: unknown field '%s' for %s (ignored)", key, backend)


def read_secret_file(path: str) -> Optional[str]:
    """Read a secret from a file path (the ``*_file`` convention).

    Returns the trimmed contents, or ``None`` (with a warning) if unreadable.
    The value itself is never logged.
    """
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("config: cannot read secret file %s: %s", path, exc)
        return None
