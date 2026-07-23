"""Uniform config-override discipline for the dwave miner.

Python mirror of the Rust ``quip_miner_core::config`` helpers so every miner
type behaves the same: config (from the coordinator's ``Configure``) overrides
CLI/env with a warning, and unrecognized keys warn. Credentials are not carried
here — the dwave miner uses D-Wave's own config (dwave.conf + env).
"""
from __future__ import annotations

import logging
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
