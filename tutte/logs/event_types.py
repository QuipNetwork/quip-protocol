"""Data definitions for the EventLog module.

Contains the EventType enum and Event dataclass. Pure data — no runtime logic,
no singleton state, no imports beyond stdlib.

See log_design.md for full design documentation.
"""

from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """Severity level for log events.

    Events below EventLog.min_level are silently counted (not stored).
    """

    DEBUG = 0
    INFO = 1
    WARN = 2


class EventType(Enum):
    """Semantic classification for engine events.

    Each value corresponds to a distinct phase or operation in the synthesis
    pipeline. Used by EventLog.filter() and EventLog.summary() for aggregation.
    """

    # --- Pipeline entry / exit ---
    SYNTHESIS_START = "synthesis_start"

    # --- Cache and lookup ---
    CACHE_HIT = "cache_hit"
    LOOKUP_HIT = "lookup_hit"

    # --- Structural decomposition ---
    BASE_CASE = "base_case"
    FAMILY_RECOGNITION = "family_recognition"
    FACTORIZE = "factorize"           # Cut vertex / disconnected split
    SERIES_PARALLEL = "series_parallel"
    TREEWIDTH_DP = "treewidth_dp"
    KSUM = "ksum"

    # --- Hierarchical tiling ---
    HIERARCHICAL = "hierarchical"
    THEOREM6 = "theorem6"

    # --- CEJ path (covering.py) ---
    CANDIDATE_FILTER = "candidate_filter"
    VF2_MATCH = "vf2_match"
    COVER_RESULT = "cover_result"

    # --- Edge addition ---
    EDGE_ADD = "edge_add"

    # --- Multigraph operations ---
    MULTIGRAPH_OP = "multigraph_op"

    # --- Verification ---
    VERIFY = "verify"

    # --- Internal ---
    SCOPE_SUMMARY = "scope_summary"


@dataclass(frozen=True)
class Event:
    """A single timestamped event recorded during synthesis.

    Attributes:
        timestamp: Seconds elapsed since the first event in this run
                   (time.perf_counter() relative to start).
        depth: Recursion depth at the time of recording. Incremented by
               push(), decremented by pop().
        event_type: Semantic category from EventType enum.
        module: Source module identifier string (e.g. "synthesis",
                "covering", "validation"). See log_design.md for the
                full list of standardized identifiers.
        message: Human-readable description of what happened.
        level: Severity level (DEBUG, INFO, WARN).
    """

    timestamp: float
    depth: int
    event_type: EventType
    module: str
    message: str
    level: LogLevel = LogLevel.INFO
