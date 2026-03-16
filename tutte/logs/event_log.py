"""EventLog singleton for capturing structured synthesis events.

Module-level singleton — any module in tutte/ can import get_log() and record
events without needing a reference to the engine instance.

Usage:
    from tutte.logs import get_log, reset_log, EventType

    reset_log()
    engine.synthesize(graph)
    get_log().print_timeline()
"""

import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tutte.logs.event_types import Event, EventType, LogLevel

# Box-drawing characters with ASCII fallback for terminals that lack UTF-8
# (e.g. older Windows cmd.exe).
_USE_UNICODE = hasattr(sys.stdout, "encoding") and sys.stdout.encoding and (
    "utf" in (sys.stdout.encoding or "").lower()
)
_DOUBLE_LINE = "=" if not _USE_UNICODE else "\u2550"
_SINGLE_LINE = "-" if not _USE_UNICODE else "\u2500"
_ARROW = "<-" if not _USE_UNICODE else "\u2190"


@dataclass
class ScopeFrame:
    """Tracks aggregated event counts for a named scope.

    When min_level filters out DEBUG events, their counts are accumulated
    here instead. On pop(), a single SCOPE_SUMMARY INFO event is emitted.
    """

    scope: str
    counts: Dict[EventType, int] = field(default_factory=lambda: defaultdict(int))


class EventLog:
    """Captures timestamped, structured events during synthesis.

    Events are stored in a flat list with a depth field for recursion tracking.
    The singleton instance is accessed via get_log(). Call reset_log() before
    each synthesize() invocation to clear state.
    """

    def __init__(self) -> None:
        self._events: List[Event] = []
        self._depth: int = 0
        self._start_time: Optional[float] = None
        self.min_level: LogLevel = LogLevel.INFO
        self._scope_stack: List[ScopeFrame] = []

    def record(
        self,
        event_type: EventType,
        module: str,
        message: str,
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        """Append a timestamped event at the current recursion depth.

        If level is below min_level, the event is not stored. Instead, the
        count is incremented on the current scope frame (if any).
        """
        if level.value < self.min_level.value:
            if self._scope_stack:
                self._scope_stack[-1].counts[event_type] += 1
            return

        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now
        timestamp = now - self._start_time
        self._events.append(
            Event(
                timestamp=timestamp,
                depth=self._depth,
                event_type=event_type,
                module=module,
                message=message,
                level=level,
            )
        )

    def push(self, scope: str = "") -> None:
        """Increment recursion depth and open a named scope.

        DEBUG events inside the scope are counted (not stored) when
        min_level > DEBUG. On pop(), a single summary event is emitted.
        """
        self._depth += 1
        self._scope_stack.append(ScopeFrame(scope))

    def pop(self) -> None:
        """Close the current scope and emit a summary if events were aggregated.

        Decrements recursion depth (floor at 0).
        """
        frame = self._scope_stack.pop() if self._scope_stack else None
        if frame and any(frame.counts.values()):
            parts = [f"{c} {t.value}" for t, c in frame.counts.items() if c > 0]
            summary = f"{frame.scope}: {', '.join(parts)}" if frame.scope else ", ".join(parts)
            self.record(
                EventType.SCOPE_SUMMARY, "log", summary, level=LogLevel.INFO
            )
        if self._depth > 0:
            self._depth -= 1

    def reset(self) -> None:
        """Clear all events and reset depth, start time, and scope stack."""
        self._events.clear()
        self._depth = 0
        self._start_time = None
        self._scope_stack.clear()

    def replay(self, events: List[Event]) -> None:
        """Load a captured event list for replay (e.g. print_timeline).

        Replaces current events and sets start_time from the first event.
        """
        self._events = list(events)
        self._start_time = events[0].timestamp if events else None
        self._depth = 0
        self._scope_stack.clear()

    @property
    def events(self) -> List[Event]:
        """Read-only access to the recorded event list."""
        return list(self._events)

    def event_count(self) -> int:
        """Return the number of recorded events without copying."""
        return len(self._events)

    def events_since(self, start: int) -> List[Event]:
        """Return events from index `start` onward (slice, no full copy)."""
        return list(self._events[start:])

    def filter(self, event_type: EventType) -> List[Event]:
        """Return events matching the given EventType."""
        return [e for e in self._events if e.event_type == event_type]

    def summary(self) -> Dict[EventType, Tuple[int, float]]:
        """Aggregate events by type: {EventType: (count, total_duration)}.

        Duration for each event is computed as the gap between that event
        and the next event in the list. The last event has zero duration.
        """
        durations: Dict[EventType, Tuple[int, float]] = {}
        for i, event in enumerate(self._events):
            if i + 1 < len(self._events):
                gap = self._events[i + 1].timestamp - event.timestamp
            else:
                gap = 0.0
            count, total = durations.get(event.event_type, (0, 0.0))
            durations[event.event_type] = (count + 1, total + gap)
        return durations

    def print_timeline(self, threshold_ms: float = 100) -> None:
        """Print formatted timeline to stdout with bottleneck highlighting.

        Args:
            threshold_ms: Duration gaps above this value (in milliseconds)
                          are marked with an arrow annotation.
        """
        if not self._events:
            print("(no events recorded)")
            return

        LEVEL_TAG = {LogLevel.DEBUG: "DBG", LogLevel.INFO: "INF", LogLevel.WARN: "WRN"}

        header = (
            f"{'Time':<11s}{'Duration':<11s}{'Depth':<7s}{'Lvl':<5s}"
            f"{'Type':<20s}{'Module':<25s}{'Message'}"
        )
        separator = _SINGLE_LINE * len(header)

        print(_DOUBLE_LINE * len(header))
        print("Engine Timeline")
        print(_DOUBLE_LINE * len(header))
        print(header)
        print(separator)

        for i, event in enumerate(self._events):
            if i + 1 < len(self._events):
                gap = self._events[i + 1].timestamp - event.timestamp
                duration_str = f"{gap:.3f}s"
            else:
                gap = 0.0
                duration_str = ""

            indent = "  " * event.depth
            time_str = f"{event.timestamp:.3f}s"
            depth_str = str(event.depth)
            level_str = LEVEL_TAG.get(event.level, "???")
            type_str = event.event_type.value
            module_str = event.module
            message_str = f"{indent}{event.message}"

            line = (
                f"{time_str:<11s}{duration_str:<11s}{depth_str:<7s}{level_str:<5s}"
                f"{type_str:<20s}{module_str:<25s}{message_str}"
            )

            if gap * 1000 >= threshold_ms:
                gap_ms = gap * 1000
                if gap_ms >= 1000:
                    line += f"  {_ARROW} {gap:.1f}s"
                else:
                    line += f"  {_ARROW} {gap_ms:.0f}ms"

            print(line)

        print(separator)

        total_time = self._events[-1].timestamp if self._events else 0.0
        max_depth = max(e.depth for e in self._events) if self._events else 0
        print(
            f"Total: {total_time:.2f}s | "
            f"Events: {len(self._events)} | "
            f"Max depth: {max_depth} | "
            f"Level: {self.min_level.name}"
        )

        # Summary by EventType
        summary = self.summary()
        if summary and total_time > 0:
            print()
            print("Summary by EventType:")
            sorted_summary = sorted(
                summary.items(), key=lambda x: -x[1][1]
            )
            for event_type, (count, total_duration) in sorted_summary:
                pct = (total_duration / total_time) * 100 if total_time > 0 else 0
                print(
                    f"  {event_type.value:<20s}"
                    f"{count:>5d} events"
                    f"{total_duration:>10.3f}s"
                    f"  ({pct:5.1f}%)"
                )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_log: EventLog = EventLog()


def get_log() -> EventLog:
    """Return the module-level EventLog singleton."""
    return _log


def reset_log() -> None:
    """Clear the singleton's events and reset depth. Call before each run."""
    _log.reset()
