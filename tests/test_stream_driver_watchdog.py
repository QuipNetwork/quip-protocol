# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parent-death watchdog for the QPU stream driver.

The driver is a non-daemon spawn child holding a live D-Wave connection; if its
parent dies ungracefully nothing else sets its stop_event, so it would keep
burning QPU as an orphan. The watchdog detects the reparent and stops it.
"""
from __future__ import annotations

import threading

from QPU.stream_driver import _parent_death_watchdog


def test_watchdog_returns_quietly_on_normal_shutdown():
    # stop_event already set (graceful shutdown) -> watchdog exits without
    # touching the hard-exit backstop.
    ev = threading.Event()
    ev.set()
    calls: list[str] = []
    _parent_death_watchdog(
        ev,
        original_ppid=100,
        getppid=lambda: 100,
        poll_s=0.0,
        sleep=lambda _s: calls.append("sleep"),
        hard_exit=lambda: calls.append("exit"),
    )
    assert calls == []


def test_watchdog_stops_and_hard_exits_on_parent_death():
    # getppid changed (reparented to init) -> set stop_event, then hard-exit
    # after the grace period if the process hasn't already died.
    ev = threading.Event()
    calls: list[str] = []
    _parent_death_watchdog(
        ev,
        original_ppid=100,
        getppid=lambda: 1,  # reparented
        poll_s=0.0,
        grace_s=0.0,
        sleep=lambda _s: calls.append("sleep"),
        hard_exit=lambda: calls.append("exit"),
    )
    assert ev.is_set(), "watchdog must signal stop so the main loop tears down"
    assert calls == ["sleep", "exit"], "grace sleep then hard exit backstop"


def test_watchdog_detects_death_after_several_clean_polls():
    # Parent alive for the first polls, then dies; watchdog must still catch it.
    ev = threading.Event()
    seq = iter([100, 100, 100, 7])  # alive, alive, alive, reparented
    last = {"v": 100}

    def _getppid() -> int:
        try:
            last["v"] = next(seq)
        except StopIteration:
            pass
        return last["v"]

    calls: list[str] = []
    _parent_death_watchdog(
        ev,
        original_ppid=100,
        getppid=_getppid,
        poll_s=0.0,
        grace_s=0.0,
        sleep=lambda _s: None,
        hard_exit=lambda: calls.append("exit"),
    )
    assert ev.is_set()
    assert calls == ["exit"]
