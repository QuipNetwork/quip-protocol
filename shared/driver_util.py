# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Shared, SDK-free helpers for the stream-driver family of processes.

Used by the single-driver pump (``QPU.stream_driver``, CPU/GPU + legacy QPU),
the connection-less feeder driver (``shared.feeder_driver``, QPU split
producer), and the isolated submitter (``QPU.dwave_submitter``). Kept in
``shared/`` so the feeder driver can import them without pulling ``QPU/__init__``
→ the D-Wave SDK.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import queue as _queue
import time
from typing import Any, Callable, Dict

log = logging.getLogger(__name__)


def _resolve(dotted: str):
    """Resolve a ``"module.path:attr"`` dotted reference to the attribute."""
    module_name, _, attr = dotted.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _parent_death_watchdog(
    stop_event,
    *,
    original_ppid: int,
    getppid: Callable[[], int] = os.getppid,
    poll_s: float = 2.0,
    grace_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    hard_exit: Callable[[], None] = lambda: os._exit(0),
) -> None:
    """Stop a driver/submitter when its parent dies (orphan → keeps burning QPU).

    These are non-daemon spawn children that may hold a live D-Wave connection.
    If the parent (miner worker) dies ungracefully — SIGKILL, crash, a ``pkill``
    that misses the spawn child — nothing else sets ``stop_event``, so the child
    keeps submitting forever. This loop polls the parent pid; a change from
    ``original_ppid`` means we were reparented (parent died). It sets
    ``stop_event`` so the main loop tears down, then force-exits after
    ``grace_s`` as a backstop. On a normal shutdown ``stop_event`` is set
    elsewhere and this returns quietly. All timing/exit hooks are injectable for
    unit testing without a real reparent.

    Note: times itself with a plain ``sleep`` and only does a brief
    ``stop_event.is_set()`` check — it must never *block* on the mp.Event from
    this side thread (holding its cross-process lock at process exit would
    deadlock the parent's ``set()``).
    """
    while not stop_event.is_set():
        if getppid() != original_ppid:
            log.warning(
                "driver: parent %d died (now reparented to %d); stopping to "
                "release resources",
                original_ppid, getppid(),
            )
            stop_event.set()
            sleep(grace_s)
            log.warning(
                "driver: teardown exceeded %.0fs after parent death; forcing "
                "exit", grace_s,
            )
            hard_exit()
            return
        sleep(poll_s)


def _start_parent_death_watchdog(stop_event):
    """Spawn the daemon watchdog thread; returns it (for tests/introspection)."""
    import threading
    t = threading.Thread(
        target=_parent_death_watchdog,
        args=(stop_event,),
        kwargs={"original_ppid": os.getppid()},
        name="parent-death-watchdog",
        daemon=True,
    )
    t.start()
    return t


def _maybe_with_stop(fn, kwargs: Dict[str, Any], stop_event) -> Dict[str, Any]:
    """Add ``stop_event`` to ``kwargs`` only if ``fn`` accepts that kwarg.

    The production factory consumes ``stop_event`` so its streaming loop can
    observe cancellation; test fakes don't declare it. Inspecting the signature
    keeps both callers working without a dual code path.
    """
    if "stop_event" in inspect.signature(fn).parameters:
        return {**kwargs, "stop_event": stop_event}
    return kwargs


def qpu_access_time_us(sampleset) -> int:
    """Sum D-Wave qpu_programming_time + qpu_sampling_time from a sampleset (µs).

    Returns 0 when the timing dict is missing, partial, or contains None values
    — happens on non-QPU fallbacks and some embedded-future code paths.
    """
    info = getattr(sampleset, "info", None) or {}
    t = info.get("timing", {})
    prog = t.get("qpu_programming_time") or 0
    sample = t.get("qpu_sampling_time") or 0
    return int(prog) + int(sample)


def _extract_qpu_us(sampleset) -> int:
    return qpu_access_time_us(sampleset)


def _coalesce_ctl_q(ctx, ctl_q) -> str:
    """Drain ctl_q, applying the newest switch + latest threshold + pause.

    Returns ``"shutdown"`` if a ``None`` sentinel was seen, else ``"ok"``.
    A burst of switches coalesces to the highest generation (dead intermediate
    rounds are skipped); a trailing threshold still applies. A ``("pause", gen)``
    (budget-exhaustion stall) stops new submissions but is superseded by a switch
    in the same drain (a fresh head outranks a stall) and ignored if stale for an
    older generation.
    """
    latest_switch = None
    latest_threshold = None
    latest_pause = None
    shutdown = False
    while True:
        try:
            cmd = ctl_q.get_nowait()
        except _queue.Empty:
            break
        if cmd is None:
            shutdown = True
            break
        if cmd[0] == "switch":
            if latest_switch is None or cmd[1] >= latest_switch[1]:
                latest_switch = cmd
        elif cmd[0] == "threshold":
            latest_threshold = cmd
        elif cmd[0] == "pause":
            if latest_pause is None or cmd[1] >= latest_pause[1]:
                latest_pause = cmd
    # Apply switch BEFORE threshold: when both target the live generation in one
    # drain, the threshold update must override the threshold embedded in the
    # switch tuple (a decay that landed in the same tick as the head change).
    if latest_switch is not None:
        ctx.apply_command(latest_switch)
    if latest_threshold is not None and (
        latest_switch is None or latest_threshold[1] >= ctx.generation
    ):
        ctx.apply_command(latest_threshold)
    if (
        latest_pause is not None
        and latest_switch is None
        and latest_pause[1] >= ctx.generation
    ):
        ctx.apply_command(latest_pause)
    return "shutdown" if shutdown else "ok"


def _wait_for_first_switch(ctx, ctl_q, stop_event) -> bool:
    """Block (polling stop_event) until the first switch arrives.

    Returns True once a round is active, False if shutdown/stop came first.
    """
    while not stop_event.is_set():
        try:
            cmd = ctl_q.get(timeout=0.1)
        except _queue.Empty:
            continue
        if cmd is None:
            return False
        if cmd[0] == "switch":
            ctx.apply_command(cmd)
            return True
        # A 'threshold' before any 'switch' is meaningless; apply + keep waiting.
        ctx.apply_command(cmd)
    return False


__all__ = [
    "_resolve",
    "_parent_death_watchdog",
    "_start_parent_death_watchdog",
    "_maybe_with_stop",
    "qpu_access_time_us",
    "_extract_qpu_us",
    "_coalesce_ctl_q",
    "_wait_for_first_switch",
]
