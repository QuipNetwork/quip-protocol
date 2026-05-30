# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Generic hardware-utilization monitor process.

Replaces the per-scheduler polling thread (see AGENTS.md). The poll function
is passed by dotted path so the spawned (re-imported) monitor process can
resolve it without pickling a bound method or a non-picklable vendor handle.
Each backend's poll fn does its own nvmlInit/IOKit setup in the child.
"""
from __future__ import annotations

import importlib
from typing import Callable


def _resolve(dotted: str) -> Callable[[], int]:
    """Resolve a ``module:attr`` dotted path to a callable.

    Args:
        dotted: String of the form ``"package.module:attr"``.

    Returns:
        The resolved callable.
    """
    module_name, _, attr = dotted.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def util_monitor_main(
    out_value,
    stop_event,
    interval_s: float,
    poll_dotted: str,
) -> None:
    """Poll ``poll_dotted()`` every interval and publish the int into out_value.

    Designed to run in a spawned child process. The poll function is resolved
    by dotted path on first call so vendor handles (NVML, IOKit) are
    initialized inside the child, never inherited from the parent.

    Args:
        out_value: A ``multiprocessing.Value("i", ...)`` shared with the parent.
        stop_event: A ``multiprocessing.Event`` — set to request shutdown.
        interval_s: Seconds between polls.
        poll_dotted: Dotted path ``"module:attr"`` for the zero-arg poll fn.
    """
    poll = _resolve(poll_dotted)
    while not stop_event.is_set():
        try:
            out_value.value = int(poll())
        except Exception:  # noqa: BLE001 — a monitor must never crash the miner
            pass
        stop_event.wait(interval_s)
