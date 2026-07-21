# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Small helpers for spawn-based worker processes.

Centralizes the spawn context + bounded shutdown so every converted
thread-to-process site behaves identically (see AGENTS.md concurrency rule).
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Iterable, Tuple

_SPAWN = mp.get_context("spawn")


def spawn_worker(
    target: Callable[..., Any],
    args: Tuple[Any, ...],
    *,
    name: str,
    daemon: bool = True,
) -> "mp.process.BaseProcess":
    """Start a daemon process on the spawn context and return it.

    Args:
        target: Callable to run in the child process. Must be picklable
            (i.e., a module-level function, not a lambda or bound method).
        args: Positional arguments to pass to target.
        name: Process name (shown in ps/Activity Monitor).
        daemon: Whether the process is a daemon (default True).

    Returns:
        The started process.
    """
    proc = _SPAWN.Process(target=target, args=args, name=name, daemon=daemon)
    proc.start()
    return proc


def terminate_join(proc: "mp.process.BaseProcess", timeout: float) -> bool:
    """Join a process, escalating to terminate then kill. True if it exited.

    Args:
        proc: The process to stop.
        timeout: Seconds to wait at each join stage.

    Returns:
        True if the process is no longer alive, False otherwise.
    """
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=min(timeout, 2.0))
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=1.0)
    return not proc.is_alive()


def drain_and_force_terminate(
    processes: "Iterable[mp.process.BaseProcess]",
    drain_queue: Callable[[], Any],
    *,
    join_timeout: float = 2.0,
) -> None:
    """Drain a shared result queue once, then terminate/join/kill all procs.

    Unlike :func:`terminate_join`, this terminates immediately with no initial
    graceful join — callers use it only after already waiting for a graceful
    stop. ``drain_queue`` is invoked once up front so any results still buffered
    on the queue are captured before the producers are killed.

    Args:
        processes: The worker processes to stop.
        drain_queue: Zero-arg callable that drains the shared result queue.
        join_timeout: Seconds to wait for each process after terminate() before
            escalating to kill().
    """
    drain_queue()
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=join_timeout)
        if p.is_alive():
            p.kill()
