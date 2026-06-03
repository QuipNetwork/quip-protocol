# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Stream-driver process: owns the sampler/feeder/generator and produces
samplesets into a SampleView for the consumer (miner worker).

Replaces the in-worker QPU pump thread (AGENTS.md). The generator is driven
by exactly one caller (this process's loop). Only a small descriptor crosses
the descriptor queue; the sample matrix is written zero-copy into the ring.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import pickle
import queue as _queue
from typing import Any, Dict

import numpy as np

from shared.ring_views import SampleView

log = logging.getLogger(__name__)


def _resolve(dotted: str):
    module_name, _, attr = dotted.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _maybe_with_stop(fn, kwargs: Dict[str, Any], stop_event) -> Dict[str, Any]:
    """Add ``stop_event`` to ``kwargs`` only if ``fn`` accepts that kwarg.

    The production factory consumes ``stop_event`` so its streaming loop can
    observe cancellation; test fakes don't declare it. Inspecting the
    signature keeps both callers working without a dual code path.
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
    t = info.get("timing", {}) if info else {}
    prog = t.get("qpu_programming_time") or 0
    sample = t.get("qpu_sampling_time") or 0
    return int(prog) + int(sample)


def _extract_qpu_us(sampleset) -> int:
    return qpu_access_time_us(sampleset)


def _coalesce_ctl_q(ctx, ctl_q) -> str:
    """Drain ctl_q, applying the newest switch + latest threshold + pause.

    Returns ``"shutdown"`` if a ``None`` sentinel was seen, else ``"ok"``.
    A burst of switches coalesces to the highest generation (dead
    intermediate rounds are skipped); a trailing threshold still applies.
    A ``("pause", gen)`` (budget-exhaustion stall) stops new submissions but
    is superseded by a switch in the same drain (a fresh head outranks a
    stall) and ignored if stale for an older generation.
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
    # Apply switch BEFORE threshold: when both target the live generation in
    # one drain, the threshold update must override the threshold embedded in
    # the switch tuple (a decay that landed in the same tick as the head
    # change). Reordering these would let the stale switch-threshold win.
    if latest_switch is not None:
        ctx.apply_command(latest_switch)
    if latest_threshold is not None and (
        latest_switch is None or latest_threshold[1] >= ctx.generation
    ):
        ctx.apply_command(latest_threshold)
    # Pause only when no switch superseded it this drain (a switch resumes the
    # driver for a new head) and it is not stale for an older generation.
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


def stream_driver_main(ring_args: Dict[str, Any], desc_q, ctl_q, stop_event,
                       stream_factory_dotted: str,
                       factory_kwargs: Dict[str, Any]) -> None:
    """Long-lived stream driver: persist the context, switch rounds via ctl_q.

    ``stream_factory_dotted`` resolves to a context factory
    (``build_persistent_context``) returning an object exposing
    ``apply_command`` / ``iter_results`` / ``cleanup`` / ``generation``.
    Descriptor tuple:
    ``(slot, n_rows, n_cols, nonce, salt, qpu_us, generation)``. A trailing
    ``None`` on ``desc_q`` signals end-of-stream (driver exit).
    """
    ring = SampleView(**ring_args)
    ctx = None
    dropped = 0
    shutdown = False
    # Context construction can raise on D-Wave auth/topology errors; keep it
    # inside the try so the finally always sends the end-of-stream sentinel.
    try:
        factory = _resolve(stream_factory_dotted)
        ctx = factory(**_maybe_with_stop(factory, factory_kwargs, stop_event))
        if not _wait_for_first_switch(ctx, ctl_q, stop_event):
            return  # shutdown/stop before any round began
        # Outer loop: ``iter_results`` returns when the round ends OR when a
        # budget-exhaustion pause has drained the in-flight queue. On a paused
        # drain we idle on ctl_q (NOT exit) until the next switch resumes us;
        # the persistent D-Wave connection is kept alive throughout.
        while not stop_event.is_set() and not shutdown:
            for model, sampleset, submit_gen in ctx.iter_results():
                if stop_event.is_set():
                    break
                if _coalesce_ctl_q(ctx, ctl_q) == "shutdown":
                    shutdown = True
                    break
                # Discard completions from a superseded round (driver-side
                # generation filter — the first of the three independent
                # filters).
                if submit_gen != ctx.generation:
                    continue
                sample = np.asarray(sampleset.record.sample, dtype=np.int8)
                energy = np.asarray(sampleset.record.energy, dtype=np.float64)
                n_rows, n_cols = sample.shape
                if n_rows > ring.max_rows or n_cols > ring.max_cols:
                    log.warning(
                        "stream driver: dropping oversized sample %dx%d "
                        "(slot capacity %dx%d)",
                        n_rows, n_cols, ring.max_rows, ring.max_cols,
                    )
                    dropped += 1
                    continue
                slot = ring.claim_free(timeout=0.0 if dropped else 0.005)
                if slot is None:
                    dropped += 1
                    continue
                ring.write(slot, sample, energy)
                try:
                    _defect = None
                    _info = getattr(sampleset, "info", None)
                    if _info:
                        _defect = _info.get("defect_info")
                    desc_q.put_nowait(
                        (slot, n_rows, n_cols, bytes(model.nonce),
                         bytes(model.salt), _extract_qpu_us(sampleset),
                         ctx.generation,
                         pickle.dumps(_defect) if _defect is not None else None))
                except _queue.Full:
                    # Consumer backpressure: release the slot and drop.
                    ring.release(slot)
                    dropped += 1
            # iter_results returned. If stop/shutdown, fall through to cleanup;
            # otherwise the driver paused-and-drained (or the round ended) —
            # idle until the next switch (resume) or None (shutdown).
            if stop_event.is_set() or shutdown:
                break
            if not _wait_for_first_switch(ctx, ctl_q, stop_event):
                break  # shutdown/stop arrived while idle
    except Exception:
        log.exception("stream driver failed")
    finally:
        if dropped:
            log.warning("stream driver dropped %d samples (backpressure / "
                        "oversized)", dropped)
        try:
            if ctx is not None:
                ctx.cleanup()
        finally:
            try:
                desc_q.put(None, timeout=2.0)
            except Exception:
                pass
            ring.close_unlink()
