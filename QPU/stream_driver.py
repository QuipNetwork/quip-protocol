# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Stream-driver process: owns the sampler/feeder/generator and produces
samplesets into a SampleView for the consumer (miner worker).

Used by the CPU/GPU backends (and the legacy single-process QPU path). The
generator is driven by exactly one caller (this process's loop). Only a small
descriptor crosses the descriptor queue; the sample matrix is written zero-copy
into the ring.

The QPU backend instead splits this across two processes — a connection-less
feeder driver (``shared.feeder_driver``) and an isolated submitter
(``QPU.dwave_submitter``) — so the SDK's GIL-held encode can't contend with
feeder-harvest + decode. The generic helpers both paths share live in
``shared.driver_util``; they are re-exported here for back-compat with existing
importers (tools + tests).
"""
from __future__ import annotations

import logging
import pickle
import queue as _queue
from typing import Any, Dict

import numpy as np

from shared.driver_util import (  # noqa: F401  (re-exported for back-compat)
    _coalesce_ctl_q,
    _extract_qpu_us,
    _maybe_with_stop,
    _parent_death_watchdog,
    _resolve,
    _start_parent_death_watchdog,
    _wait_for_first_switch,
    qpu_access_time_us,
)
from shared.logging_config import setup_child_process_logging
from shared.ring_views import SampleView

log = logging.getLogger(__name__)


def stream_driver_main(ring_args: Dict[str, Any], desc_q, ctl_q, stop_event,
                       stream_factory_dotted: str,
                       factory_kwargs: Dict[str, Any],
                       log_queue: Any = None) -> None:
    """Long-lived stream driver: persist the context, switch rounds via ctl_q.

    ``stream_factory_dotted`` resolves to a context factory
    (``build_persistent_context``) returning an object exposing
    ``apply_command`` / ``iter_results`` / ``cleanup`` / ``generation``.
    Descriptor tuple:
    ``(slot, n_rows, n_cols, nonce, salt, qpu_us, generation, defect_pickle)``.
    A trailing ``None`` on ``desc_q`` signals end-of-stream (driver exit).

    ``log_queue`` is the shared queue drained by ``log_writer_main``; this is a
    spawned process with no inherited handlers, so without configuring it here
    every producer-side INFO diagnostic (stream depth, feeder pop-wait) is
    silently dropped.
    """
    setup_child_process_logging(log_queue)
    # Orphan guard: if the parent dies ungracefully, stop instead of burning
    # QPU forever as a reparented orphan.
    _start_parent_death_watchdog(stop_event)
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
                # Always give the claim a real wait budget. This was
                # ``timeout=0.0 if dropped else 0.005``, but ``dropped`` is the
                # cumulative counter, so a single drop disabled waiting for the
                # life of the process and every later momentary ring-full was
                # discarded instantly. A dropped sample never reaches the
                # worker and is never evaluated, so a win it carried is lost
                # silently (QUI-867).
                slot = ring.claim_free(timeout=0.005)
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
