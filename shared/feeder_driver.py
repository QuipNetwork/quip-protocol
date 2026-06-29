# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Connection-less feeder driver (process A of the QPU submitter split).

Owns the ``RandomIsingFeeder`` (its gen workers run ``prepare_reduced`` —
defect-clamp + array reduction — off the submit path) and relays each reduced
problem into a shared-memory ``ProblemView`` ring for the isolated submitter
(``QPU.dwave_submitter``, process B). Holds NO D-Wave connection, so it imports
only ``shared`` modules — never ``QPU.*`` (which would pull the SDK via
``QPU/__init__``).

Startup handshake: B connects, derives the live topology (defects are known only
after the connection), and ships ``(defective_qubits, defective_edges,
live_nodes, live_edges)`` here over ``handshake_q`` (plain data — the ring's
shared free-list rides spawn inheritance, not a live queue). A then builds its
feeder with those ``prepare_reduced`` args so both sides agree on the reduced
layout.
"""
from __future__ import annotations

import logging
import pickle
import queue as _queue
from typing import Any, Optional, Tuple

import numpy as np

from shared.driver_util import (
    _coalesce_ctl_q,
    _start_parent_death_watchdog,
    _wait_for_first_switch,
)
from shared.logging_config import setup_child_process_logging
from shared.problem_prep import prepare_reduced
from shared.ring_views import ProblemView
from shared.stream_context import StreamContext

log = logging.getLogger(__name__)


def _await_handshake(handshake_q, stop_event) -> Optional[Tuple]:
    """Block (polling stop) for B's live-topology handshake; None on stop."""
    while not stop_event.is_set():
        try:
            return handshake_q.get(timeout=0.1)
        except _queue.Empty:
            continue
    return None


def _write_problem(pv: ProblemView, prob_desc_q, rp, generation: int,
                   stop_event) -> bool:
    """Pad the reduced arrays to the ring's full width, write, enqueue descriptor.

    The ``ProblemView`` is sized to the FULL topology (its free-list rides spawn
    inheritance, so it must be created before B knows the live count). The
    reduced live values occupy ``[0:K]`` in ``live_nodes``/``live_edges`` order;
    the submitter's ``rebuild_ising`` reads exactly those first K, so the zero
    pad is ignored. Returns False if the problem was dropped (stop, or consumer
    backpressure on the descriptor queue).
    """
    full_h = np.zeros(pv.n_nodes, dtype=np.float64)
    full_h[: rp.h_vec.shape[0]] = rp.h_vec
    full_j = np.zeros(pv.n_edges, dtype=np.float64)
    full_j[: rp.j_vec.shape[0]] = rp.j_vec

    slot = None
    while slot is None:
        if stop_event.is_set():
            return False
        slot = pv.claim_free(timeout=0.05)  # backpressure: wait for B to drain
    pv.write(slot, full_h, full_j)
    defect_pickle = (
        pickle.dumps(rp.defect_info) if rp.defect_info is not None else None
    )
    try:
        prob_desc_q.put_nowait(
            (slot, defect_pickle, bytes(rp.nonce), bytes(rp.salt), int(generation))
        )
    except _queue.Full:
        pv.release(slot)
        return False
    return True


def feeder_driver_main(
    prob_ring_args: dict,
    prob_desc_q: Any,
    ctl_q: Any,
    stop_event: Any,
    handshake_q: Any,
    nodes: list,
    edges: list,
    allowed_h: Any,
    feeder_buffer_size: int,
    log_queue: Any = None,
) -> None:
    """Long-lived feeder driver: relay reduced problems into the ProblemView ring.

    Mirrors ``stream_driver_main``'s ctl_q/generation/pause lifecycle but with no
    sampler: it drains the feeder (``StreamContext.iter_problems``) and writes
    each reduced problem to the ring. A ``None`` on ``prob_desc_q`` signals
    end-of-stream to the submitter on exit.
    """
    setup_child_process_logging(log_queue)
    _start_parent_death_watchdog(stop_event)
    pv = ProblemView(**prob_ring_args)  # non-owner attach (base_miner owns it)
    ctx: Optional[StreamContext] = None
    dropped = 0
    shutdown = False
    try:
        hs = _await_handshake(handshake_q, stop_event)
        if hs is None:
            return  # stop before B handed over the live topology
        dq, de, live_nodes, live_edges = hs
        ctx = StreamContext(
            sampler=None,
            nodes=nodes,
            edges=edges,
            allowed_h=allowed_h,
            feeder_buffer_size=feeder_buffer_size,
            num_reads=1,  # unused on the feeder-only path
            num_sweeps=0,
            feeder_prep_fn=prepare_reduced,
            feeder_prep_args=(dq, de, live_nodes, live_edges),
            stop_event=stop_event,
        )
        if not _wait_for_first_switch(ctx, ctl_q, stop_event):
            return
        while not stop_event.is_set() and not shutdown:
            for rp, generation in ctx.iter_problems():
                if stop_event.is_set():
                    break
                if _coalesce_ctl_q(ctx, ctl_q) == "shutdown":
                    shutdown = True
                    break
                if generation != ctx.generation:
                    continue  # round advanced while in hand — drop the stale one
                if not _write_problem(pv, prob_desc_q, rp, generation, stop_event):
                    dropped += 1
            if stop_event.is_set() or shutdown:
                break
            if not _wait_for_first_switch(ctx, ctl_q, stop_event):
                break  # shutdown/stop arrived while idle (paused-and-drained)
    except Exception:
        log.exception("feeder driver failed")
    finally:
        if dropped:
            log.warning("feeder driver dropped %d problems (backpressure)", dropped)
        try:
            if ctx is not None:
                ctx.cleanup()
        finally:
            try:
                prob_desc_q.put(None, timeout=2.0)  # end-of-stream to submitter
            except Exception:  # noqa: BLE001 — best-effort
                pass
            try:
                pv.close_unlink()  # non-owner: closes handles, base_miner unlinks
            except Exception:  # noqa: BLE001 — best-effort
                pass
