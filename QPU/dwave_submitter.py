# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Isolated D-Wave submitter process: owns the keys/connection, runs sample_bqm.

Profiling showed the streaming bottleneck was the SDK's GIL-held problem encode
inside ``sample_bqm`` (``submit_mean`` ballooned to ~11 s) contending — in one
process — with feeder-harvest, result decode, and ring writes. Splitting the
submit onto its own process removes that contention: this module's process does
*only* connection + submit + collect + decode, while the feeder/driver process
generates and reduces problems and ships them in over a shared-memory ring.

Topology (all shared-memory, zero-copy bulk; small metadata on queues):

    feeder/driver proc                     dwave_submitter proc (this module)
      RandomIsingFeeder workers              owns DWaveSamplerWrapper (connection)
      reduce_to_arrays -> ProblemView  ──►   RingProblemFeeder.pop_blocking
      (slot, meta) -> prob_desc_q            sample_ising_streaming (rebuild+sample_bqm)
                          SampleView ◄──     write results + result_desc_q -> consumer

The reduced ``(h, J)`` arrays ride a ``ProblemView`` ring (the producer claims a
slot + writes, this process reads + releases it back to the shared free-list).
Per-problem metadata (defect_info, nonce, salt, generation) rides
``prob_desc_q``; a ``None`` sentinel is end-of-stream.
"""
from __future__ import annotations

import logging
import pickle
import queue as _queue
from typing import Any, Dict, Optional

import numpy as np

from shared.driver_util import (
    _extract_qpu_us,
    _resolve,
    _start_parent_death_watchdog,
)
from shared.logging_config import setup_child_process_logging
from shared.problem_prep import ReducedProblem
from shared.ring_views import ProblemView, SampleView

log = logging.getLogger(__name__)


class RingProblemFeeder:
    """Feeder-shaped reader over a ``ProblemView`` ring + descriptor queue.

    A drop-in ``models`` source for
    :meth:`QPU.dwave_sampler.DWaveSamplerWrapper.sample_ising_streaming`: each
    ``pop_blocking`` reads the next descriptor, copies the slot's ``(h, J)``
    arrays into a :class:`~shared.problem_prep.ReducedProblem` stamped with the
    descriptor's generation, releases the slot back to the producer's free-list,
    and returns it. Raises ``StopIteration`` on the end-of-stream sentinel or
    when ``stop_event`` is set.

    The arrays are copied out of shared memory (``np.array``) before the slot is
    released so the producer can immediately reuse the slot while this problem is
    still in flight at the QPU — a 368 KB copy local to this process, far cheaper
    than holding a slot for the full QPU round-trip.
    """

    def __init__(
        self,
        problem_view: ProblemView,
        prob_desc_q: Any,
        *,
        stop_event: Any = None,
        poll_s: float = 0.1,
    ) -> None:
        self._pv = problem_view
        self._q = prob_desc_q
        self._stop_event = stop_event
        self._poll_s = poll_s
        self._popped = 0
        self._exhausted = False

    def _stopped(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def pop_blocking(self) -> ReducedProblem:
        """Return the next reduced problem; raise StopIteration when drained."""
        while True:
            if self._stopped():
                raise StopIteration
            try:
                desc = self._q.get(timeout=self._poll_s)
            except _queue.Empty:
                continue
            if desc is None:
                self._exhausted = True
                raise StopIteration
            slot, defect_pickle, nonce, salt, generation = desc
            h_view, j_view = self._pv.read(slot)
            rp = ReducedProblem(
                h_vec=np.array(h_view, dtype=np.float64),
                j_vec=np.array(j_view, dtype=np.float64),
                # defect_pickle is produced by OUR feeder/driver process and
                # crosses an internal mp.Queue (same trust boundary as the
                # existing stream_driver -> base_miner DefectInfo descriptor);
                # not untrusted input.
                defect_info=(
                    pickle.loads(defect_pickle) if defect_pickle is not None
                    else None
                ),
                nonce=nonce,
                salt=salt,
                generation=int(generation),
            )
            self._pv.release(slot)
            self._popped += 1
            return rp

    def __iter__(self) -> "RingProblemFeeder":
        return self

    def __next__(self) -> ReducedProblem:
        return self.pop_blocking()

    def stats(self) -> dict:
        """Match the RandomIsingFeeder stats surface the pump's diagnostic reads.

        The ring has no local ready/pending buffer (depth lives in the producer
        and the descriptor queue), so report zeros — the submitter's depth is
        already surfaced by the pump's own ``[QPU] stream depth`` line.
        """
        return {
            "ready": 0,
            "pending": 0,
            "buffer_size": 0,
            "drained_count": 0,
            "pop_wait_total_s": 0.0,
        }

    def stop(self) -> None:
        """API parity with RandomIsingFeeder; nothing to tear down here."""
        self._exhausted = True


def _write_result(
    ring: SampleView,
    result_desc_q: Any,
    model: ReducedProblem,
    sampleset: Any,
    *,
    dropped: int,
) -> int:
    """Write one sampleset to the SampleView ring + descriptor queue.

    Mirrors the stream-driver's result path. Returns the number of additional
    drops (0 or 1) so the caller can accumulate a drop counter. The sample
    matrix is written zero-copy into the ring; only a small descriptor —
    including the per-problem generation and pickled defect_info — crosses the
    queue.
    """
    sample = np.asarray(sampleset.record.sample, dtype=np.int8)
    energy = np.asarray(sampleset.record.energy, dtype=np.float64)
    n_rows, n_cols = sample.shape
    if n_rows > ring.max_rows or n_cols > ring.max_cols:
        log.warning(
            "dwave submitter: dropping oversized sample %dx%d (cap %dx%d)",
            n_rows, n_cols, ring.max_rows, ring.max_cols,
        )
        return 1
    slot = ring.claim_free(timeout=0.0 if dropped else 0.005)
    if slot is None:
        return 1
    ring.write(slot, sample, energy)
    defect = None
    info = getattr(sampleset, "info", None)
    if info:
        defect = info.get("defect_info")
    try:
        result_desc_q.put_nowait((
            slot, n_rows, n_cols, bytes(model.nonce), bytes(model.salt),
            _extract_qpu_us(sampleset), int(model.generation),
            pickle.dumps(defect) if defect is not None else None,
        ))
    except _queue.Full:
        ring.release(slot)
        return 1
    return 0


def dwave_submitter_main(
    sampler_factory_dotted: str,
    factory_kwargs: Dict[str, Any],
    prob_ring_args: Dict[str, Any],
    prob_desc_q: Any,
    sample_ring_args: Dict[str, Any],
    result_desc_q: Any,
    handshake_q: Any,
    stop_event: Any,
    num_reads: int,
    queue_depth: int,
    annealing_time: Optional[float] = None,
    log_queue: Any = None,
) -> None:
    """Long-lived submitter: own the connection, submit ring problems, emit results.

    ``sampler_factory_dotted`` resolves to a zero-arg-ish factory (given
    ``factory_kwargs``) returning a connected object exposing
    ``sample_ising_streaming`` + ``live_nodes``/``live_edges`` +
    ``cleanup``/``close`` (i.e. a :class:`~QPU.dwave_sampler.DWaveSamplerWrapper`
    or a thin wrapper around one). The submitter attaches the producer's
    ``ProblemView`` ring (read, non-owner) and the ``SampleView`` ring (write,
    non-owner — base_miner owns both), runs the streaming submit over a
    :class:`RingProblemFeeder`, and writes each result to the consumer.

    On connect it derives the live topology (defects known only after the
    connection) and ships it to the feeder-driver over ``handshake_q`` — plain
    lists/sets only (an mp.Queue can't ride a live Queue.put, so the ring's
    shared free-list rides spawn inheritance, not this handshake). A ``None`` on
    ``result_desc_q`` signals end-of-stream on exit.
    """
    setup_child_process_logging(log_queue)
    _start_parent_death_watchdog(stop_event)

    sampler = None
    ring = SampleView(**sample_ring_args)
    pv = ProblemView(**prob_ring_args)
    feeder = RingProblemFeeder(pv, prob_desc_q, stop_event=stop_event)
    dropped = 0
    try:
        factory = _resolve(sampler_factory_dotted)
        sampler = factory(**factory_kwargs)
        # Handshake: hand the feeder-driver the defect set + live ordering so it
        # reduces problems into the same layout this side rebuilds. Plain data
        # only (no mp.Queue / shm handle).
        handshake_q.put((
            list(sampler._defective_qubits),
            set(sampler._defective_edges),
            list(sampler.live_nodes),
            list(sampler.live_edges),
        ))
        stream = sampler.sample_ising_streaming(
            feeder,
            num_reads=num_reads,
            queue_depth=queue_depth,
            annealing_time=annealing_time,
            stop_event=stop_event,
        )
        for model, sampleset in stream:
            if stop_event.is_set():
                break
            dropped += _write_result(
                ring, result_desc_q, model, sampleset, dropped=dropped,
            )
    except Exception:
        log.exception("dwave submitter failed")
    finally:
        if dropped:
            log.warning("dwave submitter dropped %d samples", dropped)
        try:
            if sampler is not None:
                close = getattr(sampler, "cleanup", None) or getattr(
                    sampler, "close", None
                )
                if callable(close):
                    close()
        finally:
            try:
                result_desc_q.put(None, timeout=2.0)
            except Exception:  # noqa: BLE001 — best-effort end-of-stream
                pass
            ring.close_unlink()
            try:
                pv.close()
            except Exception:  # noqa: BLE001 — non-owner attach; best-effort
                pass
