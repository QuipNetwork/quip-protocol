"""Tests for the isolated D-Wave submitter's input contract.

RingProblemFeeder is the seam between the producer's ProblemView shared-memory
ring and the submitter's streaming pump. These exercise the real ProblemView
round-trip (no D-Wave connection): arrays + generation + defect_info come
through intact, the slot is released back to the producer's free-list, and the
stop / end-of-stream paths raise StopIteration.
"""

from __future__ import annotations

import pickle
import queue
import threading

import numpy as np
import pytest

from QPU.dwave_submitter import RingProblemFeeder
from shared.problem_prep import DefectInfo, ReducedProblem
from shared.ring_views import ProblemView


def _pv():
    # Owner mode (names=None): creates shared memory + a free-list of [0, 1].
    return ProblemView(slots=2, n_nodes=3, n_edges=2)


def test_pop_blocking_round_trips_arrays_generation_and_defect():
    pv = _pv()
    try:
        h = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        j = np.array([1.0, -1.0], dtype=np.float64)
        slot = pv.claim_free(timeout=1.0)
        pv.write(slot, h, j)
        di = DefectInfo(fixed_spins={5: 1}, energy_offset=2.0, removed_edges={})
        q: queue.Queue = queue.Queue()
        q.put((slot, pickle.dumps(di), b"\x01" * 32, b"\x02" * 32, 7))

        feeder = RingProblemFeeder(pv, q)
        rp = feeder.pop_blocking()

        assert isinstance(rp, ReducedProblem)
        assert np.array_equal(rp.h_vec, h)
        assert np.array_equal(rp.j_vec, j)
        assert rp.nonce == b"\x01" * 32
        assert rp.salt == b"\x02" * 32
        assert rp.generation == 7
        assert rp.defect_info.fixed_spins == {5: 1}
        assert rp.defect_info.energy_offset == 2.0

        # Slot was released back to the producer free-list: both slots claimable
        # again, a third claim times out (nothing left).
        assert pv.claim_free(timeout=0.5) is not None
        assert pv.claim_free(timeout=0.5) is not None
        assert pv.claim_free(timeout=0.05) is None
    finally:
        pv.close_unlink()


def test_pop_blocking_none_defect_yields_none():
    pv = _pv()
    try:
        slot = pv.claim_free(timeout=1.0)
        pv.write(slot, np.zeros(3), np.zeros(2))
        q: queue.Queue = queue.Queue()
        q.put((slot, None, b"\x00" * 32, b"\x00" * 32, 0))
        rp = RingProblemFeeder(pv, q).pop_blocking()
        assert rp.defect_info is None
    finally:
        pv.close_unlink()


def test_end_of_stream_sentinel_raises_stopiteration():
    pv = _pv()
    try:
        q: queue.Queue = queue.Queue()
        q.put(None)
        feeder = RingProblemFeeder(pv, q)
        with pytest.raises(StopIteration):
            feeder.pop_blocking()
        assert feeder._exhausted
    finally:
        pv.close_unlink()


def test_stop_event_raises_stopiteration_without_consuming():
    pv = _pv()
    try:
        ev = threading.Event()
        ev.set()
        q: queue.Queue = queue.Queue()
        feeder = RingProblemFeeder(pv, q, stop_event=ev)
        with pytest.raises(StopIteration):
            feeder.pop_blocking()
    finally:
        pv.close_unlink()


def test_stats_surface_matches_feeder_contract():
    pv = _pv()
    try:
        stats = RingProblemFeeder(pv, queue.Queue()).stats()
        # The pump's diagnostic reads these keys off feeder.stats().
        for key in ("ready", "buffer_size", "drained_count", "pop_wait_total_s"):
            assert key in stats
    finally:
        pv.close_unlink()
