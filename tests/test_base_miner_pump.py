# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the process-model stream driver path in BaseMiner.

The in-worker pump thread was replaced by a stream-driver PROCESS that
writes samplesets into a SharedSampleRing and enqueues small descriptors
(see QPU/stream_driver.py). These tests exercise the consumer side
(``_acquire_result`` reading the ring, slot release, budget feed, clean
teardown) without contacting a QPU — a module-level fake factory stands in
for ``build_production_stream``.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from shared.allowed_value_spec import AllowedValueSet
from shared.base_miner import (
    BaseMiner,
    _ACQUIRE_DONE,
    _ACQUIRE_OK,
    _ACQUIRE_STOP,
    _SharedSampleSet,
)
from shared.proc_util import terminate_join
from shared.shared_sample_ring import SharedSampleRing
from substrate.types import SubstrateDifficulty, SubstrateMiningContext

_FAKE_CTX = "tests.fakes.fake_stream:build_fake_persistent_context"
_FAKE_RAISING = "tests.fakes.fake_stream:build_fake_raising_context"

_BIN_SPEC = AllowedValueSet((-1000, 1000))
_TER_SPEC = AllowedValueSet((-1000, 0, 1000))


def _streaming_context(nodes=(0, 1, 2)) -> SubstrateMiningContext:
    """Small synthetic substrate context for the driver-path tests."""
    nodes = list(nodes)
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return SubstrateMiningContext(
        last_proof_block_hash=b"\xab" * 32,
        topology_hash=b"\xcd" * 32,
        nodes=nodes,
        edges=edges,
        difficulty=SubstrateDifficulty(
            min_solutions=1,
            max_energy_milli=0,
            min_diversity_milli=0,
        ),
        miner_account_bytes=b"\x42" * 32,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
        block_hash=b"\x55" * 32,
        block_number=1,
    )


class _DriverMiner(BaseMiner):
    """STREAMING_PUMP + DRIVER_OWNS_FEEDER miner driven by the fake context."""

    STREAMING_PUMP = True
    DRIVER_OWNS_FEEDER = True
    STREAM_FACTORY_DOTTED = _FAKE_CTX
    RESULT_QUEUE_MAXSIZE = 4

    def __init__(self, factory: str = _FAKE_CTX) -> None:
        super().__init__("driver-test", sampler=object(), miner_type="QPU")
        self.queue_depth = 2
        self.time_manager = None
        self.STREAM_FACTORY_DOTTED = factory

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        return {
            "num_reads": 8,
            "annealing_time": 80.0,
            "energy_threshold": requirements.difficulty_energy,
        }

    def _sample(self, *a, **k):
        raise AssertionError("single-shot _sample must not run on driver path")

    def _sample_batch(self, *a, **k):
        raise AssertionError("_sample_batch must not run on driver path")

    def evaluate_sampleset(self, *args, **kwargs):
        return None  # never a winner


# ----------------------------------------------------------------------
# Consumer-side _acquire_result against a hand-built ring + descriptor queue
# ----------------------------------------------------------------------


class _RingConsumer(BaseMiner):
    """Bare consumer used to drive ``_acquire_result`` directly."""

    STREAMING_PUMP = True
    DRIVER_OWNS_FEEDER = True

    def __init__(self) -> None:
        super().__init__("ring-consumer", sampler=object(), miner_type="QPU")
        self.time_manager = None

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        return {"num_reads": 4}

    def _sample(self, *a, **k):
        raise AssertionError("no sample on driver path")

    def _sample_batch(self, *a, **k):
        raise AssertionError("no batch on driver path")


def _sample_ctx() -> dict:
    return {
        "prev_hash": b"\x00" * 32, "miner_id": "m", "cur_index": 0,
        "nodes": [0, 1, 2], "edges": [], "num_reads": 4, "num_sweeps": 1,
        "extra": {},
    }


def _noop() -> None:
    """Module-level target so spawn can pickle it (used for the dead-driver
    liveness test — the process exits immediately)."""
    return None


def test_acquire_result_dead_driver_without_sentinel_is_done():
    """A driver that died without enqueuing None must not hang the consumer.

    The cooperative end-of-stream path sends a trailing ``None`` from the
    driver's ``finally``; a hard crash (SIGKILL/OOM/C-extension abort) skips
    it. ``_acquire_result`` must detect the dead ``driver_proc`` and return
    DONE rather than draining an empty queue forever.
    """
    consumer = _RingConsumer()
    # Empty queue: no descriptor and no None sentinel ever arrives.
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()
    stop = mp.Event()
    dead = mp.get_context("spawn").Process(target=_noop)
    dead.start()
    dead.join(timeout=5.0)
    assert not dead.is_alive()
    try:
        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
            driver_proc=dead,
        )
        assert acquired.action == _ACQUIRE_DONE
    finally:
        terminate_join(dead, 2.0)


def test_ensure_driver_spawns_non_daemon_driver():
    """The persistent driver must be non-daemon (its feeder forks children)."""
    miner = _DriverMiner()
    sample_ctx = {
        "num_reads": 8, "nodes": [0, 1, 2], "edges": [],
        "annealing_time": 80.0, "energy_threshold": -1.0,
        "last_proof_block_hash": b"\xab" * 32, "miner_bytes": b"\x42" * 32,
    }
    try:
        assert miner._ensure_driver(sample_ctx) is True
        assert miner._driver_proc is not None
        assert miner._driver_proc.daemon is False
        # Idempotent: a second call reuses the same process.
        pid = miner._driver_proc.pid
        assert miner._ensure_driver(sample_ctx) is True
        assert miner._driver_proc.pid == pid
    finally:
        miner._close_driver()
    assert miner._ring is None
    assert miner._driver_proc is None


def test_acquire_result_reads_descriptor_from_ring():
    consumer = _RingConsumer()
    ring = SharedSampleRing(slots=2, max_rows=4, max_cols=3)
    consumer._ring = ring
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()
    stop = mp.Event()
    try:
        sample = np.ones((4, 3), np.int8)
        energy = np.array([-1.0, -2.0, -3.0, -4.0], np.float64)
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, sample, energy)
        desc_q.put((slot, 4, 3, b"\x01" * 32, b"\x02" * 32, 51010, 1))

        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
            generation=1,
        )
        assert acquired.action == _ACQUIRE_OK
        assert acquired.ring_slot == slot
        assert isinstance(acquired.sampleset, _SharedSampleSet)
        # Zero-copy view sees what the producer wrote.
        np.testing.assert_array_equal(acquired.sampleset.record.energy, energy)
        np.testing.assert_array_equal(acquired.sampleset.record.sample, sample)
        # QPU access time rode the descriptor and fed timing_stats.
        assert acquired.qpu_access_time_us == 51010
        assert consumer.timing_stats["qpu_access_time"][-1] == 51010
        # Release the slot before closing (drop the exported view first).
        acquired = None
        ring.release(slot)
    finally:
        ring.close_unlink()


def test_acquire_result_none_descriptor_is_done():
    consumer = _RingConsumer()
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()
    desc_q.put(None)
    stop = mp.Event()
    acquired = consumer._acquire_result(
        stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
    )
    assert acquired.action == _ACQUIRE_DONE


def test_acquire_result_stop_event_returns_stop():
    consumer = _RingConsumer()
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()  # left empty
    stop = mp.Event()
    stop.set()
    acquired = consumer._acquire_result(
        stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
    )
    assert acquired.action == _ACQUIRE_STOP


def test_acquire_result_feeds_budget_time_manager():
    class _Mgr:
        def __init__(self):
            self.recorded = []

        def record_block_time(self, us):
            self.recorded.append(us)

    consumer = _RingConsumer()
    consumer.time_manager = _Mgr()
    ring = SharedSampleRing(slots=2, max_rows=4, max_cols=3)
    consumer._ring = ring
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()
    stop = mp.Event()
    try:
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, np.ones((4, 3), np.int8),
                   np.zeros(4, np.float64))
        desc_q.put((slot, 4, 3, b"\x01" * 32, b"\x02" * 32, 4242, 1))
        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
            generation=1,
        )
        assert acquired.action == _ACQUIRE_OK
        assert consumer.time_manager.recorded == [4242]
        acquired = None
        ring.release(slot)
    finally:
        ring.close_unlink()


def test_acquire_result_drops_stale_generation_descriptor():
    """A descriptor from a superseded round is released + skipped, not OK."""
    consumer = _RingConsumer()
    ring = SharedSampleRing(slots=2, max_rows=4, max_cols=3)
    consumer._ring = ring
    desc_q = mp.get_context("spawn").Queue()
    stop = mp.Event()
    try:
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, np.ones((4, 3), np.int8), np.zeros(4, np.float64))
        # Stale gen=1 descriptor, then end-of-stream. Consumer wants gen=2.
        desc_q.put((slot, 4, 3, b"\x01" * 32, b"\x02" * 32, 0, 1))
        desc_q.put(None)
        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
            generation=2,
        )
        # Stale one skipped; stream then ended -> DONE.
        assert acquired.action == _ACQUIRE_DONE
        # The skipped slot was released back to the free-list (no leak): both
        # slots of the 2-slot ring are claimable again. (claim_free is a FIFO
        # so the reclaimed index need not equal ``slot``.)
        claimed = {ring.claim_free(timeout=1.0) for _ in range(2)}
        assert claimed == {0, 1}
        for s in claimed:
            ring.release(s)
    finally:
        ring.close_unlink()


# ----------------------------------------------------------------------
# Integration: mine_work_item spawns the driver, consumes, and tears down
# ----------------------------------------------------------------------


def test_mine_work_item_persists_driver_across_dispatches():
    """Two dispatches reuse ONE driver process (same pid, not respawned)."""
    ctx = _streaming_context()
    miner = _DriverMiner(factory=_FAKE_CTX)  # infinite stream (n=0)
    try:
        import threading
        import time as _t

        for _ in range(2):
            stop = mp.Event()
            done = threading.Event()

            def _run(s=stop, d=done):
                miner.mine_work_item(ctx, s)
                d.set()

            t = threading.Thread(target=_run)
            t.start()
            _t.sleep(0.4)
            pid = miner._driver_proc.pid
            stop.set()
            t.join(timeout=15.0)
            assert not t.is_alive()
            # Driver persists after a dispatch ends.
            assert miner._driver_proc is not None
            assert miner._driver_proc.is_alive()
            assert miner._driver_proc.pid == pid
            assert miner._ring is not None  # ring persists too
        # Two dispatches bumped the generation twice.
        assert miner._generation == 2
    finally:
        miner._close_driver()
    assert miner._ring is None
    assert miner._driver_proc is None


def test_mine_work_item_stops_promptly_on_stop_event():
    ctx = _streaming_context()
    miner = _DriverMiner(factory=_FAKE_CTX)  # infinite
    stop = mp.Event()

    import threading
    import time

    done = threading.Event()
    raised: list = []

    def _run():
        try:
            miner.mine_work_item(ctx, stop)
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=_run, name="mine-loop")
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=15.0)
    try:
        assert not t.is_alive(), "mine_work_item did not stop on stop_event"
        assert done.is_set()
        assert raised == [], f"mine_work_item raised: {raised[0] if raised else None}"
        # The driver is NOT torn down by a dispatch stop — it persists.
        assert miner._driver_proc is not None and miner._driver_proc.is_alive()
    finally:
        miner._close_driver()
