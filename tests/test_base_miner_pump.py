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

_FAKE_FACTORY = "tests.fakes.fake_stream:build_fake_production_stream"
_FAKE_INFINITE = "tests.fakes.fake_stream:build_fake_infinite_stream"

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
    """STREAMING_PUMP + DRIVER_OWNS_FEEDER miner driven by the fake factory.

    ``evaluate_sampleset`` never wins so ``mine_work_item`` runs until the
    stream ends or stop fires; the fake factory supplies real ndarrays into
    the ring so the consumer's zero-copy read path is exercised end to end.
    """

    STREAMING_PUMP = True
    DRIVER_OWNS_FEEDER = True
    STREAM_FACTORY_DOTTED = _FAKE_FACTORY
    RESULT_QUEUE_MAXSIZE = 4

    def __init__(self, factory: str = _FAKE_FACTORY) -> None:
        super().__init__("driver-test", sampler=object(), miner_type="QPU")
        self.queue_depth = 2
        self.time_manager = None
        self.STREAM_FACTORY_DOTTED = factory

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        # num_reads sizes the ring rows; the fake factory matches it.
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
        desc_q.put((slot, 4, 3, b"\x01" * 32, b"\x02" * 32, 51010))

        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
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
        desc_q.put((slot, 4, 3, b"\x01" * 32, b"\x02" * 32, 4242))
        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
        )
        assert acquired.action == _ACQUIRE_OK
        assert consumer.time_manager.recorded == [4242]
        acquired = None
        ring.release(slot)
    finally:
        ring.close_unlink()


# ----------------------------------------------------------------------
# Integration: mine_work_item spawns the driver, consumes, and tears down
# ----------------------------------------------------------------------


def test_mine_work_item_drives_stream_process_and_tears_down_cleanly():
    ctx = _streaming_context()
    miner = _DriverMiner()  # bounded fake (n=5)
    stop = mp.Event()

    # Bounded fake stream (5 results) → mine_work_item exits on its own
    # (no winner) once the driver enqueues the trailing None.
    result = miner.mine_work_item(ctx, stop)
    assert result is None
    # Driver process reaped, ring closed, no dangling handles.
    assert miner._ring is None
    assert miner._driver_stop is None
    # Consumed real shared-ring results (no exceptions leaked from the
    # _sample / _sample_batch asserts means the driver path was taken).
    assert miner.timing_stats["blocks_attempted"] >= 1


def test_mine_work_item_stops_promptly_on_stop_event():
    ctx = _streaming_context()
    # Infinite fake stream → runs until stop_event fires.
    miner = _DriverMiner(factory=_FAKE_INFINITE)
    stop = mp.Event()

    import threading

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
    import time
    time.sleep(0.5)
    stop.set()
    t.join(timeout=15.0)

    assert not t.is_alive(), "mine_work_item did not stop on stop_event"
    assert done.is_set()
    assert raised == [], f"mine_work_item raised: {raised[0] if raised else None}"
    assert miner._ring is None


def test_stream_driver_drops_under_backpressure_without_blocking():
    """A tiny ring forces the driver to drop when the consumer never reads.

    Drives stream_driver_main directly with the fake factory and a 1-slot
    ring; with no consumer releasing slots the driver must drop (rather than
    block) and still terminate, enqueueing the trailing None.
    """
    from QPU.stream_driver import stream_driver_main

    spawn = mp.get_context("spawn")
    ring = SharedSampleRing(slots=1, max_rows=8, max_cols=3)
    desc_q = spawn.Queue()
    stop = spawn.Event()
    proc = spawn.Process(
        target=stream_driver_main,
        args=(ring.attach_args(), desc_q, stop, _FAKE_FACTORY,
              {"num_reads": 8, "nodes": [0, 1, 2], "n": 50}),
        daemon=True,
    )
    proc.start()
    try:
        # Read exactly one descriptor (fills the single slot, never released)
        # then let the producer run into the full ring and drop the rest.
        first = desc_q.get(timeout=10.0)
        assert first is not None
        stop.set()
        # Drain whatever is queued until the trailing None; must not hang.
        saw_none = False
        for _ in range(100):
            item = desc_q.get(timeout=10.0)
            if item is None:
                saw_none = True
                break
        assert saw_none, "driver never enqueued end-of-stream None"
    finally:
        stop.set()
        assert terminate_join(proc, 5.0)
        ring.close_unlink()
