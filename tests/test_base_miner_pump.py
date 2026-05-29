# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the out-of-band result pump in BaseMiner."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time

import dimod

from shared.allowed_value_spec import AllowedValueSet
from shared.base_miner import BaseMiner, _PUMP_DONE, _PumpedResult
from substrate.types import SubstrateDifficulty, SubstrateMiningContext


class _FakeStreamMiner(BaseMiner):
    """Minimal concrete miner whose _sample_batch yields fast fake results."""

    STREAMING_PUMP = True

    def __init__(self, n_results: int) -> None:
        super().__init__("pump-test", sampler=object(), miner_type="QPU")
        self._n = n_results
        self._produced = 0

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        return {"num_reads": 1, "num_sweeps": 1}

    def _sample(self, h, J, *, num_reads, num_sweeps, **kwargs):
        raise AssertionError("single-shot _sample must not be called in pump mode")

    def _sample_batch(self, prev_hash, miner_id, cur_index, nodes, edges,
                      *, num_reads, num_sweeps, **kwargs):
        if self._produced >= self._n:
            return None
        self._produced += 1
        # Fake "sampleset" stand-in; the pump only forwards it.
        return [(self._produced, b"\x00" * 32, f"ss{self._produced}")]


class _BlockingStreamMiner(BaseMiner):
    """Concrete miner whose _sample_batch blocks briefly and never ends."""

    STREAMING_PUMP = True

    def __init__(self) -> None:
        super().__init__("pump-block", sampler=object(), miner_type="QPU")
        self._produced = 0

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        return {"num_reads": 1, "num_sweeps": 1}

    def _sample(self, h, J, *, num_reads, num_sweeps, **kwargs):
        raise AssertionError("single-shot _sample must not be called in pump mode")

    def _sample_batch(self, prev_hash, miner_id, cur_index, nodes, edges,
                      *, num_reads, num_sweeps, **kwargs):
        time.sleep(0.01)  # simulate per-call QPU latency
        self._produced += 1
        return [(self._produced, b"\x00" * 32, f"ss{self._produced}")]


class _RaisingStreamMiner(BaseMiner):
    """Concrete miner whose _sample_batch raises on the first call."""

    STREAMING_PUMP = True

    def __init__(self) -> None:
        super().__init__("pump-raise", sampler=object(), miner_type="QPU")

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        return {"num_reads": 1, "num_sweeps": 1}

    def _sample(self, h, J, *, num_reads, num_sweeps, **kwargs):
        raise AssertionError("single-shot _sample must not be called in pump mode")

    def _sample_batch(self, prev_hash, miner_id, cur_index, nodes, edges,
                      *, num_reads, num_sweeps, **kwargs):
        raise RuntimeError("qpu blip")


def _sample_kwargs() -> dict:
    return {
        "prev_hash": b"\x00" * 32, "miner_id": "m", "cur_index": 0,
        "nodes": [0], "edges": [], "num_reads": 1, "num_sweeps": 1,
        "extra": {},
    }


def test_pump_drains_all_results_into_queue():
    miner = _FakeStreamMiner(n_results=5)
    q: "queue.Queue" = queue.Queue(maxsize=10)
    pump_stop = threading.Event()
    t = threading.Thread(
        target=miner._result_pump, args=(q, pump_stop, _sample_kwargs()),
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "pump did not terminate when stream exhausted"

    drained = []
    while True:
        item = q.get_nowait()
        if item is _PUMP_DONE:
            break
        drained.append(item)
    assert len(drained) == 5
    assert all(isinstance(r, _PumpedResult) for r in drained)
    assert [r.sampleset for r in drained] == [f"ss{i}" for i in range(1, 6)]


def test_pump_drops_newest_on_full_queue_and_counts():
    miner = _FakeStreamMiner(n_results=20)
    q: "queue.Queue" = queue.Queue(maxsize=2)  # tiny: force drops
    pump_stop = threading.Event()
    t = threading.Thread(
        target=miner._result_pump, args=(q, pump_stop, _sample_kwargs()),
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    # 20 produced, queue holds 2. The first 2 fill the queue; the next 18
    # are dropped on put. At exit the sentinel push finds the queue full,
    # evicts one buffered result (counted) to make room. So:
    #   18 (put-time drops) + 1 (sentinel-fallback eviction) == 19.
    assert miner._dropped_results == 19


def test_pump_exits_when_stop_signalled():
    miner = _BlockingStreamMiner()
    q: "queue.Queue" = queue.Queue(maxsize=10)
    pump_stop = threading.Event()
    t = threading.Thread(
        target=miner._result_pump, args=(q, pump_stop, _sample_kwargs()),
    )
    t.start()
    time.sleep(0.05)  # let a few iterations run
    pump_stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "pump did not terminate after pump_stop.set()"
    # Sentinel must always land so the consumer's blocking get() unblocks.
    items = []
    while True:
        item = q.get_nowait()
        items.append(item)
        if item is _PUMP_DONE:
            break
    assert items[-1] is _PUMP_DONE


def test_pump_pushes_sentinel_on_sampler_exception(caplog):
    miner = _RaisingStreamMiner()
    q: "queue.Queue" = queue.Queue(maxsize=10)
    pump_stop = threading.Event()
    with caplog.at_level(logging.ERROR):
        t = threading.Thread(
            target=miner._result_pump, args=(q, pump_stop, _sample_kwargs()),
        )
        t.start()
        t.join(timeout=2.0)
    assert not t.is_alive(), "pump did not terminate after sampler exception"
    # Sentinel must still be pushed so a waiting consumer can't deadlock.
    assert q.get_nowait() is _PUMP_DONE
    assert q.empty()
    assert any("result pump error" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# Integration: mine_work_item consumes pump results in STREAMING_PUMP mode
# ----------------------------------------------------------------------


_BIN_SPEC = AllowedValueSet((-1000, 1000))
_TER_SPEC = AllowedValueSet((-1000, 0, 1000))


def _streaming_context() -> SubstrateMiningContext:
    """Small synthetic substrate context for the pump integration test.

    A 3-node / 2-edge topology keeps ``make_feeder`` cheap; the fake
    miner's ``_sample_batch`` doesn't actually use the feeder models.
    """
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
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


def _fake_sampleset(nodes):
    """A real dimod.SampleSet over the given nodes (all-ones spin)."""
    return dimod.SampleSet.from_samples(
        {n: 1 for n in nodes}, vartype=dimod.SPIN, energy=0.0,
    )


class _FastProducerSlowConsumerMiner(BaseMiner):
    """STREAMING_PUMP miner: fast batch producer, deliberately slow eval.

    ``_sample_batch`` returns instantly so the pump thread outruns the
    consumer (whose ``evaluate_sampleset`` sleeps ~20ms and never wins).
    With a small RESULT_QUEUE_MAXSIZE this forces backpressure drops.
    """

    STREAMING_PUMP = True
    RESULT_QUEUE_MAXSIZE = 2

    def __init__(self, nodes) -> None:
        super().__init__("pump-integ", sampler=object(), miner_type="QPU")
        self._nodes = nodes
        self._produced = 0

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        return {"num_reads": 1, "num_sweeps": 1}

    def _sample(self, h, J, *, num_reads, num_sweeps, **kwargs):
        raise AssertionError("single-shot _sample must not run in pump mode")

    def _sample_batch(self, prev_hash, miner_id, cur_index, nodes, edges,
                      *, num_reads, num_sweeps, **kwargs):
        # ~1ms per "QPU call": fast enough to outrun the 20ms consumer
        # (so the bounded queue fills and drops) without spinning a CPU
        # core flat-out and starving the consumer/stop check.
        time.sleep(0.001)
        self._produced += 1
        return [(self._produced, b"\x07" * 32, _fake_sampleset(self._nodes))]

    def evaluate_sampleset(self, *args, **kwargs):
        time.sleep(0.02)  # slow consumer: outpaced by the fast pump
        return None       # never a winner


def test_mine_work_item_streams_results_and_drops_under_backpressure():
    ctx = _streaming_context()
    miner = _FastProducerSlowConsumerMiner(ctx.nodes)
    stop = mp.Event()

    done = threading.Event()

    def _run():
        try:
            miner.mine_work_item(ctx, stop)
        finally:
            done.set()

    t = threading.Thread(target=_run, name="mine-loop")
    t.start()
    # Let the fast pump outrun the slow consumer so the bounded queue
    # fills and the pump starts dropping the newest results.
    time.sleep(0.5)
    stop.set()
    t.join(timeout=10.0)

    assert not t.is_alive(), "mine_work_item did not stop on stop_event"
    assert done.is_set()
    # Fast producer (instant batch) outran the 20ms-per-result consumer
    # with a depth-2 queue: the pump must have dropped results rather
    # than stalling lockstep with the consumer.
    assert miner._dropped_results > 0, (
        "expected backpressure drops; pump appears to be running in "
        "lockstep with the consumer instead of streaming ahead"
    )
    # The single-shot path must never run when STREAMING_PUMP is on.
    # (_sample raises if called — covered by the AssertionError above.)
