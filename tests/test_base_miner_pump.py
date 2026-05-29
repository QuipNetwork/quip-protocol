# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the out-of-band result pump in BaseMiner."""
from __future__ import annotations

import logging
import queue
import threading
import time

from shared.base_miner import BaseMiner, _PUMP_DONE, _PumpedResult


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
