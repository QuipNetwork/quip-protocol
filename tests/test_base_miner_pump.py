# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the out-of-band result pump in BaseMiner."""
from __future__ import annotations

import queue
import threading
import time

from shared.base_miner import BaseMiner, _PumpedResult


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


def test_pump_drains_all_results_into_queue():
    miner = _FakeStreamMiner(n_results=5)
    q: "queue.Queue" = queue.Queue(maxsize=10)
    pump_stop = threading.Event()
    sample_kwargs = {
        "prev_hash": b"\x00" * 32, "miner_id": "m", "cur_index": 0,
        "nodes": [0], "edges": [], "num_reads": 1, "num_sweeps": 1,
        "extra": {},
    }
    t = threading.Thread(
        target=miner._result_pump, args=(q, pump_stop, sample_kwargs),
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "pump did not terminate when stream exhausted"

    drained = []
    while True:
        item = q.get_nowait()
        if item is BaseMiner._PUMP_DONE:
            break
        drained.append(item)
    assert len(drained) == 5
    assert all(isinstance(r, _PumpedResult) for r in drained)
    assert [r.sampleset for r in drained] == [f"ss{i}" for i in range(1, 6)]


def test_pump_drops_newest_on_full_queue_and_counts():
    miner = _FakeStreamMiner(n_results=20)
    q: "queue.Queue" = queue.Queue(maxsize=2)  # tiny: force drops
    pump_stop = threading.Event()
    sample_kwargs = {
        "prev_hash": b"\x00" * 32, "miner_id": "m", "cur_index": 0,
        "nodes": [0], "edges": [], "num_reads": 1, "num_sweeps": 1,
        "extra": {},
    }
    t = threading.Thread(
        target=miner._result_pump, args=(q, pump_stop, sample_kwargs),
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    # 20 produced, queue holds at most 2 + sentinel; the rest are dropped.
    assert miner._dropped_results >= 1
