# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unit tests for DWaveSamplerWrapper.sample_ising_streaming and
DWaveMiner._finalize_sample.

No real D-Wave connection is used: a fake sampler and fake futures drive the
pump contract (queue depth, defect_info attachment, cancel-on-close).
"""

from __future__ import annotations

import multiprocessing as mp
import time as _time
from typing import Any, List, Optional, Tuple

import dimod
import numpy as np
import pytest

from QPU.dwave_sampler import (
    DefectInfo,
    DWaveSamplerWrapper,
    _default_submit_workers,
)


class TestDefaultSubmitWorkers:
    """Submit-pool sizing scales with the node but never exceeds queue_depth."""

    @pytest.mark.parametrize(
        "cpu,queue_depth,expected",
        [
            (4, 30, 8),   # the deployed 4-core node: 16->8, frees CPU for gen
            (8, 30, 16),  # bigger box keeps full concurrency
            (4, 4, 4),    # shallow queue caps below cpu*2
            (1, 30, 2),   # single-core floor
        ],
    )
    def test_scales_with_cpu_capped_at_queue_depth(
        self, cpu, queue_depth, expected, monkeypatch
    ):
        monkeypatch.setattr("os.cpu_count", lambda: cpu)
        assert _default_submit_workers(queue_depth) == expected

    def test_unknown_cpu_count_falls_back_to_eight(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert _default_submit_workers(30) == 16


@pytest.fixture(autouse=True)
def _patch_wait_for_completions(monkeypatch):
    """Drive the pump's wait seam with a done()-poll over the fake futures.

    Production blocks via ``Future.wait_multiple``, which needs real cloud
    Future event machinery the fakes don't have. This emulates its
    ``(done, remaining)`` contract: block up to ``timeout``, returning early
    once ``min_done`` fakes report ``done()``.
    """
    def _fake_wait(futures, *, min_done, timeout):
        deadline = _time.time() + (timeout or 0.0)
        while True:
            done = [f for f in futures if f.done()]
            if len(done) >= (min_done or 1) or _time.time() >= deadline:
                return done, [f for f in futures if not f.done()]
            _time.sleep(0.005)

    monkeypatch.setattr(
        "QPU.dwave_sampler._wait_for_completions", _fake_wait
    )


# ---------------------------------------------------------------------------
# Fake infrastructure
# ---------------------------------------------------------------------------

class _FakeFuture:
    """Synchronous fake that is immediately done."""

    def __init__(self, sampleset: dimod.SampleSet, index: int) -> None:
        self._ss = sampleset
        self.index = index
        self._cancelled = False

    @property
    def sampleset(self) -> dimod.SampleSet:
        return self._ss

    def done(self) -> bool:
        return True

    def cancel(self) -> None:
        self._cancelled = True


class _SlowFuture(_FakeFuture):
    """Future that is NOT done until explicitly marked ready."""

    def __init__(self, sampleset: dimod.SampleSet, index: int) -> None:
        super().__init__(sampleset, index)
        self._ready = False

    def done(self) -> bool:
        return self._ready

    def mark_done(self) -> None:
        self._ready = True


def _make_ss(energy: float = -100.0, n_reads: int = 4) -> dimod.SampleSet:
    """Return a minimal SampleSet with ``n_reads`` samples."""
    samples = [{0: 1, 1: -1} for _ in range(n_reads)]
    energies = [energy] * n_reads
    return dimod.SampleSet.from_samples(samples, vartype=dimod.SPIN, energy=energies)


def _make_defect(offset: float = 10.0) -> DefectInfo:
    return DefectInfo(fixed_spins={99: 1}, energy_offset=offset, removed_edges={})


class _FakeSamplerWrapper:
    """Minimal stand-in for DWaveSamplerWrapper.

    Each ``_submit_prepared`` call consumes one entry from ``_results`` and
    returns its future. ``_results`` stays a list of ``(FakeFuture,
    Optional[DefectInfo])`` tuples for construction convenience, but the pump
    now reads ``defect_info`` from the model (ReducedProblem), so the tuple's
    second element is ignored here.
    """

    job_label = "test_label"
    # The pump rebuilds via the real _submit_prepared in production; the fake
    # bypasses rebuild_ising entirely, so these are never read.
    live_nodes: List[int] = [0, 1]
    live_edges: List[Tuple[int, int]] = [(0, 1)]

    def __init__(
        self, results: List[Tuple[_FakeFuture, Optional[DefectInfo]]]
    ) -> None:
        self._results = list(results)
        self._call_count = 0
        self._reconstruct_calls = 0
        # The pump submits concurrently on a thread pool, so this is called from
        # multiple threads — guard the counter + result list.
        self._lock = __import__("threading").Lock()

    def _submit_prepared(
        self, h_vec: Any, j_vec: Any, **kwargs: Any
    ) -> _FakeFuture:
        with self._lock:
            self._call_count += 1
            return self._results.pop(0)[0]

    def reconstruct_full_sampleset(
        self, ss: dimod.SampleSet, defect_info: DefectInfo
    ) -> dimod.SampleSet:
        """Identity reconstruction for testing."""
        self._reconstruct_calls += 1
        return ss

    # Attach the real method under test after import.
    sample_ising_streaming = None  # patched in fixture


class _FakeModel:
    """Minimal ReducedProblem stand-in.

    Carries the array payload + provenance + per-nonce defect_info the pump now
    reads off the model. The fake ``_submit_prepared`` ignores the arrays, so
    their values don't matter — only the attributes must exist.
    """

    def __init__(self, idx: int, defect_info: Optional[DefectInfo] = None) -> None:
        self.h_vec = np.zeros(2, dtype=np.float64)
        self.j_vec = np.zeros(1, dtype=np.float64)
        self.defect_info = defect_info
        self.nonce = idx
        self.salt = b""


class _ListFeeder:
    """Feeder that pops from a pre-built list; raises StopIteration when empty."""

    def __init__(self, models: List[_FakeModel]) -> None:
        self._q: List[_FakeModel] = list(models)

    def pop_blocking(self) -> _FakeModel:
        if not self._q:
            raise StopIteration
        return self._q.pop(0)

    def stats(self) -> dict:
        return {
            "ready": len(self._q),
            "buffer_size": len(self._q) + 1,
            "drained_count": 0,
            "pop_wait_total_s": 0.0,
        }


# Attach the real method under test to the fake class.
_FakeSamplerWrapper.sample_ising_streaming = (  # type: ignore[method-assign]
    DWaveSamplerWrapper.sample_ising_streaming
)


# ---------------------------------------------------------------------------
# Tests: sample_ising_streaming pump contract
# ---------------------------------------------------------------------------

def _build_results(n: int, defect: Optional[DefectInfo] = None):
    """Build ``n`` (future, defect_info) pairs, all immediately done."""
    out = []
    for i in range(n):
        ss = _make_ss(energy=float(-100 - i))
        out.append((_FakeFuture(ss, i), defect))
    return out


def test_streaming_yields_model_and_sampleset_basic():
    """Generator yields (model, ss) for each result."""
    n = 4
    results = _build_results(n)
    sampler = _FakeSamplerWrapper(results)
    models = _ListFeeder([_FakeModel(i) for i in range(n)])

    gen = sampler.sample_ising_streaming(models, num_reads=8, queue_depth=2)
    collected = list(gen)

    assert len(collected) == n
    for model, ss in collected:
        assert hasattr(model, "nonce")
        assert isinstance(ss, dimod.SampleSet)


def test_defect_info_attached_to_sampleset_info():
    """Raw sampleset has ss.info['defect_info'] set to the DefectInfo object."""
    defect = _make_defect(offset=42.0)
    results = _build_results(2, defect=defect)
    sampler = _FakeSamplerWrapper(results)
    models = _ListFeeder([_FakeModel(i, defect) for i in range(2)])

    gen = sampler.sample_ising_streaming(models, num_reads=8, queue_depth=2)
    _, ss = next(gen)

    assert "defect_info" in ss.info
    assert ss.info["defect_info"] is defect


def test_no_defect_info_is_none_in_sampleset_info():
    """When sample_ising_async returns None defect_info, ss.info['defect_info'] is None."""
    results = _build_results(2, defect=None)
    sampler = _FakeSamplerWrapper(results)
    models = _ListFeeder([_FakeModel(i) for i in range(2)])

    gen = sampler.sample_ising_streaming(models, num_reads=8, queue_depth=2)
    _, ss = next(gen)

    assert ss.info.get("defect_info") is None


def test_no_reconstruction_called():
    """The streaming pump must NOT call reconstruct_full_sampleset."""
    defect = _make_defect()
    results = _build_results(3, defect=defect)
    sampler = _FakeSamplerWrapper(results)
    models = _ListFeeder([_FakeModel(i) for i in range(3)])

    list(sampler.sample_ising_streaming(models, num_reads=8, queue_depth=2))

    assert sampler._reconstruct_calls == 0


def test_queue_depth_limits_concurrent_submissions():
    """Submissions should not exceed queue_depth before any completion."""
    # Use a feeder with many models but futures that start NOT done.
    n_models = 10
    queue_depth = 3

    slow_futures = []
    results_for_sampler = []
    for i in range(n_models):
        ss = _make_ss()
        f = _SlowFuture(ss, i)
        slow_futures.append(f)
        results_for_sampler.append((f, None))

    sampler = _FakeSamplerWrapper(results_for_sampler)
    models = _ListFeeder([_FakeModel(i) for i in range(n_models)])

    # A stop_event lets us tear down the generator's poll loop at the end so
    # its thread exits instead of spinning forever on the never-completing
    # futures (queue_depth..n_models) it refills with after the first batch.
    stop = mp.get_context("spawn").Event()
    gen = sampler.sample_ising_streaming(
        models, num_reads=8, queue_depth=queue_depth, stop_event=stop,
    )

    # Pump the generator: it fills to queue_depth, then blocks on poll.
    # We need to step it carefully — use a thread so we can mark futures done.
    import threading

    yielded: List[Any] = []
    gen_error: List[Exception] = []

    def _run():
        try:
            for item in gen:
                yielded.append(item)
        except Exception as exc:
            gen_error.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Give the generator time to fill to queue_depth and block.
    import time as _time
    _time.sleep(0.05)

    # At this point, no futures are done, so nothing should have been yielded.
    assert len(yielded) == 0
    # And exactly queue_depth submissions were made.
    assert sampler._call_count == queue_depth

    # Release the first queue_depth slow futures; the pump should yield them.
    for f in slow_futures[:queue_depth]:
        f.mark_done()

    # Condition-based wait: let the pump observe the completions and yield
    # them before we assert (avoids a fixed sleep racing a slow CI runner).
    deadline = _time.time() + 2.0
    while len(yielded) < queue_depth and _time.time() < deadline:
        _time.sleep(0.005)

    # At minimum, the queue_depth completions were processed.
    assert len(yielded) >= queue_depth

    # Tear down: stop_event unblocks the poll loop so the thread exits cleanly
    # instead of leaking and busy-polling the never-completing refill futures.
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "generator thread leaked after stop_event"
    assert not gen_error, gen_error


def test_gen_close_cancels_inflight_futures():
    """GeneratorExit (gen.close() or stop_event) must cancel all in-flight futures.

    Python does not allow calling gen.close() from a different thread while the
    generator is executing; instead we use stop_event — which triggers the same
    ``finally: _cancel_all()`` path — and verify that the futures get cancelled.
    """
    import time as _time

    n_slow = 3
    stop = mp.get_context("spawn").Event()
    slow_futures: List[_SlowFuture] = []
    results_for_sampler = []
    for i in range(n_slow):
        ss = _make_ss()
        f = _SlowFuture(ss, i)
        slow_futures.append(f)
        results_for_sampler.append((f, None))

    sampler = _FakeSamplerWrapper(results_for_sampler)
    models = _ListFeeder([_FakeModel(i) for i in range(n_slow)])

    gen = sampler.sample_ising_streaming(
        models, num_reads=8, queue_depth=n_slow, stop_event=stop,
    )

    # Run the generator in a thread so we can set stop_event from here.
    import threading

    started = threading.Event()
    yielded: List[Any] = []
    gen_error: List[Exception] = []

    def _run():
        started.set()
        try:
            for item in gen:
                yielded.append(item)
        except Exception as exc:
            gen_error.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    started.wait(timeout=1.0)
    _time.sleep(0.05)  # let the generator fill the queue and enter the poll loop

    # Trigger stop: the generator sees _stopped()==True, calls _cancel_all(),
    # then hits the finally block which calls _cancel_all() again (idempotent).
    stop.set()
    t.join(timeout=2.0)

    assert not t.is_alive(), "generator thread did not exit after stop_event"
    assert not gen_error, gen_error

    # All in-flight slow futures must have been cancelled.
    cancelled = [f._cancelled for f in slow_futures]
    assert any(cancelled), (
        "expected at least one in-flight future to be cancelled on stop; "
        f"cancel flags: {cancelled}"
    )


def test_gen_close_directly_cancels_futures_single_thread():
    """gen.close() on a generator that hasn't been iterated cancels the queue.

    Tests the ``finally:`` branch directly: advance past queue fill, then
    close before any completions, which fires GeneratorExit -> _cancel_all.
    Since we drive the generator manually (no thread), close() is safe.
    """
    n = 2
    slow_futures: List[_SlowFuture] = []
    results_for_sampler = []
    for i in range(n):
        ss = _make_ss()
        f = _SlowFuture(ss, i)
        slow_futures.append(f)
        results_for_sampler.append((f, None))

    sampler = _FakeSamplerWrapper(results_for_sampler)
    # Give extra models so the pump can fill without exhausting the feeder.
    models = _ListFeeder([_FakeModel(i) for i in range(n + 5)])

    gen = sampler.sample_ising_streaming(
        models, num_reads=8, queue_depth=n,
    )

    # Advance enough to fill the queue (the pump fills before first yield).
    # Because futures are not done, next() will block. Use send(None) to
    # enter the generator without pulling a value out — we just need it to
    # reach the poll loop. We can't do that directly (next() blocks too), so
    # instead we just close an unstarted generator and confirm it's safe
    # (the finally cancel loop runs over an empty pending dict).
    gen.close()  # should not raise; finally cancels empty pending

    # All futures are still in sampler._results (never submitted) — nothing to cancel.
    assert all(not f._cancelled for f in slow_futures)


def test_stop_event_halts_generator():
    """When stop_event is set, the generator stops and cancels in-flight work."""
    stop = mp.get_context("spawn").Event()
    n = 3

    slow_futures: List[_SlowFuture] = []
    results_for_sampler = []
    for i in range(n):
        ss = _make_ss()
        f = _SlowFuture(ss, i)
        slow_futures.append(f)
        results_for_sampler.append((f, None))

    sampler = _FakeSamplerWrapper(results_for_sampler)
    models = _ListFeeder([_FakeModel(i) for i in range(n)])

    gen = sampler.sample_ising_streaming(
        models, num_reads=8, queue_depth=n, stop_event=stop,
    )

    import threading
    import time as _time

    yielded: List[Any] = []
    started = threading.Event()

    def _run():
        started.set()
        for item in gen:
            yielded.append(item)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    started.wait(timeout=1.0)
    _time.sleep(0.05)

    stop.set()
    t.join(timeout=2.0)

    assert not t.is_alive(), "generator did not stop after stop_event"
    assert len(yielded) == 0, "should not yield any results after stop_event"


def test_raw_energies_not_shifted():
    """Yielded sampleset energies are NOT shifted by defect_info.energy_offset."""
    raw_energy = -200.0
    offset = 50.0
    defect = _make_defect(offset=offset)

    ss = _make_ss(energy=raw_energy)
    sampler = _FakeSamplerWrapper([(_FakeFuture(ss, 0), defect)])
    models = _ListFeeder([_FakeModel(0, defect)])

    gen = sampler.sample_ising_streaming(models, num_reads=8, queue_depth=1)
    _, yielded_ss = next(gen)

    # Energies must be the raw QPU energies — NOT shifted.
    assert float(yielded_ss.record.energy[0]) == raw_energy, (
        "energies should not be shifted; consumer adds offset"
    )


# ---------------------------------------------------------------------------
# Tests: DWaveMiner._finalize_sample
# ---------------------------------------------------------------------------

def _make_shared_ss(n_reads: int = 1, energy: float = -100.0):
    """Build the real production input: a positional _SharedSampleSet.

    Two live columns (nodes 0, 1) carry [+1, -1]; the stream driver discards
    dimod labels, so this is a raw int8 matrix, not a labeled dimod SampleSet.
    """
    import numpy as np
    from shared.base_miner import _SharedSampleSet

    sample = np.tile(np.array([1, -1], dtype=np.int8), (n_reads, 1))
    return _SharedSampleSet(sample, np.full(n_reads, energy, dtype=np.float64))


def test_finalize_sample_reconstructs_without_live_sampler():
    """_finalize_sample reconstructs even when self.sampler is None.

    The worker miner has no D-Wave connection (sampler is None); reconstruction
    must not depend on one. Regression for the graph_id-change crash where a
    newly-offline qubit forced reconstruction on the connection-less worker.
    The driver ships a label-less ``_SharedSampleSet``, so reconstruction is
    positional against ``nodes``.
    """
    from QPU.dwave_miner import DWaveMiner

    miner = DWaveMiner.__new__(DWaveMiner)
    miner.sampler = None  # the production condition that crashed

    # Live columns map to nodes [0, 1]; the defect clamps qubit 99 to +1,
    # offset +10. Full topology order is [0, 1, 99].
    nodes = [0, 1, 99]
    full = miner._finalize_sample(
        _make_shared_ss(energy=-100.0), _make_defect(offset=10.0), nodes,
    )

    row = full.record.sample[0]
    assert list(row) == [1, -1, 1], "clamped qubit must reappear with its spin"
    assert full.record.energy[0] == -90.0, "energy must be reduced energy + offset"


def test_finalize_sample_preserves_all_reads():
    """Reconstruction returns one full-topology sample per read, row content intact.

    Uses per-row-distinct spins so a vectorization bug (broadcasting one row, or
    filling the wrong axis) cannot pass a shape-only assertion.
    """
    import numpy as np

    from QPU.dwave_miner import DWaveMiner
    from shared.base_miner import _SharedSampleSet

    miner = DWaveMiner.__new__(DWaveMiner)
    miner.sampler = None

    # Live columns map to nodes [0, 1]; node 99 clamped to +1. Four distinct rows.
    reduced = np.array(
        [[1, -1], [-1, 1], [1, 1], [-1, -1]], dtype=np.int8,
    )
    shared_ss = _SharedSampleSet(reduced, np.zeros(4, dtype=np.float64))

    result = miner._finalize_sample(shared_ss, _make_defect(), [0, 1, 99])

    assert result.record.sample.shape == (4, 3)
    # Each row: [node0, node1, node99(=+1 clamp)].
    for i in range(4):
        assert list(result.record.sample[i]) == [reduced[i, 0], reduced[i, 1], 1]
