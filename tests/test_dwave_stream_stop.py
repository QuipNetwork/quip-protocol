"""Unit tests for DWaveMiner's streaming-iterator cancellation behaviour.

Covers two modes:
- Default discard: stop_event fires → iterator abandons pending futures
  and returns within one poll cycle. This is what the production node
  wants so we can pivot to the next block immediately when a peer wins.
- drain_on_stop=True: stop_event fires → iterator stops submitting new
  QPU jobs but yields results for whatever is already in flight. This is
  the test-only escape hatch for workflows that want to inspect partial
  results.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import pytest

from QPU.dwave_miner import DWaveMiner
from shared.ising_model import IsingModel


class _FakeFuture:
    """Controllable stand-in for a D-Wave SampleFuture."""

    def __init__(self, ready_after: float | None = None):
        self._ready_at = (
            None if ready_after is None else time.monotonic() + ready_after
        )
        self.cancelled = False

    def done(self) -> bool:
        if self.cancelled:
            return False
        if self._ready_at is None:
            return False
        return time.monotonic() >= self._ready_at

    def cancel(self) -> None:
        self.cancelled = True

    @property
    def sampleset(self):
        return SimpleNamespace(
            record=SimpleNamespace(energy=[0.0]),
            info={},
        )


class _FakeFeeder:
    """Pops deterministic IsingModels for the streaming iterator."""

    def __init__(self):
        self._counter = 0

    def pop_blocking(self) -> IsingModel:
        self._counter += 1
        return IsingModel(h={}, J={}, nonce=self._counter, salt=b"s")


class _FakeSampler:
    """Hands out fake futures and records submissions for inspection."""

    def __init__(self, future_factory):
        self._future_factory = future_factory
        self.job_label = "test"
        self.submissions = 0

    def sample_ising_async(self, h, J, **_):
        self.submissions += 1
        future = self._future_factory()
        return future, None


def _miner_with_sampler(sampler, drain_on_stop: bool) -> DWaveMiner:
    """Build a DWaveMiner bypassing the real __init__ (no D-Wave needed)."""
    miner = object.__new__(DWaveMiner)
    miner.sampler = sampler
    miner.drain_on_stop = drain_on_stop
    miner._stop_event = None
    # BaseMiner.__init__ sets self.logger = getLogger(f'miner.{miner_id}');
    # we skip __init__ here, so wire up an equivalent so the streaming
    # iterator's debug/info logs don't AttributeError.
    miner.logger = logging.getLogger("miner.test-dwave-stream")
    miner.timing_stats = {
        "preprocessing": [], "sampling": [], "postprocessing": [],
        "quantum_annealing_time": [], "per_sample_overhead": [],
        "qpu_access_time": [], "total_samples": 0, "blocks_attempted": 0,
    }
    miner.time_manager = None
    return miner


@pytest.mark.timeout(10)
def test_streaming_discards_in_flight_jobs_on_stop():
    """Default behaviour: stop_event aborts the stream and cancels pending futures."""
    # Futures never complete on their own.
    sampler = _FakeSampler(lambda: _FakeFuture(ready_after=None))
    miner = _miner_with_sampler(sampler, drain_on_stop=False)

    stop_event = mp.Event()
    miner._stop_event = stop_event

    stream = miner.sample_ising_streaming(
        feeder=_FakeFeeder(),
        num_reads=1,
        annealing_time=1.0,
        queue_depth=3,
        energy_threshold=0.0,
    )

    # Fire stop from another thread so the iterator's polling loop sees it.
    def _trigger_stop():
        time.sleep(0.1)
        stop_event.set()

    threading.Thread(target=_trigger_stop, daemon=True).start()

    t0 = time.monotonic()
    with pytest.raises(StopIteration):
        next(stream)
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, f"stream did not abort promptly (took {elapsed:.2f}s)"


@pytest.mark.timeout(10)
def test_streaming_drains_on_stop_when_flag_set():
    """drain_on_stop=True: stop halts new submissions but yields completions."""
    # Each future is ready 50ms after creation, so the three queued jobs
    # will all finish within a few hundred ms.
    sampler = _FakeSampler(lambda: _FakeFuture(ready_after=0.05))
    miner = _miner_with_sampler(sampler, drain_on_stop=True)

    stop_event = mp.Event()
    miner._stop_event = stop_event

    stream = miner.sample_ising_streaming(
        feeder=_FakeFeeder(),
        num_reads=1,
        annealing_time=1.0,
        queue_depth=3,
        energy_threshold=0.0,
    )

    stop_event.set()

    # Drain mode should yield the three already-submitted jobs, then stop.
    drained = list(stream)
    assert len(drained) == 3, (
        f"drain_on_stop should flush in-flight jobs, got {len(drained)}"
    )
    # And it should NOT have submitted any replacements after stop.
    assert sampler.submissions == 3, (
        f"drain_on_stop must stop submitting new jobs; "
        f"submissions={sampler.submissions}"
    )
