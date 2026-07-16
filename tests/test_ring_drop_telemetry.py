# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Ring drops must be visible on telemetry, and must not lie when unmeasured.

A sample the stream driver drops never reaches the worker and is never
evaluated, so a winning solution it carried is lost. Before this the only
signal was a warning at driver exit. The driver counts drops on a shared
``multiprocessing.Value`` (rather than a descriptor field, which would go
silent exactly when a saturated ring drops everything), the worker pushes the
count at the progress cadence, and the controller surfaces it as
``miners[].ring_drops``.

The distinction these tests protect: None means "not measured", 0 means
"measured, no drops". Collapsing them would report a clean bill of health for
a path that isn't counting.
"""

from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

import pytest

import QPU.stream_driver as stream_driver


class _FakeRingFull:
    """Ring that never has a free slot, so every result drops."""

    max_rows = 64
    max_cols = 64

    def claim_free(self, timeout: float = 0.0):
        return None

    def write(self, slot, sample, energy):  # pragma: no cover - never reached
        raise AssertionError("write must not run when no slot was claimed")

    def release(self, slot):  # pragma: no cover - never reached
        pass

    def close_unlink(self):
        pass


class _FakeRingOversized:
    """Ring too small for the produced sample, so results drop as oversized."""

    max_rows = 1
    max_cols = 1

    def claim_free(self, timeout: float = 0.0):  # pragma: no cover
        raise AssertionError("oversized results must drop before claiming")

    def write(self, slot, sample, energy):  # pragma: no cover
        pass

    def release(self, slot):  # pragma: no cover
        pass

    def close_unlink(self):
        pass


class _SelfStoppingCtx:
    """Yields *n* results, then sets stop_event so the driver exits cleanly.

    Stopping from inside the context (rather than pre-queuing a ctl_q shutdown)
    is what lets the driver actually process the results first: the driver
    checks for shutdown at the top of each result iteration, so a sentinel
    queued up front would end the run before anything could drop.
    """

    generation = 1

    def __init__(self, *, n, stop_event, rows=4, cols=3, **_ignored):
        self._n = n
        self._stop_event = stop_event
        self._rows, self._cols = rows, cols
        self.cleaned_up = False

    def apply_command(self, cmd):
        if cmd[0] == "switch":
            self.generation = int(cmd[1])

    def iter_results(self):
        import numpy as np

        for i in range(self._n):
            ss = SimpleNamespace(
                record=SimpleNamespace(
                    sample=np.ones((self._rows, self._cols), np.int8),
                    energy=np.zeros(self._rows, np.float64),
                ),
                info={},
            )
            model = SimpleNamespace(nonce=bytes([i % 256]) * 32, salt=b"\3" * 32)
            yield model, ss, self.generation
        # Every result produced: end the run without a ctl_q sentinel.
        self._stop_event.set()

    def cleanup(self):
        self.cleaned_up = True


def build_self_stopping_ctx(*, n, stop_event=None, **kwargs):
    """Factory resolved by stream_driver_main via its dotted path.

    ``stop_event`` is named explicitly because ``_maybe_with_stop`` inspects
    this signature to decide whether to inject it.
    """
    return _SelfStoppingCtx(n=n, stop_event=stop_event, **kwargs)


def _run_driver(monkeypatch, ring, drops, n_results: int):
    """Drive stream_driver_main in-process with *ring* and *drops*."""
    monkeypatch.setattr(stream_driver, "SampleView", lambda **_kw: ring)
    monkeypatch.setattr(
        stream_driver, "_start_parent_death_watchdog", lambda _stop: None,
    )
    monkeypatch.setattr(
        stream_driver, "setup_child_process_logging", lambda _q: None,
    )
    ctx = mp.get_context("spawn")
    desc_q = ctx.Queue(maxsize=64)
    ctl_q = ctx.Queue(maxsize=64)
    stop = ctx.Event()
    ctl_q.put(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 8, 80.0))
    stream_driver.stream_driver_main(
        ring_args={},
        desc_q=desc_q,
        ctl_q=ctl_q,
        stop_event=stop,
        stream_factory_dotted=(
            "tests.test_ring_drop_telemetry:build_self_stopping_ctx"
        ),
        factory_kwargs={"n": n_results},
        log_queue=None,
        drops=drops,
    )


class TestDriverCountsDrops:
    """The driver must record every drop on the shared counter."""

    def test_backpressure_drop_increments_shared_counter(self, monkeypatch):
        drops = mp.get_context("spawn").Value("L", 0)
        _run_driver(monkeypatch, _FakeRingFull(), drops, n_results=1)
        assert drops.value >= 1

    def test_oversized_drop_increments_shared_counter(self, monkeypatch):
        drops = mp.get_context("spawn").Value("L", 0)
        _run_driver(monkeypatch, _FakeRingOversized(), drops, n_results=1)
        assert drops.value >= 1

    def test_driver_runs_without_a_counter(self, monkeypatch):
        """drops=None must stay supported; the counter is optional."""
        _run_driver(monkeypatch, _FakeRingFull(), None, n_results=1)


class TestRingDropsAccessor:
    """BaseMiner.ring_drops must distinguish unmeasured from zero."""

    def _miner(self, drops):
        from shared.base_miner import BaseMiner

        m = SimpleNamespace(_drops=drops)
        return BaseMiner.ring_drops.fget(m)

    def test_no_driver_reports_none_not_zero(self):
        assert self._miner(None) is None

    def test_measured_zero_reports_zero(self):
        assert self._miner(mp.get_context("spawn").Value("L", 0)) == 0

    def test_measured_count_reports_count(self):
        assert self._miner(mp.get_context("spawn").Value("L", 7)) == 7


class _Handle:
    miner_id = "GPU-1"
    miner_type = "GPU"


class _Controller:
    """Minimal stand-in exposing just the drop-store surface."""

    def __init__(self):
        self._latest_drops = {}

    _store_drops = None  # bound below


class TestStoreDrops:
    """The controller must accept good pushes and reject junk."""

    def _controller(self):
        from substrate.miner_controller import SubstrateMinerController

        c = _Controller()
        c._store_drops = SubstrateMinerController._store_drops.__get__(
            c, _Controller,
        )
        return c

    def test_valid_push_is_stored(self):
        c = self._controller()
        c._store_drops(_Handle(), {"op": "drops", "data": {"ring_drops": 5}})
        assert c._latest_drops["GPU-1"] == 5

    def test_zero_is_stored(self):
        """A measured zero is a real reading, not an absent one."""
        c = self._controller()
        c._store_drops(_Handle(), {"op": "drops", "data": {"ring_drops": 0}})
        assert c._latest_drops["GPU-1"] == 0

    @pytest.mark.parametrize("data", [None, "nope", 5, []])
    def test_malformed_data_ignored(self, data):
        c = self._controller()
        c._store_drops(_Handle(), {"op": "drops", "data": data})
        assert c._latest_drops == {}

    @pytest.mark.parametrize("count", [None, "5", -1, 1.5, True])
    def test_bad_count_ignored(self, count):
        """A stray payload must not poison the dashboard.

        True is rejected explicitly: bool is an int subclass, so a naive
        isinstance check would store True as a drop count of 1.
        """
        c = self._controller()
        c._store_drops(_Handle(), {"op": "drops", "data": {"ring_drops": count}})
        assert c._latest_drops == {}
