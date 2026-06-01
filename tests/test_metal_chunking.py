# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for adaptive betas_per_chunk sizing + per-batch path selection.

``compute_betas_per_chunk`` is pure math (no Metal) and runs anywhere. The
path-selection tests instantiate the real Metal sampler and so are gated on a
Metal device.
"""
from __future__ import annotations

import sys

import pytest

from GPU.metal_sa import compute_betas_per_chunk

pytestmark_metal = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal tests require macOS",
)


# ── pure chunk-sizing math ──────────────────────────────────────────────

class TestComputeBetasPerChunk:
    def test_targets_burst_budget(self):
        # 8ms budget / 1ms per beta -> 8 betas per chunk.
        assert compute_betas_per_chunk(8.0, 1.0, 100) == 8

    def test_rounds_to_nearest(self):
        # 8 / 3 = 2.67 -> 3.
        assert compute_betas_per_chunk(8.0, 3.0, 100) == 3

    def test_clamps_to_at_least_one(self):
        # Single beta already exceeds the burst budget -> floor at 1.
        assert compute_betas_per_chunk(8.0, 50.0, 100) == 1

    def test_clamps_to_total_betas(self):
        # Tiny per-beta time would want a huge chunk -> cap at total.
        assert compute_betas_per_chunk(8.0, 0.001, 40) == 40

    def test_uncalibrated_ema_returns_total(self):
        # ema <= 0 means "not yet measured" -> dispatch all (monolithic-like).
        assert compute_betas_per_chunk(8.0, 0.0, 64) == 64

    def test_grows_when_betas_get_cheaper(self):
        small = compute_betas_per_chunk(8.0, 4.0, 100)
        big = compute_betas_per_chunk(8.0, 1.0, 100)
        assert big > small


# ── path selection (Metal device required) ─────────────────────────────

@pytestmark_metal
class TestPerBatchPathSelection:
    def _sampler_and_models(self, n=2):
        from GPU.metal_sa import MetalSASampler
        from tests.test_metal_yielding import _make_models
        s = MetalSASampler()
        return s, _make_models(s, n)

    class _FakeScheduler:
        def __init__(self, targets):
            self._targets = list(targets)
            self._i = 0

        def get_target_pct(self):
            t = self._targets[min(self._i, len(self._targets) - 1)]
            self._i += 1
            return t

        def get_cached_utilization(self):
            return 0

    def test_target_100_uses_monolithic(self, monkeypatch):
        s, models = self._sampler_and_models()
        calls = {"mono": 0, "chunk": 0}

        def fake_mono(batch, **kw):
            calls["mono"] += 1
            return [None] * len(batch)

        def fake_chunk(batch, **kw):
            calls["chunk"] += 1
            return [None] * len(batch)

        monkeypatch.setattr(s, "_dispatch_batch", fake_mono)
        monkeypatch.setattr(s, "_dispatch_batch_chunked", fake_chunk)
        out = list(s.sample_ising_streaming(
            iter(models), num_reads=8, num_sweeps=64, max_threadgroups=4,
            seed=1, scheduler=self._FakeScheduler([100]),
        ))
        assert calls == {"mono": 1, "chunk": 0}
        assert len(out) == len(models)

    def test_target_50_uses_chunked(self, monkeypatch):
        from GPU.metal_scheduler import DutyCycleController
        s, models = self._sampler_and_models()
        calls = {"mono": 0, "chunk": 0}

        def fake_mono(batch, **kw):
            calls["mono"] += 1
            return [None] * len(batch)

        def fake_chunk(batch, **kw):
            calls["chunk"] += 1
            return [None] * len(batch)

        monkeypatch.setattr(s, "_dispatch_batch", fake_mono)
        monkeypatch.setattr(s, "_dispatch_batch_chunked", fake_chunk)
        out = list(s.sample_ising_streaming(
            iter(models), num_reads=8, num_sweeps=64, max_threadgroups=4,
            seed=1, scheduler=self._FakeScheduler([50]),
            duty_cycle=DutyCycleController(target_pct=100),
        ))
        assert calls == {"mono": 0, "chunk": 1}
        assert len(out) == len(models)

    def test_target_zero_pauses_then_dispatches(self, monkeypatch):
        s, models = self._sampler_and_models(1)
        slept = {"n": 0}

        def fake_mono(batch, **kw):
            return [None] * len(batch)

        monkeypatch.setattr(s, "_dispatch_batch", fake_mono)
        monkeypatch.setattr("GPU.metal_sa.time.sleep",
                            lambda _s: slept.__setitem__("n", slept["n"] + 1))
        # Pause twice (0), then flat-out (100).
        sched = self._FakeScheduler([0, 0, 100])
        out = list(s.sample_ising_streaming(
            iter(models), num_reads=8, num_sweeps=64, max_threadgroups=4,
            seed=1, scheduler=sched,
        ))
        assert slept["n"] >= 2          # paused before dispatching
        assert len(out) == 1            # model still dispatched after resume

    def test_pause_returns_promptly_when_stop_set(self):
        """A permanent PAUSE (target 0) must not hang: a set stop_event ends
        the generator instead of spinning forever (battery / critical-thermal
        full-stop must never block teardown)."""
        s, models = self._sampler_and_models(2)

        class _StopSet:
            def is_set(self):
                return True

        out = list(s.sample_ising_streaming(
            iter(models), num_reads=8, num_sweeps=64, max_threadgroups=2,
            seed=1, scheduler=self._FakeScheduler([0]), stop_event=_StopSet(),
        ))
        assert out == []

    def test_adaptive_chunked_matches_monolithic(self):
        """Adaptive chunk sizing must stay state-equivalent to monolithic
        (persistent device buffers + beta_start/beta_count make chunk size
        irrelevant to the result given the same seed)."""
        from GPU.metal_scheduler import DutyCycleController
        s, models = self._sampler_and_models(3)
        direct = s.sample_ising(
            [m.h for m in models], [m.J for m in models],
            num_reads=32, num_sweeps=128, seed=7,
        )
        streamed = list(s.sample_ising_streaming(
            iter(models), num_reads=32, num_sweeps=128, max_threadgroups=3,
            seed=7, scheduler=self._FakeScheduler([50]),
            duty_cycle=DutyCycleController(target_pct=100), burst_ms=4.0,
        ))
        for i in range(len(models)):
            assert min(direct[i].record.energy) == min(streamed[i][1].record.energy)
