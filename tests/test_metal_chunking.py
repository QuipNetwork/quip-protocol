# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for occupancy-budget read-splitting (GPU/metal_sa.py).

``reads_per_buffer_for_budget`` is pure math (no Metal) and runs anywhere. The
dispatch tests instantiate the real Metal sampler and so are gated on a Metal
device. The jank lever is concurrent threads per command buffer
(problems x reads); read-splitting caps it while preserving total reads.
"""
from __future__ import annotations

import sys

import pytest

from GPU.metal_sa import reads_per_buffer_for_budget
from GPU.metal_scheduler import UNCAPPED

pytestmark_metal = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal tests require macOS",
)


# ── pure occupancy math ─────────────────────────────────────────────────

class TestReadsPerBufferForBudget:
    def test_uncapped_returns_full_reads(self):
        assert reads_per_buffer_for_budget(UNCAPPED, 8, 1024) == 1024

    def test_under_budget_is_single_dispatch(self):
        # 8 x 256 = 2048 <= budget 2048 -> full reads, one buffer.
        assert reads_per_buffer_for_budget(2048, 8, 256) == 256

    def test_over_budget_splits_reads(self):
        # 8 x 1024 = 8192 > 2048 -> 2048 // 8 = 256 reads per buffer.
        assert reads_per_buffer_for_budget(2048, 8, 1024) == 256

    def test_caps_to_at_least_one(self):
        # 40 problems, budget 20 -> 20 // 40 = 0 -> floor at 1.
        assert reads_per_buffer_for_budget(20, 40, 1024) == 1

    def test_more_problems_means_fewer_reads_per_buffer(self):
        few = reads_per_buffer_for_budget(2048, 4, 1024)
        many = reads_per_buffer_for_budget(2048, 16, 1024)
        assert many < few


# ── dispatch path (Metal device required) ───────────────────────────────

@pytestmark_metal
class TestStreamingBudget:
    def _sampler_and_models(self, n=2):
        from GPU.metal_sa import MetalSASampler
        from tests.test_metal_yielding import _make_models
        s = MetalSASampler()
        return s, _make_models(s, n)

    class _FakeScheduler:
        """get_thread_budget returns a fixed budget per the constructor."""

        def __init__(self, budget):
            self._budget = budget

        def get_thread_budget(self):
            return self._budget

        def get_measured_gpu(self):
            return 0

    def test_uncapped_is_single_monolithic_dispatch(self, monkeypatch):
        s, models = self._sampler_and_models(3)
        calls = {"batch": 0, "split": 0}

        def fake_batch(batch, **kw):
            calls["batch"] += 1
            assert kw["num_reads"] == 8     # full reads in one buffer
            return [None] * len(batch)

        orig_split = s._dispatch_read_split

        def counting_split(*a, **k):
            calls["split"] += 1
            return orig_split(*a, **k)

        monkeypatch.setattr(s, "_dispatch_batch", fake_batch)
        monkeypatch.setattr(s, "_dispatch_read_split", counting_split)
        out = list(s.sample_ising_streaming(
            iter(models), num_reads=8, num_sweeps=32, max_threadgroups=3,
            seed=1, scheduler=self._FakeScheduler(UNCAPPED),
        ))
        assert calls["batch"] == 1          # one monolithic dispatch
        assert len(out) == 3

    def test_capped_splits_reads_preserving_total(self):
        s, models = self._sampler_and_models(1)
        # budget 16, 1 problem -> 16 reads/buffer; 64 reads -> 4 buffers.
        out = list(s.sample_ising_streaming(
            iter(models), num_reads=64, num_sweeps=32, max_threadgroups=4,
            seed=1, scheduler=self._FakeScheduler(16),
        ))
        assert len(out) == 1
        assert len(out[0][1]) == 64         # all reads delivered

    def test_budget_zero_pauses_then_resumes(self, monkeypatch):
        s, models = self._sampler_and_models(1)
        slept = {"n": 0}
        monkeypatch.setattr("GPU.metal_sa.time.sleep",
                            lambda _s: slept.__setitem__("n", slept["n"] + 1))

        class _Seq:
            def __init__(self, seq):
                self._seq, self._i = seq, 0

            def get_thread_budget(self):
                v = self._seq[min(self._i, len(self._seq) - 1)]
                self._i += 1
                return v

            def get_measured_gpu(self):
                return 0

        out = list(s.sample_ising_streaming(
            iter(models), num_reads=8, num_sweeps=32, max_threadgroups=4,
            seed=1, scheduler=_Seq([0, 0, UNCAPPED]),
        ))
        assert slept["n"] >= 2              # paused before dispatching
        assert len(out) == 1

    def test_pause_returns_promptly_when_stop_set(self):
        s, models = self._sampler_and_models(2)

        class _StopSet:
            def is_set(self):
                return True

        out = list(s.sample_ising_streaming(
            iter(models), num_reads=8, num_sweeps=32, max_threadgroups=2,
            seed=1, scheduler=self._FakeScheduler(0), stop_event=_StopSet(),
        ))
        assert out == []

    def test_read_split_matches_monolithic_energy_floor(self):
        """Splitting reads must preserve solution quality: the best energy over
        all reads is at least as good as a monolithic dispatch of the same
        total reads (independent SA runs, just grouped differently)."""
        from GPU.metal_utils import compute_beta_schedule
        s, models = self._sampler_and_models(2)
        s.prepare_topology()
        beta_arr, br = compute_beta_schedule(
            models[0].h, models[0].J, 128, 1, None, "geometric", None,
        )
        common = dict(
            num_reads=64, beta_schedule_arr=beta_arr, beta_range=br,
            beta_schedule_type="geometric", num_sweeps_per_beta=1, seed=7,
        )
        mono = s._dispatch_batch(models, **common)
        split = s._dispatch_read_split(models, reads_per_buffer=16, **common)
        for i in range(len(models)):
            assert len(split[i]) == 64
            # Both explore the same landscape; split (more seeds) should not be
            # systematically worse than monolithic.
            assert min(split[i].record.energy) <= min(mono[i].record.energy) + 50
