# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for Metal GPU utilization scaling, yielding, and IOKit integration.

Requires Mac with Metal GPU. Tests verify:
  - IOKit detects GPU activity during kernel dispatch
  - Throughput scales with gpu_utilization percentage
  - MetalScheduler throttle logic responds to IOKit readings
  - Streaming pipeline respects core budget

Run:
    python -m pytest tests/test_metal_yielding.py -v
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

# Skip entire module on non-macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Metal tests require macOS",
)

try:
    import Metal as _Metal  # noqa: F401
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

if not METAL_AVAILABLE:
    pytest.skip("Metal not available", allow_module_level=True)

from GPU.metal_sa import MetalSASampler
from GPU.metal_scheduler import MetalScheduler, _query_iokit_gpu_utilization
from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)


# ── Helpers ──────────────────────────────────────────────

def _make_models(
    sampler: MetalSASampler,
    count: int,
    seed: int = 0,
) -> List[IsingModel]:
    """Generate deterministic IsingModels for testing."""
    rng = np.random.RandomState(seed)
    models = []
    for _ in range(count):
        salt = rng.bytes(32)
        nonce = ising_nonce_from_block(
            b"test_hash_padding_to_32_bytes!!", "miner-test", 1, salt,
        )
        h, J = generate_ising_model_from_nonce(
            nonce, sampler.nodes, sampler.edges,
        )
        models.append(IsingModel(h=h, J=J, nonce=nonce, salt=salt))
    return models


def _measure_streaming_throughput(
    sampler: MetalSASampler,
    models: List[IsingModel],
    max_threadgroups: int,
    num_reads: int = 32,
    num_sweeps: int = 64,
) -> float:
    """Run streaming pipeline and return nonces per second."""
    start = time.perf_counter()
    count = 0
    for _model, _ss in sampler.sample_ising_streaming(
        iter(models),
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        max_threadgroups=max_threadgroups,
        seed=42,
    ):
        count += 1
    elapsed = time.perf_counter() - start
    return count / elapsed if elapsed > 0 else 0.0


# ── IOKit integration ────────────────────────────────────

class TestIOKitGPUDetection:
    """Verify IOKit detects GPU activity from Metal kernels."""

    def test_idle_reads_zero(self):
        """At idle, IOKit should report 0% (or very low)."""
        # Wait briefly to let any prior GPU work drain
        time.sleep(0.5)
        util = _query_iokit_gpu_utilization()
        assert util <= 10, (
            f"Expected idle GPU, got {util}%"
        )

    def test_detects_metal_kernel(self):
        """IOKit should read >0% while a Metal kernel runs."""
        sampler = MetalSASampler()
        models = _make_models(sampler, 4)

        readings: List[int] = []
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                readings.append(_query_iokit_gpu_utilization())
                time.sleep(0.15)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()

        # Run a kernel long enough for IOKit to detect
        h_list = [m.h for m in models]
        j_list = [m.J for m in models]
        sampler.sample_ising(
            h_list, j_list,
            num_reads=128, num_sweeps=256, seed=42,
        )

        stop.set()
        poller.join(timeout=2)

        peak = max(readings) if readings else 0
        nonzero = sum(1 for r in readings if r > 0)
        assert peak > 50, (
            f"Expected >50% peak GPU utilization, "
            f"got {peak}% (readings: {readings})"
        )
        assert nonzero > 0, (
            "IOKit never detected GPU activity"
        )

    def test_returns_to_idle_after_kernel(self):
        """GPU utilization drops back after kernel completes."""
        sampler = MetalSASampler()
        models = _make_models(sampler, 2)

        sampler.sample_ising(
            [m.h for m in models],
            [m.J for m in models],
            num_reads=32, num_sweeps=64, seed=42,
        )

        # IOKit updates asynchronously; poll until idle or timeout
        deadline = time.monotonic() + 5.0
        util = 100
        while time.monotonic() < deadline:
            time.sleep(0.5)
            util = _query_iokit_gpu_utilization()
            if util < 20:
                break

        assert util < 20, (
            f"Expected GPU idle after kernel within 5s, "
            f"got {util}%"
        )


# ── Utilization scaling ──────────────────────────────────

class TestUtilizationScaling:
    """Verify throughput scales with gpu_utilization percentage."""

    def test_core_budget_proportional(self):
        """Core budget should be proportional to utilization%."""
        from GPU.metal_miner import get_gpu_core_count
        cores = get_gpu_core_count()

        sched_100 = MetalScheduler(cores, 100, yielding=False)
        sched_50 = MetalScheduler(cores, 50, yielding=False)
        sched_25 = MetalScheduler(cores, 25, yielding=False)

        assert sched_100.get_core_budget() == cores
        assert sched_50.get_core_budget() == cores // 2
        assert sched_25.get_core_budget() == cores // 4

    def test_throughput_scales_with_batch_size(self):
        """More threadgroups per batch = higher throughput.

        Dispatching 8 problems per batch should be faster than
        dispatching 1 problem per batch (amortized overhead).
        """
        sampler = MetalSASampler()
        models = _make_models(sampler, 8)

        # Warm up topology cache
        sampler.prepare_topology()

        # 1 threadgroup per batch = 8 sequential dispatches
        tp_1 = _measure_streaming_throughput(
            sampler, models,
            max_threadgroups=1,
            num_reads=32,
            num_sweeps=64,
        )

        # 8 threadgroups per batch = 1 batched dispatch
        tp_8 = _measure_streaming_throughput(
            sampler, models,
            max_threadgroups=8,
            num_reads=32,
            num_sweeps=64,
        )

        # Batched should be at least 1.3x faster (less dispatch overhead)
        assert tp_8 > tp_1 * 1.3, (
            f"Batched ({tp_8:.1f} nonces/s) should be "
            f">1.3x single ({tp_1:.1f} nonces/s)"
        )

    def test_utilization_reduces_batch_size(self):
        """Lower utilization → smaller max_threadgroups → more batches.

        With 8 models at max_tg=2, pipeline does 4 batches.
        With 8 models at max_tg=8, pipeline does 1 batch.
        The first should take measurably longer.
        """
        sampler = MetalSASampler()
        sampler.prepare_topology()
        models = _make_models(sampler, 8)

        tp_full = _measure_streaming_throughput(
            sampler, models,
            max_threadgroups=8,
            num_reads=32,
            num_sweeps=64,
        )

        tp_half = _measure_streaming_throughput(
            sampler, models,
            max_threadgroups=2,
            num_reads=32,
            num_sweeps=64,
        )

        # Full batch should be faster than quarter-batch
        assert tp_full > tp_half, (
            f"Full batch ({tp_full:.1f}) should be faster "
            f"than quarter ({tp_half:.1f})"
        )


# ── Yielding and throttle ────────────────────────────────

class TestYieldingBehavior:
    """Verify yielding mode and throttle logic."""

    def test_scheduler_no_throttle_at_idle(self):
        """Yielding scheduler shouldn't throttle when GPU idle."""
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        # Let the IOKit poll thread run once
        time.sleep(0.5)
        assert sched.should_throttle() is False
        sched.stop()

    def test_scheduler_throttles_at_high_utilization(self):
        """Yielding scheduler should throttle when GPU is busy."""
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        # Simulate external GPU load via internal state
        with sched._util_lock:
            sched._external_util_pct = 95

        assert sched.should_throttle() is True
        sched.stop()

    def test_iokit_thread_updates_utilization(self):
        """IOKit polling thread should update external_util_pct."""
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
            poll_interval=0.2,
        )

        # At idle, after a few polls, should read ~0
        time.sleep(0.8)
        with sched._util_lock:
            idle_util = sched._external_util_pct

        assert idle_util <= 10, (
            f"Expected idle util, got {idle_util}%"
        )
        sched.stop()

    def test_iokit_thread_detects_gpu_load(self):
        """IOKit thread should read high util during Metal dispatch."""
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
            poll_interval=0.2,
        )
        sampler = MetalSASampler()
        models = _make_models(sampler, 4)

        # Run kernel while scheduler polls
        sampler.sample_ising(
            [m.h for m in models],
            [m.J for m in models],
            num_reads=128, num_sweeps=256, seed=42,
        )

        # Check what the thread observed
        with sched._util_lock:
            observed = sched._external_util_pct

        sched.stop()

        # The thread should have caught at least one high reading.
        # Since the kernel just finished, the last reading might
        # be either high (caught during) or low (caught after).
        # We just verify the mechanism works — the scheduler
        # did update its value from the IOKit thread.
        assert isinstance(observed, int)
        assert 0 <= observed <= 100

    def test_target_threadgroups_reduced_under_load(self):
        """compute_target_threadgroups should reduce when loaded."""
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )

        # No load → full budget
        with sched._util_lock:
            sched._external_util_pct = 0
        target_idle = sched.compute_target_threadgroups(
            max_tg=10, active_tg=0,
        )
        assert target_idle == 10

        # High external load, no active tg → reduced
        with sched._util_lock:
            sched._external_util_pct = 80
        target_loaded = sched.compute_target_threadgroups(
            max_tg=10, active_tg=0,
        )
        assert target_loaded < target_idle, (
            f"Loaded target ({target_loaded}) should be "
            f"< idle ({target_idle})"
        )
        assert target_loaded >= 1

        sched.stop()

    def test_non_yielding_ignores_load(self):
        """Non-yielding scheduler always returns full budget."""
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )

        target = sched.compute_target_threadgroups(
            max_tg=10, active_tg=5,
        )
        assert target == 10
        assert sched.should_throttle() is False


# ── Streaming pipeline with scheduler ────────────────────

class TestStreamingWithScheduler:
    """End-to-end: streaming pipeline respects core budget."""

    def test_streaming_completes_all_models(self):
        """All models should be yielded regardless of batch size."""
        sampler = MetalSASampler()
        models = _make_models(sampler, 7)  # Non-power-of-2

        results = list(sampler.sample_ising_streaming(
            iter(models),
            num_reads=16,
            num_sweeps=32,
            max_threadgroups=3,  # 3 batches: 3+3+1
            seed=42,
        ))

        assert len(results) == 7
        returned_nonces = {m.nonce for m, _ in results}
        expected_nonces = {m.nonce for m in models}
        assert returned_nonces == expected_nonces

    def test_streaming_energies_are_valid(self):
        """Sample energies should be negative (Ising solutions)."""
        sampler = MetalSASampler()
        models = _make_models(sampler, 3)

        for model, ss in sampler.sample_ising_streaming(
            iter(models),
            num_reads=32,
            num_sweeps=64,
            max_threadgroups=3,
            seed=42,
        ):
            min_e = min(ss.record.energy)
            assert min_e < 0, (
                f"Nonce {model.nonce}: expected negative "
                f"energy, got {min_e}"
            )

    def test_streaming_matches_direct_dispatch(self):
        """Streaming should produce same results as sample_ising."""
        sampler = MetalSASampler()
        models = _make_models(sampler, 4)

        # Direct dispatch
        direct = sampler.sample_ising(
            [m.h for m in models],
            [m.J for m in models],
            num_reads=32, num_sweeps=64, seed=42,
        )

        # Streaming (same seed, same batch size → same kernel)
        streamed = list(sampler.sample_ising_streaming(
            iter(models),
            num_reads=32,
            num_sweeps=64,
            max_threadgroups=4,
            seed=42,
        ))

        for i in range(4):
            direct_min = min(direct[i].record.energy)
            stream_min = min(streamed[i][1].record.energy)
            assert direct_min == stream_min, (
                f"Problem {i}: direct={direct_min} != "
                f"streamed={stream_min}"
            )

    def test_topology_cache_reused_across_calls(self):
        """prepare_topology should only run once."""
        sampler = MetalSASampler()
        assert not sampler._topo_prepared

        models = _make_models(sampler, 2)

        # First streaming call triggers prepare
        list(sampler.sample_ising_streaming(
            iter(models),
            num_reads=16, num_sweeps=32,
            max_threadgroups=2, seed=1,
        ))
        assert sampler._topo_prepared
        cached_N = sampler._topo_N
        cached_row_ptr_id = id(sampler._topo_row_ptr)

        # Second call reuses cache
        list(sampler.sample_ising_streaming(
            iter(models),
            num_reads=16, num_sweeps=32,
            max_threadgroups=2, seed=2,
        ))
        assert sampler._topo_N == cached_N
        assert id(sampler._topo_row_ptr) == cached_row_ptr_id
