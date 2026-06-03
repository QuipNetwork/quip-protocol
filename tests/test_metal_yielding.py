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

import sys
import threading
import time
from typing import List

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
from GPU.metal_scheduler import _query_iokit_gpu_utilization
from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    derive_nonce,
    generate_ising_model_from_nonce,
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
        # Pad the test miner name to 32 bytes so derive_nonce accepts it.
        miner_bytes = b"miner-test".ljust(32, b"\x00")
        last_proof_block_hash = b"test_hash_padding_to_32_bytes!!!"
        nonce = derive_nonce(last_proof_block_hash, miner_bytes, salt)
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
        models = _make_models(sampler, 8)

        readings: List[int] = []
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                readings.append(_query_iokit_gpu_utilization())
                time.sleep(0.1)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()

        # Run a heavy kernel so IOKit has time to register
        h_list = [m.h for m in models]
        j_list = [m.J for m in models]
        sampler.sample_ising(
            h_list, j_list,
            num_reads=256, num_sweeps=512, seed=42,
        )

        stop.set()
        poller.join(timeout=2)

        peak = max(readings) if readings else 0
        nonzero = sum(1 for r in readings if r > 0)
        assert peak > 30, (
            f"Expected >30% peak GPU utilization, "
            f"got {peak}% (readings: {readings})"
        )
        assert nonzero > 0, (
            "IOKit never detected GPU activity"
        )


# ── Utilization scaling ──────────────────────────────────

class TestUtilizationScaling:
    """Verify throughput scales with gpu_utilization percentage."""

    def test_throughput_scales_with_batch_size(self):
        """More threadgroups per batch = higher throughput.

        This intentionally exercises the FLAT-OUT (idle/headless, target=100)
        path — batching only applies there; when throttled the governor caps
        to one problem per command buffer. Kept small to avoid saturating the
        GPU during the test run.
        """
        sampler = MetalSASampler()
        models = _make_models(sampler, 8)

        # Warm up topology cache
        sampler.prepare_topology()

        # 1 threadgroup per batch = 8 sequential dispatches
        tp_1 = _measure_streaming_throughput(
            sampler, models,
            max_threadgroups=1,
            num_reads=16,
            num_sweeps=24,
        )

        # 8 threadgroups per batch = 1 batched dispatch
        tp_8 = _measure_streaming_throughput(
            sampler, models,
            max_threadgroups=8,
            num_reads=16,
            num_sweeps=24,
        )

        # Batched should be at least 1.3x faster (less dispatch overhead)
        assert tp_8 > tp_1 * 1.3, (
            f"Batched ({tp_8:.1f} nonces/s) should be "
            f">1.3x single ({tp_1:.1f} nonces/s)"
        )

    def test_utilization_reduces_batch_size(self):
        """Smaller max_threadgroups → more batches → lower flat-out throughput.

        Also the FLAT-OUT (idle/headless, target=100) path; kept small to avoid
        saturating the GPU during the test run.
        """
        sampler = MetalSASampler()
        sampler.prepare_topology()
        models = _make_models(sampler, 8)

        tp_full = _measure_streaming_throughput(
            sampler, models,
            max_threadgroups=8,
            num_reads=16,
            num_sweeps=24,
        )

        tp_half = _measure_streaming_throughput(
            sampler, models,
            max_threadgroups=2,
            num_reads=16,
            num_sweeps=24,
        )

        # Full batch should be faster than quarter-batch
        assert tp_full > tp_half, (
            f"Full batch ({tp_full:.1f}) should be faster "
            f"than quarter ({tp_half:.1f})"
        )


# ── Yielding and throttle ────────────────────────────────

class TestYieldingBehavior:
    """Verify yielding mode and throttle logic."""

    def test_iokit_thread_updates_utilization(self):
        """util_monitor_main process should publish IOKit readings into shared Value."""
        import multiprocessing as mp

        from GPU.util_monitor import util_monitor_main
        from shared.proc_util import terminate_join

        ctx = mp.get_context("spawn")
        val = ctx.Value("i", -1)
        stop = ctx.Event()
        proc = ctx.Process(
            target=util_monitor_main,
            args=(val, stop, 0.1,
                  "GPU.metal_scheduler:poll_iokit_gpu_util"),
            daemon=True,
        )
        proc.start()

        # Wait up to 3s for at least one poll to write a valid value
        deadline = time.monotonic() + 3.0
        while val.value == -1 and time.monotonic() < deadline:
            time.sleep(0.05)
        stop.set()
        assert terminate_join(proc, 2.0)

        # After at least one poll the value must be a valid 0-100 reading
        assert 0 <= val.value <= 100, (
            f"Expected 0-100 from IOKit poll, got {val.value}"
        )


# ── Streaming pipeline through the governor (yielding) ───

class _ThrottleScheduler:
    """Fake governor that caps occupancy to a fixed thread budget.

    Drives ``sample_ising_streaming`` down the GOVERNED path: reads are split
    so ``problems x reads <= budget`` per command buffer, keeping the GPU below
    the occupancy that janks the UI rather than saturating it. That's the whole
    point of yielding.
    """

    def __init__(self, budget: int = 256):
        self._budget = budget

    def get_thread_budget(self) -> int:
        return self._budget

    def get_measured_gpu(self) -> int:
        return 0


class TestStreamingWithScheduler:
    """End-to-end: streaming pipeline runs through the governor (throttled)."""

    def _governed_kwargs(self):
        return {"scheduler": _ThrottleScheduler(256)}

    def test_streaming_completes_all_models(self):
        """All models are yielded through the throttled (yielding) path."""
        sampler = MetalSASampler()
        models = _make_models(sampler, 7)  # Non-power-of-2

        results = list(sampler.sample_ising_streaming(
            iter(models),
            num_reads=16,
            num_sweeps=32,
            max_threadgroups=3,
            seed=42,
            **self._governed_kwargs(),
        ))

        assert len(results) == 7
        returned_nonces = {m.nonce for m, _ in results}
        expected_nonces = {m.nonce for m in models}
        assert returned_nonces == expected_nonces

    def test_streaming_energies_are_valid(self):
        """Throttled streaming still yields valid (negative) Ising energies."""
        sampler = MetalSASampler()
        models = _make_models(sampler, 3)

        for model, ss in sampler.sample_ising_streaming(
            iter(models),
            num_reads=32,
            num_sweeps=64,
            max_threadgroups=3,
            seed=42,
            **self._governed_kwargs(),
        ):
            assert len(ss) == 32, "throttled path must preserve total reads"
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
