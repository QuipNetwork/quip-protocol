# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""GPU regression test for CUDA nonce-starvation liveness (QUI-828 / gh-19).

The CUDA SA self-feeding kernel must survive a *transient* feeder stall. The
original kernel searched for the next READY slot with a bounded ~100ms retry
(``for retry < 10000: __nanosleep(10us)``) and then let the block ``return``.
Any starvation longer than that killed the GPU block permanently, so the host
could no longer revive that nonce -- the miner's attempt rate decayed
block-by-block until restart. The fix makes the kernel wait through starvation
and exit only on the host ``EXIT_NOW`` flag.

This drives the real ``_run_streaming_loop`` on a real GPU under induced
starvation and asserts completions *sustain* well past the cold-start ceiling
(``2 * num_k``). Pre-fix, completions freeze at that ceiling once every block
has starved out; post-fix they keep flowing through repeated idle->revive
cycles.

Skipped when no CUDA device is present.
"""
from __future__ import annotations

import os
import time

import pytest

cp = pytest.importorskip("cupy")

try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

import GPU.base_cuda_sampler as bcs
from GPU.cuda_sa import CudaSASampler
from GPU.slot_rotation import SlotState as _RealSlotState
from shared.ising_feeder import RandomIsingFeeder
from tools.baseline_utils import load_baseline_topology


class _CountingSlotState(_RealSlotState):
    """SlotState that tallies idle-rotations and revivals across all nonces."""

    idle_rotations = 0
    revivals = 0
    _cycled = False

    def rotate_on_completion(self) -> None:
        going_idle = self.next_model is None
        super().rotate_on_completion()
        self._cycled = True
        if going_idle:
            type(self).idle_rotations += 1

    def assign_active(self, slot, model) -> None:
        if self._cycled and self.active_model is None:
            type(self).revivals += 1
        super().assign_active(slot, model)

    @classmethod
    def reset(cls) -> None:
        cls.idle_rotations = 0
        cls.revivals = 0


class _StarvingFeeder:
    """Injects bursty wall-clock starvation into try_pop().

    A time-based (not call-based) duty cycle is essential: the consumer
    busy-polls, so a call-counted starve window would elapse in microseconds --
    far shorter than a kernel completion -- and no completion would ever land
    inside it. The starve half returns None (buffer momentarily empty, NOT
    exhausted); the supply half returns real models. pop_blocking() always
    yields (cold start).
    """

    def __init__(self, inner: RandomIsingFeeder, window: float) -> None:
        self.inner = inner
        self.window = window
        self._t0 = time.monotonic()

    def __iter__(self):
        return self

    def __next__(self):
        return self.pop_blocking()

    def _starving(self) -> bool:
        return int((time.monotonic() - self._t0) / self.window) % 2 == 0

    def pop_blocking(self):
        return self.inner.pop_blocking()

    def try_pop(self):
        if self._starving():
            return None
        return self.inner.try_pop() or self.inner.pop_blocking()


def test_cuda_nonce_survives_transient_starvation(monkeypatch):
    """A nonce must keep producing after a >100ms feeder stall, not die."""
    num_k = 4
    num_reads, num_sweeps = 16, 64
    window = 2.0
    poll_timeout = 6.0  # > window, so a supply-window gap never false-trips
    safety_deadline_s = 45.0
    cold_start_ceiling = 2 * num_k  # active + next filled before any revive
    target = 3 * cold_start_ceiling

    monkeypatch.setattr(bcs, "SlotState", _CountingSlotState)
    _CountingSlotState.reset()

    nodes, edges, _ = load_baseline_topology()
    inner = RandomIsingFeeder(
        last_proof_block_hash=os.urandom(32),
        miner_bytes=os.urandom(32),
        nodes=nodes,
        edges=edges,
        buffer_size=8,
        max_workers=4,
    )
    feeder = _StarvingFeeder(inner, window=window)
    sampler = CudaSASampler()
    try:
        sampler.prepare(
            num_reads=num_reads, num_sweeps=num_sweeps, num_sweeps_per_beta=1,
        )
        sampler.prepare_self_feeding(
            num_nonces=num_k, reads_per_nonce=num_reads, num_sweeps=num_sweeps,
            num_sweeps_per_beta=1, **sampler._self_feeding_kwargs(),
        )
        first = inner.pop_blocking()
        num_betas, _ = sampler.upload_beta_schedule(
            first.h, first.J, num_sweeps, 1, None, "geometric",
        )
        # poll_timeout turns a stall into a TimeoutError instead of an
        # unbounded block, so the generator can be driven (and cleanly torn
        # down) from this single thread. Pre-fix, every block suicides on the
        # first starve window and no completion arrives for poll_timeout ->
        # TimeoutError; post-fix, supply windows keep completions flowing.
        gen = sampler._run_streaming_loop(
            feeder, num_k=num_k, num_betas=num_betas, seed=1234,
            poll_timeout=poll_timeout,
        )
        yielded = 0
        t0 = time.monotonic()
        try:
            for _model, _ss in gen:
                yielded += 1
                if yielded >= target:
                    break
                if time.monotonic() - t0 > safety_deadline_s:
                    break
        except TimeoutError:
            pass  # stall -> pre-fix; `yielded` stays low, asserted below
        finally:
            gen.close()
    finally:
        inner.stop()
        sampler.close()

    # The starvation path must actually have been exercised (else vacuous).
    assert _CountingSlotState.idle_rotations >= 1, (
        "no nonce ever went idle -- starvation was not induced"
    )
    assert _CountingSlotState.revivals >= 1, "no idle nonce was revived"
    # The decisive regression check: completions must sustain far past the
    # cold-start ceiling. Pre-fix, blocks suicide after the first starve and
    # yield freezes near cold_start_ceiling; post-fix it keeps climbing.
    assert yielded >= target, (
        f"stream stalled after starvation: {yielded} completions "
        f"(cold-start ceiling {cold_start_ceiling}); the GPU block likely "
        f"exited on transient starvation instead of waiting"
    )
