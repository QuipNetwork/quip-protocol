# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unit tests for the QPU reconstruction gate + persistent context plumbing."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from QPU.dwave_miner import PersistentStreamContext, _should_reconstruct


def test_should_reconstruct_strict_below_threshold():
    # approx = best + offset = -14910; threshold -14900 -> below -> reconstruct.
    assert _should_reconstruct(-14950.0, 40.0, -14900.0) is True


def test_should_reconstruct_above_threshold_is_false():
    # approx = -14850; threshold -14900 -> above -> do not reconstruct.
    assert _should_reconstruct(-14890.0, 40.0, -14900.0) is False


def test_should_reconstruct_tracks_loosened_threshold():
    """A candidate yielded raw at a strict threshold reconstructs once the
    threshold loosens past it — the decay 'near-live' guarantee (rule 2)."""
    best, offset = -14890.0, 40.0  # approx = -14850
    assert _should_reconstruct(best, offset, -14900.0) is False  # strict
    assert _should_reconstruct(best, offset, -14800.0) is True  # loosened


class _FakeSampler:
    """Minimal DWaveMiner.sampler stand-in: no defect path, echoes energies."""

    job_label = "fake"

    def __init__(self):
        self.closed = False

    def sample_ising_async(
        self,
        h,
        J,
        *,
        num_reads,
        answer_mode,
        annealing_time,
        label,
        nonce_seed,
    ):
        rec = type("R", (), {})()
        rec.sample = np.ones((num_reads, 3), np.int8)
        rec.energy = np.full(num_reads, -14900.0, np.float64)
        ss = type("SS", (), {})()
        ss.record = rec
        ss.info = {"timing": {"qpu_programming_time": 1, "qpu_sampling_time": 2}}
        fut = type("F", (), {})()
        fut.sampleset = ss
        fut.done = lambda: True
        fut.cancel = lambda: None
        # defect_info None -> raw sampleset path (no reconstruction gate).
        return fut, None

    def close(self):
        self.closed = True


class _FakeMiner:
    def __init__(self):
        self.sampler = _FakeSampler()


def _make_ctx(stop_event):
    ctx = PersistentStreamContext(
        miner=_FakeMiner(),
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        feeder_buffer_size=4,
        num_reads=4,
        annealing_time=80.0,
        energy_threshold_milli=0,
        precheck_margin_milli=2000,
        queue_depth=2,
        stop_event=stop_event,
    )
    return ctx


def test_context_tags_results_with_current_generation():
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(
            ("switch", 7, b"\x01" * 32, b"\x02" * 32, 0, 4, 80.0),
        )
        gen_before = ctx.generation
        results = ctx.iter_results()
        model, ss, submit_gen = next(results)
        assert submit_gen == gen_before == 7
        assert ss.record.sample.shape[1] == 3
    finally:
        stop.set()
        ctx.cleanup()


def test_context_reseed_keeps_pool_across_switch():
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 4, 80.0))
        pool = ctx._feeder._pool
        ctx.apply_command(("switch", 2, b"\x09" * 32, b"\x02" * 32, 0, 4, 80.0))
        assert ctx.generation == 2
        assert ctx._feeder._pool is pool  # reseed, not re-fork
    finally:
        stop.set()
        ctx.cleanup()


def test_context_threshold_command_does_not_bump_generation():
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(
            ("switch", 5, b"\x01" * 32, b"\x02" * 32, -14900_000, 4, 80.0)
        )
        ctx.apply_command(("threshold", 5, -14800_000))
        assert ctx.generation == 5
        assert ctx._energy_threshold_milli == -14800_000
    finally:
        stop.set()
        ctx.cleanup()
