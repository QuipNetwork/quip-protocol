# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unit tests for the QPU reconstruction gate + persistent context plumbing."""

from __future__ import annotations

import multiprocessing as mp

import dimod
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


def test_threshold_command_leaves_feeder_and_pending_untouched():
    """A 'threshold' update must NOT reseed the feeder or cancel in-flight work.

    Decay (rule 1) is a gate-only update: it changes what the driver
    reconstructs but leaves the feeder object and every in-flight submission
    in place. A switch (rule: head change) is the only thing that may rebuild
    the feeder / clear ``_pending``. This pins that a threshold command is a
    pure no-op against both.
    """
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(
            ("switch", 3, b"\x01" * 32, b"\x02" * 32, -14900_000, 4, 80.0)
        )
        # Populate _pending with a real in-flight submission (the fake future
        # is already 'done', but it stays in _pending until iter_results pops
        # it — we never iterate here, so it persists for the assertion).
        ctx._submit_one()
        feeder_before = ctx._feeder
        pending_before = dict(ctx._pending)
        assert pending_before, "expected one in-flight submission"

        ctx.apply_command(("threshold", 3, -14800_000))

        assert ctx._feeder is feeder_before  # same object, no reseed/re-fork
        assert ctx._pending == pending_before  # no cancel, in-flight intact
        assert ctx.generation == 3  # no generation bump
        assert ctx._energy_threshold_milli == -14800_000  # gate did move
    finally:
        stop.set()
        ctx.cleanup()


class _GatingSampler(_FakeSampler):
    """Sampler whose results carry a defect so the reconstruction gate runs."""

    def __init__(self, best_energy):
        super().__init__()
        self._best = best_energy
        self.reconstructed_calls = 0

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
        # Build a real dimod SampleSet: the non-reconstruct branch of the gate
        # runs ``_shift_energies`` which needs a recarray-backed sampleset
        # (record.shape/dtype, variables, vartype). A bare stub object would
        # raise AttributeError there before the loosened threshold could be
        # exercised. (Plan-test correction.)
        ss = dimod.SampleSet.from_samples(
            (np.ones((num_reads, 3), np.int8), [0, 1, 2]),
            vartype="SPIN",
            energy=np.full(num_reads, self._best, np.float64),
        )
        fut = type("F", (), {})()
        fut.sampleset = ss
        fut.done = lambda: True
        fut.cancel = lambda: None
        defect = type("D", (), {})()
        defect.energy_offset = 0.0
        return fut, defect

    def reconstruct_full_sampleset(self, raw_ss, defect_info):
        self.reconstructed_calls += 1
        raw_ss.reconstructed = True
        return raw_ss


def test_driver_gate_reconstructs_after_loosening_threshold():
    """First defect-bearing sample enters the empty floor and is reconstructed;
    subsequent samples with the same energy are NOT (floor already full of
    better-or-equal entries). This replaces the old threshold-driven assertion
    now that reconstruction is energy-relative (running best-5 floor), not
    threshold-relative.
    """
    stop = mp.get_context("spawn").Event()
    miner = _FakeMiner()
    miner.sampler = _GatingSampler(best_energy=-14850.0)
    ctx = PersistentStreamContext(
        miner=miner,
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        feeder_buffer_size=4,
        num_reads=4,
        annealing_time=80.0,
        energy_threshold_milli=-14900_000,
        queue_depth=1,
        stop_event=stop,
    )
    try:
        ctx.apply_command(
            ("switch", 1, b"\x01" * 32, b"\x02" * 32, -14900_000, 4, 80.0)
        )
        results = ctx.iter_results()
        # First sample: floor empty → enters best-5 → reconstructed.
        _m, ss, _g = next(results)
        assert ss.reconstructed is True
        # Seed the floor with 5 entries equal to the sample's energy so the
        # floor is now full and the next same-energy sample won't enter it.
        ctx._recon_floor.clear()
        for e in [-14850.0, -14851.0, -14852.0, -14853.0, -14854.0]:
            ctx._recon_floor.append(e)
        ctx._recon_floor.sort()
        # Next sample (-14850.0) is not better than the floor's worst (-14850.0)
        # → NOT reconstructed (only energy-shifted).
        _m, ss2, _g = next(results)
        assert getattr(ss2, "reconstructed", False) is False
    finally:
        stop.set()
        ctx.cleanup()


def test_driver_reconstructs_by_running_best5_floor():
    """The first defect-bearing sample is always reconstructed (floor not full).
    After seeding the floor with 5 better energies, a worse sample is NOT
    reconstructed — only energy-shifted.
    """
    stop = mp.get_context("spawn").Event()
    miner = _FakeMiner()
    # Sampler always returns energy=-14850.0
    miner.sampler = _GatingSampler(best_energy=-14850.0)
    ctx = PersistentStreamContext(
        miner=miner,
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        feeder_buffer_size=4,
        num_reads=4,
        annealing_time=80.0,
        energy_threshold_milli=0,
        queue_depth=1,
        stop_event=stop,
    )
    try:
        ctx.apply_command(
            ("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 4, 80.0)
        )
        # Floor starts empty — first sample enters best-5 and is reconstructed.
        results = ctx.iter_results()
        _m, ss, _g = next(results)
        assert ss.reconstructed is True
        assert miner.sampler.reconstructed_calls == 1

        # Seed the floor with 5 energies all better than -14850 so the floor
        # is full and the next -14850 sample won't enter it.
        ctx._recon_floor.clear()
        for e in [-14900.0, -14895.0, -14890.0, -14885.0, -14880.0]:
            ctx._recon_floor.append(e)
        ctx._recon_floor.sort()

        before = miner.sampler.reconstructed_calls
        _m, ss2, _g = next(results)
        assert getattr(ss2, "reconstructed", False) is False
        assert miner.sampler.reconstructed_calls == before  # no new reconstruction
    finally:
        stop.set()
        ctx.cleanup()


def test_switch_clears_recon_floor():
    """apply_command(("switch", ...)) must reset the running best-5 floor."""
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 4, 80.0))
        # Seed the floor with some values.
        ctx._recon_floor.extend([-100.0, -200.0, -300.0])
        assert len(ctx._recon_floor) == 3

        # A new switch (new head) must clear the floor.
        ctx.apply_command(("switch", 2, b"\x09" * 32, b"\x02" * 32, 0, 4, 80.0))
        assert ctx._recon_floor == []
    finally:
        stop.set()
        ctx.cleanup()


def test_pause_command_does_not_cancel_inflight():
    """A 'pause' is a drain-and-idle signal: it stops NEW submissions but must
    leave every in-flight future in place so we still consume what we paid for.
    """
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(("switch", 4, b"\x01" * 32, b"\x02" * 32, 0, 4, 80.0))
        ctx._submit_one()
        ctx._submit_one()
        pending_before = dict(ctx._pending)
        assert len(pending_before) == 2

        ctx.apply_command(("pause", 4))

        assert ctx._paused is True
        assert ctx._pending == pending_before  # NOT cancelled — drains normally
        assert ctx.generation == 4  # pause does not bump the generation
    finally:
        stop.set()
        ctx.cleanup()


def test_pause_stops_new_submissions_and_drains_then_returns():
    """While paused, iter_results refills nothing new, yields the existing
    in-flight set until it empties, then returns so the driver can idle.
    """
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(("switch", 9, b"\x01" * 32, b"\x02" * 32, 0, 4, 80.0))
        ctx._submit_one()
        ctx._submit_one()
        submitted_before = ctx._job_index
        assert submitted_before == 2

        ctx.apply_command(("pause", 9))
        drained = list(ctx.iter_results())  # fake futures are 'done' → drains

        assert len(drained) == 2  # exactly the two in-flight, nothing more
        assert ctx._job_index == submitted_before  # no new _submit_one calls
        assert not ctx._pending  # fully drained
    finally:
        stop.set()
        ctx.cleanup()


def test_pause_with_empty_pending_returns_immediately():
    """Paused with nothing in flight → iter_results returns at once (idle)."""
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 4, 80.0))
        ctx.apply_command(("pause", 1))
        assert list(ctx.iter_results()) == []
        assert ctx._job_index == 0  # never submitted anything
    finally:
        stop.set()
        ctx.cleanup()


def test_switch_clears_paused_to_resume():
    """A subsequent switch (new head) resumes a paused context."""
    stop = mp.get_context("spawn").Event()
    ctx = _make_ctx(stop)
    try:
        ctx.apply_command(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 4, 80.0))
        ctx.apply_command(("pause", 1))
        assert ctx._paused is True
        ctx.apply_command(("switch", 2, b"\x09" * 32, b"\x02" * 32, 0, 4, 80.0))
        assert ctx._paused is False
        assert ctx.generation == 2
    finally:
        stop.set()
        ctx.cleanup()


def test_build_persistent_context_forwards_topology():
    from unittest.mock import patch
    from dwave_topologies import DEFAULT_TOPOLOGY
    import QPU.dwave_miner as dm

    with patch.object(dm, "DWaveMiner") as mk:
        dm.build_persistent_context(
            miner_id="m", queue_depth=2, nodes=[0, 1, 2], edges=[(0, 1)],
            feeder_buffer_size=4, num_reads=4, annealing_time=80.0,
            energy_threshold_milli=0,
            topology=DEFAULT_TOPOLOGY,
        )
    _a, kwargs = mk.call_args
    assert kwargs["topology"] is DEFAULT_TOPOLOGY


def test_build_persistent_context_requires_topology():
    import pytest
    import QPU.dwave_miner as dm

    with pytest.raises(ValueError, match="requires a topology"):
        dm.build_persistent_context(
            miner_id="m", queue_depth=2, nodes=[0, 1, 2], edges=[(0, 1)],
            feeder_buffer_size=4, num_reads=4, annealing_time=80.0,
            energy_threshold_milli=0,
            topology=None,
        )
