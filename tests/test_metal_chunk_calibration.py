# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Off-Metal unit tests for MetalSASampler sweep-chunk self-calibration.

Exercise the pure chunk-sizing arithmetic (``_next_beta_chunk_size`` /
``_record_chunk_timing`` / ``_maybe_reset_chunk_calibration``) without a GPU by
constructing the sampler via ``__new__`` (skips the Metal-device ``__init__``),
so they run in CI on any platform — the real-Metal path only proves bit-identity
(``test_metal_chunking.py``), never the controller's own sizing decisions.
"""
from GPU.metal_sa import MetalSASampler, _INITIAL_BETAS_PER_CHUNK


def _calibrator() -> MetalSASampler:
    """A MetalSASampler with only the chunk-calibration state initialised."""
    s = MetalSASampler.__new__(MetalSASampler)
    s._betas_per_chunk = None
    s._chunk_cal_threads = None
    return s


def test_next_beta_chunk_size_seeds_on_first_use():
    s = _calibrator()
    assert s._next_beta_chunk_size(1000) == _INITIAL_BETAS_PER_CHUNK
    assert s._betas_per_chunk == _INITIAL_BETAS_PER_CHUNK


def test_next_beta_chunk_size_bounded_by_remaining():
    s = _calibrator()
    s._betas_per_chunk = 50
    assert s._next_beta_chunk_size(10) == 10   # final short chunk
    assert s._next_beta_chunk_size(1) == 1     # last beta
    assert s._next_beta_chunk_size(80) == 50   # capped by converged size


def test_record_chunk_timing_shrinks_when_slower_than_target():
    s = _calibrator()
    s._betas_per_chunk = 10
    # 10 betas took 40 ms against an 8 ms target → shrink toward ~2.
    s._record_chunk_timing(elapsed_s=0.040, betas=10, target_ms=8.0)
    assert 1 <= s._betas_per_chunk < 10


def test_record_chunk_timing_grows_when_faster_than_target():
    s = _calibrator()
    s._betas_per_chunk = 4
    # 4 betas took 2 ms against an 8 ms target → grow toward ~16.
    s._record_chunk_timing(elapsed_s=0.002, betas=4, target_ms=8.0)
    assert s._betas_per_chunk > 4


def test_record_chunk_timing_floors_elapsed_to_avoid_div_zero():
    s = _calibrator()
    s._betas_per_chunk = 5
    # elapsed_s == 0 must not raise / produce inf; the 0.05 ms floor caps growth.
    s._record_chunk_timing(elapsed_s=0.0, betas=5, target_ms=8.0)
    assert 1 <= s._betas_per_chunk <= 1_000_000


def test_record_chunk_timing_never_below_one():
    s = _calibrator()
    s._betas_per_chunk = 1
    # A single beta that massively overruns the target stays floored at 1.
    s._record_chunk_timing(elapsed_s=5.0, betas=1, target_ms=8.0)
    assert s._betas_per_chunk == 1


def test_maybe_reset_chunk_calibration_resets_on_workload_change():
    s = _calibrator()
    s._betas_per_chunk = 200      # converged on a cheaper prior workload
    s._chunk_cal_threads = 1000
    # A heavier workload (different thread count) must drop the carried size so
    # the first new chunk can't run oversized and reintroduce the UI freeze.
    s._maybe_reset_chunk_calibration(45000)
    assert s._chunk_cal_threads == 45000
    assert s._betas_per_chunk is None


def test_maybe_reset_chunk_calibration_keeps_size_when_workload_stable():
    s = _calibrator()
    s._betas_per_chunk = 12
    s._chunk_cal_threads = 45000
    # Same workload → keep the converged size (don't discard calibration).
    s._maybe_reset_chunk_calibration(45000)
    assert s._betas_per_chunk == 12
