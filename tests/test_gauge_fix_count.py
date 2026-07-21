# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 QUIP Protocol Contributors

"""Flip-invariant (Z2-gauge) count gate for zero-field Ising instances.

With ``h = 0`` the Ising energy is exactly flip-invariant (``E(s) == E(-s)``),
so every solution row has an equal-energy twin ``-s``. The raw row dedup in
``compute_solution_meta`` counts twins separately, inflating
``n_unique_below_threshold``. ``gauge_fix=True`` canonicalizes rows (anchor
spin = +1) before dedup so twins collapse. These tests pin:

  * ``gauge_canonicalize`` maps ``-s`` onto ``s`` and leaves canonical rows be;
  * ``gauge_fix=True`` halves a pure-twin below-threshold count (the fix);
  * with no twins (the ``h != 0`` case) the flag is a no-op for both the count
    and ``top_5_diversity`` — so production / ternary callers are unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import shared.quantum_proof_of_work as q


def _sampleset(sample: np.ndarray, energy: np.ndarray):
    """Minimal dimod-style sampleset exposing ``record.sample`` / ``.energy``."""
    return SimpleNamespace(record=SimpleNamespace(sample=sample, energy=energy))


# A pair of distinct base solutions and their spin-flip twins (anchor spin is
# column 0; A and B are already canonical with column0 = +1).
_A = np.array([1, 1, 1, -1, -1, -1], dtype=np.int8)
_B = np.array([1, -1, 1, -1, 1, -1], dtype=np.int8)


def test_gauge_canonicalize_collapses_twins():
    rows = np.stack([_A, -_A, _B, -_B])
    out = q.gauge_canonicalize(rows)
    # Every row now has anchor spin +1.
    assert np.all(out[:, 0] == 1)
    # Twins mapped onto their canonical base; canonical rows untouched.
    np.testing.assert_array_equal(out[0], _A)
    np.testing.assert_array_equal(out[1], _A)
    np.testing.assert_array_equal(out[2], _B)
    np.testing.assert_array_equal(out[3], _B)


def test_gauge_canonicalize_empty_is_safe():
    empty = np.empty((0, 6), dtype=np.int8)
    out = q.gauge_canonicalize(empty)
    assert out.shape == (0, 6)


def test_gauge_fix_halves_twin_count():
    # Four below-threshold rows: A, -A, B, -B — two physical solutions.
    sample = np.stack([_A, -_A, _B, -_B])
    energy = np.array([-10.0, -10.0, -10.0, -10.0])
    ss = _sampleset(sample, energy)

    raw, _, _ = q.compute_solution_meta(ss, threshold=0.0, gauge_fix=False)
    fixed, _, _ = q.compute_solution_meta(ss, threshold=0.0, gauge_fix=True)

    # Raw dedup sees four "unique" rows; the gauge collapses the two twin pairs.
    assert raw["n_unique_below_threshold"] == 4
    assert fixed["n_unique_below_threshold"] == 2
    assert fixed["n_unique_total"] == 2


def test_gauge_fix_noop_without_twins():
    # Distinct, already-canonical rows (no twins) — the h != 0 regime. The flag
    # must not change the count or the diversity: raw ≈ flip-invariant.
    c = np.array([1, 1, -1, 1, -1, 1], dtype=np.int8)
    sample = np.stack([_A, _B, c])
    energy = np.array([-10.0, -9.0, -8.0])
    ss = _sampleset(sample, energy)

    raw, _, _ = q.compute_solution_meta(ss, threshold=0.0, gauge_fix=False)
    fixed, _, _ = q.compute_solution_meta(ss, threshold=0.0, gauge_fix=True)

    assert raw["n_unique_below_threshold"] == fixed["n_unique_below_threshold"] == 3
    assert raw["top_5_diversity"] == fixed["top_5_diversity"]


def test_gauge_fix_default_is_off():
    # Default call (no gauge_fix kwarg) matches gauge_fix=False exactly, so
    # every existing caller is byte-for-byte unchanged.
    sample = np.stack([_A, -_A, _B])
    energy = np.array([-10.0, -10.0, -9.0])
    ss = _sampleset(sample, energy)

    default, _, _ = q.compute_solution_meta(ss, threshold=0.0)
    explicit_off, _, _ = q.compute_solution_meta(ss, threshold=0.0, gauge_fix=False)
    assert default == explicit_off
