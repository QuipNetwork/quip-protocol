# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 QUIP Protocol Contributors

"""End-to-end QPU rescore pipeline on a synthetic archive (no QPU).

Chain under test: SubmissionArchiver (real writer) -> Cell loader ->
gate_replica Layer A/B -> scores npz -> exact replay against the REAL
``evaluate_sampleset`` -> estimators. The exactness claim of the whole
offline study rests on the validate step returning zero mismatches, so
this test IS the "exact, with proof" gate, run on planted-structure
data where dedup, Z2 twins, column permutation, and pool-size edges all
occur by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_STUDY = Path(__file__).resolve().parent.parent / "test_results" / "qpu_reads_diversity"
sys.path.insert(0, str(_STUDY))

from make_synthetic_archive import make_cell  # noqa: E402
from qpu_archive_lib import Cell  # noqa: E402
from rescore import rescore_cell  # noqa: E402
from rescore_validate import validate  # noqa: E402
import estimators  # noqa: E402

E_GRID = [-105.0, -100.0, -95.0, -90.0, -85.0]
R_GRID = [1, 2, 4, 8]  # plus the automatic full-reads rung (16)


@pytest.fixture(scope="module")
def cell_and_scores(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("qpu_synth")
    cell_dir = make_cell(tmp / "synth_a080_r016", n_sub=60, num_reads=16,
                         n_nodes=64, seed=42, zero_field=True)
    out = {}
    for gauge in ("raw", "fixed"):
        out[gauge] = rescore_cell(
            cell_dir, gauge, "random", seed=1, workers=1,
            out_dir=tmp / "scores", e_grid=E_GRID, r_grid=R_GRID,
        )
    return cell_dir, out


def test_validate_raw_lens_exact(cell_and_scores):
    cell_dir, scores = cell_and_scores
    n_checked, n_flagged, mismatches = validate(
        cell_dir, scores["raw"], n_combos=150, kd_per_combo=3, seed=7,
    )
    assert n_checked > 300
    assert mismatches == [], mismatches[:5]


def test_validate_fixed_lens_exact(cell_and_scores):
    cell_dir, scores = cell_and_scores
    n_checked, n_flagged, mismatches = validate(
        cell_dir, scores["fixed"], n_combos=150, kd_per_combo=3, seed=11,
    )
    assert n_checked > 300
    assert mismatches == [], mismatches[:5]


def test_gauge_lens_collapses_twins(cell_and_scores):
    _, scores = cell_and_scores
    raw = np.load(scores["raw"])
    fixed = np.load(scores["fixed"])
    # Same subsets (same seed) -> the gauge-fixed unique count can never
    # exceed the raw count, and with 30% planted twins it must be
    # strictly lower somewhere.
    for r in [1, 2, 4, 8, 16]:
        nu_raw, nu_fix = raw[f"n_unique_r{r}"], fixed[f"n_unique_r{r}"]
        assert (nu_fix <= nu_raw).all()
    assert (fixed["n_unique_r16"] < raw["n_unique_r16"]).any()


def test_hypergeometric_anchor(cell_and_scores):
    cell_dir, scores = cell_and_scores
    cell = Cell(cell_dir)
    sc = np.load(scores["raw"])
    # k=1, D=0 yield from random subsets must match the closed form
    # within Monte Carlo noise — the variance-free check of the whole
    # subset machinery.
    for r in (2, 8):
        a = estimators.hypergeom_anchor(cell, sc, e=-95.0, r=r)
        assert abs(a["z"]) < 4.0, a
        assert a["y_exact"] > 0  # planted band guarantees events


def test_yield_monotone_in_reads_at_k1_d0(cell_and_scores):
    _, scores = cell_and_scores
    sc = np.load(scores["raw"])
    e_idx = 2  # -95
    ys = [float(estimators.yield_hat(sc, r, 1, 0.0)[e_idx])
          for r in [1, 2, 4, 8, 16]]
    # More reads can only help P(any read below E'); allow MC slack.
    for lo, hi in zip(ys, ys[1:]):
        assert hi >= lo - 0.02, ys


def test_access_cost_model(cell_and_scores):
    cell_dir, _ = cell_and_scores
    cell = Cell(cell_dir)
    acc8 = cell.access_us(8)
    acc16 = cell.access_us(16)
    per_read = 245.0 + 20.5 + 80.0
    np.testing.assert_allclose(acc16 - acc8, 8 * per_read)
    measured = cell.access_us(None)
    np.testing.assert_allclose(measured, acc16, rtol=0.05)
