# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Equivalence: _SharedSampleSet feeds evaluate/meta identically to dimod."""
from __future__ import annotations

from types import SimpleNamespace

import dimod
import numpy as np

import shared.quantum_proof_of_work as q
from shared.base_miner import _SharedSampleSet


def _req():
    return SimpleNamespace(difficulty_energy=-14850.0, min_diversity=0.2,
                           min_solutions=5)


def test_shim_matches_dimod_for_meta_and_evaluate():
    rng = np.random.default_rng(11)
    N, R = 200, 112
    sample = rng.choice(np.array([-1, 1], np.int8), size=(R, N))
    energy = rng.normal(-14800, 60, size=R).astype(np.float64)
    nodes = list(range(N))
    edges = [(i, i + 1) for i in range(0, N - 1, 2)]
    h = {i: 1.0 for i in nodes}
    J = {(u, v): 1.0 for (u, v) in edges}

    shim = _SharedSampleSet(sample, energy)
    dss = dimod.SampleSet.from_samples(sample, vartype=dimod.SPIN, energy=energy)

    m_shim = q.compute_solution_meta(shim, -14850.0)[0]
    m_dss = q.compute_solution_meta(dss, -14850.0)[0]
    assert m_shim == m_dss

    def run(ss):
        np.random.seed(7)
        return q.evaluate_sampleset(ss, _req(), nodes, edges, nonce=42,
                                    salt=b"\1" * 32, prev_timestamp=0, start_time=0.0,
                                    miner_id="m", miner_type="QPU", h=h, J=J,
                                    skip_validation=True, strict_energy=False,
                                    live_threshold_energy=-14900.0)
    r_shim, r_dss = run(shim), run(dss)
    assert (r_shim is None) == (r_dss is None)
    if r_shim is not None:
        assert round(r_shim.energy, 6) == round(r_dss.energy, 6)
        assert round(r_shim.diversity, 9) == round(r_dss.diversity, 9)
        assert r_shim.num_valid == r_dss.num_valid
        assert [list(map(int, s)) for s in r_shim.solutions] == \
               [list(map(int, s)) for s in r_dss.solutions]
