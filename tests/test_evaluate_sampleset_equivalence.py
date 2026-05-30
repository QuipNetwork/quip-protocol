# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Golden-value equivalence tests for evaluate_sampleset + compute_solution_meta.

Pins the current (vectorized) behavior of both functions so Phase 3
multiprocessing integration cannot silently regress chain outputs.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import shared.quantum_proof_of_work as q

# ---------------------------------------------------------------------------
# Scenario definitions: (N, R, seed, bias, diff, live, mindiv)
# ---------------------------------------------------------------------------
_SCENARIOS = [
    (200, 112, 1, -14760, -14850, -14900, 0.2),
    (200, 112, 2, -14920, -14850, -14900, 0.2),
    (150,  80, 3, -14760, -14850, -14900, 0.2),
    (200, 112, 4, -14760, -14850,   None, 0.2),
    (200, 112, 5, -14990, -14850, -14900, 0.2),
    (120,  40, 6, -14760, -14850, -14900, 0.2),
    (200, 112, 7, -14760, -14850, -14900, 0.9),
    (60,   20, 8, -14760, -14850, -14900, 0.2),
]

# ---------------------------------------------------------------------------
# Golden tuples: (energy, diversity, num_valid, submit_floor_energy, n_solutions)
# Captured on 2026-05-30 with the committed vectorized implementation.
# sid is the seed value from _SCENARIOS[i][2].
# ---------------------------------------------------------------------------
_GOLDEN: dict = {
    1: (-14812.204339, 0.495,       0, 30.0, 5),
    2: (-14977.995458, 0.493,       0, 22.0, 5),
    3: (-14846.160425, 0.494666667, 0, 17.0, 5),
    4: (-14808.66764,  0.495,       8, 22.0, 5),
    5: (-15099.460023, 0.493,       0, 16.0, 5),
    6: (-14851.452318, 0.49,        0, 24.0, 5),
    7: None,
    8: (-14874.263836, 0.483333333, 0,  4.0, 5),
}

_COMPUTE_SOLUTION_META_KEYS = frozenset({
    "n_unique_total",
    "n_unique_below_threshold",
    "top_5_diversity",
    "top_5_energy_ceiling",
})


def _run(sid: int, N: int, R: int, seed: int, bias: float,
         diff: float, live, mindiv: float):
    """Reproduce one scenario deterministically and return the MiningResult or None."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N))
    if R > 10:
        samples[3] = samples[2]
        samples[7] = samples[2]
    energies = rng.normal(bias, 60, size=R).astype(np.float64)
    nodes = list(range(N))
    edges = (
        [(i, i + 1) for i in range(0, N - 1, 2)]
        + [(i, i + 3) for i in range(0, N - 3, 5)]
    )
    h = {i: float(rng.choice([-1, 1])) for i in nodes}
    J = {(u, v): float(rng.choice([-1, 1])) for (u, v) in edges}
    ss = SimpleNamespace(record=SimpleNamespace(energy=energies, sample=samples))
    req = SimpleNamespace(difficulty_energy=diff, min_diversity=mindiv, min_solutions=5)
    np.random.seed(999)
    return q.evaluate_sampleset(
        ss, req, nodes, edges,
        nonce=10 ** 6 + sid,
        salt=b"\5" * 32,
        prev_timestamp=0,
        start_time=0.0,
        miner_id="m",
        miner_type="QPU",
        h=h,
        J=J,
        skip_validation=True,
        strict_energy=False,
        live_threshold_energy=live,
    ), ss, diff


@pytest.mark.parametrize("row", _SCENARIOS)
def test_scenario_outputs_match_golden(row):
    N, R, seed, bias, diff, live, mindiv = row
    sid = seed  # seed doubles as scenario id
    r, ss, diff_energy = _run(sid, N, R, seed, bias, diff, live, mindiv)
    golden = _GOLDEN[sid]

    if golden is None:
        assert r is None, f"sid={sid}: expected None result, got {r!r}"
        return

    assert r is not None, f"sid={sid}: expected non-None result"
    actual = (
        round(float(r.energy), 6),
        round(float(r.diversity), 9),
        int(r.num_valid),
        round(float(r.submit_floor_energy), 6),
        len(r.solutions),
    )
    assert actual == golden, (
        f"sid={sid}: golden mismatch\n  expected: {golden}\n  actual:   {actual}"
    )

    # Also assert compute_solution_meta returns expected keys
    meta, _top5, _top5e = q.compute_solution_meta(ss, diff_energy)
    assert set(meta.keys()) == _COMPUTE_SOLUTION_META_KEYS, (
        f"sid={sid}: compute_solution_meta keys mismatch: {set(meta.keys())}"
    )
