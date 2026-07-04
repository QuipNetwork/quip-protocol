"""Selection contract for ``min_solutions == 1`` rounds.

The chain pallet filters each submitted solution with strict
``energy < max_energy_milli`` and needs only ``min_solutions`` survivors,
so a 1-solution round should ship exactly the best-energy row. Before the
fast path in ``_select_diverse_with_fallback``, ``select_diverse_solutions``
seeded farthest-point sampling with the two most-distant rows and never
trimmed, so ``target_count=1`` shipped 2 solutions and
``submit_floor_energy`` (max over the selected set's chain-recomputed
energies) was dragged above the best row by a farthest-point sibling —
observed in production 2026-07-04 where a floor of -14503 gated a -14519
candidate that the chain would have accepted.

The fast path is guarded on ``min_diversity <= 0``: the chain scores
1-solution diversity as 0 and gates ``diversity >= min_diversity``
unconditionally, so with ``min_diversity > 0`` the padded pair must be
kept (a single solution would be structurally rejected on-chain).
"""

import time

import dimod
import pytest

from shared.miner_types import BlockRequirements
from shared.quantum_proof_of_work import evaluate_sampleset

# Ferromagnetic chain on 4 spins: h all -1, J all -1.
NODES = [0, 1, 2, 3]
EDGES = [(0, 1), (1, 2), (2, 3)]
H = {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}
J = {(0, 1): -1.0, (1, 2): -1.0, (2, 3): -1.0}

# True Ising energies (manually verified):
#   A = [1, 1, 1, 1]   -> -7
#   B = [1, 1, 1, -1]  -> -3
#   C = [1, -1, -1, 1] -> +1
SAMPLE_A = [1, 1, 1, 1]
SAMPLE_B = [1, 1, 1, -1]
SAMPLE_C = [1, -1, -1, 1]


def _evaluate(samples, energies, *, min_solutions, min_diversity):
    sampleset = dimod.SampleSet.from_samples(
        samples, vartype=dimod.SPIN, energy=energies,
    )
    requirements = BlockRequirements(
        difficulty_energy=-2.0,
        min_diversity=min_diversity,
        min_solutions=min_solutions,
        timeout_to_difficulty_adjustment_decay=10_000,
    )
    return evaluate_sampleset(
        sampleset, requirements, NODES, EDGES,
        nonce=b"\x00" * 32, salt=b"\x00" * 32,
        prev_timestamp=0, start_time=time.time(),
        miner_id="test", miner_type="CPU",
        h=H, J=J,
        # Lenient — mirrors the substrate ratchet path.
        strict_energy=False,
    )


def test_min_solutions_one_selects_single_best_energy_row():
    """min_solutions=1 + diversity unenforced ships exactly the argmin row.

    Sampler energies are order-preserving but slightly off the true Ising
    values, so the assertions distinguish the two sources: ``energy`` is
    the selected row's sampler value, ``submit_floor_energy`` is that same
    row's chain recompute — NOT a farthest-point sibling's (+1 for C).
    """
    result = _evaluate(
        [SAMPLE_A, SAMPLE_B, SAMPLE_C],
        [-6.9, -2.9, 1.1],
        min_solutions=1, min_diversity=0.0,
    )
    assert result is not None
    assert len(result.solutions) == 1, (
        "min_solutions=1 must ship exactly one solution, "
        f"got {len(result.solutions)}"
    )
    assert result.solutions[0] == SAMPLE_A
    assert result.energy == pytest.approx(-6.9)
    assert result.submit_floor_energy == pytest.approx(-7.0), (
        "floor must be the selected row's own chain recompute, not a "
        f"farthest-point sibling's; got {result.submit_floor_energy}"
    )
    assert result.diversity == 0.0


def test_min_solutions_one_with_min_diversity_keeps_pair():
    """With min_diversity > 0 the padded farthest pair is preserved.

    Flip-invariant distances: d(A,B)=0.25, d(B,C)=0.25, d(A,C)=0.5, so
    farthest-point seeding picks (A, C) with diversity 0.5 — a single
    solution would score 0 and be rejected by the chain's diversity gate.
    """
    result = _evaluate(
        [SAMPLE_A, SAMPLE_B, SAMPLE_C],
        [-7.0, -3.0, 1.0],
        min_solutions=1, min_diversity=0.3,
    )
    assert result is not None
    assert len(result.solutions) == 2, (
        "min_diversity>0 must keep the diverse pair, "
        f"got {len(result.solutions)}"
    )
    assert result.diversity == pytest.approx(0.5)
    # Floor is still the worst of the selected set (C's recompute).
    assert result.submit_floor_energy == pytest.approx(1.0)


def test_min_solutions_one_single_valid_row():
    """A pool of exactly one row round-trips through the fast path."""
    result = _evaluate(
        [SAMPLE_A], [-7.0],
        min_solutions=1, min_diversity=0.0,
    )
    assert result is not None
    assert len(result.solutions) == 1
    assert result.solutions[0] == SAMPLE_A
    assert result.submit_floor_energy == pytest.approx(-7.0)
    assert result.diversity == 0.0
