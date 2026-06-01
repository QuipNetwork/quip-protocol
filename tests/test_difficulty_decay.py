"""Parity tests for ``substrate.difficulty_decay`` against the Rust impl.

The Rust source of truth lives at
``pallets/quantum-pow/src/difficulty.rs``; structural invariants are
exercised in ``pallets/quantum-pow/src/tests.rs``. This file mirrors
the same invariants in Python and pins a few exact (input, output)
pairs at the milli — drift here means a future miner build will
silently disagree with the chain about which proofs are eligible.

The ``test_curve_*`` set use the same 2-node / 1-edge mock topology
the Rust tests use (``test_curve()`` in tests.rs) so anyone can cross-
check by reading off corresponding Rust assertions.
"""
from __future__ import annotations

import math

import pytest

from substrate.difficulty_decay import (
    DECAY_RATE_MILLI,
    MIN_DECAY_DELTA_MILLI,
    Direction,
    EnergyCurve,
    adjust_energy_along_curve,
    apply_decay,
    current_difficulty,
)
from substrate.types import SubstrateDifficulty


# ----------------------------------------------------------------------
# Fixtures mirroring tests.rs::test_curve()
# ----------------------------------------------------------------------

# tests.rs::test_curve = EnergyCurve::new(2, 1, 700, 750, 800)
TEST_NODES = 2
TEST_EDGES = 1
TEST_C_EASY_MILLI = 700
TEST_C_KNEE_MILLI = 750
TEST_C_HARD_MILLI = 800


@pytest.fixture
def test_curve() -> EnergyCurve:
    return EnergyCurve.from_topology(
        TEST_NODES,
        TEST_EDGES,
        TEST_C_EASY_MILLI,
        TEST_C_KNEE_MILLI,
        TEST_C_HARD_MILLI,
    )


@pytest.fixture
def base_difficulty() -> SubstrateDifficulty:
    return SubstrateDifficulty(
        min_solutions=5,
        max_energy_milli=-2_500,
        min_diversity_milli=200,
    )


# ----------------------------------------------------------------------
# EnergyCurve.from_topology — numerical anchors
# ----------------------------------------------------------------------


def _expected_gse_milli(n: int, m: int, c: float) -> int:
    """Independent recomputation of the chain's GSE formula."""
    avg_degree = (2.0 * m) / n
    sqrt_avg = math.sqrt(avg_degree)
    j_contribution = -c * sqrt_avg * n
    h_contribution = -c * 0.88 * (2.0 / 3.0) * n / sqrt_avg
    raw = (j_contribution + h_contribution) * 1000.0
    # Half-away-from-zero, matching libm::round.
    if raw >= 0.0:
        return int(math.floor(raw + 0.5))
    return int(math.ceil(raw - 0.5))


def test_energy_curve_matches_chain_formula() -> None:
    curve = EnergyCurve.from_topology(
        TEST_NODES,
        TEST_EDGES,
        TEST_C_EASY_MILLI,
        TEST_C_KNEE_MILLI,
        TEST_C_HARD_MILLI,
    )
    assert curve.min_milli == _expected_gse_milli(TEST_NODES, TEST_EDGES, 0.800)
    assert curve.knee_milli == _expected_gse_milli(TEST_NODES, TEST_EDGES, 0.750)
    assert curve.max_milli == _expected_gse_milli(TEST_NODES, TEST_EDGES, 0.700)
    # Sanity: ordering invariant the Rust EnergyCurve doc claims.
    assert curve.min_milli < curve.knee_milli < curve.max_milli


def test_energy_curve_realistic_topology() -> None:
    """Z(9,2) numbers — anchor for the production curve.

    Production uses ~1368 nodes / 7692 edges. The expected_gse_with_c
    formula should produce results in the documented Z(9,2) range
    of roughly -4100 to -3870 milli.
    """
    curve = EnergyCurve.from_topology(1368, 7692, 700, 750, 800)
    # The exact values depend on the formula; pin them by recomputing.
    assert curve.min_milli == _expected_gse_milli(1368, 7692, 0.800)
    assert curve.max_milli == _expected_gse_milli(1368, 7692, 0.700)
    assert curve.min_milli < curve.knee_milli < curve.max_milli


def test_energy_curve_zero_nodes_or_edges() -> None:
    """Mirrors the Rust ``if num_nodes == 0 || num_edges == 0`` guard."""
    curve = EnergyCurve.from_topology(0, 1, 700, 750, 800)
    assert curve.min_milli == 0
    assert curve.knee_milli == 0
    assert curve.max_milli == 0
    curve = EnergyCurve.from_topology(2, 0, 700, 750, 800)
    assert curve.min_milli == 0


# ----------------------------------------------------------------------
# current_difficulty — short-circuits + decay step accounting
# ----------------------------------------------------------------------


def test_current_difficulty_passes_through_when_no_elapsed(
    test_curve, base_difficulty,
) -> None:
    """Mirrors tests.rs::current_difficulty_passes_through_when_no_decay_steps."""
    # block_number == last_proof_block: zero elapsed → no decay.
    assert current_difficulty(100, base_difficulty, 100, 10, test_curve) == base_difficulty
    # Less than one full epoch elapsed: still no decay.
    assert current_difficulty(109, base_difficulty, 100, 10, test_curve) == base_difficulty


def test_current_difficulty_applies_decay_per_full_epoch(
    test_curve, base_difficulty,
) -> None:
    """Mirrors tests.rs::current_difficulty_applies_decay_per_full_epoch.

    25 blocks elapsed at epoch_length=10 → 2 decay steps.
    """
    result = current_difficulty(125, base_difficulty, 100, 10, test_curve)
    expected = apply_decay(base_difficulty, 2, test_curve)
    assert result == expected


def test_current_difficulty_short_circuits_without_curve(base_difficulty) -> None:
    """Mirrors tests.rs::current_difficulty_short_circuits_without_curve."""
    assert current_difficulty(200, base_difficulty, 100, 10, None) == base_difficulty


def test_current_difficulty_short_circuits_at_genesis(test_curve, base_difficulty) -> None:
    """Mirrors tests.rs::current_difficulty_short_circuits_at_genesis.

    ``last_proof_block == 0`` is the genesis sentinel: no proof has
    ever won, so no decay applies.
    """
    assert current_difficulty(500, base_difficulty, 0, 10, test_curve) == base_difficulty


def test_current_difficulty_short_circuits_at_zero_epoch_length(
    test_curve, base_difficulty,
) -> None:
    """``epoch_length == 0`` would divide-by-zero; chain returns base."""
    assert current_difficulty(500, base_difficulty, 100, 0, test_curve) == base_difficulty


# ----------------------------------------------------------------------
# apply_decay — invariants from tests.rs
# ----------------------------------------------------------------------


def test_apply_decay_only_mutates_max_energy(test_curve) -> None:
    """Mirrors tests.rs::apply_decay_only_mutates_max_energy."""
    before = SubstrateDifficulty(
        min_solutions=7,
        max_energy_milli=-2_500,
        min_diversity_milli=400,
    )
    after = apply_decay(before, 3, test_curve)
    assert after.min_solutions == before.min_solutions
    assert after.min_diversity_milli == before.min_diversity_milli
    # Decay eases (moves toward zero).
    assert after.max_energy_milli > before.max_energy_milli


def test_apply_decay_zero_steps_is_identity(test_curve, base_difficulty) -> None:
    assert apply_decay(base_difficulty, 0, test_curve) == base_difficulty


def test_apply_decay_is_monotonic_easing(test_curve, base_difficulty) -> None:
    """Each step must move toward (or stay at) the easier side."""
    current = base_difficulty
    for _ in range(10):
        next_d = apply_decay(current, 1, test_curve)
        assert next_d.max_energy_milli >= current.max_energy_milli
        current = next_d


# ----------------------------------------------------------------------
# adjust_energy_along_curve — algorithm-level edge cases
# ----------------------------------------------------------------------


def test_adjust_energy_along_curve_degenerate_curve_is_no_op() -> None:
    """``total_range <= 0`` → leave current alone (matches Rust guard)."""
    flat = EnergyCurve(min_milli=-1000, knee_milli=-1000, max_milli=-1000)
    assert adjust_energy_along_curve(
        -1000, DECAY_RATE_MILLI, Direction.EASIER, flat, MIN_DECAY_DELTA_MILLI,
    ) == -1000


def test_adjust_energy_along_curve_out_of_range_is_linear(test_curve) -> None:
    """current outside [min, max] → linear fallback (matches Rust)."""
    below = test_curve.min_milli - 100
    result = adjust_energy_along_curve(
        below, DECAY_RATE_MILLI, Direction.EASIER, test_curve, MIN_DECAY_DELTA_MILLI,
    )
    total_range = test_curve.max_milli - test_curve.min_milli
    expected_delta = max(
        MIN_DECAY_DELTA_MILLI,
        _libm_round(total_range * DECAY_RATE_MILLI / 1000.0),
    )
    assert result == below + expected_delta


def test_adjust_energy_along_curve_min_delta_floor_applies(test_curve) -> None:
    """A tiny rate that would round to 0 still moves by min_delta_milli.

    Mirrors tests.rs::adjust_energy_along_curve_applies_min_delta_floor
    behaviour: ``raw_delta_f`` in (0, 0.5) rounds to 0, the floor kicks
    in and the adjustment proceeds.
    """
    start = (test_curve.min_milli + test_curve.knee_milli) // 2
    # Rate of 1‰ → raw_delta would be sub-1, well under the round
    # boundary on the test_curve's narrow range.
    result = adjust_energy_along_curve(
        start, 1, Direction.EASIER, test_curve, MIN_DECAY_DELTA_MILLI,
    )
    # Must have moved by exactly the floor.
    assert result == start + MIN_DECAY_DELTA_MILLI


def test_adjust_energy_along_curve_harder_decreases(test_curve) -> None:
    """Direction.HARDER moves toward min_milli (more negative)."""
    start = (test_curve.min_milli + test_curve.max_milli) // 2
    harder = adjust_energy_along_curve(
        start, DECAY_RATE_MILLI, Direction.HARDER, test_curve, MIN_DECAY_DELTA_MILLI,
    )
    assert harder < start


def test_adjust_energy_along_curve_easier_increases(test_curve) -> None:
    """Direction.EASIER moves toward max_milli (less negative)."""
    start = (test_curve.min_milli + test_curve.max_milli) // 2
    easier = adjust_energy_along_curve(
        start, DECAY_RATE_MILLI, Direction.EASIER, test_curve, MIN_DECAY_DELTA_MILLI,
    )
    assert easier > start


# ----------------------------------------------------------------------
# libm::round semantics — banker's rounding would fail these
# ----------------------------------------------------------------------


def _libm_round(x: float) -> int:
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


# ----------------------------------------------------------------------
# build_decay_schedule + step_for_energy
# ----------------------------------------------------------------------


def _curve() -> EnergyCurve:
    return EnergyCurve(min_milli=-16000_000, knee_milli=-15500_000, max_milli=-14000_000)


def _base() -> SubstrateDifficulty:
    return SubstrateDifficulty(
        min_solutions=5, max_energy_milli=-15123_000, min_diversity_milli=200,
    )


def test_schedule_matches_stepwise_apply_decay() -> None:
    from substrate.difficulty_decay import build_decay_schedule

    base, curve, horizon = _base(), _curve(), 20
    sched = build_decay_schedule(base.max_energy_milli, curve, horizon)
    assert len(sched) == horizon + 1
    assert sched[0] == base.max_energy_milli
    for s in range(horizon + 1):
        assert sched[s] == apply_decay(base, s, curve).max_energy_milli
    assert all(sched[i] <= sched[i + 1] for i in range(horizon))


def test_step_for_energy_first_strictly_greater() -> None:
    from substrate.difficulty_decay import step_for_energy

    sched = [-15123_000, -15100_000, -15000_000, -14900_000, -14800_000]
    assert step_for_energy(sched, -15000_000) == 3
    assert step_for_energy(sched, -16000_000) == 0
    assert step_for_energy(sched, -14000_000) is None


def test_none_curve_is_flat_schedule() -> None:
    from substrate.difficulty_decay import build_decay_schedule

    sched = build_decay_schedule(-15123_000, None, 5)
    assert sched == [-15123_000] * 6


def test_libm_round_half_away_from_zero() -> None:
    """If we ever swap to Python's built-in round, these would flip.

    Python ``round(0.5) == 0`` (banker's), C ``round(0.5) == 1``. The
    decay path threads ``libm::round`` semantics through every
    integer-conversion step; a mismatch would land us one milli off
    in roughly half of all decay computations.
    """
    assert _libm_round(0.5) == 1
    assert _libm_round(1.5) == 2
    assert _libm_round(-0.5) == -1
    assert _libm_round(-1.5) == -2
    assert _libm_round(0.49999) == 0
    assert _libm_round(2.5) == 3  # banker's would give 2
