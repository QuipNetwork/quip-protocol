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

from shared.allowed_value_spec import AllowedValueSet
from substrate.difficulty_decay import (
    DECAY_RATE_MILLI,
    MIN_ENERGY_DELTA_MILLI,
    Direction,
    EnergyCurve,
    _expected_gse_for_specs,
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
# Spec-aware curve — mirrors expected_gse_for_specs (quip-protocol-rs MR !35)
# ----------------------------------------------------------------------

# Legacy registered specs: ternary h {-1,0,+1} (⟨|h|⟩ = 2/3) and binary J.
_LEGACY_H_SET = AllowedValueSet((-1000, 0, 1000))
_LEGACY_J_SET = AllowedValueSet((-1000, 1000))


def test_expected_gse_for_specs_matches_legacy_for_default_specs() -> None:
    """Feeding the legacy ternary/binary specs must reproduce the old
    hardcoded formula bit-for-bit — the Python mirror of the Rust test
    ``expected_gse_for_specs_matches_legacy_for_default_specs``."""
    for n, m in ((TEST_NODES, TEST_EDGES), (1368, 7692), (4800, 48000)):
        for c in (0.700, 0.725, 0.750, 0.800):
            assert _expected_gse_for_specs(
                n, m, c, _LEGACY_H_SET, _LEGACY_J_SET
            ) == _expected_gse_milli(n, m, c)


def test_expected_gse_for_specs_zero_field_drops_h_term() -> None:
    """h = {0} → pure ±J spin glass. n=1024,m=2048,c=0.75: avg degree 4,
    √4=2 → -0.75·1.0·2·1024 = -1536.0 → -1_536_000 milli. Mirrors the
    Rust ``expected_gse_for_specs_zero_field_drops_h_term``."""
    zero_h = AllowedValueSet((0,))
    assert _expected_gse_for_specs(1024, 2048, 0.75, zero_h, _LEGACY_J_SET) == -1_536_000


def test_from_topology_default_specs_match_explicit_legacy() -> None:
    """``from_topology`` with no specs must equal passing the legacy specs
    explicitly — old call sites stay valid against the legacy default
    topology with zero behavior change."""
    implicit = EnergyCurve.from_topology(1368, 7692, 700, 750, 800)
    explicit = EnergyCurve.from_topology(
        1368, 7692, 700, 750, 800,
        allowed_h=_LEGACY_H_SET, allowed_j=_LEGACY_J_SET,
    )
    assert implicit == explicit


def test_from_topology_zero_field_curve_less_negative() -> None:
    """A zero-field (h={0}) topology yields a strictly less-negative curve
    than the ternary-field curve of the same graph — the chain credits no
    energy the puzzle cannot produce."""
    zero_h = AllowedValueSet((0,))
    legacy = EnergyCurve.from_topology(1368, 7692, 700, 750, 800)
    zero_field = EnergyCurve.from_topology(
        1368, 7692, 700, 750, 800,
        allowed_h=zero_h, allowed_j=_LEGACY_J_SET,
    )
    assert zero_field.min_milli > legacy.min_milli
    assert zero_field.knee_milli > legacy.knee_milli
    assert zero_field.max_milli > legacy.max_milli


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


# Production-scale curve (milli) for the geometric walk-up tests. The tiny
# test_curve range (~318 milli) sits below the 1000-milli floor, which would
# flatten every step to the floor and hide the geometric behaviour. Mirrors
# tests.rs::walkup_curve().
def _walkup_curve() -> EnergyCurve:
    return EnergyCurve(min_milli=-16_000_000, knee_milli=-15_600_000, max_milli=-14_000_000)


def test_adjust_energy_along_curve_degenerate_curve_is_no_op() -> None:
    """``max_milli <= min_milli`` → leave current alone (matches Rust guard)."""
    flat = EnergyCurve(min_milli=-1000, knee_milli=-1000, max_milli=-1000)
    assert adjust_energy_along_curve(
        -1000, DECAY_RATE_MILLI, Direction.EASIER, flat, MIN_ENERGY_DELTA_MILLI,
    ) == -1000


def test_adjust_energy_along_curve_min_delta_floor_applies(test_curve) -> None:
    """A tiny geometric step that rounds to 0 still moves by min_delta_milli.

    Mirrors tests.rs::difficulty_adjust_applies_min_delta_for_small_positive_floats:
    harden from max_milli (greatest room to the hard cap) with rate 1‰ rounds to
    0, the floor lifts it to min_delta, and the result stays inside the cap.
    """
    current = test_curve.max_milli
    result = adjust_energy_along_curve(
        current, 1, Direction.HARDER, test_curve, 100,
    )
    assert result == current - 100
    assert result > test_curve.min_milli


def test_adjust_harder_geometric_fraction_in_interior() -> None:
    """Harden steps exactly round(room*rate) of the gap to the hard cap and
    stays strictly inside it. Mirrors tests.rs::harden_steps_geometric_fraction
    _in_the_interior."""
    curve = _walkup_curve()
    starts = [curve.max_milli, curve.knee_milli, -15_000_000, curve.min_milli + 100_000]
    for current in starts:
        for rate_milli in (50, 350, 650):
            result = adjust_energy_along_curve(
                current, rate_milli, Direction.HARDER, curve, 5000,
            )
            room = current - curve.min_milli
            expected_delta = _libm_round(room * rate_milli / 1000.0)
            assert result == current - expected_delta
            assert result > curve.min_milli


def test_adjust_harder_tail_walks_past_cap_by_min_delta() -> None:
    """Near/at/below the hard cap the geometric term is below the floor, so a
    harden steps one floor — uncapped, crossing past min_milli. Mirrors
    tests.rs::harden_tail_walks_past_cap_by_min_delta."""
    curve = _walkup_curve()
    floor = MIN_ENERGY_DELTA_MILLI
    for room in (5 * floor, floor, 1, 0, -floor):
        current = curve.min_milli + room
        result = adjust_energy_along_curve(current, 1, Direction.HARDER, curve, floor)
        assert result == current - floor
    at_cap = adjust_energy_along_curve(curve.min_milli, 1, Direction.HARDER, curve, floor)
    assert at_cap < curve.min_milli


def test_adjust_easier_capped_at_max() -> None:
    """Easier never eases past the easy cap, and is a no-op at/past it."""
    curve = _walkup_curve()
    # A large rate from just below max clamps to the remaining gap (lands on max).
    near_max = curve.max_milli - 100
    assert adjust_energy_along_curve(
        near_max, 650, Direction.EASIER, curve, MIN_ENERGY_DELTA_MILLI,
    ) == curve.max_milli
    # At the cap: no-op.
    assert adjust_energy_along_curve(
        curve.max_milli, DECAY_RATE_MILLI, Direction.EASIER, curve, MIN_ENERGY_DELTA_MILLI,
    ) == curve.max_milli


def test_fast_wins_walk_up_the_curve_over_many_steps() -> None:
    """A 35% geometric harden takes many wins to reach the cap, never one.
    Mirrors tests.rs::fast_wins_walk_up_the_curve_over_many_steps."""
    curve = _walkup_curve()
    current = curve.knee_milli
    wins = 0
    while current - curve.min_milli > 1000 and wins < 1000:
        current = adjust_energy_along_curve(current, 350, Direction.HARDER, curve, 5000)
        wins += 1
    assert wins >= 10
    assert current <= curve.min_milli + 1000


def test_decay_recovers_fastest_from_the_hard_cap() -> None:
    """Easing is geometric in the gap to the easy cap, so the largest step is at
    the hard ceiling. Mirrors tests.rs::decay_recovers_fastest_from_the_hard_cap."""
    curve = _walkup_curve()

    def decay_step(start: int) -> int:
        return adjust_energy_along_curve(
            start, DECAY_RATE_MILLI, Direction.EASIER, curve, 3000,
        ) - start

    from_cap = decay_step(curve.min_milli)
    from_mid = decay_step(curve.knee_milli)
    from_near_max = decay_step(curve.max_milli - 50_000)
    assert from_cap > from_mid > from_near_max


def test_observed_win_series_never_pins_the_hard_cap() -> None:
    """Replay the documented inter-win gap series and assert the base threshold
    stays strictly interior. Decay-only mirror (the miner has no on-proof
    hardening); the easing keeps the threshold off both caps."""
    curve = _walkup_curve()
    base = SubstrateDifficulty(
        min_solutions=1, max_energy_milli=curve.knee_milli, min_diversity_milli=0,
    )
    for gap in (29, 280, 27, 200, 100, 300, 1401):
        steps = gap // 100
        base = apply_decay(base, steps, curve)
        assert curve.min_milli < base.max_energy_milli <= curve.max_milli


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
