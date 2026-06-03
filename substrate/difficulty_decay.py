"""Python port of ``pallets/quantum-pow/src/difficulty.rs``.

The chain's difficulty adjustment is block-based: after every
``EpochLength`` blocks elapse since the last winning proof, the energy
threshold eases by one decay step. The runtime computes this on every
``submit_proof`` (see ``lib.rs::current_difficulty``); the miner needs
the same algorithm to know which proofs are eligible *right now*
without an RPC round-trip per iteration.

The base ``DifficultyConfig`` value in storage only changes when a
proof wins (``adjust_on_proof``); decay is purely a function of
``block_number - last_proof_block``, the per-pallet ``EpochLength``,
and a per-topology ``EnergyCurve``. So the parent process can cache
the inputs and re-decay locally on every head.

Parity with the Rust must hold *to the milli*, because the chain
validates ``best_energy_milli < max_energy_milli`` strictly. A
one-milli drift between the Python view and the runtime would either
discard valid proofs (miner stricter) or submit doomed proofs (miner
looser).

This module deliberately uses i64 milli arithmetic and ``libm::round``
semantics (round-half-away-from-zero) rather than Python's float-based
``adjust_energy_along_curve`` in :mod:`shared.energy_utils` (the v0.1
helper, still used by the legacy tests).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from substrate.types import SubstrateDifficulty


# Mirrors `pallets/quantum-pow/src/difficulty.rs`.
DECAY_RATE_MILLI: int = 25
MIN_DECAY_DELTA_MILLI: int = 3
MIN_HARDENING_DELTA_MILLI: int = 5

# i64 saturation bounds. Used so ``saturating_add``/``saturating_sub``
# match the Rust exactly; in practice ``max_energy_milli`` lives in the
# low millions and never approaches these.
_I64_MIN: int = -(1 << 63)
_I64_MAX: int = (1 << 63) - 1


class Direction(Enum):
    """Direction the energy threshold moves under one adjustment step."""

    HARDER = "harder"
    EASIER = "easier"


@dataclass(frozen=True)
class EnergyCurve:
    """Topology-derived bounds for the difficulty energy curve.

    Mirrors ``pallets/quantum-pow/src/difficulty.rs::EnergyCurve``.

    All three fields are milli precision i64. The Rust invariant
    ``min_milli < knee_milli < max_milli`` (all negative) holds for any
    legitimate topology + c-triple; degenerate inputs cause
    ``adjust_energy_along_curve`` to fall back to a linear / no-op path
    rather than panicking.
    """

    min_milli: int
    knee_milli: int
    max_milli: int

    @classmethod
    def from_topology(
        cls,
        num_nodes: int,
        num_edges: int,
        c_easy_milli: int,
        c_knee_milli: int,
        c_hard_milli: int,
    ) -> "EnergyCurve":
        """Build a curve matching ``EnergyCurve::new`` in the Rust.

        c values are stored on chain as scaled u32 (e.g. 700 = 0.70)
        because pallet constants must implement ``Get<_>`` and ``f64``
        does not implement ``Encode``. They divide by 1000 here before
        feeding into ``expected_gse_with_c``.
        """
        return cls(
            min_milli=_expected_gse_with_c(num_nodes, num_edges, c_hard_milli / 1000.0),
            knee_milli=_expected_gse_with_c(num_nodes, num_edges, c_knee_milli / 1000.0),
            max_milli=_expected_gse_with_c(num_nodes, num_edges, c_easy_milli / 1000.0),
        )


def adjust_energy_along_curve(
    current_milli: int,
    rate_milli: int,
    direction: Direction,
    curve: EnergyCurve,
    min_delta_milli: int,
) -> int:
    """Move ``current_milli`` along the curve by ``rate_milli`` per-mille.

    Mirrors ``adjust_energy_along_curve`` in
    ``pallets/quantum-pow/src/difficulty.rs`` (Phase 1 port for the
    Python miner). Behaviour at boundaries / degenerate inputs / the
    ``min_delta_milli`` floor matches the Rust to the milli.

    A degenerate curve (``total_range <= 0``) leaves ``current`` alone.
    A current value outside ``[min, max]`` falls back to linear motion
    (``total_range * rate``). Otherwise the curve compresses motion
    near the boundaries and peaks at the knee, per the same sqrt
    schedule.
    """
    min_f = float(curve.min_milli)
    max_f = float(curve.max_milli)
    knee_f = float(curve.knee_milli)
    cur_f = float(current_milli)
    total_range = max_f - min_f
    rate = float(rate_milli) / 1000.0

    if total_range <= 0.0:
        return current_milli

    if cur_f < min_f or cur_f > max_f:
        raw_delta_f = total_range * rate
    else:
        normalized = (cur_f - min_f) / total_range
        knee_pos = (knee_f - min_f) / total_range
        if knee_pos <= 0.0 or knee_pos >= 1.0:
            curve_factor = 1.0
        elif normalized <= knee_pos:
            curve_factor = 0.1 + 0.9 * math.sqrt(normalized / knee_pos)
        else:
            curve_factor = 1.0 - 0.9 * math.sqrt(
                (normalized - knee_pos) / (1.0 - knee_pos)
            )
        raw_delta_f = total_range * rate * curve_factor

    delta = _libm_round(raw_delta_f)
    # Floor on the raw float, not the rounded int — see Rust comment:
    # a raw_delta_f in (0, 0.5) rounds to 0, which would skip the floor
    # and stall difficulty progress.
    if raw_delta_f > 0.0 and delta < min_delta_milli:
        delta = min_delta_milli
    if delta == 0:
        return current_milli

    if direction is Direction.HARDER:
        return _saturating_sub(current_milli, delta)
    return _saturating_add(current_milli, delta)


def apply_decay(
    current: SubstrateDifficulty,
    steps: int,
    curve: EnergyCurve,
) -> SubstrateDifficulty:
    """Apply ``steps`` decay-easing steps to ``current``.

    Only ``max_energy_milli`` mutates; ``min_solutions`` and
    ``min_diversity_milli`` are chain-static and only ``set_difficulty``
    (root-only) can change them. Matches the Rust invariant exercised
    by ``apply_decay_only_mutates_max_energy`` in ``tests.rs``.
    """
    max_energy_milli = current.max_energy_milli
    for _ in range(steps):
        max_energy_milli = adjust_energy_along_curve(
            max_energy_milli,
            DECAY_RATE_MILLI,
            Direction.EASIER,
            curve,
            MIN_DECAY_DELTA_MILLI,
        )
    return SubstrateDifficulty(
        min_solutions=current.min_solutions,
        max_energy_milli=max_energy_milli,
        min_diversity_milli=current.min_diversity_milli,
    )


def current_difficulty(
    block_number: int,
    base_difficulty: SubstrateDifficulty,
    last_proof_block: int,
    epoch_length: int,
    curve: Optional[EnergyCurve],
) -> SubstrateDifficulty:
    """Compute the active difficulty for ``block_number``.

    Mirrors ``difficulty::current_difficulty`` in the Rust. Returns the
    same value the chain uses to validate proofs submitted at
    ``block_number`` — base ``Difficulty<T>`` storage with per-epoch
    decay applied for blocks elapsed since the last winning proof.

    ``last_proof_block == 0`` is the genesis sentinel: no proof has
    ever won, so no decay applies (matches ``current_difficulty_short_
    circuits_at_genesis`` in tests.rs). A ``None`` curve disables decay
    (genesis or defensive fallback).
    """
    if last_proof_block == 0 or epoch_length == 0:
        return base_difficulty
    elapsed = max(0, block_number - last_proof_block)
    steps = elapsed // epoch_length
    if curve is None or steps == 0:
        return base_difficulty
    return apply_decay(base_difficulty, steps, curve)


def build_decay_schedule(
    base_max_energy_milli: int,
    curve: Optional[EnergyCurve],
    horizon: int,
) -> list[int]:
    """Max-energy threshold at each decay step ``0..horizon`` (inclusive).

    Built incrementally — each step is one ``adjust_energy_along_curve`` from
    the prior — so the whole array is O(horizon). Monotonic non-decreasing
    (decay only eases ``max_energy_milli`` upward). A ``None`` curve yields a
    flat schedule (decay disabled), matching ``current_difficulty``.
    """
    sched = [base_max_energy_milli]
    if curve is None:
        return sched + [base_max_energy_milli] * horizon
    cur = base_max_energy_milli
    for _ in range(horizon):
        cur = adjust_energy_along_curve(
            cur, DECAY_RATE_MILLI, Direction.EASIER, curve, MIN_DECAY_DELTA_MILLI,
        )
        sched.append(cur)
    return sched


def step_for_energy(
    decay_schedule: list[int], floor_energy_milli: int
) -> Optional[int]:
    """First decay step ``s`` where ``schedule[s] > floor_energy_milli``.

    Mirrors the chain's strict ``best_energy_milli < max_energy_milli`` gate:
    the candidate clears at the first step whose threshold is *strictly* above
    its floor. ``None`` when it never clears within the schedule's horizon.
    The schedule is monotonic non-decreasing, so this is a binary search.
    """
    i = bisect.bisect_right(decay_schedule, floor_energy_milli)
    return i if i < len(decay_schedule) else None


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _libm_round(x: float) -> int:
    """C's ``round()`` semantics — round half away from zero.

    Python's ``round`` does banker's rounding (half to even). The chain
    uses ``libm::round``, which is the C-standard half-away-from-zero
    rule. A milli of drift here would propagate one-for-one into the
    proof-acceptance check and silently desync miner and chain.
    """
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def _saturating_add(a: int, b: int) -> int:
    result = a + b
    if result > _I64_MAX:
        return _I64_MAX
    if result < _I64_MIN:
        return _I64_MIN
    return result


def _saturating_sub(a: int, b: int) -> int:
    return _saturating_add(a, -b)


def _expected_gse_with_c(num_nodes: int, num_edges: int, c: float) -> int:
    """Port of ``quantum_validation::expected_gse_with_c``.

    Constants and formula must match ``crates/quantum-validation/src/
    energy.rs`` exactly — the chain registers topologies with curve
    bounds it computes via this function, so any Python drift would
    yield a different ``EnergyCurve`` and a different decay trajectory.
    """
    if num_nodes == 0 or num_edges == 0:
        return 0
    DEFAULT_H_NONZERO_FRACTION = 2.0 / 3.0
    DEFAULT_H_ALPHA = 0.88
    MILLI_SCALE = 1_000
    n = float(num_nodes)
    m = float(num_edges)
    avg_degree = (2.0 * m) / n
    sqrt_avg_degree = math.sqrt(avg_degree)
    j_contribution = -c * sqrt_avg_degree * n
    h_contribution = -c * DEFAULT_H_ALPHA * DEFAULT_H_NONZERO_FRACTION * n / sqrt_avg_degree
    return _libm_round((j_contribution + h_contribution) * MILLI_SCALE)


__all__ = [
    "DECAY_RATE_MILLI",
    "MIN_DECAY_DELTA_MILLI",
    "MIN_HARDENING_DELTA_MILLI",
    "Direction",
    "EnergyCurve",
    "adjust_energy_along_curve",
    "apply_decay",
    "current_difficulty",
    "build_decay_schedule",
    "step_for_energy",
]
