//! Pure difficulty-decay math (agm.2.1).
//!
//! The chain eases `max_energy_milli` upward each epoch since the last winning
//! proof, so a candidate that does not clear the current threshold eventually
//! wins as difficulty decays. This module ports the runtime-110 decay curve
//! (`pallets/quantum-pow/src/difficulty.rs`, mirrored by the v0.2
//! `substrate/difficulty_decay.py`) to the milli: `ease_step` walks
//! `max_energy_milli` a geometric fraction of the remaining gap toward the easy
//! cap; `build_decay_schedule` unrolls it; `step_for_energy` finds a candidate's
//! win step. No chain dependency — the caller supplies the curve + base.
//!
//! Golden-faithful: `round` is half-away-from-zero (Rust `f64::round`, matching
//! `libm::round`), and every add/sub saturates at i64 bounds.

/// Geometric ease fraction, in milli (rate = 25/1000 = 0.025).
pub const DECAY_RATE_MILLI: i64 = 25;
/// Minimum per-step easing when the geometric term rounds below it (one energy
/// unit in milli).
pub const MIN_ENERGY_DELTA_MILLI: i64 = 1000;

/// GSE-estimate bounds for the decay curve (all negative). `ease_step` only
/// references `min_milli`/`max_milli`; `knee_milli` is retained for parity with
/// the chain curve and the (separate) curve-construction path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnergyCurve {
    pub min_milli: i64,
    pub knee_milli: i64,
    pub max_milli: i64,
}

/// `round(room * rate)` (half away from zero) floored at `min_delta`.
fn geometric_floored(room: i64, rate_milli: i64, min_delta: i64) -> i64 {
    let rate = rate_milli as f64 / 1000.0;
    // f64::round is half-away-from-zero (libm::round); float→int cast saturates.
    let stepped = (room as f64 * rate).round() as i64;
    stepped.max(min_delta)
}

/// One EASIER decay step on `current` max-energy toward `curve.max_milli`,
/// capped there (never ease past the easy bound). A degenerate curve
/// (`max <= min`) or a current already at/past the cap is a no-op.
pub fn ease_step(current: i64, curve: &EnergyCurve) -> i64 {
    if curve.max_milli <= curve.min_milli {
        return current;
    }
    let room = curve.max_milli.saturating_sub(current);
    if room <= 0 {
        return current;
    }
    let g = geometric_floored(room, DECAY_RATE_MILLI, MIN_ENERGY_DELTA_MILLI);
    current.saturating_add(g.min(room))
}

/// Apply `steps` EASIER steps to `base_max_energy_milli`.
pub fn apply_decay(base_max_energy_milli: i64, steps: u64, curve: &EnergyCurve) -> i64 {
    let mut cur = base_max_energy_milli;
    for _ in 0..steps {
        cur = ease_step(cur, curve);
    }
    cur
}

/// Active `max_energy_milli` at `block_number`: base difficulty with per-epoch
/// decay for blocks elapsed since the last winning proof. `last_proof_block == 0`
/// (genesis) or `epoch_length == 0` disables decay, as does a `None` curve.
pub fn current_max_energy(
    block_number: u64,
    base_max_energy_milli: i64,
    last_proof_block: u64,
    epoch_length: u64,
    curve: Option<&EnergyCurve>,
) -> i64 {
    if last_proof_block == 0 || epoch_length == 0 {
        return base_max_energy_milli;
    }
    let elapsed = block_number.saturating_sub(last_proof_block);
    let steps = elapsed / epoch_length;
    match curve {
        Some(c) if steps > 0 => apply_decay(base_max_energy_milli, steps, c),
        _ => base_max_energy_milli,
    }
}

/// `max_energy_milli` threshold at each decay step `0..=horizon` (inclusive),
/// built incrementally. Monotonic non-decreasing (decay only eases upward). A
/// `None` curve yields a flat schedule (decay disabled).
pub fn build_decay_schedule(
    base_max_energy_milli: i64,
    curve: Option<&EnergyCurve>,
    horizon: usize,
) -> Vec<i64> {
    let mut sched = Vec::with_capacity(horizon + 1);
    sched.push(base_max_energy_milli);
    match curve {
        None => {
            for _ in 0..horizon {
                sched.push(base_max_energy_milli);
            }
        }
        Some(c) => {
            let mut cur = base_max_energy_milli;
            for _ in 0..horizon {
                cur = ease_step(cur, c);
                sched.push(cur);
            }
        }
    }
    sched
}

/// First step `s` where `schedule[s] > floor_energy_milli` (the strict
/// `best_energy_milli < max_energy_milli` gate), or `None` if a candidate with
/// that floor never clears within the schedule's horizon. `schedule` is
/// monotonic non-decreasing, so this is a binary search (bisect_right).
pub fn step_for_energy(schedule: &[i64], floor_energy_milli: i64) -> Option<usize> {
    let i = schedule.partition_point(|&t| t <= floor_energy_milli);
    (i < schedule.len()).then_some(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A wide curve so the geometric term (not the cap or floor) drives the walk.
    fn curve() -> EnergyCurve {
        EnergyCurve {
            min_milli: -100_000,
            knee_milli: -50_000,
            max_milli: -1_000,
        }
    }

    #[test]
    fn ease_step_walks_geometric_fraction_toward_cap() {
        let c = curve();
        // room = -1000 - (-50000) = 49000; round(49000*0.025)=1225 → -48775
        assert_eq!(ease_step(-50_000, &c), -48_775);
        // room = 47775; round(1194.375)=1194 → -47581
        assert_eq!(ease_step(-48_775, &c), -47_581);
        // room = 46581; round(1164.525)=1165 → -46416
        assert_eq!(ease_step(-47_581, &c), -46_416);
    }

    #[test]
    fn ease_step_is_capped_and_noop_at_easy_bound() {
        let c = curve();
        // room = -1000 - (-1200) = 200; geometric floors to 1000, min(1000,200)=200
        // → clamped exactly to the cap, then a no-op.
        assert_eq!(ease_step(-1_200, &c), -1_000);
        assert_eq!(ease_step(-1_000, &c), -1_000);
        // Degenerate curve leaves current alone.
        let degenerate = EnergyCurve {
            min_milli: -1_000,
            knee_milli: -1_000,
            max_milli: -1_000,
        };
        assert_eq!(ease_step(-50_000, &degenerate), -50_000);
    }

    #[test]
    fn apply_decay_matches_step_by_step() {
        let c = curve();
        assert_eq!(apply_decay(-50_000, 0, &c), -50_000);
        assert_eq!(apply_decay(-50_000, 3, &c), -46_416);
    }

    #[test]
    fn build_schedule_is_prefix_of_apply_decay() {
        let c = curve();
        let sched = build_decay_schedule(-50_000, Some(&c), 3);
        assert_eq!(sched, vec![-50_000, -48_775, -47_581, -46_416]);
        // None curve → flat schedule.
        assert_eq!(
            build_decay_schedule(-50_000, None, 3),
            vec![-50_000, -50_000, -50_000, -50_000]
        );
    }

    #[test]
    fn step_for_energy_finds_first_clearing_step() {
        let sched = vec![-50_000, -48_775, -47_581, -46_416];
        // A floor equal to the base clears at step 1 (strict >).
        assert_eq!(step_for_energy(&sched, -50_000), Some(1));
        // -48000 clears when the threshold first exceeds it → step 2.
        assert_eq!(step_for_energy(&sched, -48_000), Some(2));
        // A floor the schedule never exceeds within the horizon → None.
        assert_eq!(step_for_energy(&sched, -1), None);
    }

    #[test]
    fn current_max_energy_applies_epoch_decay() {
        let c = curve();
        // Genesis / no epoch → no decay.
        assert_eq!(current_max_energy(1_000, -50_000, 0, 10, Some(&c)), -50_000);
        assert_eq!(
            current_max_energy(1_000, -50_000, 100, 0, Some(&c)),
            -50_000
        );
        // elapsed 25, epoch 10 → 2 steps.
        assert_eq!(
            current_max_energy(125, -50_000, 100, 10, Some(&c)),
            apply_decay(-50_000, 2, &c)
        );
        // No curve → base regardless of elapsed.
        assert_eq!(current_max_energy(125, -50_000, 100, 10, None), -50_000);
    }
}
