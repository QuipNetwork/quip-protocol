//! Adaptive sampling parameters: map a difficulty target to a sampling budget.
//!
//! Port of `shared/energy_utils.py` (GSE model) + `base_miner.adapt_parameters`.
//! The coordinator advertises the target energy; the miner computes its own
//! `num_reads` / `num_sweeps` from it. Cross-language parity is pinned by
//! `conformance/golden_adapt.json`.

/// SA efficiency range `(c_easy, c_hard)` — verbatim from `energy_utils.py`.
const C_EASY: f64 = 0.7;
const C_HARD: f64 = 0.75;
/// Empirical h-field scaling constant.
const ALPHA: f64 = 0.88;
/// Fallback h set when a topology advertises none (matches Python default).
const DEFAULT_H: [f64; 3] = [-1.0, 0.0, 1.0];

/// Expected ground-state energy: `GSE ≈ -c·√(avg_degree)·N` plus an h-field term.
fn expected_solution_energy(num_nodes: usize, num_edges: usize, c: f64, h_values: &[f64]) -> f64 {
    if num_nodes == 0 || num_edges == 0 {
        return 0.0;
    }
    let n = num_nodes as f64;
    let avg_degree = (2.0 * num_edges as f64) / n;
    let j_contribution = -c * avg_degree.sqrt() * n;
    let h_contribution = if h_values.len() == 1 && h_values[0] == 0.0 {
        0.0
    } else {
        let nonzero =
            h_values.iter().filter(|&&v| v != 0.0).count() as f64 / h_values.len() as f64;
        -c * ALPHA * nonzero * n / avg_degree.sqrt()
    };
    j_contribution + h_contribution
}

/// `(min_energy @ c_hard, knee @ c_mid, max_energy @ c_easy)`.
fn calc_energy_range(num_nodes: usize, num_edges: usize, h_values: &[f64]) -> (f64, f64, f64) {
    let c_knee = (C_EASY + C_HARD) / 2.0;
    (
        expected_solution_energy(num_nodes, num_edges, C_HARD, h_values),
        expected_solution_energy(num_nodes, num_edges, c_knee, h_values),
        expected_solution_energy(num_nodes, num_edges, C_EASY, h_values),
    )
}

/// Convert a target energy (milli) to a normalized difficulty in `[0, 1]`.
///
/// `0.0` = easiest (target ≥ max_energy), `1.0` = hardest (target ≤ min_energy),
/// with a piecewise curve: concave `sqrt` below the knee, convex `square` above.
#[must_use]
pub fn energy_to_difficulty(
    target_milli: i64,
    num_nodes: usize,
    num_edges: usize,
    allowed_h_milli: &[i32],
) -> f64 {
    let target = target_milli as f64 / 1000.0;
    let h_owned: Vec<f64>;
    let h_values: &[f64] = if allowed_h_milli.is_empty() {
        &DEFAULT_H
    } else {
        h_owned = allowed_h_milli.iter().map(|&v| f64::from(v) / 1000.0).collect();
        &h_owned
    };

    let (min_energy, knee_energy, max_energy) = calc_energy_range(num_nodes, num_edges, h_values);

    if target <= min_energy {
        return 1.0;
    }
    if target >= max_energy {
        return 0.0;
    }

    let total_range = max_energy - min_energy;
    let normalized_pos = (max_energy - target) / total_range;
    let knee_pos = (max_energy - knee_energy) / total_range;

    if normalized_pos <= knee_pos {
        0.5 * (normalized_pos / knee_pos).sqrt()
    } else {
        let progress = (normalized_pos - knee_pos) / (1.0 - knee_pos);
        0.5 + 0.5 * progress * progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn energy_to_difficulty_matches_python_golden() {
        let text = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../conformance/golden_adapt.json"
        ));
        let golden: Value = serde_json::from_str(text).expect("parse golden");
        let cases = golden["energy_to_difficulty"].as_array().expect("cases");
        assert!(!cases.is_empty());
        for c in cases {
            let target = c["target_milli"].as_i64().unwrap();
            let n = c["num_nodes"].as_u64().unwrap() as usize;
            let m = c["num_edges"].as_u64().unwrap() as usize;
            let h: Vec<i32> = c["allowed_h_milli"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            let expected = c["difficulty"].as_f64().unwrap();
            let got = energy_to_difficulty(target, n, m, &h);
            assert!(
                (got - expected).abs() < 1e-9,
                "target={target} n={n} m={m} h={h:?}: got {got}, want {expected}"
            );
        }
    }
}
