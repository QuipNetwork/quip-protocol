//! Geometric beta (inverse-temperature) ladder for annealing.
//!
//! Port of neal / `shared/beta_schedule._default_ising_beta_range` and the
//! geometric schedule used by the CPU miner and Python GPU samplers.

use crate::csr::IsingGraph;

/// Default hot/cold beta from per-variable effective-field magnitudes.
pub fn default_ising_beta_range(graph: &IsingGraph) -> (f64, f64) {
    let n = graph.num_nodes();
    if n == 0 {
        return (0.1, 1.0);
    }

    let mut sum_abs = vec![0.0f64; n];
    let mut min_abs: Vec<Option<f64>> = vec![None; n];

    for (i, &hi) in graph.h.iter().enumerate() {
        let a = hi.abs();
        sum_abs[i] += a;
        if a > 0.0 {
            min_abs[i] = Some(a);
        }
    }
    for (k, &(u, v)) in graph.edges.iter().enumerate() {
        if u >= n || v >= n {
            continue;
        }
        let a = graph.j.get(k).copied().unwrap_or(0.0).abs();
        sum_abs[u] += a;
        sum_abs[v] += a;
        if a > 0.0 {
            for node in [u, v] {
                min_abs[node] = Some(match min_abs[node] {
                    Some(m) => m.min(a),
                    None => a,
                });
            }
        }
    }

    let min_gaps: Vec<f64> = min_abs.into_iter().flatten().collect();
    if min_gaps.is_empty() {
        return (0.1, 1.0);
    }

    let max_eff = sum_abs.iter().cloned().fold(0.0f64, f64::max);
    let hot_beta = if max_eff == 0.0 {
        1.0
    } else {
        std::f64::consts::LN_2 / (2.0 * max_eff)
    };

    let min_eff = min_gaps.iter().cloned().fold(f64::INFINITY, f64::min);
    let number_min_gaps = min_gaps.iter().filter(|&&g| g == min_eff).count() as f64;
    let max_excitation = 0.01f64;
    let cold_beta = (number_min_gaps / max_excitation).ln() / (2.0 * min_eff);

    (hot_beta, cold_beta.max(hot_beta))
}

/// Geometric beta ladder with `num_betas` points from hot → cold.
pub fn geometric_beta_schedule(hot: f64, cold: f64, num_betas: usize) -> Vec<f32> {
    if num_betas == 0 {
        return Vec::new();
    }
    if num_betas == 1 {
        return vec![cold as f32];
    }
    let hot = hot.max(f64::MIN_POSITIVE);
    let cold = cold.max(f64::MIN_POSITIVE);
    let log_hot = hot.ln();
    let log_cold = cold.ln();
    (0..num_betas)
        .map(|i| {
            let t = i as f64 / (num_betas - 1) as f64;
            (log_hot + t * (log_cold - log_hot)).exp() as f32
        })
        .collect()
}

/// Build the schedule used for a sample request.
pub fn build_beta_schedule(
    graph: &IsingGraph,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
) -> (Vec<f32>, usize) {
    let sweeps_per = sweeps_per_beta.max(1);
    let num_betas = (num_sweeps / sweeps_per).max(1);
    let (hot, cold) = beta_range.unwrap_or_else(|| default_ising_beta_range(graph));
    (geometric_beta_schedule(hot, cold, num_betas), sweeps_per)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_schedule_endpoints() {
        let s = geometric_beta_schedule(0.1, 10.0, 5);
        assert_eq!(s.len(), 5);
        assert!((s[0] as f64 - 0.1).abs() < 1e-5);
        assert!((s[4] as f64 - 10.0).abs() < 1e-4);
        for w in s.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
}
