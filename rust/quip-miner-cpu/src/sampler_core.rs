//! Neal-style SA and single-site heat-bath Gibbs over a geometric beta ladder.
//!
//! Algorithm notes (match dwave-neal / GPU ports):
//! - Beta schedule: geometric from auto hot/cold range (or explicit range).
//! - SA: per-read random restart; sequential Metropolis flips per sweep.
//! - Gibbs: heat-bath resample `P(s=+1) = 1/(1+exp(2 β h_eff))`.
//! - Solution energies are always scored with
//!   [`quip_protocol::scoring::energy_milli`] (positive sign, trunc toward 0).
//! - Parallelism: rayon over reads (no multiprocessing).

use quip_protocol::scoring::energy_milli;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Sampling algorithm selected by the binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// Metropolis-Hastings simulated annealing (neal default).
    Sa,
    /// Single-site heat-bath Gibbs along the same beta ladder.
    Gibbs,
}

/// Per-job sampling knobs.
#[derive(Clone, Debug)]
pub struct SampleParams {
    pub num_reads: usize,
    pub num_sweeps: usize,
    pub sweeps_per_beta: usize,
    /// Optional `(hot_beta, cold_beta)`. `None` → auto from biases.
    pub beta_range: Option<(f64, f64)>,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            num_reads: 1,
            num_sweeps: 64,
            sweeps_per_beta: 1,
            beta_range: None,
            seed: 0,
        }
    }
}

/// One completed read: spins in {-1,+1} and consensus milli-energy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerResult {
    pub spins: Vec<i8>,
    pub energy_milli: i64,
}

/// Dense Ising problem with CSR-style adjacency for O(degree) local fields.
#[derive(Clone, Debug)]
pub struct IsingGraph {
    pub h: Vec<f64>,
    pub j: Vec<f64>,
    pub edges: Vec<(usize, usize)>,
    /// Per-variable neighbor list: `(neighbor, coupling)`.
    pub neighbors: Vec<Vec<(usize, f64)>>,
}

impl IsingGraph {
    /// Build adjacency from flat h / j / edge lists.
    ///
    /// Edges whose endpoints are out of range for `h.len()` are skipped (same
    /// defensive posture as `energy_milli`). Couplings shorter than edges are
    /// treated as 0 for the missing entries.
    pub fn new(h: Vec<f64>, j: Vec<f64>, edges: Vec<(usize, usize)>) -> Self {
        let n = h.len();
        let mut neighbors = vec![Vec::new(); n];
        for (k, &(u, v)) in edges.iter().enumerate() {
            if u >= n || v >= n {
                continue;
            }
            let coup = j.get(k).copied().unwrap_or(0.0);
            neighbors[u].push((v, coup));
            neighbors[v].push((u, coup));
        }
        Self {
            h,
            j,
            edges,
            neighbors,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.h.len()
    }
}

/// Default hot/cold beta from per-variable effective-field magnitudes.
///
/// Port of neal / `shared/beta_schedule._default_ising_beta_range`:
/// - hot: `ln(2) / (2 * max_abs_field)` so worst-case flip ≈ 50%
/// - cold: low single-qubit excitation rate on the smallest non-zero gap
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

    // Ensure cold is at least as large as hot (schedule increases β).
    (hot_beta, cold_beta.max(hot_beta))
}

/// Geometric beta ladder with `num_betas` points from hot → cold.
pub fn geometric_beta_schedule(hot: f64, cold: f64, num_betas: usize) -> Vec<f64> {
    if num_betas == 0 {
        return Vec::new();
    }
    if num_betas == 1 {
        return vec![cold];
    }
    // Guard geometric schedule: both ends must be strictly positive.
    let hot = hot.max(f64::MIN_POSITIVE);
    let cold = cold.max(f64::MIN_POSITIVE);
    let log_hot = hot.ln();
    let log_cold = cold.ln();
    (0..num_betas)
        .map(|i| {
            let t = i as f64 / (num_betas - 1) as f64;
            (log_hot + t * (log_cold - log_hot)).exp()
        })
        .collect()
}

fn build_beta_schedule(graph: &IsingGraph, params: &SampleParams) -> Vec<f64> {
    let sweeps_per = params.sweeps_per_beta.max(1);
    let num_betas = (params.num_sweeps / sweeps_per).max(1);
    let (hot, cold) = params
        .beta_range
        .unwrap_or_else(|| default_ising_beta_range(graph));
    geometric_beta_schedule(hot, cold, num_betas)
}

fn spin_sign(s: i8) -> f64 {
    if s > 0 {
        1.0
    } else {
        -1.0
    }
}

/// Local field `h_i + Σ_j J_ij s_j`.
fn effective_field(var: usize, spins: &[i8], graph: &IsingGraph) -> f64 {
    let mut heff = graph.h.get(var).copied().unwrap_or(0.0);
    for &(nbr, jij) in &graph.neighbors[var] {
        heff += jij * spin_sign(spins[nbr]);
    }
    heff
}

/// ΔE for flipping spin `var` (positive-sign Ising convention).
fn flip_energy(var: usize, spins: &[i8], graph: &IsingGraph) -> f64 {
    let s = spin_sign(spins[var]);
    -2.0 * s * effective_field(var, spins, graph)
}

fn random_spins(n: usize, rng: &mut SmallRng) -> Vec<i8> {
    (0..n)
        .map(|_| if rng.gen::<bool>() { 1i8 } else { -1i8 })
        .collect()
}

fn metropolis_accept(delta: f64, beta: f64, rng: &mut SmallRng) -> bool {
    if delta <= 0.0 {
        return true;
    }
    let accept_prob = (-delta * beta).exp();
    rng.gen::<f64>() < accept_prob
}

/// Heat-bath: sample new spin from conditional Boltzmann.
fn gibbs_sample_spin(heff: f64, beta: f64, rng: &mut SmallRng) -> i8 {
    // P(s = +1) = 1 / (1 + exp(2 β h_eff))
    let arg = (2.0 * beta * heff).clamp(-500.0, 500.0);
    let p_plus = 1.0 / (1.0 + arg.exp());
    if rng.gen::<f64>() < p_plus {
        1
    } else {
        -1
    }
}

fn anneal_one_read(
    graph: &IsingGraph,
    beta_schedule: &[f64],
    sweeps_per_beta: usize,
    algorithm: Algorithm,
    rng: &mut SmallRng,
) -> Vec<i8> {
    let n = graph.num_nodes();
    let mut spins = random_spins(n, rng);
    if n == 0 {
        return spins;
    }

    for &beta in beta_schedule {
        for _ in 0..sweeps_per_beta {
            for var in 0..n {
                match algorithm {
                    Algorithm::Sa => {
                        let delta = flip_energy(var, &spins, graph);
                        if metropolis_accept(delta, beta, rng) {
                            spins[var] = -spins[var];
                        }
                    }
                    Algorithm::Gibbs => {
                        let heff = effective_field(var, &spins, graph);
                        spins[var] = gibbs_sample_spin(heff, beta, rng);
                    }
                }
            }
        }
    }
    spins
}

fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
}

/// Run `num_reads` independent anneals in parallel (rayon).
pub fn sample_ising(
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Vec<SamplerResult> {
    let num_reads = params.num_reads.max(1);
    let beta_schedule = build_beta_schedule(graph, params);
    let sweeps_per = params.sweeps_per_beta.max(1);
    let base_seed = params.seed;

    (0..num_reads)
        .into_par_iter()
        .map(|read_idx| {
            // Distinct stream per read; seed 0 still diversifies via read index.
            let seed = base_seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(read_idx as u64)
                .wrapping_add(1);
            let mut rng = SmallRng::seed_from_u64(seed);
            let spins = anneal_one_read(graph, &beta_schedule, sweeps_per, algorithm, &mut rng);
            score_spins(&spins, graph)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ferro2() -> IsingGraph {
        // Prefer aligned spins: J=-1 means E = J s0 s1 is lower when s0==s1.
        IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)])
    }

    #[test]
    fn geometric_schedule_endpoints() {
        let s = geometric_beta_schedule(0.1, 10.0, 5);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 0.1).abs() < 1e-12);
        assert!((s[4] - 10.0).abs() < 1e-12);
        // Strictly increasing for hot < cold.
        for w in s.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn energy_of_known_spins_matches_scoring() {
        let g = ferro2();
        let spins = vec![1i8, 1i8];
        let r = score_spins(&spins, &g);
        assert_eq!(r.energy_milli, energy_milli(&spins, &g.h, &g.j, &g.edges));
        // E = -1 * 1 * 1 = -1 → -1000 milli
        assert_eq!(r.energy_milli, -1000);
    }

    #[test]
    fn sa_finds_ground_state_on_ferro() {
        let g = ferro2();
        let params = SampleParams {
            num_reads: 8,
            num_sweeps: 128,
            seed: 42,
            ..Default::default()
        };
        let results = sample_ising(&g, &params, Algorithm::Sa);
        assert_eq!(results.len(), 8);
        // At least one read should land on a ground state (energy -1000).
        assert!(
            results.iter().any(|r| r.energy_milli == -1000),
            "SA failed to find ferro ground state: {:?}",
            results.iter().map(|r| r.energy_milli).collect::<Vec<_>>()
        );
        // Every reported energy must equal consensus scoring of the spins.
        for r in &results {
            assert_eq!(r.energy_milli, energy_milli(&r.spins, &g.h, &g.j, &g.edges));
        }
    }

    #[test]
    fn gibbs_reports_consensus_energies() {
        let g = ferro2();
        let params = SampleParams {
            num_reads: 4,
            num_sweeps: 64,
            seed: 7,
            ..Default::default()
        };
        let results = sample_ising(&g, &params, Algorithm::Gibbs);
        for r in &results {
            assert_eq!(r.energy_milli, energy_milli(&r.spins, &g.h, &g.j, &g.edges));
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
        }
    }

    #[test]
    fn empty_graph_is_safe() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        let params = SampleParams {
            num_reads: 2,
            num_sweeps: 4,
            seed: 1,
            ..Default::default()
        };
        let results = sample_ising(&g, &params, Algorithm::Sa);
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|r| r.spins.is_empty() && r.energy_milli == 0));
    }
}
