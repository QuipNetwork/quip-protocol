//! Neal-style SA and single-site heat-bath Gibbs over a geometric beta ladder.
//!
//! Algorithm notes (match dwave-neal / GPU ports):
//! - Beta schedule: geometric from auto hot/cold range (or explicit range).
//! - SA: per-read random restart; sequential Metropolis flips per sweep.
//! - Gibbs: heat-bath resample `P(s=+1) = 1/(1+exp(2 β h_eff))`.
//! - Solution energies are always scored with
//!   [`quip_protocol::scoring::energy_milli`] (positive sign, trunc toward 0).
//! - Parallelism: model-level (one model per core via the streaming pump);
//!   reads run sequentially and cache-local on a single core.
//!
//! Types (`Algorithm`, `SampleParams`, `SamplerResult`, base `IsingGraph`) and
//! the beta schedule come from `quip-miner-core`; this module keeps only the
//! CPU annealing kernels and a private adjacency ([`CpuGraph`]) built from the
//! base graph.

use quip_miner_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
use quip_protocol::scoring::energy_milli;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Per-variable neighbor lists in CSR layout for O(degree) local fields.
///
/// A flat CSR (`nbr_start` offsets into contiguous `nbr_node`/`nbr_coup`)
/// keeps each variable's neighbors cache-local in the annealing hot loop,
/// unlike a `Vec<Vec<_>>` whose rows are scattered heap allocations.
///
/// Built from the base [`IsingGraph`] with the same defensive posture as
/// `energy_milli`: edges out of range for `h.len()` are skipped, and couplings
/// shorter than the edge list are treated as 0.
struct CpuGraph {
    h: Vec<f64>,
    /// CSR row offsets, length `n + 1`.
    nbr_start: Vec<u32>,
    /// Flattened neighbor node ids.
    nbr_node: Vec<u32>,
    /// Flattened couplings, parallel to `nbr_node`.
    nbr_coup: Vec<f64>,
}

impl CpuGraph {
    fn from_base(g: &IsingGraph) -> Self {
        let n = g.h.len();
        let mut deg = vec![0u32; n];
        for &(u, v) in &g.edges {
            if u >= n || v >= n {
                continue;
            }
            deg[u] += 1;
            deg[v] += 1;
        }
        let mut nbr_start = vec![0u32; n + 1];
        for i in 0..n {
            nbr_start[i + 1] = nbr_start[i] + deg[i];
        }
        let total = nbr_start[n] as usize;
        let mut nbr_node = vec![0u32; total];
        let mut nbr_coup = vec![0.0f64; total];
        let mut cursor: Vec<u32> = nbr_start[..n].to_vec();
        for (k, &(u, v)) in g.edges.iter().enumerate() {
            if u >= n || v >= n {
                continue;
            }
            let coup = g.j.get(k).copied().unwrap_or(0.0);
            let pu = cursor[u] as usize;
            nbr_node[pu] = v as u32;
            nbr_coup[pu] = coup;
            cursor[u] += 1;
            let pv = cursor[v] as usize;
            nbr_node[pv] = u as u32;
            nbr_coup[pv] = coup;
            cursor[v] += 1;
        }
        Self {
            h: g.h.clone(),
            nbr_start,
            nbr_node,
            nbr_coup,
        }
    }

    fn num_nodes(&self) -> usize {
        self.h.len()
    }

    /// `(neighbor_ids, couplings)` slices for `var`.
    #[inline]
    fn neighbors(&self, var: usize) -> (&[u32], &[f64]) {
        let s = self.nbr_start[var] as usize;
        let e = self.nbr_start[var + 1] as usize;
        (&self.nbr_node[s..e], &self.nbr_coup[s..e])
    }
}

/// Geometric beta schedule for one sample request (f64 for CPU precision).
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

/// Local field `h_i + Σ_j J_ij s_j` (full recompute; used once to seed the
/// incremental `heff` cache).
fn effective_field(var: usize, spins: &[i8], graph: &CpuGraph) -> f64 {
    let mut heff = graph.h[var];
    let (nodes, coups) = graph.neighbors(var);
    for i in 0..nodes.len() {
        heff += coups[i] * spin_sign(spins[nodes[i] as usize]);
    }
    heff
}

/// Propagate a spin change at `var` (sign delta `ds`) into its neighbors' cached
/// effective fields. `var`'s own field is unaffected (it excludes its own spin).
#[inline]
fn apply_field_delta(graph: &CpuGraph, heff: &mut [f64], var: usize, ds: f64) {
    let (nodes, coups) = graph.neighbors(var);
    for i in 0..nodes.len() {
        heff[nodes[i] as usize] += coups[i] * ds;
    }
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
    graph: &CpuGraph,
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

    // Incremental effective-field cache: `heff[var]` stays equal to
    // `effective_field(var, spins)` across the whole anneal. Seeded once
    // (O(edges)); each accepted flip updates only its neighbors (O(degree)),
    // so a sweep costs O(n + accepts·degree) instead of O(n·degree) every
    // time. ΔE and the Gibbs conditional both read the cache in O(1). The
    // accept/reject RNG stream is unchanged (ΔE is identical), so results are
    // bit-for-bit the same as the recompute-every-flip version.
    let mut heff: Vec<f64> = (0..n).map(|v| effective_field(v, &spins, graph)).collect();

    match algorithm {
        Algorithm::Sa => {
            for &beta in beta_schedule {
                for _ in 0..sweeps_per_beta {
                    for var in 0..n {
                        let s = spin_sign(spins[var]);
                        let delta = -2.0 * s * heff[var];
                        if metropolis_accept(delta, beta, rng) {
                            spins[var] = -spins[var];
                            apply_field_delta(graph, &mut heff, var, -2.0 * s);
                        }
                    }
                }
            }
        }
        Algorithm::Gibbs => {
            for &beta in beta_schedule {
                for _ in 0..sweeps_per_beta {
                    for var in 0..n {
                        let new = gibbs_sample_spin(heff[var], beta, rng);
                        if new != spins[var] {
                            let ds = spin_sign(new) - spin_sign(spins[var]);
                            spins[var] = new;
                            apply_field_delta(graph, &mut heff, var, ds);
                        }
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

/// Run `num_reads` independent anneals sequentially on one core.
///
/// Reads stay on a single core so the model's arrays (h/j/spins/edges) stay hot
/// in that core's cache — fanning reads across cores bounced those cache lines
/// and measured slower. Model-level parallelism (one model per core) lives in
/// the streaming pump (`CpuSampler::sample_stream`).
pub fn sample_ising(
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Vec<SamplerResult> {
    let num_reads = params.num_reads.max(1);
    let cpu = CpuGraph::from_base(graph);
    let beta_schedule = build_beta_schedule(graph, params);
    let sweeps_per = params.sweeps_per_beta.max(1);
    let base_seed = params.seed;

    (0..num_reads)
        .map(|read_idx| {
            // Distinct stream per read; seed 0 still diversifies via read index.
            let seed = base_seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(read_idx as u64)
                .wrapping_add(1);
            let mut rng = SmallRng::seed_from_u64(seed);
            let spins = anneal_one_read(&cpu, &beta_schedule, sweeps_per, algorithm, &mut rng);
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
