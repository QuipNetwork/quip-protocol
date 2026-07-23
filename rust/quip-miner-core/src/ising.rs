//! Base Ising problem type and per-job sampling knobs, shared by all backends.
//!
//! The base [`IsingGraph`] holds the wire-parsed `(h, j, edges)`. Backends
//! derive their own representation from it: the CPU miner builds an adjacency
//! list, the GPU miners build [`crate::csr::CsrGraph`].

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

/// Wire-parsed Ising problem: dense biases, flat couplings, and edge list.
#[derive(Clone, Debug)]
pub struct IsingGraph {
    pub h: Vec<f64>,
    pub j: Vec<f64>,
    pub edges: Vec<(usize, usize)>,
}

impl IsingGraph {
    /// Store flat `h` / `j` / edge lists as the base problem.
    pub fn new(h: Vec<f64>, j: Vec<f64>, edges: Vec<(usize, usize)>) -> Self {
        Self { h, j, edges }
    }

    pub fn num_nodes(&self) -> usize {
        self.h.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_nodes_counts_biases() {
        let g = IsingGraph::new(
            vec![1.0, -0.5, 0.0, 0.25],
            vec![1.0, -1.0, -1.0, 1.0],
            vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        );
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.edges.len(), 4);
    }
}
