//! CPU Ising samplers.
//!
//! Two binaries share this library:
//! - `quip-cpu-sa` — neal-style geometric SA (Metropolis)
//! - `quip-cpu-gibbs` — heat-bath single-site Gibbs over the same ladder
//!
//! The coordinator session loop lives in `quip-miner-core`; this crate provides
//! the [`CpuSampler`] backend and the two binaries.

pub mod sampler_core;

pub use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use sampler_core::sample_ising;

use quip_miner_core::{BackendIdentity, Sampler};
use quip_proto::v1::RejectReason;

const DEFAULT_MAX_NODES: u32 = 100_000;
const DEFAULT_MAX_EDGES: u32 = 1_000_000;

/// Backend identity for `quip-cpu-sa`.
pub const CPU_SA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "sa",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
};

/// Backend identity for `quip-cpu-gibbs`.
pub const CPU_GIBBS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "gibbs",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
};

/// CPU sampler backend. No device, no governor, uncapped reads.
pub struct CpuSampler {
    algorithm: Algorithm,
}

impl CpuSampler {
    pub fn new(algorithm: Algorithm) -> Self {
        Self { algorithm }
    }
}

impl Sampler for CpuSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        Ok(sample_ising(graph, params, self.algorithm))
    }
}
