//! CUDA Ising samplers.
//!
//! Two binaries share this library:
//! - `quip-cuda-sa` — Metropolis simulated annealing on one GPU
//! - `quip-cuda-gibbs` — single-site heat-bath Gibbs on one GPU
//!
//! Kernels take **explicit per-job** CSR buffers from the host (no kernel-side
//! nonce economy / rotating slots). Solution energies are always scored with
//! [`quip_protocol::scoring::energy_milli`] for consensus. The coordinator
//! session loop lives in `quip-miner-core`.

pub mod cuda_device;
pub mod nvml_gov;
pub mod sampler;

pub use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use sampler::{gpu_energy_milli, sample_ising};

use cuda_device::CudaDevice;
use nvml_gov::UtilGovernor;
use quip_miner_core::{BackendIdentity, Sampler};
use quip_proto::v1::RejectReason;

const DEFAULT_MAX_NODES: u32 = 100_000;
const DEFAULT_MAX_EDGES: u32 = 1_000_000;
/// Device-memory bound on reads (each read allocates `num_reads * N` device
/// bytes, twice). A job over this is rejected `TooLarge`.
const DEFAULT_MAX_READS: u32 = 100_000;

/// Backend identity for `quip-cuda-sa`.
/// CUDA adapt envelope (from `GPU/cuda_miner.py`).
const CUDA_ADAPT: quip_miner_core::adapt::AdaptBounds = quip_miner_core::adapt::AdaptBounds {
    min_sweeps: 256,
    max_sweeps: 2048,
    min_reads: 64,
    max_reads: 256,
    reads_solution_min_factor: 0,
    reads_solution_max_factor: 0,
    reads_solution_floor_factor: 0,
};

pub const CUDA_SA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cuda",
    algorithm: "sa",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: CUDA_ADAPT,
};

/// Backend identity for `quip-cuda-gibbs`.
pub const CUDA_GIBBS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cuda",
    algorithm: "gibbs",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: CUDA_ADAPT,
};

/// CUDA sampler backend: one GPU device plus an NVML utilization governor.
pub struct CudaSampler {
    device: CudaDevice,
    gov: UtilGovernor,
    algorithm: Algorithm,
}

impl CudaSampler {
    pub fn new(device: CudaDevice, gov: UtilGovernor, algorithm: Algorithm) -> Self {
        Self {
            device,
            gov,
            algorithm,
        }
    }
}

impl Sampler for CudaSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        sample_ising(&self.device, graph, params, self.algorithm).map_err(|e| {
            eprintln!("cuda sample failed: {e}");
            RejectReason::Overloaded
        })
    }

    fn utilization(&self) -> f64 {
        self.gov.utilization() as f64
    }

    fn should_throttle(&self) -> bool {
        self.gov.should_throttle()
    }

    fn max_reads(&self) -> u32 {
        DEFAULT_MAX_READS
    }
}
