//! CUDA Ising samplers.
//!
//! Two binaries share this library:
//! - `quip-cuda-sa` — Metropolis simulated annealing on one GPU
//! - `quip-cuda-gibbs` — single-site heat-bath Gibbs on one GPU
//!
//! Kernels are the v0.2 self-feeding persistent kernels (`GPU/cuda_sa.cu` /
//! `GPU/cuda_gibbs.cu`, copied verbatim into `kernels/`): a kernel-side
//! nonce economy with 3-slot rotating buffers keeps many models in flight
//! across one persistent launch (see [`streaming`]). Solution energies are
//! always scored with [`quip_protocol::scoring::energy_milli`] for
//! consensus — the kernel's own int8-quantized energy tracking only drives
//! its internal annealing accept/reject decisions. The coordinator session
//! loop lives in `quip-miner-core`.

pub mod cuda_device;
pub mod nvml_gov;
pub mod sampler;
pub mod streaming;
pub mod topology;

pub use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use sampler::{gpu_energy_milli, sample_ising};

use cuda_device::CudaDevice;
use nvml_gov::UtilGovernor;
use quip_miner_core::{BackendIdentity, Sampler, StreamJob, StreamResult};
use quip_proto::v1::RejectReason;

const DEFAULT_MAX_EDGES: u32 = 1_000_000;

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

/// Backend identity for `quip-cuda-sa`. `max_nodes` is the self-feeding SA
/// kernel's hard limit (`unpacked_state[5000]` thread-local array in
/// `kernels/sa.cu`) — a job over this would overrun kernel-local storage,
/// not just run slowly, so it must reject `TooLarge` rather than clamp.
pub const CUDA_SA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cuda",
    algorithm: "sa",
    max_nodes: 5000,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: CUDA_ADAPT,
};

/// Backend identity for `quip-cuda-gibbs`. `max_nodes` is the self-feeding
/// Gibbs kernel's hard limit (`shared_state[4800]` in `kernels/gibbs.cu`).
pub const CUDA_GIBBS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cuda",
    algorithm: "gibbs",
    max_nodes: 4800,
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

    /// Self-feeding kernel instances: `max_sms / sms_per_nonce` (1 for SA,
    /// 4 for Gibbs), each a fully independent nonce group.
    fn sample_stream(&self, jobs: tokio::sync::mpsc::Receiver<StreamJob>, out: tokio::sync::mpsc::Sender<StreamResult>) {
        streaming::run_stream(&self.device, self.algorithm, jobs, out);
    }

    fn stream_width(&self) -> usize {
        streaming::stream_width(&self.device, self.algorithm)
    }

    fn utilization(&self) -> f64 {
        self.gov.utilization() as f64
    }

    fn should_throttle(&self) -> bool {
        self.gov.should_throttle()
    }

    fn max_reads(&self) -> u32 {
        streaming::max_reads(self.algorithm)
    }
}
