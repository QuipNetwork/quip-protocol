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
pub use sampler::sample_ising;

use cuda_device::CudaDevice;
use nvml_gov::UtilGovernor;
use quip_miner_core::config::{config_override, warn_unknown_fields};
use quip_miner_core::{BackendIdentity, Sampler, StreamJob, StreamResult};
use quip_proto::v1::RejectReason;
use std::collections::BTreeMap;

const DEFAULT_MAX_EDGES: u32 = 1_000_000;

/// CUDA backend config, parsed from the verbatim `config.toml` subsection in
/// `Configure.backend_toml`. Unrecognized keys land in `unknown`;
/// [`warn_unknown_fields`] filters session-level keys (e.g. `num_sweeps`) so
/// only genuine typos are reported.
#[derive(serde::Deserialize, Default)]
struct CudaConfig {
    /// GPU utilization ceiling 1–100 (governor throttle threshold when yielding).
    utilization: Option<u32>,
    /// Yield the GPU to siblings when util exceeds the ceiling.
    yielding: Option<bool>,
    #[serde(flatten)]
    unknown: BTreeMap<String, toml::Value>,
}

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
    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
    ) {
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

    fn apply_config(&self, backend_toml: &str) {
        let cfg: CudaConfig = toml::from_str(backend_toml).unwrap_or_default();
        warn_unknown_fields("cuda", cfg.unknown.keys());
        // config over CLI (the governor holds the CLI-set values until now).
        let ceiling = config_override(
            "utilization",
            self.gov.utilization_ceiling(),
            cfg.utilization,
        );
        let yielding = config_override("yielding", self.gov.yielding(), cfg.yielding);
        self.gov.reconfigure(ceiling, yielding);
    }
}

#[cfg(test)]
mod config_tests {
    use super::CudaConfig;

    #[test]
    fn parses_known_fields_and_collects_unknown() {
        let cfg: CudaConfig =
            toml::from_str("utilization = 60\nyielding = true\nnum_sweeps = 100\nfoo = 1\n")
                .unwrap();
        assert_eq!(cfg.utilization, Some(60));
        assert_eq!(cfg.yielding, Some(true));
        // num_sweeps + foo are not CUDA fields -> flattened into `unknown`;
        // warn_unknown_fields later drops num_sweeps (a session key), keeps foo.
        let unknown: Vec<&str> = cfg.unknown.keys().map(String::as_str).collect();
        assert_eq!(unknown, vec!["foo", "num_sweeps"]);
    }

    #[test]
    fn empty_config_is_all_none() {
        let cfg: CudaConfig = toml::from_str("").unwrap();
        assert!(cfg.utilization.is_none());
        assert!(cfg.yielding.is_none());
        assert!(cfg.unknown.is_empty());
    }
}
