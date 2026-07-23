//! Metal Ising samplers.
//!
//! Two binaries share this library:
//! - `quip-metal-sa` — Metropolis simulated annealing on one Apple GPU
//! - `quip-metal-gibbs` — single-site heat-bath Gibbs on one Apple GPU
//!
//! Kernels take **explicit per-job** CSR buffers from the host (no kernel-side
//! nonce economy / rotating slots). Solution energies are always scored with
//! [`quip_protocol::scoring::energy_milli`] for consensus (host f64 — Metal
//! has no fp64). The coordinator session loop lives in `quip-miner-core`.
//!
//! GPU modules (`metal_device`, `iokit_gov`) are macOS-only so Linux can still
//! build the CLI stubs and host math.

pub mod sampler;

#[cfg(target_os = "macos")]
pub mod iokit_gov;
#[cfg(target_os = "macos")]
pub mod metal_device;
#[cfg(target_os = "macos")]
pub mod streaming;
#[cfg(target_os = "macos")]
pub mod topology;

pub use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};

#[cfg(target_os = "macos")]
pub use sampler::sample_ising;

use quip_miner_core::{run, BackendIdentity, CommonArgs};
use std::process::ExitCode;

/// SA kernel `N` cap: `thread int8_t delta_energy[4593]` in `kernels/sa.metal`.
/// A job over this would overrun kernel-local storage, so it must reject
/// `TooLarge` rather than clamp.
const SA_MAX_NODES: u32 = 4593;
/// Gibbs kernel `N` cap: `thread int8_t packed_state[600]` (600*8) in
/// `kernels/gibbs.metal`.
const GIBBS_MAX_NODES: u32 = 4800;
const DEFAULT_MAX_EDGES: u32 = 1_000_000;

/// Backend identity for `quip-metal-sa`.
/// Metal adapt envelope (from `GPU/metal_miner.py`).
const METAL_ADAPT: quip_miner_core::adapt::AdaptBounds = quip_miner_core::adapt::AdaptBounds {
    min_sweeps: 256,
    max_sweeps: 2048,
    min_reads: 64,
    max_reads: 256,
    reads_solution_min_factor: 0,
    reads_solution_max_factor: 0,
    reads_solution_floor_factor: 0,
};

pub const METAL_SA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "metal",
    algorithm: "sa",
    max_nodes: SA_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: METAL_ADAPT,
};

/// Backend identity for `quip-metal-gibbs`.
pub const METAL_GIBBS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "metal",
    algorithm: "gibbs",
    max_nodes: GIBBS_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: METAL_ADAPT,
};

/// Metal sampler backend: one Apple GPU device plus an IOKit utilization
/// governor. macOS-only.
#[cfg(target_os = "macos")]
pub struct MetalSampler {
    device: crate::metal_device::MetalDevice,
    gov: crate::iokit_gov::UtilGovernor,
    algorithm: Algorithm,
}

/// Metal backend config, parsed from the verbatim `config.toml` subsection in
/// `Configure.backend_toml`. Unrecognized keys land in `unknown`;
/// `warn_unknown_fields` filters session-level keys (e.g. `num_sweeps`).
#[cfg(target_os = "macos")]
#[derive(serde::Deserialize, Default)]
struct MetalConfig {
    /// GPU utilization ceiling 1–100 (governor throttle threshold when yielding).
    utilization: Option<u32>,
    /// Yield the GPU to siblings when util exceeds the ceiling.
    yielding: Option<bool>,
    #[serde(flatten)]
    unknown: std::collections::BTreeMap<String, toml::Value>,
}

#[cfg(target_os = "macos")]
impl MetalSampler {
    pub fn new(
        device: crate::metal_device::MetalDevice,
        gov: crate::iokit_gov::UtilGovernor,
        algorithm: Algorithm,
    ) -> Self {
        Self {
            device,
            gov,
            algorithm,
        }
    }
}

#[cfg(target_os = "macos")]
impl quip_miner_core::Sampler for MetalSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, quip_proto::v1::RejectReason> {
        sample_ising(&self.device, graph, params, self.algorithm).map_err(|e| {
            eprintln!("metal sample failed: {e}");
            quip_proto::v1::RejectReason::Overloaded
        })
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<quip_miner_core::StreamJob>,
        out: tokio::sync::mpsc::Sender<quip_miner_core::StreamResult>,
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
        use quip_miner_core::config::{config_override, warn_unknown_fields};
        let cfg: MetalConfig = toml::from_str(backend_toml).unwrap_or_default();
        warn_unknown_fields("metal", cfg.unknown.keys());
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

/// Run a Metal miner binary. macOS opens the GPU and governor; other platforms
/// support `--capabilities`/`--version` but return `EnvIncompatible` for
/// `--check` and session mode.
pub fn run_metal(
    id: BackendIdentity,
    algorithm: Algorithm,
    common: &CommonArgs,
    device: usize,
    utilization: u32,
    yielding: bool,
) -> ExitCode {
    #[cfg(target_os = "macos")]
    {
        use crate::iokit_gov::UtilGovernor;
        use crate::metal_device::MetalDevice;
        use quip_miner_core::OpenError;
        run(id, common, || {
            let dev = MetalDevice::open(device)
                .map_err(|e| OpenError(format!("device {device}: {e}")))?;
            let gov = UtilGovernor::start(device as u32, utilization, yielding);
            Ok(MetalSampler::new(dev, gov, algorithm))
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (algorithm, device, utilization, yielding);
        struct Unsupported;
        impl quip_miner_core::Sampler for Unsupported {
            fn sample(
                &self,
                _graph: &IsingGraph,
                _params: &SampleParams,
            ) -> Result<Vec<SamplerResult>, quip_proto::v1::RejectReason> {
                Err(quip_proto::v1::RejectReason::Overloaded)
            }
        }
        run(id, common, || {
            Err::<Unsupported, _>(quip_miner_core::OpenError(
                "metal miners require macOS".into(),
            ))
        })
    }
}
