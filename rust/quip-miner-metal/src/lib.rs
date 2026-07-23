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

pub use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};

#[cfg(target_os = "macos")]
pub use sampler::sample_ising;

use quip_miner_core::{run, BackendIdentity, CommonArgs};
use std::process::ExitCode;

const DEFAULT_MAX_NODES: u32 = 100_000;
const DEFAULT_MAX_EDGES: u32 = 1_000_000;
/// Device-memory bound on reads; a job over this is rejected `TooLarge`.
#[cfg(target_os = "macos")]
const DEFAULT_MAX_READS: u32 = 100_000;

/// Backend identity for `quip-metal-sa`.
pub const METAL_SA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "metal",
    algorithm: "sa",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
};

/// Backend identity for `quip-metal-gibbs`.
pub const METAL_GIBBS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "metal",
    algorithm: "gibbs",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
};

/// Metal sampler backend: one Apple GPU device plus an IOKit utilization
/// governor. macOS-only.
#[cfg(target_os = "macos")]
pub struct MetalSampler {
    device: crate::metal_device::MetalDevice,
    gov: crate::iokit_gov::UtilGovernor,
    algorithm: Algorithm,
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
