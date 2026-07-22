//! Metal Ising samplers and the shared coordinator session loop.
//!
//! Two binaries share this library:
//! - `quip-metal-sa` — Metropolis simulated annealing on one Apple GPU
//! - `quip-metal-gibbs` — single-site heat-bath Gibbs on one Apple GPU
//!
//! Kernels take **explicit per-job** CSR buffers from the host (no kernel-side
//! nonce economy / rotating slots). Solution energies are always scored with
//! [`quip_protocol::scoring::energy_milli`] for consensus (host f64 — Metal
//! has no fp64).
//!
//! GPU modules (`metal_device`, `iokit_gov`) are macOS-only so Linux CI can
//! still build the CLI stubs and host math.

pub mod beta;
pub mod csr;
pub mod sampler;
pub mod session;

#[cfg(target_os = "macos")]
pub mod iokit_gov;
#[cfg(target_os = "macos")]
pub mod metal_device;

pub use sampler::{Algorithm, SampleParams, SamplerResult};
pub use session::{run_cli, AlgorithmIdentity};

#[cfg(target_os = "macos")]
pub use sampler::sample_ising;
