//! CUDA Ising samplers and the shared coordinator session loop.
//!
//! Two binaries share this library:
//! - `quip-cuda-sa` — Metropolis simulated annealing on one GPU
//! - `quip-cuda-gibbs` — single-site heat-bath Gibbs on one GPU
//!
//! Kernels take **explicit per-job** CSR buffers from the host (no kernel-side
//! nonce economy / rotating slots). Solution energies are always scored with
//! [`quip_protocol::scoring::energy_milli`] for consensus.

pub mod beta;
pub mod csr;
pub mod cuda_device;
pub mod nvml_gov;
pub mod sampler;
pub mod session;

pub use sampler::{sample_ising, Algorithm, SampleParams, SamplerResult};
pub use session::{run_cli, AlgorithmIdentity};
