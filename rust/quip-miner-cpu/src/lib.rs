//! CPU Ising samplers and the shared coordinator session loop.
//!
//! Two binaries share this library:
//! - `quip-cpu-sa` — neal-style geometric SA (Metropolis)
//! - `quip-cpu-gibbs` — heat-bath single-site Gibbs over the same ladder

pub mod sampler_core;
pub mod session;

pub use sampler_core::{sample_ising, Algorithm, SampleParams, SamplerResult};
pub use session::{run_cli, AlgorithmIdentity};
