//! Shared miner harness for the v0.3 Ising miners.
//!
//! Holds the generic gRPC session loop, job validation, the base Ising types,
//! the CSR representation used by GPU backends, and the beta schedule. Each
//! miner provides a [`Sampler`] and calls [`run`]; the harness owns everything
//! else (Hello/Welcome, Configure, credits, Reject reasons, Status, Shutdown,
//! idle timeout, exit codes).

pub mod adapt;
pub mod beta;
pub mod cli;
pub mod csr;
pub mod ising;
mod job;
mod session;

pub use cli::CommonArgs;
pub use csr::CsrGraph;
pub use ising::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use session::{run, BackendIdentity, OpenError};

use quip_proto::v1::RejectReason;

/// A backend that samples Ising problems for the miner harness.
///
/// Implementations own their device and algorithm. Only [`sample`](Sampler::sample)
/// is required; the other methods default to a no-governor, uncapped backend
/// (the CPU miner's shape).
pub trait Sampler: Send + Sync + 'static {
    /// Sample one job. Device errors map to a reject reason
    /// (`Overloaded`, `TooLarge`, …).
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason>;

    /// Current utilization for Status messages. `0.0` when no governor.
    fn utilization(&self) -> f64 {
        0.0
    }

    /// Whether to briefly back off before the next job (governor backpressure).
    fn should_throttle(&self) -> bool {
        false
    }

    /// Largest `num_reads` this backend accepts. Defaults to no cap; a backend
    /// with a device-memory bound overrides it.
    fn max_reads(&self) -> u32 {
        u32::MAX
    }
}
