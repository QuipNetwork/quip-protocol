//! Single-job sampling entry point.
//!
//! `sample_ising` drives the self-feeding kernel (see [`crate::streaming`])
//! through a dedicated one-nonce session: same kernels, same host-side
//! quantization/coloring as the streaming path, just without the 3-slot
//! rotation across multiple concurrent models. Energies are always scored
//! host-side with [`quip_protocol::scoring::energy_milli`] for consensus;
//! the kernel's own (int8-quantized) energy tracking only drives its
//! internal accept/reject decisions during annealing.

use crate::cuda_device::CudaDevice;
use crate::streaming;
use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SampleError {
    #[error(transparent)]
    Cuda(#[from] crate::cuda_device::CudaError),
    #[error("CUDA driver: {0}")]
    Driver(String),
}

impl From<cudarc::driver::DriverError> for SampleError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        SampleError::Driver(e.to_string())
    }
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem.
pub fn sample_ising(
    device: &CudaDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<Vec<SamplerResult>, SampleError> {
    streaming::sample_one(device, graph, params, algorithm)
}
