//! Single-job sampling entry point + GPU energy-kernel verification.
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

/// Evaluate energy_milli for a fixed spin configuration on the GPU.
///
/// Used by golden-parity tests. Host `energy_milli` remains the consensus
/// scorer for production Results; this path verifies the GPU double kernel.
pub fn gpu_energy_milli(
    device: &CudaDevice,
    spins: &[i8],
    h: &[f64],
    j: &[f64],
    edges: &[(usize, usize)],
) -> Result<i64, SampleError> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let n = h.len();
    let m = edges.len();
    let stream = &device.stream;

    if n == 0 {
        return Ok(0);
    }

    let edges_u: Vec<i32> = edges.iter().map(|&(u, _)| u as i32).collect();
    let edges_v: Vec<i32> = edges.iter().map(|&(_, v)| v as i32).collect();
    // Pad empty edge buffers so device pointers are non-null.
    let (edges_u, edges_v, j_host) = if m == 0 {
        (vec![0i32], vec![0i32], vec![0.0f64])
    } else {
        (edges_u, edges_v, j.to_vec())
    };

    let d_spins = stream.clone_htod(spins)?;
    let d_h = stream.clone_htod(h)?;
    let d_j = stream.clone_htod(&j_host)?;
    let d_u = stream.clone_htod(&edges_u)?;
    let d_v = stream.clone_htod(&edges_v)?;
    let mut d_out = stream.alloc_zeros::<i64>(1)?;

    let n_i = n as i32;
    let m_i = m as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&device.energy);
    builder.arg(&d_spins);
    builder.arg(&d_h);
    builder.arg(&d_j);
    builder.arg(&d_u);
    builder.arg(&d_v);
    builder.arg(&n_i);
    builder.arg(&m_i);
    builder.arg(&mut d_out);
    unsafe { builder.launch(cfg) }?;
    stream.synchronize()?;

    let out: Vec<i64> = stream.clone_dtoh(&d_out)?;
    Ok(out[0])
}
