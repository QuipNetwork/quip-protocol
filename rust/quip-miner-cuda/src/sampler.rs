//! Launch SA / Gibbs kernels and score solutions with consensus energy_milli.

use crate::beta::build_beta_schedule;
use crate::csr::IsingGraph;
use crate::cuda_device::{CudaDevice, CudaError};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use quip_protocol::scoring::energy_milli;
use thiserror::Error;

/// Sampling algorithm selected by the binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// Metropolis-Hastings simulated annealing.
    Sa,
    /// Single-site heat-bath Gibbs along the same beta ladder.
    Gibbs,
}

/// Per-job sampling knobs.
#[derive(Clone, Debug)]
pub struct SampleParams {
    pub num_reads: usize,
    pub num_sweeps: usize,
    pub sweeps_per_beta: usize,
    /// Optional `(hot_beta, cold_beta)`. `None` → auto from biases.
    pub beta_range: Option<(f64, f64)>,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            num_reads: 1,
            num_sweeps: 64,
            sweeps_per_beta: 1,
            beta_range: None,
            seed: 0,
        }
    }
}

/// One completed read: spins in {-1,+1} and consensus milli-energy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerResult {
    pub spins: Vec<i8>,
    pub energy_milli: i64,
}

#[derive(Debug, Error)]
pub enum SampleError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("CUDA driver: {0}")]
    Driver(String),
}

impl From<cudarc::driver::DriverError> for SampleError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        SampleError::Driver(e.to_string())
    }
}

fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem.
pub fn sample_ising(
    device: &CudaDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<Vec<SamplerResult>, SampleError> {
    let num_reads = params.num_reads.max(1);
    let n = graph.num_nodes();
    let (beta, sweeps_per) = build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );
    let num_betas = beta.len() as i32;
    let sweeps_per_beta = sweeps_per as i32;
    let num_reads_i = num_reads as i32;
    let n_i = n as i32;
    let base_seed = (params.seed as u32).wrapping_add(1);

    let stream = &device.stream;

    // Empty problem: no kernel needed.
    if n == 0 {
        return Ok((0..num_reads)
            .map(|_| SamplerResult {
                spins: vec![],
                energy_milli: 0,
            })
            .collect());
    }

    // Host CSR may be empty (no edges); still need non-empty device buffers
    // for row_ptr (N+1) and h (N). Pad col/j with a dummy if nnz==0.
    let row_ptr = &graph.row_ptr;
    let (col_ind, j_csr) = if graph.nnz() == 0 {
        (vec![0i32], vec![0.0f32])
    } else {
        (graph.col_ind.clone(), graph.j_csr.clone())
    };

    let d_row = stream.clone_htod(row_ptr)?;
    let d_col = stream.clone_htod(&col_ind)?;
    let d_j = stream.clone_htod(&j_csr)?;
    let d_h = stream.clone_htod(&graph.h_f32)?;
    let d_beta = stream.clone_htod(&beta)?;
    let mut d_work = stream.alloc_zeros::<i8>(num_reads * n)?;
    let mut d_out = stream.alloc_zeros::<i8>(num_reads * n)?;

    let func = match algorithm {
        Algorithm::Sa => &device.sa,
        Algorithm::Gibbs => &device.gibbs,
    };

    let threads = 256u32;
    let blocks = (num_reads as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(func);
    builder.arg(&d_row);
    builder.arg(&d_col);
    builder.arg(&d_j);
    builder.arg(&d_h);
    builder.arg(&mut d_work);
    builder.arg(&mut d_out);
    builder.arg(&d_beta);
    builder.arg(&num_betas);
    builder.arg(&sweeps_per_beta);
    builder.arg(&num_reads_i);
    builder.arg(&n_i);
    builder.arg(&base_seed);
    unsafe { builder.launch(cfg) }?;
    stream.synchronize()?;

    let flat: Vec<i8> = stream.clone_dtoh(&d_out)?;
    let mut results = Vec::with_capacity(num_reads);
    for r in 0..num_reads {
        let start = r * n;
        let spins = flat[start..start + n].to_vec();
        // Normalize any non-±1 garbage to sign.
        let spins: Vec<i8> = spins
            .into_iter()
            .map(|s| if s >= 0 { 1 } else { -1 })
            .collect();
        results.push(score_spins(&spins, graph));
    }
    Ok(results)
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
