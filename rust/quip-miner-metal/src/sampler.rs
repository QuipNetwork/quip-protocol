//! Launch SA / Gibbs Metal kernels and score solutions with consensus energy_milli.
//!
//! Dynamics run on the GPU in f32. Final spins are always scored on the host
//! with [`quip_protocol::scoring::energy_milli`] (f64 consensus). There is no
//! GPU energy kernel — Metal Shading Language has no `double`.

use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
use thiserror::Error;

#[cfg(target_os = "macos")]
use quip_miner_core::beta::{default_ising_beta_range, geometric_beta_schedule};
#[cfg(target_os = "macos")]
use quip_miner_core::CsrGraph;
#[cfg(target_os = "macos")]
use quip_protocol::scoring::energy_milli;

#[derive(Debug, Error)]
pub enum SampleError {
    #[cfg(target_os = "macos")]
    #[error(transparent)]
    Metal(#[from] crate::metal_device::MetalError),
    #[error("Metal driver: {0}")]
    Driver(String),
    #[error("Metal unavailable on this platform")]
    Unavailable,
}

#[cfg(target_os = "macos")]
fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
}

/// Packed kernel scalars — layout must match `KernelParams` in `.metal` sources.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct KernelParams {
    num_betas: i32,
    sweeps_per_beta: i32,
    num_reads: i32,
    n: i32,
    base_seed: u32,
}

/// Geometric beta schedule cast to f32 for kernel upload, plus sweeps-per-beta.
///
/// Uses the shared f64 schedule and casts each element to f32 — bit-identical
/// to the prior in-crate f32 schedule.
#[cfg(target_os = "macos")]
fn build_beta_schedule(
    graph: &IsingGraph,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
) -> (Vec<f32>, usize) {
    let sweeps_per = sweeps_per_beta.max(1);
    let num_betas = (num_sweeps / sweeps_per).max(1);
    let (hot, cold) = beta_range.unwrap_or_else(|| default_ising_beta_range(graph));
    let sched: Vec<f32> = geometric_beta_schedule(hot, cold, num_betas)
        .iter()
        .map(|&b| b as f32)
        .collect();
    (sched, sweeps_per)
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem.
///
/// TODO(metal-tts): anneal-depth / sweeps tuning to close the TTS gap vs the
/// v0.2 Python Metal miner — follow-up measurement work, not this scaffold.
#[cfg(target_os = "macos")]
pub fn sample_ising(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<Vec<SamplerResult>, SampleError> {
    use metal::{MTLCommandBufferStatus, MTLSize, NSUInteger};

    let num_reads = params.num_reads.max(1);
    let n = graph.num_nodes();
    let (beta, sweeps_per) = build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );
    let base_seed = (params.seed as u32).wrapping_add(1);

    // Empty problem: no kernel needed.
    if n == 0 {
        return Ok((0..num_reads)
            .map(|_| SamplerResult {
                spins: vec![],
                energy_milli: 0,
            })
            .collect());
    }

    let csr = CsrGraph::from_base(graph);

    // Host CSR may be empty (no edges); still need non-empty device buffers
    // for row_ptr (N+1) and h (N). Pad col/j with a dummy if nnz==0.
    let row_ptr = &csr.row_ptr;
    let (col_ind, j_csr) = if csr.nnz() == 0 {
        (vec![0i32], vec![0.0f32])
    } else {
        (csr.col_ind.clone(), csr.j_csr.clone())
    };

    let d_row = device.new_buffer_from_slice(row_ptr);
    let d_col = device.new_buffer_from_slice(&col_ind);
    let d_j = device.new_buffer_from_slice(&j_csr);
    let d_h = device.new_buffer_from_slice(&csr.h_f32);
    let d_beta = device.new_buffer_from_slice(&beta);
    let workspace_bytes = (num_reads * n) as u64;
    let d_work = device.new_zeroed_buffer(workspace_bytes);
    let d_out = device.new_zeroed_buffer(workspace_bytes);

    let kparams = KernelParams {
        num_betas: beta.len() as i32,
        sweeps_per_beta: sweeps_per as i32,
        num_reads: num_reads as i32,
        n: n as i32,
        base_seed,
    };
    let d_params = device.new_buffer_from_slice(std::slice::from_ref(&kparams));

    let pipeline = match algorithm {
        Algorithm::Sa => &device.sa,
        Algorithm::Gibbs => &device.gibbs,
    };

    let cmd = device.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    // Buffer indices match [[buffer(N)]] in sa.metal / gibbs.metal.
    encoder.set_buffer(0, Some(&d_row), 0);
    encoder.set_buffer(1, Some(&d_col), 0);
    encoder.set_buffer(2, Some(&d_j), 0);
    encoder.set_buffer(3, Some(&d_h), 0);
    encoder.set_buffer(4, Some(&d_work), 0);
    encoder.set_buffer(5, Some(&d_out), 0);
    encoder.set_buffer(6, Some(&d_beta), 0);
    encoder.set_buffer(7, Some(&d_params), 0);

    let max_tg = pipeline.max_total_threads_per_threadgroup() as NSUInteger;
    let tg_w = (256u64.min(max_tg as u64).max(1)) as NSUInteger;
    let grid = MTLSize {
        width: num_reads as NSUInteger,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // A GPU-side failure (OOM binding, device reset, kernel fault, timeout)
    // leaves d_out in its allocated-zero state; without this check the sign map
    // would turn those zeros into an all-`+1` config and score it as a real
    // solution. Reject instead, matching the CUDA path's `synchronize()?`.
    if cmd.status() != MTLCommandBufferStatus::Completed {
        return Err(SampleError::Driver(format!(
            "metal command buffer did not complete: status {:?}",
            cmd.status()
        )));
    }

    let flat = read_i8_buffer(&d_out, num_reads * n)?;
    let mut results = Vec::with_capacity(num_reads);
    for r in 0..num_reads {
        let start = r * n;
        let spins: Vec<i8> = flat[start..start + n]
            .iter()
            .map(|&s| if s >= 0 { 1 } else { -1 })
            .collect();
        results.push(score_spins(&spins, graph));
    }
    Ok(results)
}

#[cfg(target_os = "macos")]
fn read_i8_buffer(buf: &metal::Buffer, count: usize) -> Result<Vec<i8>, SampleError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let ptr = buf.contents() as *const i8;
    if ptr.is_null() {
        return Err(SampleError::Driver(
            "Metal buffer contents() returned null".into(),
        ));
    }
    let mut out = vec![0i8; count];
    // SAFETY: buffer is StorageModeShared, length >= count bytes, and we
    // waited for the command buffer before calling.
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
    }
    Ok(out)
}

/// Stub for non-macOS: sample path is never reached by the harness (`open`
/// fails first with `Unavailable`).
#[cfg(not(target_os = "macos"))]
pub fn sample_ising(
    _device: &(),
    _graph: &IsingGraph,
    _params: &SampleParams,
    _algorithm: Algorithm,
) -> Result<Vec<SamplerResult>, SampleError> {
    Err(SampleError::Unavailable)
}
