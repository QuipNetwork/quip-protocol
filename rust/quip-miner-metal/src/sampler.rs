//! Launch the v0.2 Metal SA / Gibbs kernels and score with consensus energy.
//!
//! Kernels are the original v0.2 Metal sources (`GPU/metal_kernels.metal` /
//! `GPU/metal_gibbs.metal`, copied verbatim into `kernels/`): int8-quantized
//! CSR, D-Wave incremental delta-energy SA / color-block Gibbs, bit-packed
//! thread-local state, one thread per read. Solution energies are always
//! scored on the host with [`quip_protocol::scoring::energy_milli`] (f64
//! consensus) — the kernel's own int8 energy tracking only drives its internal
//! accept/reject decisions. There is no GPU energy kernel (MSL has no `double`).
//!
//! [`encode_job`] builds one job's buffers and encodes the dispatch without
//! committing; [`harvest`] reads the bit-packed samples and host-scores them.
//! The synchronous [`sample_ising`] commits + waits; the streaming loop
//! ([`crate::streaming`]) commits with a completion handler and harvests on the
//! wakeup. Both keep every Metal object on one owner thread.

use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
use thiserror::Error;

#[cfg(target_os = "macos")]
use crate::topology::{fill_h_j, SelfFeedingTopology};
#[cfg(target_os = "macos")]
use quip_miner_core::beta::{default_ising_beta_range, geometric_beta_schedule};
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

/// Largest `num_reads` a single command buffer allocates for (mirrors CUDA).
#[cfg(target_os = "macos")]
pub(crate) const MAX_READS: usize = 256;

/// SA kernel `N` cap: `thread int8_t delta_energy[4593]` in `kernels/sa.metal`.
#[cfg(target_os = "macos")]
pub(crate) const SA_MAX_NODES: usize = 4593;
/// Gibbs kernel `N` cap: `thread int8_t packed_state[600]` (600*8) in
/// `kernels/gibbs.metal`.
#[cfg(target_os = "macos")]
pub(crate) const GIBBS_MAX_NODES: usize = 4800;

#[cfg(target_os = "macos")]
fn algo_max_nodes(algorithm: Algorithm) -> usize {
    match algorithm {
        Algorithm::Sa => SA_MAX_NODES,
        Algorithm::Gibbs => GIBBS_MAX_NODES,
    }
}

#[cfg(target_os = "macos")]
fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
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

/// Unpack one read's bit-packed spins (LSB-first per byte, bit=1 -> -1,
/// bit=0 -> +1; matches the kernel's `set_spin_packed`). Identical to the
/// CUDA crate's unpacker — both kernels share the v0.2 packing convention.
#[cfg(target_os = "macos")]
fn unpack_spins(packed: &[i8], n: usize) -> Vec<i8> {
    let mut spins = vec![1i8; n];
    for (i, s) in spins.iter_mut().enumerate() {
        let byte = packed[i >> 3] as u8;
        let bit = (byte >> (i & 7)) & 1;
        *s = if bit == 1 { -1 } else { 1 };
    }
    spins
}

/// One encoded-but-uncommitted job: the command buffer plus the metadata and
/// device buffers needed to harvest it. Input/scratch buffers are held in
/// `_keep` so they outlive the GPU execution (the command buffer references
/// them; freeing them early would be a use-after-free on the GPU).
#[cfg(target_os = "macos")]
pub(crate) struct EncodedJob {
    pub(crate) cmd: metal::CommandBuffer,
    d_samples: metal::Buffer,
    n: usize,
    num_reads: usize,
    packed_size: usize,
    _keep: Vec<metal::Buffer>,
}

#[cfg(target_os = "macos")]
fn set_bytes_i32(enc: &metal::ComputeCommandEncoderRef, index: u64, val: i32) {
    enc.set_bytes(
        index as metal::NSUInteger,
        4,
        &val as *const i32 as *const std::ffi::c_void,
    );
}

#[cfg(target_os = "macos")]
fn set_bytes_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, val: u32) {
    enc.set_bytes(
        index as metal::NSUInteger,
        4,
        &val as *const u32 as *const std::ffi::c_void,
    );
}

/// Build one job's device buffers and encode its dispatch. Does **not** commit.
///
/// Caller must handle the empty graph (`N == 0`) before calling — this always
/// dispatches a kernel. `num_reads` is clamped to [`MAX_READS`]; with
/// `num_problems = 1` the launch runs `num_reads` threads, one per read.
#[cfg(target_os = "macos")]
pub(crate) fn encode_job(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<EncodedJob, SampleError> {
    use metal::{MTLSize, NSUInteger};

    let n = graph.num_nodes();
    let cap = algo_max_nodes(algorithm);
    if n > cap {
        // Defense in depth: the harness rejects N > max_nodes (identity const)
        // before the sampler; this catches drift before it overruns the
        // kernel's fixed-size thread-local arrays.
        return Err(SampleError::Driver(format!(
            "graph N={n} exceeds {algorithm:?} kernel limit {cap}"
        )));
    }

    let num_reads = params.num_reads.clamp(1, MAX_READS);
    let num_threads = num_reads; // num_problems = 1
    let packed_size = n.div_ceil(8).max(1);

    let (beta, sweeps_per) = build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );
    let num_betas = beta.len() as i32;
    let base_seed = (params.seed as u32).wrapping_add(1);

    let topo = SelfFeedingTopology::build(graph);
    let (j_csr, h_i8) = fill_h_j(&topo, graph);

    // Non-empty device buffers for row_ptr (N+1) and h (N); pad col/J when the
    // graph has no edges so the device pointers stay non-null.
    let row_ptr = if topo.row_ptr.is_empty() {
        vec![0i32]
    } else {
        topo.row_ptr.clone()
    };
    let (col_ind, j_vals) = if topo.nnz == 0 {
        (vec![0i32], vec![0i8])
    } else {
        (topo.col_ind.clone(), j_csr)
    };
    // Single problem: offsets into the concatenated CSR both start at 0.
    let row_offsets = [0i32, row_ptr.len() as i32];
    let col_offsets = [0i32, col_ind.len() as i32];

    let d_row = device.new_buffer_from_slice(&row_ptr);
    let d_col = device.new_buffer_from_slice(&col_ind);
    let d_j = device.new_buffer_from_slice(&j_vals);
    let d_h = device.new_buffer_from_slice(&h_i8);
    let d_row_off = device.new_buffer_from_slice(&row_offsets);
    let d_col_off = device.new_buffer_from_slice(&col_offsets);
    let d_beta = device.new_buffer_from_slice(&beta);
    let d_samples = device.new_zeroed_buffer((num_threads * packed_size) as u64);
    let d_energies = device.new_zeroed_buffer((num_threads * 4) as u64); // i32

    let pipeline = match algorithm {
        Algorithm::Sa => &device.sa,
        Algorithm::Gibbs => &device.gibbs,
    };

    let cmd = device.queue.new_command_buffer().to_owned();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    // Shared buffer layout (indices 0..15) — identical in both kernels.
    encoder.set_buffer(0, Some(&d_row), 0);
    encoder.set_buffer(1, Some(&d_col), 0);
    encoder.set_buffer(2, Some(&d_j), 0);
    encoder.set_buffer(3, Some(&d_row_off), 0);
    encoder.set_buffer(4, Some(&d_col_off), 0);
    set_bytes_i32(encoder, 5, n as i32);
    set_bytes_i32(encoder, 6, num_betas);
    set_bytes_i32(encoder, 7, sweeps_per as i32);
    set_bytes_u32(encoder, 8, base_seed);
    encoder.set_buffer(9, Some(&d_beta), 0);
    encoder.set_buffer(10, Some(&d_samples), 0);
    encoder.set_buffer(11, Some(&d_energies), 0);
    set_bytes_i32(encoder, 12, num_threads as i32);
    set_bytes_i32(encoder, 13, 1); // num_problems
    set_bytes_i32(encoder, 14, num_reads as i32);
    encoder.set_buffer(15, Some(&d_h), 0);

    let mut keep = vec![
        d_row, d_col, d_j, d_h, d_row_off, d_col_off, d_beta, d_energies,
    ];

    match algorithm {
        Algorithm::Sa => {
            // Chunked-dispatch persistent buffers (16..21). Single-shot run:
            // beta_start = 0, beta_count = num_betas, so these are written but
            // never re-read — allocated as zeroed scratch.
            let d_pstate = device.new_zeroed_buffer((num_threads * packed_size) as u64);
            let d_pdelta = device.new_zeroed_buffer((num_threads * n.max(1)) as u64);
            let d_prng = device.new_zeroed_buffer((num_threads * 4) as u64); // uint
            let d_penergy = device.new_zeroed_buffer((num_threads * 4) as u64); // int
            set_bytes_i32(encoder, 16, 0); // beta_start
            set_bytes_i32(encoder, 17, num_betas); // beta_count (>= num_betas)
            encoder.set_buffer(18, Some(&d_pstate), 0);
            encoder.set_buffer(19, Some(&d_pdelta), 0);
            encoder.set_buffer(20, Some(&d_prng), 0);
            encoder.set_buffer(21, Some(&d_penergy), 0);
            keep.extend([d_pstate, d_pdelta, d_prng, d_penergy]);
        }
        Algorithm::Gibbs => {
            let starts = pad_i32(&topo.colors.starts);
            let counts = pad_i32(&topo.colors.counts);
            let nodes = pad_i32(&topo.colors.nodes);
            let d_cstart = device.new_buffer_from_slice(&starts);
            let d_ccount = device.new_buffer_from_slice(&counts);
            let d_cnodes = device.new_buffer_from_slice(&nodes);
            encoder.set_buffer(16, Some(&d_cstart), 0);
            encoder.set_buffer(17, Some(&d_ccount), 0);
            encoder.set_buffer(18, Some(&d_cnodes), 0);
            set_bytes_i32(encoder, 19, 0); // update_mode = heat-bath Gibbs
            set_bytes_i32(encoder, 20, topo.colors.num_colors);
            keep.extend([d_cstart, d_ccount, d_cnodes]);
        }
    }

    // Uniform threadgroups so `thread_id = tg * tgw + tp` is exact; the kernel
    // guards `thread_id >= num_threads` for the last partial group.
    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let tgw = (num_threads as u64).clamp(1, max_tg.min(256));
    let groups = (num_threads as u64).div_ceil(tgw);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tgw as NSUInteger,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(EncodedJob {
        cmd,
        d_samples,
        n,
        num_reads,
        packed_size,
        _keep: keep,
    })
}

#[cfg(target_os = "macos")]
fn pad_i32(v: &[i32]) -> Vec<i32> {
    if v.is_empty() {
        vec![0i32]
    } else {
        v.to_vec()
    }
}

/// Read a completed job's bit-packed samples and host-score each read.
#[cfg(target_os = "macos")]
pub(crate) fn harvest(
    job: &EncodedJob,
    graph: &IsingGraph,
) -> Result<Vec<SamplerResult>, SampleError> {
    let count = job.num_reads * job.packed_size;
    let packed = read_i8_buffer(&job.d_samples, count)?;
    let mut out = Vec::with_capacity(job.num_reads);
    for r in 0..job.num_reads {
        let start = r * job.packed_size;
        let spins = unpack_spins(&packed[start..start + job.packed_size], job.n);
        out.push(score_spins(&spins, graph));
    }
    Ok(out)
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem
/// (synchronous single-job path; used by [`crate::MetalSampler::sample`] and
/// the golden-parity tests).
#[cfg(target_os = "macos")]
pub fn sample_ising(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<Vec<SamplerResult>, SampleError> {
    use metal::MTLCommandBufferStatus;

    let n = graph.num_nodes();
    if n == 0 {
        let reads = params.num_reads.max(1);
        return Ok((0..reads)
            .map(|_| SamplerResult {
                spins: vec![],
                energy_milli: 0,
            })
            .collect());
    }

    let job = encode_job(device, graph, params, algorithm)?;
    job.cmd.commit();
    job.cmd.wait_until_completed();

    // A GPU-side failure (device reset, kernel fault, timeout) leaves d_samples
    // in its allocated-zero state; without this check the unpack would turn
    // those zeros into an all-`+1` config and score it as a real solution.
    if job.cmd.status() != MTLCommandBufferStatus::Completed {
        return Err(SampleError::Driver(format!(
            "metal command buffer did not complete: status {:?}",
            job.cmd.status()
        )));
    }

    harvest(&job, graph)
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
    // SAFETY: buffer is StorageModeShared, length >= count bytes, and the
    // command buffer that wrote it completed before this call.
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
