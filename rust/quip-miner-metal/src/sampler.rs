//! Launch the v0.2 Metal SA / Gibbs kernels and score with consensus energy.
//!
//! Kernels are the original v0.2 Metal sources (`GPU/metal_kernels.metal` /
//! `GPU/metal_gibbs.metal`, copied verbatim into `kernels/`): int8-quantized
//! CSR, D-Wave incremental delta-energy SA / color-block Gibbs, bit-packed
//! thread-local state, one thread per read. Solution energies are always
//! scored on the host with [`quip_protocol::scoring::energy_milli`] (f64
//! consensus). There is no GPU energy kernel (MSL has no `double`).
//!
//! # Batched dispatch (throughput)
//!
//! The kernel maps **one threadgroup per problem, one thread per read**
//! (`thread_id = threadgroup·num_reads + read`, `problem_id = thread_id /
//! num_reads`). A dispatch of `P` problems is
//! `dispatchThreadgroups(P, num_reads)` → `P` threadgroups occupy `P` GPU
//! cores, so a batch of ≈`gpu_cores` problems fills the whole GPU and hides
//! the kernel's thread-private memory latency. This mirrors v0.2
//! `metal_sa.py::_dispatch_batch`. Driving `P = 1` (one problem per command
//! buffer) leaves all but one core idle — a ~`gpu_cores`× slowdown — which is
//! why the streaming loop batches ([`crate::streaming`]).
//!
//! [`encode_batch`] builds one batch's buffers and encodes the dispatch without
//! committing; [`harvest_batch`] reads the bit-packed samples per problem. The
//! synchronous [`sample_ising`] runs a single-problem batch and waits.

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

/// Largest `num_reads` a dispatch allocates for (mirrors CUDA). Also the
/// per-threadgroup thread count, well under `maxTotalThreadsPerThreadgroup`.
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
pub(crate) fn algo_max_nodes(algorithm: Algorithm) -> usize {
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

/// One encoded-but-uncommitted batch of `num_problems` problems sharing a
/// topology, plus the metadata and device buffers needed to harvest it.
/// Input/scratch buffers are held in `_keep` so they outlive the GPU execution.
#[cfg(target_os = "macos")]
pub(crate) struct EncodedBatch {
    pub(crate) cmd: metal::CommandBuffer,
    d_samples: metal::Buffer,
    n: usize,
    num_reads: usize,
    num_problems: usize,
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

/// Tile a slice `times` times into one contiguous `Vec`.
#[cfg(target_os = "macos")]
fn tile_i32(src: &[i32], times: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(src.len() * times);
    for _ in 0..times {
        out.extend_from_slice(src);
    }
    out
}

/// Build one batch's device buffers and encode its dispatch. Does **not** commit.
///
/// `graphs` must be non-empty and share a topology (same `N` and `edges`) — the
/// caller ([`crate::streaming`]) guarantees this by batch key; the topology is
/// built from `graphs[0]`. `params` (reads, sweeps, beta) is shared across the
/// batch, matching v0.2 (`compute_beta_schedule(h[0], J[0], ...)`). Dispatches
/// `dispatchThreadgroups(num_problems, num_reads)`.
#[cfg(target_os = "macos")]
pub(crate) fn encode_batch(
    device: &crate::metal_device::MetalDevice,
    graphs: &[&IsingGraph],
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<EncodedBatch, SampleError> {
    use metal::{MTLSize, NSUInteger};

    let num_problems = graphs.len();
    debug_assert!(num_problems >= 1, "encode_batch needs at least one graph");
    let n = graphs[0].num_nodes();
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
    let num_threads = num_problems * num_reads;
    let packed_size = n.div_ceil(8).max(1);

    let (beta, sweeps_per) = build_beta_schedule(
        graphs[0],
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );
    let num_betas = beta.len() as i32;
    let base_seed = (params.seed as u32).wrapping_add(1);

    let topo = SelfFeedingTopology::build(graphs[0]);
    let nnz_alloc = topo.nnz.max(1);
    let rp_len = topo.row_ptr.len().max(1);

    // Shared CSR structure, tiled per problem; per-problem J / h values.
    let base_row = if topo.row_ptr.is_empty() {
        vec![0i32]
    } else {
        topo.row_ptr.clone()
    };
    let base_col = if topo.nnz == 0 {
        vec![0i32]
    } else {
        topo.col_ind.clone()
    };
    let all_row_ptr = tile_i32(&base_row, num_problems);
    let all_col_ind = tile_i32(&base_col, num_problems);
    let mut all_j = vec![0i8; num_problems * nnz_alloc];
    let mut all_h = vec![0i8; num_problems * n];
    for (p, graph) in graphs.iter().enumerate() {
        let (j_csr, h_i8) = fill_h_j(&topo, graph);
        // `j_csr` has length `topo.nnz`; pad region is the trailing slot when
        // nnz == 0. `h_i8` has length N.
        all_j[p * nnz_alloc..p * nnz_alloc + j_csr.len()].copy_from_slice(&j_csr);
        all_h[p * n..p * n + h_i8.len()].copy_from_slice(&h_i8);
    }
    let row_ptr_offsets: Vec<i32> = (0..=num_problems).map(|p| (p * rp_len) as i32).collect();
    let col_ind_offsets: Vec<i32> = (0..=num_problems).map(|p| (p * nnz_alloc) as i32).collect();

    let d_row = device.new_buffer_from_slice(&all_row_ptr);
    let d_col = device.new_buffer_from_slice(&all_col_ind);
    let d_j = device.new_buffer_from_slice(&all_j);
    let d_h = device.new_buffer_from_slice(&all_h);
    let d_row_off = device.new_buffer_from_slice(&row_ptr_offsets);
    let d_col_off = device.new_buffer_from_slice(&col_ind_offsets);
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
    set_bytes_i32(encoder, 13, num_problems as i32);
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
            // Color blocks are shared across the batch (same topology → same
            // coloring); the kernel indexes them globally, not per problem.
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

    // One threadgroup per problem, `num_reads` threads (one per read) each:
    // `problem_id = thread_id / num_reads` = the threadgroup index.
    encoder.dispatch_thread_groups(
        MTLSize {
            width: num_problems as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: num_reads as NSUInteger,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(EncodedBatch {
        cmd,
        d_samples,
        n,
        num_reads,
        num_problems,
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

/// Read a completed batch's bit-packed samples and host-score each read,
/// returning one `Vec<SamplerResult>` per problem (in `graphs` order).
///
/// `graphs` must be the exact slice passed to [`encode_batch`] (same order and
/// length) so each problem's spins are scored against its own `h`/`J`.
#[cfg(target_os = "macos")]
pub(crate) fn harvest_batch(
    batch: &EncodedBatch,
    graphs: &[&IsingGraph],
) -> Result<Vec<Vec<SamplerResult>>, SampleError> {
    debug_assert_eq!(graphs.len(), batch.num_problems);
    let count = batch.num_problems * batch.num_reads * batch.packed_size;
    let packed = read_i8_buffer(&batch.d_samples, count)?;
    let mut out = Vec::with_capacity(batch.num_problems);
    for (p, graph) in graphs.iter().enumerate() {
        let mut reads = Vec::with_capacity(batch.num_reads);
        for r in 0..batch.num_reads {
            let start = (p * batch.num_reads + r) * batch.packed_size;
            let spins = unpack_spins(&packed[start..start + batch.packed_size], batch.n);
            reads.push(score_spins(&spins, graph));
        }
        out.push(reads);
    }
    Ok(out)
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem
/// (synchronous single-problem batch; used by [`crate::MetalSampler::sample`]
/// and the golden-parity tests).
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

    let batch = encode_batch(device, &[graph], params, algorithm)?;
    batch.cmd.commit();
    batch.cmd.wait_until_completed();

    // A GPU-side failure (device reset, kernel fault, timeout) leaves d_samples
    // in its allocated-zero state; without this check the unpack would turn
    // those zeros into an all-`+1` config and score it as a real solution.
    if batch.cmd.status() != MTLCommandBufferStatus::Completed {
        return Err(SampleError::Driver(format!(
            "metal command buffer did not complete: status {:?}",
            batch.cmd.status()
        )));
    }

    let mut per_problem = harvest_batch(&batch, &[graph])?;
    Ok(per_problem.remove(0))
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
