//! Self-feeding streaming session: persistent kernel + 3-slot rotation.
//!
//! Rust port of `GPU/base_cuda_sampler.py`'s `prepare_self_feeding` /
//! `upload_slot` / `download_slot` / `launch_self_feeding` /
//! `_run_streaming_loop`, adapted to the `Sampler::sample_stream` contract
//! (`blocking_recv`/`blocking_send` over bounded tokio mpsc channels) and to
//! `GPU/slot_rotation.py`'s `SlotState` bookkeeping.
//!
//! Deviation from the reference driver: the Python cold start blocks for
//! `num_k` models unconditionally, which assumes an effectively-infinite
//! feeder. Jobs here arrive credit-gated from a coordinator that may send
//! fewer than `stream_width()` jobs in total (e.g. a short drive run), so an
//! unconditional blocking cold start could hang forever. This driver instead
//! blocks for the first job, then drains whatever else is already queued
//! (bounded wait), and launches with however many nonces that filled — still
//! correct, just not guaranteed to hit full width on a very short run.

use crate::cuda_device::CudaDevice;
use crate::sampler::SampleError;
use crate::topology::{fill_h_j, SelfFeedingTopology};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use quip_miner_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_miner_core::{
    Algorithm, IsingGraph, SampleParams, SamplerResult, StreamJob, StreamResult,
};
use quip_proto::v1::RejectReason;
use quip_protocol::scoring::energy_milli;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{Receiver, Sender};

const CTRL_STRIDE: usize = 8;
const CTRL_EXIT_NOW: usize = 6;
const SLOTS_PER_NONCE: usize = 3;
const SLOT_READY: i32 = 1;
const SLOT_COMPLETE: i32 = 3;

/// Per-algorithm constants dictated by the verbatim kernel's fixed-size
/// arrays / thread-block shape. Not tunable without editing the kernel.
struct AlgoLimits {
    /// CUDA blocks (SMs) launched per nonce.
    sms_per_nonce: usize,
    /// Largest `N` the kernel's fixed-size per-thread/shared state supports.
    max_nodes: usize,
    /// Largest reads-per-nonce this driver allocates for.
    max_reads: usize,
}

fn algo_limits(algorithm: Algorithm) -> AlgoLimits {
    match algorithm {
        // 1 block (1 SM) per nonce; `if (tid < num_reads)` in a 256-thread
        // block hard-caps reads/nonce; `unpacked_state[5000]` caps N.
        Algorithm::Sa => AlgoLimits {
            sms_per_nonce: 1,
            max_nodes: 5000,
            max_reads: 256,
        },
        // `shared_state[4800]` caps N. reads/nonce isn't block-capped (work
        // is chunked across `sms_per_nonce` blocks) but is held to the same
        // 256 for a uniform, generous device-memory bound.
        Algorithm::Gibbs => AlgoLimits {
            sms_per_nonce: 4,
            max_nodes: 4800,
            max_reads: 256,
        },
    }
}

/// Backend-facing read cap for `Sampler::max_reads`. `max_nodes` is instead
/// hardcoded directly on [`crate::CUDA_SA_IDENTITY`] /
/// [`crate::CUDA_GIBBS_IDENTITY`] (kept next to `algo_limits` in spirit —
/// `BackendIdentity` is a `const`, so it can't call a non-`const fn` here).
pub fn max_reads(algorithm: Algorithm) -> u32 {
    algo_limits(algorithm).max_reads as u32
}

fn tile_i32(src: &[i32], times: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(src.len() * times);
    for _ in 0..times {
        out.extend_from_slice(src);
    }
    out
}

/// Geometric beta schedule cast to f32 for kernel upload, plus sweeps-per-beta.
pub(crate) fn build_beta_schedule(
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

pub(crate) fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
}

/// Unpack one read's bit-packed spins (LSB-first per byte, bit=1 -> -1,
/// bit=0 -> +1; matches the kernel's `set_spin_packed`).
fn unpack_spins(packed: &[i8], n: usize) -> Vec<i8> {
    let mut spins = vec![1i8; n];
    for (i, s) in spins.iter_mut().enumerate() {
        let byte = packed[i >> 3] as u8;
        let bit = (byte >> (i & 7)) & 1;
        *s = if bit == 1 { -1 } else { 1 };
    }
    spins
}

/// A running self-feeding kernel + its 3-slot rotating buffers.
///
/// Buffers are allocated with the device's event tracking disabled (see
/// `CudaDevice::open`): the persistent kernel on `stream_compute` and the
/// slot upload/download traffic on `stream_transfer` intentionally race at
/// the byte-range level, arbitrated by the kernel's own volatile ctrl
/// protocol, not by stream ordering. `Drop` signals exit and synchronizes
/// `stream_compute` before any buffer is freed, so this is safe as long as
/// no other code independently frees these slices first.
struct SelfFeedingSession<'a> {
    device: &'a CudaDevice,
    algorithm: Algorithm,
    topology: SelfFeedingTopology,
    num_nonces: usize,
    active_nonces: usize,
    reads_per_nonce: usize,
    max_packed_size: usize,

    stream_compute: Arc<CudaStream>,
    stream_transfer: Arc<CudaStream>,

    d_row_ptr: CudaSlice<i32>,
    d_col_ind: CudaSlice<i32>,
    d_j: CudaSlice<i8>,
    d_h: CudaSlice<i8>,
    d_samples: CudaSlice<i8>,
    d_energies: CudaSlice<i32>,
    d_ctrl: CudaSlice<i32>,
    d_beta: CudaSlice<f32>,

    algo_state: AlgoState,

    // Host staging, reused across uploads to avoid a realloc per model.
    stage_j: Vec<i8>,
    stage_h: Vec<i8>,

    launched: bool,
}

/// Algorithm-specific buffers, kept out of `Option`s: which variant is
/// populated always matches `SelfFeedingSession::algorithm` by construction,
/// so `launch()` destructures it directly instead of unwrapping an `Option`
/// known-Some-by-invariant.
enum AlgoState {
    Sa {
        d_delta_energy: CudaSlice<i8>,
    },
    Gibbs {
        d_block_starts: CudaSlice<i32>,
        d_block_counts: CudaSlice<i32>,
        d_color_nodes: CudaSlice<i32>,
        num_colors: i32,
        chunks_per_model: i32,
        reads_per_chunk: i32,
    },
}

impl<'a> SelfFeedingSession<'a> {
    fn build(
        device: &'a CudaDevice,
        algorithm: Algorithm,
        topology: SelfFeedingTopology,
        num_nonces: usize,
        reads_per_nonce: usize,
        max_num_betas: usize,
    ) -> Result<Self, SampleError> {
        let limits = algo_limits(algorithm);
        let n = topology.n;
        // Defense in depth: `CUDA_SA_IDENTITY`/`CUDA_GIBBS_IDENTITY.max_nodes`
        // already reject an oversized job in `job.rs` before it reaches the
        // sampler; this catches any future drift between those consts and
        // the kernel's actual fixed-size array bounds before it becomes a
        // kernel-side buffer overrun instead of a clean error.
        if n > limits.max_nodes {
            return Err(SampleError::Driver(format!(
                "graph N={n} exceeds self-feeding kernel limit {}",
                limits.max_nodes
            )));
        }
        let nnz_alloc = topology.nnz.max(1);
        let max_packed_size = n.div_ceil(8).max(1);
        let total_slots = num_nonces * SLOTS_PER_NONCE;

        let ctx = &device.ctx;
        let stream_compute = ctx.new_stream()?;
        let stream_transfer = ctx.new_stream()?;

        let row_ptr = if topology.row_ptr.is_empty() {
            vec![0i32]
        } else {
            topology.row_ptr.clone()
        };
        let col_ind = if topology.nnz == 0 {
            vec![0i32]
        } else {
            topology.col_ind.clone()
        };
        let d_row_ptr = stream_compute.clone_htod(&row_ptr)?;
        let d_col_ind = stream_compute.clone_htod(&col_ind)?;

        let d_j = stream_compute.alloc_zeros::<i8>(total_slots * nnz_alloc)?;
        let d_h = stream_compute.alloc_zeros::<i8>(total_slots * n.max(1))?;
        let d_samples =
            stream_compute.alloc_zeros::<i8>(total_slots * reads_per_nonce * max_packed_size)?;
        let d_energies =
            stream_compute.alloc_zeros::<i32>((total_slots * reads_per_nonce).max(1))?;
        let d_ctrl = stream_compute.alloc_zeros::<i32>(num_nonces * CTRL_STRIDE)?;
        let d_beta = stream_compute.alloc_zeros::<f32>(max_num_betas.max(1))?;

        let algo_state = match algorithm {
            Algorithm::Sa => {
                let total_threads = num_nonces * 256;
                AlgoState::Sa {
                    d_delta_energy: stream_compute.alloc_zeros::<i8>(total_threads * n.max(1))?,
                }
            }
            Algorithm::Gibbs => {
                let starts = tile_i32(&topology.colors.starts, num_nonces);
                let counts = tile_i32(&topology.colors.counts, num_nonces);
                let starts = if starts.is_empty() {
                    vec![0i32]
                } else {
                    starts
                };
                let counts = if counts.is_empty() {
                    vec![0i32]
                } else {
                    counts
                };
                let nodes = if topology.colors.nodes.is_empty() {
                    vec![0i32]
                } else {
                    topology.colors.nodes.clone()
                };
                AlgoState::Gibbs {
                    d_block_starts: stream_compute.clone_htod(&starts)?,
                    d_block_counts: stream_compute.clone_htod(&counts)?,
                    d_color_nodes: stream_compute.clone_htod(&nodes)?,
                    num_colors: topology.colors.num_colors,
                    chunks_per_model: limits.sms_per_nonce as i32,
                    reads_per_chunk: reads_per_nonce.div_ceil(limits.sms_per_nonce) as i32,
                }
            }
        };

        Ok(Self {
            device,
            algorithm,
            topology,
            num_nonces,
            active_nonces: 0,
            reads_per_nonce,
            max_packed_size,
            stream_compute,
            stream_transfer,
            d_row_ptr,
            d_col_ind,
            d_j,
            d_h,
            d_samples,
            d_energies,
            d_ctrl,
            d_beta,
            algo_state,
            stage_j: vec![0i8; nnz_alloc],
            stage_h: vec![0i8; n.max(1)],
            launched: false,
        })
    }

    fn upload_beta_schedule(&mut self, sched: &[f32]) -> Result<(), SampleError> {
        if sched.is_empty() {
            return Ok(());
        }
        self.stream_transfer
            .memcpy_htod(sched, &mut self.d_beta.slice_mut(0..sched.len()))?;
        Ok(())
    }

    /// Upload one job's `h`/`J` into `(nonce_id, slot_id)` and mark it READY.
    /// The job's graph must already be known compatible with `self.topology`
    /// (same `N`/edges) — checked by the caller via [`SessionKey`].
    fn upload_slot(
        &mut self,
        nonce_id: usize,
        slot_id: usize,
        graph: &IsingGraph,
    ) -> Result<(), SampleError> {
        let slot_idx = nonce_id * SLOTS_PER_NONCE + slot_id;
        let nnz = self.topology.nnz;
        let n = self.topology.n;

        self.stage_j[..nnz.max(1)].fill(0);
        self.stage_h[..n.max(1)].fill(0);
        let (j, h) = fill_h_j(&self.topology, graph);
        self.stage_j[..j.len()].copy_from_slice(&j);
        self.stage_h[..h.len()].copy_from_slice(&h);

        let nnz_alloc = self.topology.nnz.max(1);
        let n_alloc = self.topology.n.max(1);
        let j_start = slot_idx * nnz_alloc;
        let h_start = slot_idx * n_alloc;
        self.stream_transfer.memcpy_htod(
            &self.stage_j[..nnz_alloc],
            &mut self.d_j.slice_mut(j_start..j_start + nnz_alloc),
        )?;
        self.stream_transfer.memcpy_htod(
            &self.stage_h[..n_alloc],
            &mut self.d_h.slice_mut(h_start..h_start + n_alloc),
        )?;

        // Zero this slot's output region so a stale prior model's samples
        // can't leak through if the kernel writes fewer bytes than expected.
        let sample_start = slot_idx * self.reads_per_nonce * self.max_packed_size;
        let sample_len = self.reads_per_nonce * self.max_packed_size;
        let zeros = vec![0i8; sample_len];
        self.stream_transfer.memcpy_htod(
            &zeros,
            &mut self
                .d_samples
                .slice_mut(sample_start..sample_start + sample_len),
        )?;

        let ctrl_offset = nonce_id * CTRL_STRIDE + slot_id;
        self.stream_transfer.memcpy_htod(
            &[SLOT_READY],
            &mut self.d_ctrl.slice_mut(ctrl_offset..ctrl_offset + 1),
        )?;
        Ok(())
    }

    /// Download and unpack one COMPLETE slot's samples into per-read spins.
    fn download_slot(&self, nonce_id: usize, slot_id: usize) -> Result<Vec<Vec<i8>>, SampleError> {
        let slot_idx = nonce_id * SLOTS_PER_NONCE + slot_id;
        let sample_start = slot_idx * self.reads_per_nonce * self.max_packed_size;
        let sample_len = self.reads_per_nonce * self.max_packed_size;
        let packed: Vec<i8> = self.stream_transfer.clone_dtoh(
            &self
                .d_samples
                .slice(sample_start..sample_start + sample_len),
        )?;
        let n = self.topology.n;
        Ok((0..self.reads_per_nonce)
            .map(|r| {
                let start = r * self.max_packed_size;
                unpack_spins(&packed[start..start + self.max_packed_size], n)
            })
            .collect())
    }

    fn poll_ctrl(&self) -> Result<Vec<i32>, SampleError> {
        Ok(self.stream_transfer.clone_dtoh(&self.d_ctrl)?)
    }

    fn launch(
        &mut self,
        active_nonces: usize,
        num_betas: i32,
        sweeps_per_beta: i32,
        seed: u32,
    ) -> Result<(), SampleError> {
        self.active_nonces = active_nonces;
        let limits = algo_limits(self.algorithm);
        let num_blocks = (active_nonces * limits.sms_per_nonce) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = self.topology.n as i32;
        let nnz = self.topology.nnz as i32;
        let max_packed = self.max_packed_size as i32;
        let num_nonces = self.num_nonces as i32;
        let reads_per_nonce = self.reads_per_nonce as i32;

        // Buffer args are pushed as shared refs regardless of which side
        // (host driver vs. persistent kernel) mutates them: event tracking
        // is disabled for this device (see `CudaDevice::open`), so
        // cudarc's read/write distinction is inert here — the kernel's own
        // volatile ctrl protocol is the actual synchronization.
        match &self.algo_state {
            AlgoState::Sa { d_delta_energy } => {
                let mut b = self.stream_compute.launch_builder(&self.device.sa);
                b.arg(&self.d_row_ptr);
                b.arg(&self.d_col_ind);
                b.arg(&self.d_j);
                b.arg(&self.d_h);
                b.arg(&self.d_samples);
                b.arg(&self.d_energies);
                b.arg(&self.d_beta);
                b.arg(&num_betas);
                b.arg(&sweeps_per_beta);
                b.arg(&self.d_ctrl);
                b.arg(&num_nonces);
                b.arg(&reads_per_nonce);
                b.arg(&n);
                b.arg(&nnz);
                b.arg(&max_packed);
                b.arg(&seed);
                b.arg(d_delta_energy);
                b.arg(&n);
                unsafe { b.launch(cfg) }?;
            }
            AlgoState::Gibbs {
                d_block_starts,
                d_block_counts,
                d_color_nodes,
                num_colors,
                chunks_per_model,
                reads_per_chunk,
            } => {
                let mut b = self.stream_compute.launch_builder(&self.device.gibbs);
                b.arg(&self.d_row_ptr);
                b.arg(&self.d_col_ind);
                b.arg(d_block_starts);
                b.arg(d_block_counts);
                b.arg(d_color_nodes);
                b.arg(num_colors);
                b.arg(&self.d_beta);
                b.arg(&num_betas);
                b.arg(&sweeps_per_beta);
                b.arg(&self.d_j);
                b.arg(&self.d_h);
                b.arg(&self.d_samples);
                b.arg(&self.d_energies);
                b.arg(&self.d_ctrl);
                b.arg(&num_nonces);
                let sms_per_nonce = limits.sms_per_nonce as i32;
                b.arg(&sms_per_nonce);
                b.arg(&reads_per_nonce);
                b.arg(&n);
                b.arg(&nnz);
                b.arg(&max_packed);
                b.arg(chunks_per_model);
                b.arg(reads_per_chunk);
                b.arg(&seed);
                let update_mode = 0i32; // heat-bath Gibbs (no metropolis knob on this wire path)
                b.arg(&update_mode);
                unsafe { b.launch(cfg) }?;
            }
        }
        self.launched = true;
        Ok(())
    }

    fn signal_exit(&mut self) -> Result<(), SampleError> {
        if !self.launched {
            return Ok(());
        }
        let exit = vec![1i32; 1];
        for nonce_id in 0..self.active_nonces {
            let off = nonce_id * CTRL_STRIDE + CTRL_EXIT_NOW;
            self.stream_transfer
                .memcpy_htod(&exit, &mut self.d_ctrl.slice_mut(off..off + 1))?;
        }
        Ok(())
    }

    fn wait_exit(&mut self) -> Result<(), SampleError> {
        if self.launched {
            self.stream_compute.synchronize()?;
            self.launched = false;
        }
        Ok(())
    }
}

impl Drop for SelfFeedingSession<'_> {
    fn drop(&mut self) {
        // Kernel must genuinely exit before any CudaSlice field is freed:
        // event tracking is disabled for this device (see CudaDevice::open),
        // so there is no automatic wait built into the Drop of those fields.
        let _ = self.signal_exit();
        let _ = self.wait_exit();
    }
}

/// Structural + sampling-config identity a self-feeding session is built
/// for. A job can reuse the running session iff it matches: same graph
/// (topology, so the CSR/coloring/edge positions stay valid) and same
/// beta-schedule shape (so the shared beta buffer stays valid). `num_reads`
/// only needs to fit under the session's established per-slot capacity.
#[derive(Clone)]
struct SessionKey {
    n: usize,
    edges: Vec<(usize, usize)>,
    reads_per_nonce: usize,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
}

impl SessionKey {
    fn seed(job: &StreamJob, reads_per_nonce: usize) -> Self {
        Self {
            n: job.graph.h.len(),
            edges: job.graph.edges.clone(),
            reads_per_nonce,
            num_sweeps: job.params.num_sweeps,
            sweeps_per_beta: job.params.sweeps_per_beta.max(1),
            beta_range: job.params.beta_range,
        }
    }

    fn matches(&self, job: &StreamJob) -> bool {
        self.n == job.graph.h.len()
            && self.edges == job.graph.edges
            && self.num_sweeps == job.params.num_sweeps
            && self.sweeps_per_beta == job.params.sweeps_per_beta.max(1)
            && self.beta_range == job.params.beta_range
            && job.params.num_reads.max(1) <= self.reads_per_nonce
    }
}

/// Run one job through a dedicated single-nonce self-feeding session:
/// upload to slot 0, launch, poll to completion, download, tear down.
/// Used by [`crate::sampler::sample_ising`] (the `Sampler::sample` path)
/// and as the streaming loop's oversized/incompatible-job fallback.
pub fn sample_one(
    device: &CudaDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<Vec<SamplerResult>, SampleError> {
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

    let limits = algo_limits(algorithm);
    let reads_per_nonce = params.num_reads.max(1).min(limits.max_reads);
    let (beta, sweeps_per_beta) = build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );
    let num_betas = beta.len() as i32;
    let topology = SelfFeedingTopology::build(graph);

    let mut sess =
        SelfFeedingSession::build(device, algorithm, topology, 1, reads_per_nonce, beta.len())?;
    sess.upload_beta_schedule(&beta)?;
    sess.upload_slot(0, 0, graph)?;
    let seed = (params.seed as u32).wrapping_add(1);
    sess.launch(1, num_betas, sweeps_per_beta as i32, seed)?;

    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let ctrl = sess.poll_ctrl()?;
        if ctrl[0] == SLOT_COMPLETE {
            break;
        }
        if Instant::now() > deadline {
            return Err(SampleError::Driver("self-feeding kernel timed out".into()));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let reads = sess.download_slot(0, 0)?;
    sess.signal_exit()?;
    sess.wait_exit()?;

    Ok(reads
        .into_iter()
        .take(params.num_reads.max(1))
        .map(|spins| score_spins(&spins, graph))
        .collect())
}

/// Per-nonce ACTIVE/NEXT slot bookkeeping. Port of `GPU/slot_rotation.py`.
struct SlotState {
    active_slot: i32,
    active_job: Option<StreamJob>,
    active_started: Option<Instant>,
    next_slot: i32,
    next_job: Option<StreamJob>,
}

impl SlotState {
    fn new() -> Self {
        Self {
            active_slot: -1,
            active_job: None,
            active_started: None,
            next_slot: -1,
            next_job: None,
        }
    }

    fn is_idle(&self) -> bool {
        self.active_job.is_none()
    }

    fn needs_next(&self) -> bool {
        self.active_job.is_some() && self.next_job.is_none()
    }

    fn free_slot(&self) -> i32 {
        for i in 0..SLOTS_PER_NONCE as i32 {
            if i != self.active_slot && i != self.next_slot {
                return i;
            }
        }
        -1
    }

    fn assign_active(&mut self, slot: i32, job: StreamJob) {
        self.active_slot = slot;
        self.active_started = Some(Instant::now());
        self.active_job = Some(job);
    }

    fn assign_next(&mut self, slot: i32, job: StreamJob) {
        self.next_slot = slot;
        self.next_job = Some(job);
    }

    fn rotate_on_completion(&mut self) {
        if self.next_job.is_some() {
            self.active_slot = self.next_slot;
            self.active_started = Some(Instant::now());
            self.active_job = self.next_job.take();
        } else {
            self.active_slot = -1;
            self.active_started = None;
            self.active_job = None;
        }
        self.next_slot = -1;
    }
}

/// `Sampler::stream_width()` for the CUDA backend: `max_sms / sms_per_nonce`.
pub fn stream_width(device: &CudaDevice, algorithm: Algorithm) -> usize {
    (device.max_sms / algo_limits(algorithm).sms_per_nonce).max(1)
}

enum Pull {
    Job(StreamJob),
    Mismatch(StreamJob),
    Empty,
    Closed,
}

fn try_pull(jobs: &mut Receiver<StreamJob>, key: &SessionKey) -> Pull {
    match jobs.try_recv() {
        Ok(j) if key.matches(&j) => Pull::Job(j),
        Ok(j) => Pull::Mismatch(j),
        Err(TryRecvError::Empty) => Pull::Empty,
        Err(TryRecvError::Disconnected) => Pull::Closed,
    }
}

fn send_reject(out: &Sender<StreamResult>, job: StreamJob, reason: RejectReason) {
    let _ = out.blocking_send(StreamResult {
        job_id: job.job_id,
        result: Err(reason),
        device_access_time_us: 0,
    });
}

/// Drive the self-feeding streaming loop for the lifetime of `jobs`: keep up
/// to [`stream_width`] models in flight across one (or, on a topology/param
/// change, successive) persistent kernel launches, emitting results in
/// completion order.
pub fn run_stream(
    device: &CudaDevice,
    algorithm: Algorithm,
    mut jobs: Receiver<StreamJob>,
    out: Sender<StreamResult>,
) {
    let width = stream_width(device, algorithm);
    let limits = algo_limits(algorithm);
    let mut pending_seed: Option<StreamJob> = match jobs.blocking_recv() {
        Some(j) => Some(j),
        None => return,
    };

    'session: while let Some(seed) = pending_seed.take() {
        if seed.graph.num_nodes() == 0 {
            // Degenerate empty-graph job: no kernel needed, answer directly.
            let reads = seed.params.num_reads.max(1);
            let _ = out.blocking_send(StreamResult {
                job_id: seed.job_id,
                result: Ok((0..reads)
                    .map(|_| SamplerResult {
                        spins: vec![],
                        energy_milli: 0,
                    })
                    .collect()),
                device_access_time_us: 0,
            });
            pending_seed = match jobs.blocking_recv() {
                Some(j) => Some(j),
                None => return,
            };
            continue 'session;
        }

        let reads_per_nonce = seed.params.num_reads.max(1).min(limits.max_reads);
        let key = SessionKey::seed(&seed, reads_per_nonce);
        let (beta, sweeps_per_beta) = build_beta_schedule(
            &seed.graph,
            seed.params.num_sweeps,
            seed.params.sweeps_per_beta,
            seed.params.beta_range,
        );
        let num_betas = beta.len() as i32;
        let topology = SelfFeedingTopology::build(&seed.graph);

        let mut sess = match SelfFeedingSession::build(
            device,
            algorithm,
            topology,
            width,
            reads_per_nonce,
            beta.len(),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cuda streaming session build failed: {e}");
                send_reject(&out, seed, RejectReason::Overloaded);
                pending_seed = jobs.blocking_recv();
                continue 'session;
            }
        };
        if sess.upload_beta_schedule(&beta).is_err() {
            send_reject(&out, seed, RejectReason::Overloaded);
            pending_seed = jobs.blocking_recv();
            continue 'session;
        }

        let mut slots: Vec<SlotState> = (0..width).map(|_| SlotState::new()).collect();

        // Cold start: the first job is already in hand; drain whatever else
        // shows up so the launch starts with as much concurrency as
        // possible. `launch_self_feeding`'s grid size is fixed for the
        // kernel's whole lifetime (no adding nonces after launch), so it's
        // worth waiting past the first quiet moment: this keeps pulling
        // until either `width` is reached, a hard cap elapses, or a full
        // `idle_timeout` passes with nothing new arriving (a burst source
        // like a coordinator dispatching its whole staged queue arrives in
        // well under `idle_timeout`; a genuinely slow/empty source gives up
        // after it).
        let mut cold: Vec<StreamJob> = vec![seed];
        let mut mismatch: Option<StreamJob> = None;
        let mut closed = false;
        let cold_hard_cap = Instant::now() + Duration::from_secs(3);
        let idle_timeout = Duration::from_millis(150);
        let mut last_arrival = Instant::now();
        while cold.len() < width && Instant::now() < cold_hard_cap {
            match try_pull(&mut jobs, &key) {
                Pull::Job(j) => {
                    cold.push(j);
                    last_arrival = Instant::now();
                }
                Pull::Mismatch(j) => {
                    mismatch = Some(j);
                    break;
                }
                Pull::Empty => {
                    if last_arrival.elapsed() > idle_timeout {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                Pull::Closed => {
                    closed = true;
                    break;
                }
            }
        }

        let active_nonces = cold.len();
        eprintln!(
            "quip-miner-cuda: self-feeding session launching with {active_nonces}/{width} nonces active"
        );
        for (nonce_id, job) in cold.into_iter().enumerate() {
            if sess.upload_slot(nonce_id, 0, &job.graph).is_err() {
                send_reject(&out, job, RejectReason::Overloaded);
                continue;
            }
            slots[nonce_id].assign_active(0, job);
        }
        let seed_val = (Instant::now().elapsed().as_nanos() as u32) ^ 0x9E3779B9;
        if sess
            .launch(active_nonces, num_betas, sweeps_per_beta as i32, seed_val)
            .is_err()
        {
            // Every cold-start job already handed off; nothing more to reject
            // here beyond what upload_slot rejected above.
            pending_seed = if closed { None } else { jobs.blocking_recv() };
            continue 'session;
        }

        let mut exhausted = closed || mismatch.is_some();

        loop {
            // Fill NEXT / revive idle nonces (non-blocking) before polling.
            if !exhausted {
                'nonces: for (nonce_id, slot) in slots.iter_mut().enumerate().take(active_nonces) {
                    while slot.is_idle() || slot.needs_next() {
                        let free = slot.free_slot();
                        if free < 0 {
                            break;
                        }
                        match try_pull(&mut jobs, &key) {
                            Pull::Job(j) => {
                                if sess.upload_slot(nonce_id, free as usize, &j.graph).is_err() {
                                    send_reject(&out, j, RejectReason::Overloaded);
                                    continue;
                                }
                                if slot.is_idle() {
                                    slot.assign_active(free, j);
                                } else {
                                    slot.assign_next(free, j);
                                }
                            }
                            Pull::Mismatch(j) => {
                                mismatch = Some(j);
                                exhausted = true;
                                break 'nonces;
                            }
                            Pull::Empty => break,
                            Pull::Closed => {
                                exhausted = true;
                                break 'nonces;
                            }
                        }
                    }
                }
            }

            if exhausted
                && slots[..active_nonces]
                    .iter()
                    .all(|s| s.is_idle() && s.next_job.is_none())
            {
                break;
            }

            let ctrl = match sess.poll_ctrl() {
                Ok(c) => c,
                Err(_) => break,
            };
            let mut found = false;
            for (nonce_id, slot) in slots.iter_mut().enumerate().take(active_nonces) {
                if slot.is_idle() {
                    continue;
                }
                let active_slot = slot.active_slot as usize;
                let idx = nonce_id * CTRL_STRIDE + active_slot;
                if ctrl[idx] != SLOT_COMPLETE {
                    continue;
                }
                // `!slot.is_idle()` above guarantees `active_job.is_some()`;
                // fall through gracefully instead of asserting it, so a
                // future refactor that breaks the invariant degrades to a
                // skipped completion rather than a panic.
                let Some(completed) = slot.active_job.take() else {
                    continue;
                };
                found = true;
                let reads = sess
                    .download_slot(nonce_id, active_slot)
                    .unwrap_or_default();
                let device_access_time_us = slot
                    .active_started
                    .map(|t| t.elapsed().as_micros() as u64)
                    .unwrap_or(0);
                slot.rotate_on_completion();

                let results: Vec<SamplerResult> = reads
                    .into_iter()
                    .take(completed.params.num_reads.max(1))
                    .map(|spins| score_spins(&spins, &completed.graph))
                    .collect();
                if out
                    .blocking_send(StreamResult {
                        job_id: completed.job_id,
                        result: Ok(results),
                        device_access_time_us,
                    })
                    .is_err()
                {
                    exhausted = true;
                }
            }
            if !found {
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        drop(sess); // signals exit + synchronizes stream_compute (Drop impl)

        pending_seed = mismatch.or_else(|| if closed { None } else { jobs.blocking_recv() });
    }
}
