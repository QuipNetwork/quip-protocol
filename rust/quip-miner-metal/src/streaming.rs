//! Host-driven batched Metal streaming.
//!
//! Mirrors v0.2 `GPU/metal_sa.py::stream_read_split_batches` /
//! `_dispatch_batch`: collect up to [`stream_width`] queued jobs that share a
//! topology + sampling params, dispatch them as **one** command buffer with
//! `num_problems = batch size` (one threadgroup per problem → one GPU core per
//! problem, filling the GPU), wait, then host-score and emit one result per
//! job. Command buffers run serially; each is internally maximal.
//!
//! This is the throughput mechanism: on Metal the GPU is filled *inside* one
//! dispatch (many threadgroups), not by committing many small command buffers
//! to a single queue (which serialize). Driving one problem per dispatch leaves
//! all but one core idle.
//!
//! # Threading
//!
//! `run_stream` runs on the single blocking thread the harness gives
//! `Sampler::sample_stream`; every Metal object stays on that thread.

use crate::metal_device::MetalDevice;
use crate::sampler::{self, algo_max_nodes};
use quip_miner_core::{Algorithm, IsingGraph, SamplerResult, StreamJob, StreamResult};
use quip_proto::v1::RejectReason;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{Receiver, Sender};

/// Fallback GPU core count when IOKit can't report `gpu-core-count`. Sizes the
/// batch (problems per dispatch), so a miss costs throughput tuning, not
/// correctness.
const DEFAULT_GPU_CORES: usize = 10;

/// Backend-facing read cap for [`crate::MetalSampler::max_reads`] (mirrors CUDA).
pub fn max_reads(_algorithm: Algorithm) -> u32 {
    sampler::MAX_READS as u32
}

/// Problems per dispatch = GPU core count (one threadgroup per core fills the
/// GPU). Advertised to the harness as `stream_width` so it keeps roughly a
/// batch's worth of jobs queued.
fn batch_size(_algorithm: Algorithm) -> usize {
    crate::iokit_gov::gpu_core_count()
        .unwrap_or(DEFAULT_GPU_CORES)
        .max(1)
}

/// `Sampler::stream_width`: how many models the backend keeps in flight. Two
/// batches' worth, so the harness buffers the next batch while one dispatches.
pub fn stream_width(_device: &MetalDevice, algorithm: Algorithm) -> usize {
    (batch_size(algorithm) * 2).max(1)
}

/// Structural + sampling identity a single dispatch batches over: same topology
/// (so the shared CSR/coloring stay valid) and same reads/beta shape (one
/// `num_reads`, `num_betas`, `beta_schedule` per dispatch).
struct BatchKey {
    n: usize,
    edges: Vec<(usize, usize)>,
    num_reads: usize,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
}

impl BatchKey {
    fn from_job(job: &StreamJob) -> Self {
        Self {
            n: job.graph.h.len(),
            edges: job.graph.edges.clone(),
            num_reads: job.params.num_reads.clamp(1, sampler::MAX_READS),
            num_sweeps: job.params.num_sweeps,
            sweeps_per_beta: job.params.sweeps_per_beta.max(1),
            beta_range: job.params.beta_range,
        }
    }

    fn matches(&self, job: &StreamJob) -> bool {
        self.n == job.graph.h.len()
            && self.num_reads == job.params.num_reads.clamp(1, sampler::MAX_READS)
            && self.num_sweeps == job.params.num_sweeps
            && self.sweeps_per_beta == job.params.sweeps_per_beta.max(1)
            && self.beta_range == job.params.beta_range
            && self.edges == job.graph.edges
    }
}

/// Emit an empty-graph job's answer directly (no GPU work needed).
fn answer_empty(out: &Sender<StreamResult>, job: StreamJob) {
    let reads = job.params.num_reads.max(1);
    let _ = out.blocking_send(StreamResult {
        job_id: job.job_id,
        result: Ok((0..reads)
            .map(|_| SamplerResult {
                spins: vec![],
                energy_milli: 0,
            })
            .collect()),
        device_access_time_us: 0,
    });
}

fn send_reject(out: &Sender<StreamResult>, job: StreamJob, reason: RejectReason) {
    let _ = out.blocking_send(StreamResult {
        job_id: job.job_id,
        result: Err(reason),
        device_access_time_us: 0,
    });
}

/// Pull the next non-empty, in-range job to seed a batch. Empty-graph jobs are
/// answered inline and oversized jobs rejected without occupying a batch slot.
///
/// `blocking` waits for a job (returns `None` only on channel close); otherwise
/// returns `None` the moment no job is immediately queued — the overlap path
/// uses this so it never blocks while an in-flight batch is un-harvested.
fn next_seed(
    jobs: &mut Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    pending: &mut Option<StreamJob>,
    algorithm: Algorithm,
    blocking: bool,
) -> Option<StreamJob> {
    loop {
        let job = match pending.take() {
            Some(j) => j,
            None if blocking => jobs.blocking_recv()?,
            None => jobs.try_recv().ok()?,
        };
        if job.graph.num_nodes() == 0 {
            answer_empty(out, job);
            continue;
        }
        if job.graph.num_nodes() > algo_max_nodes(algorithm) {
            send_reject(out, job, RejectReason::TooLarge);
            continue;
        }
        return Some(job);
    }
}

/// Collect jobs matching `key` into `batch` (already seeded) up to `cap`,
/// draining what's queued and briefly waiting for a fuller batch. A
/// topology/param mismatch is stashed in `pending` to seed the next batch.
/// Returns `false` once the channel has closed.
fn fill_batch(
    jobs: &mut Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    key: &BatchKey,
    batch: &mut Vec<StreamJob>,
    pending: &mut Option<StreamJob>,
    cap: usize,
    algorithm: Algorithm,
) -> bool {
    // Steady state: the previous batch's GPU time already filled the channel,
    // so try_recv drains a near-full batch immediately. The short idle wait
    // only matters at cold start.
    let hard_cap = Instant::now() + Duration::from_secs(2);
    let idle_timeout = Duration::from_millis(50);
    let mut last_arrival = Instant::now();
    while batch.len() < cap && Instant::now() < hard_cap {
        match jobs.try_recv() {
            Ok(job) if job.graph.num_nodes() == 0 => answer_empty(out, job),
            Ok(job) if job.graph.num_nodes() > algo_max_nodes(algorithm) => {
                send_reject(out, job, RejectReason::TooLarge)
            }
            Ok(job) if key.matches(&job) => {
                batch.push(job);
                last_arrival = Instant::now();
            }
            Ok(job) => {
                *pending = Some(job); // different topology/params → next batch
                return true;
            }
            Err(TryRecvError::Empty) => {
                if last_arrival.elapsed() > idle_timeout {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(TryRecvError::Disconnected) => return false,
        }
    }
    true
}

/// One committed batch awaiting completion + harvest.
struct InFlight {
    encoded: sampler::EncodedBatch,
    jobs: Vec<StreamJob>,
}

/// Drive the batched streaming loop for the lifetime of `jobs`.
///
/// Double-buffered: each iteration forms and **commits** the next batch (host
/// work — collect, build buffers, enqueue) while the previous batch is still
/// executing on the GPU, then waits on and harvests the previous batch (its GPU
/// compute overlaps this iteration's host work + the next batch's execution).
/// The GPU stays continuously fed; host encode + parallel scoring are hidden
/// behind GPU compute.
pub fn run_stream(
    device: &MetalDevice,
    algorithm: Algorithm,
    mut jobs: Receiver<StreamJob>,
    out: Sender<StreamResult>,
) {
    let cap = batch_size(algorithm);
    let mut pending: Option<StreamJob> = None;

    // Prime the pipeline with the first batch (blocking for its seed).
    let mut inflight = form_and_commit(device, algorithm, &mut jobs, &out, &mut pending, cap, true);

    while let Some(cur) = inflight.take() {
        // Form + commit the next batch WITHOUT blocking, so it overlaps `cur`'s
        // GPU compute. Never block here: `cur` is still un-harvested, and its
        // results must flow (freeing coordinator credits) before more jobs come.
        let next = form_and_commit(device, algorithm, &mut jobs, &out, &mut pending, cap, false);
        // Wait on `cur`, host-score (rayon), emit. Its GPU compute overlapped
        // the `next` form above and now overlaps `next`'s execution.
        finish_batch(cur, &out);
        inflight = match next {
            Some(f) => Some(f),
            // Nothing was queued to overlap; now that `cur` freed credits, block
            // for the next batch (or exit when the channel closes).
            None => form_and_commit(device, algorithm, &mut jobs, &out, &mut pending, cap, true),
        };
    }
}

/// Collect the next batch and commit it to the GPU without waiting. With
/// `blocking`, waits for the seed (returns `None` only on channel close); the
/// non-blocking overlap path returns `None` if no job is immediately queued or
/// on an encode failure (the caller finishes the in-flight batch, then retries).
fn form_and_commit(
    device: &MetalDevice,
    algorithm: Algorithm,
    jobs: &mut Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    pending: &mut Option<StreamJob>,
    cap: usize,
    blocking: bool,
) -> Option<InFlight> {
    let seed = next_seed(jobs, out, pending, algorithm, blocking)?;
    let key = BatchKey::from_job(&seed);
    let mut batch = vec![seed];
    fill_batch(jobs, out, &key, &mut batch, pending, cap, algorithm);

    // Scope `graphs` so its borrow of `batch` ends before `batch` moves.
    let encoded = {
        let graphs: Vec<&IsingGraph> = batch.iter().map(|j| &j.graph).collect();
        sampler::encode_batch(device, &graphs, &batch[0].params, algorithm)
    };
    match encoded {
        Ok(enc) => {
            enc.cmd.commit();
            Some(InFlight {
                encoded: enc,
                jobs: batch,
            })
        }
        Err(e) => {
            eprintln!("metal batch encode failed: {e}");
            for job in batch {
                send_reject(out, job, RejectReason::Overloaded);
            }
            None
        }
    }
}

/// Wait on a committed batch, then host-score (rayon) and emit one result per
/// job. `device_access_time_us` is the true GPU execution time
/// (`GPUEndTime - GPUStartTime`), not the wall clock — the wall includes this
/// batch's overlap with host work on either side.
fn finish_batch(inflight: InFlight, out: &Sender<StreamResult>) {
    use metal::MTLCommandBufferStatus;
    let InFlight { encoded, jobs } = inflight;

    encoded.cmd.wait_until_completed();
    let device_access_time_us = gpu_time_us(&encoded.cmd);

    if encoded.cmd.status() != MTLCommandBufferStatus::Completed {
        eprintln!(
            "metal batch command buffer did not complete: status {:?}",
            encoded.cmd.status()
        );
        for job in jobs {
            send_reject(out, job, RejectReason::Overloaded);
        }
        return;
    }

    let per_problem = {
        let graphs: Vec<&IsingGraph> = jobs.iter().map(|j| &j.graph).collect();
        sampler::harvest_batch(&encoded, &graphs)
    };
    let per_problem = match per_problem {
        Ok(p) => p,
        Err(e) => {
            eprintln!("metal batch harvest failed: {e}");
            for job in jobs {
                send_reject(out, job, RejectReason::Overloaded);
            }
            return;
        }
    };

    for (job, results) in jobs.into_iter().zip(per_problem) {
        let take = job.params.num_reads.max(1);
        let result = Ok(results
            .into_iter()
            .take(take)
            .collect::<Vec<SamplerResult>>());
        if out
            .blocking_send(StreamResult {
                job_id: job.job_id,
                result,
                device_access_time_us,
            })
            .is_err()
        {
            break; // consumer gone
        }
    }
}

/// True GPU execution time of a completed command buffer, in microseconds,
/// from `GPUEndTime - GPUStartTime` (`CFTimeInterval` seconds). metal-rs 0.33
/// exposes no accessor, so read the properties via `objc`. Returns 0 if the
/// timestamps are unavailable / non-positive.
// `unexpected_cfgs`: objc 0.2's `msg_send!` expands to a `cfg(cargo-clippy)`
// check the compiler no longer recognizes — a macro-internal quirk, not our cfg.
#[allow(unexpected_cfgs)]
fn gpu_time_us(cmd: &metal::CommandBufferRef) -> u64 {
    use objc::{msg_send, sel, sel_impl};
    // SAFETY: `GPUStartTime`/`GPUEndTime` are `CFTimeInterval` (f64) properties
    // on a completed `MTLCommandBuffer`; `cmd` implements `objc::Message`.
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
    let dur = end - start;
    if dur.is_finite() && dur > 0.0 {
        (dur * 1_000_000.0) as u64
    } else {
        0
    }
}
