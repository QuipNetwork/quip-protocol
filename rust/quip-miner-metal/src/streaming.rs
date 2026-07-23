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
use quip_miner_core::{
    Algorithm, IsingGraph, SampleParams, SamplerResult, StreamJob, StreamResult,
};
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
/// Returns `None` when the channel closes with nothing pending.
fn next_seed(
    jobs: &mut Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    pending: &mut Option<StreamJob>,
    algorithm: Algorithm,
) -> Option<StreamJob> {
    loop {
        let job = match pending.take() {
            Some(j) => j,
            None => jobs.blocking_recv()?,
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

/// Drive the batched streaming loop for the lifetime of `jobs`.
pub fn run_stream(
    device: &MetalDevice,
    algorithm: Algorithm,
    mut jobs: Receiver<StreamJob>,
    out: Sender<StreamResult>,
) {
    let cap = batch_size(algorithm);
    let mut pending: Option<StreamJob> = None;

    while let Some(seed) = next_seed(&mut jobs, &out, &mut pending, algorithm) {
        let key = BatchKey::from_job(&seed);
        let mut batch = vec![seed];
        let open = fill_batch(
            &mut jobs,
            &out,
            &key,
            &mut batch,
            &mut pending,
            cap,
            algorithm,
        );

        dispatch_batch(device, algorithm, batch, &out);

        if !open && pending.is_none() {
            break;
        }
    }
}

/// Encode + commit + wait for one batch, then host-score and emit per job.
fn dispatch_batch(
    device: &MetalDevice,
    algorithm: Algorithm,
    batch: Vec<StreamJob>,
    out: &Sender<StreamResult>,
) {
    use metal::MTLCommandBufferStatus;

    let graphs: Vec<&IsingGraph> = batch.iter().map(|j| &j.graph).collect();
    // Every job in the batch shares params by construction (BatchKey).
    let params: &SampleParams = &batch[0].params;

    let encoded = match sampler::encode_batch(device, &graphs, params, algorithm) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("metal batch encode failed: {e}");
            for job in batch {
                send_reject(out, job, RejectReason::Overloaded);
            }
            return;
        }
    };

    let t0 = Instant::now();
    encoded.cmd.commit();
    encoded.cmd.wait_until_completed();
    let device_access_time_us = t0.elapsed().as_micros() as u64;

    if encoded.cmd.status() != MTLCommandBufferStatus::Completed {
        eprintln!(
            "metal batch command buffer did not complete: status {:?}",
            encoded.cmd.status()
        );
        for job in batch {
            send_reject(out, job, RejectReason::Overloaded);
        }
        return;
    }

    let per_problem = match sampler::harvest_batch(&encoded, &graphs) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("metal batch harvest failed: {e}");
            for job in batch {
                send_reject(out, job, RejectReason::Overloaded);
            }
            return;
        }
    };

    for (job, results) in batch.into_iter().zip(per_problem) {
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
