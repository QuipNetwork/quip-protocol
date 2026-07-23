//! Host-driven Metal streaming: keep up to [`stream_width`] command buffers in
//! flight, harvested via `MTLCommandBuffer.addCompletedHandler`.
//!
//! Unlike the CUDA self-feeding kernel (device-persistent, 3-slot rotation),
//! Metal streams host-side. Reference: `GPU/metal_stream.py` /
//! `GPU/metal_scheduler.py`. Each job is one independent command buffer with
//! its own uploaded CSR/h/J; concurrency comes from committing several command
//! buffers before waiting. Each completion fires a handler that signals the
//! owner thread, which scores the freed slot's result and refills it.
//!
//! # Threading
//!
//! `run_stream` runs on the single blocking thread the harness gives
//! `Sampler::sample_stream`; every Metal object (device, command buffers,
//! buffers) is created, encoded, committed, and read on that one thread. The
//! completion handler runs on a Metal-internal thread, so it must be `Send`:
//! it captures only a `crossbeam` sender, a slot index, and a start `Instant`,
//! and moves a plain [`Done`] back to the owner thread — never a Metal object.

use crate::metal_device::MetalDevice;
use crate::sampler::{self, EncodedJob, MAX_READS};
use quip_miner_core::{Algorithm, SamplerResult, StreamJob, StreamResult};
use quip_proto::v1::RejectReason;
use std::collections::VecDeque;
use std::time::Instant;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{Receiver, Sender};

/// Fallback GPU core count when IOKit can't report `gpu-core-count`. Only sizes
/// the command-buffer budget, so a miss costs concurrency tuning, not
/// correctness.
const DEFAULT_GPU_CORES: usize = 10;

/// Backend-facing read cap for [`crate::MetalSampler::max_reads`] (mirrors CUDA).
pub fn max_reads(_algorithm: Algorithm) -> u32 {
    MAX_READS as u32
}

/// How many command buffers to keep in flight.
///
/// Budget = `maxTotalThreadsPerThreadgroup * gpu_cores` (the kernel's peak
/// concurrent-thread capacity, per `GPU/metal_stream.py::active_threads_for_util`);
/// each job's command buffer uses up to [`MAX_READS`] threads (one per read),
/// so width = budget / `MAX_READS`. Operator-tunable ceiling for throughput.
pub fn stream_width(device: &MetalDevice, algorithm: Algorithm) -> usize {
    let pipeline = match algorithm {
        Algorithm::Sa => &device.sa,
        Algorithm::Gibbs => &device.gibbs,
    };
    let per_tg = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    let cores = crate::iokit_gov::gpu_core_count().unwrap_or(DEFAULT_GPU_CORES);
    let budget = per_tg.saturating_mul(cores);
    (budget / MAX_READS).max(1)
}

/// Plain data a completion handler sends back to the owner thread.
struct Done {
    slot: usize,
    device_access_time_us: u64,
}

/// An in-flight job occupying one slot: the pending job (for scoring +
/// reply) and its encoded command buffer / buffers (kept alive until harvest).
struct InFlight {
    job: StreamJob,
    encoded: EncodedJob,
}

/// Non-blocking drain of the job channel into `queued`. Returns `false` once
/// the channel is closed (no more jobs will ever arrive).
fn drain_queued(jobs: &mut Receiver<StreamJob>, queued: &mut VecDeque<StreamJob>) -> bool {
    loop {
        match jobs.try_recv() {
            Ok(j) => queued.push_back(j),
            Err(TryRecvError::Empty) => return true,
            Err(TryRecvError::Disconnected) => return false,
        }
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

/// Drive the streaming loop for the lifetime of `jobs`: keep up to
/// [`stream_width`] command buffers in flight, emitting each result to `out`
/// in completion order. Blocks until `jobs` closes and all in-flight finish.
pub fn run_stream(
    device: &MetalDevice,
    algorithm: Algorithm,
    mut jobs: Receiver<StreamJob>,
    out: Sender<StreamResult>,
) {
    let width = stream_width(device, algorithm);
    let (done_tx, done_rx) = crossbeam_channel::unbounded::<Done>();
    let mut slots: Vec<Option<InFlight>> = (0..width).map(|_| None).collect();
    let mut in_flight = 0usize;

    // Block for the first job, then drain whatever else is already queued.
    let mut queued: VecDeque<StreamJob> = VecDeque::new();
    match jobs.blocking_recv() {
        Some(j) => queued.push_back(j),
        None => return,
    }
    let mut jobs_open = drain_queued(&mut jobs, &mut queued);

    loop {
        // Fill idle slots. Stay on the same slot until it's actually filled so
        // an empty-graph reply or an encode reject doesn't waste it this pass.
        let mut slot = 0;
        while slot < width {
            if slots[slot].is_some() {
                slot += 1;
                continue;
            }
            let Some(job) = queued.pop_front() else {
                break;
            };
            if job.graph.num_nodes() == 0 {
                answer_empty(&out, job);
                continue;
            }
            match sampler::encode_job(device, &job.graph, &job.params, algorithm) {
                Ok(encoded) => {
                    let tx = done_tx.clone();
                    let started = Instant::now();
                    let handler =
                        block::ConcreteBlock::new(move |_cb: &metal::CommandBufferRef| {
                            let _ = tx.send(Done {
                                slot,
                                device_access_time_us: started.elapsed().as_micros() as u64,
                            });
                        })
                        .copy();
                    encoded.cmd.add_completed_handler(&handler);
                    encoded.cmd.commit();
                    slots[slot] = Some(InFlight { job, encoded });
                    in_flight += 1;
                    slot += 1;
                }
                Err(e) => {
                    eprintln!("metal streaming encode failed: {e}");
                    let _ = out.blocking_send(StreamResult {
                        job_id: job.job_id,
                        result: Err(RejectReason::Overloaded),
                        device_access_time_us: 0,
                    });
                }
            }
        }

        // Top up the queue without blocking so the next pass can keep filling.
        if jobs_open && queued.is_empty() {
            jobs_open = drain_queued(&mut jobs, &mut queued);
        }

        if in_flight == 0 {
            if !queued.is_empty() {
                continue; // fill the jobs we just drained
            }
            if !jobs_open {
                break; // channel closed and nothing left in flight
            }
            // Idle: block for the next job (or exit when the channel closes).
            match jobs.blocking_recv() {
                Some(j) => {
                    queued.push_back(j);
                    jobs_open = drain_queued(&mut jobs, &mut queued);
                    continue;
                }
                None => break,
            }
        }

        // Wait for a completion, then harvest + reply. Refill happens next pass.
        let done = match done_rx.recv() {
            Ok(d) => d,
            Err(_) => break, // all senders dropped (should not happen while in_flight > 0)
        };
        let Some(inflight) = slots[done.slot].take() else {
            continue; // spurious/duplicate signal for an already-harvested slot
        };
        in_flight -= 1;

        let result = match sampler::harvest(&inflight.encoded, &inflight.job.graph) {
            Ok(reads) => Ok(reads
                .into_iter()
                .take(inflight.job.params.num_reads.max(1))
                .collect::<Vec<SamplerResult>>()),
            Err(e) => {
                eprintln!("metal streaming harvest failed: {e}");
                Err(RejectReason::Overloaded)
            }
        };
        if out
            .blocking_send(StreamResult {
                job_id: inflight.job.job_id,
                result,
                device_access_time_us: done.device_access_time_us,
            })
            .is_err()
        {
            break; // consumer gone
        }
    }
}
