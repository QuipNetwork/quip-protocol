//! CPU Ising samplers.
//!
//! Two binaries share this library:
//! - `quip-cpu-sa` — neal-style geometric SA (Metropolis)
//! - `quip-cpu-gibbs` — heat-bath single-site Gibbs over the same ladder
//!
//! The coordinator session loop lives in `quip-miner-core`; this crate provides
//! the [`CpuSampler`] backend and the two binaries.

pub mod sampler_core;

pub use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use sampler_core::sample_ising;

use quip_miner_core::adapt::AdaptBounds;
use quip_miner_core::{BackendIdentity, Sampler, StreamJob, StreamResult};
use quip_proto::v1::RejectReason;

const DEFAULT_MAX_NODES: u32 = 100_000;
const DEFAULT_MAX_EDGES: u32 = 1_000_000;

/// Backend identity for `quip-cpu-sa`.
/// CPU adapt envelope (from `CPU/sa_miner.py`).
const CPU_ADAPT: AdaptBounds = AdaptBounds {
    min_sweeps: 64,
    max_sweeps: 4096,
    min_reads: 64,
    max_reads: 512,
    reads_solution_min_factor: 4,
    reads_solution_max_factor: 8,
    reads_solution_floor_factor: 0,
};

pub const CPU_SA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "sa",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: CPU_ADAPT,
};

/// Backend identity for `quip-cpu-gibbs`.
pub const CPU_GIBBS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: "cpu",
    algorithm: "gibbs",
    max_nodes: DEFAULT_MAX_NODES,
    max_edges: DEFAULT_MAX_EDGES,
    adapt: CPU_ADAPT,
};

/// CPU sampler backend. No device, no governor, uncapped reads.
pub struct CpuSampler {
    algorithm: Algorithm,
}

impl CpuSampler {
    pub fn new(algorithm: Algorithm) -> Self {
        Self { algorithm }
    }
}

impl Sampler for CpuSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        Ok(sample_ising(graph, params, self.algorithm))
    }

    /// One model per core: `sample`'s reads are sequential and cache-local, so
    /// throughput comes from running `stream_width` models concurrently, each
    /// pinned to a worker thread. Fanning a single model's reads across cores
    /// bounced the shared arrays' cache lines and measured slower.
    fn stream_width(&self) -> usize {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }

    fn sample_stream(
        &self,
        mut jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
    ) {
        let width = self.stream_width();
        let algorithm = self.algorithm;
        // MPMC hand-off: this thread (dispatcher) pulls from the async job
        // channel; `width` worker threads each take one model at a time.
        let (work_tx, work_rx) = crossbeam_channel::bounded::<StreamJob>(width);
        let workers: Vec<_> = (0..width)
            .map(|_| {
                let work_rx = work_rx.clone();
                let out = out.clone();
                std::thread::spawn(move || {
                    for j in work_rx.iter() {
                        let t0 = std::time::Instant::now();
                        let result = Ok(sample_ising(&j.graph, &j.params, algorithm));
                        let device_access_time_us = t0.elapsed().as_micros() as u64;
                        if out
                            .blocking_send(StreamResult {
                                job_id: j.job_id,
                                result,
                                device_access_time_us,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
            })
            .collect();
        drop(work_rx);
        drop(out);

        while let Some(j) = jobs.blocking_recv() {
            if work_tx.send(j).is_err() {
                break;
            }
        }
        drop(work_tx); // close -> workers drain and exit
        for w in workers {
            let _ = w.join();
        }
    }
}
