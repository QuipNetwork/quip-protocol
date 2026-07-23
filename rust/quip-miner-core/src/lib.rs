//! Shared miner harness for the v0.3 Ising miners.
//!
//! Holds the generic gRPC session loop, job validation, the base Ising types,
//! the CSR representation used by GPU backends, and the beta schedule. Each
//! miner provides a [`Sampler`] and calls [`run`]; the harness owns everything
//! else (Hello/Welcome, Configure, credits, Reject reasons, Status, Shutdown,
//! idle timeout, exit codes).

pub mod adapt;
pub mod beta;
pub mod cli;
pub mod config;
pub mod csr;
pub mod ising;
mod job;
mod session;

pub use cli::CommonArgs;
pub use csr::CsrGraph;
pub use ising::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use session::{run, BackendIdentity, OpenError};

use quip_proto::v1::RejectReason;

/// One job entering the streaming sampler.
pub struct StreamJob {
    pub job_id: Vec<u8>,
    pub graph: IsingGraph,
    pub params: SampleParams,
}

/// One completed job leaving the streaming sampler, in completion order.
pub struct StreamResult {
    pub job_id: Vec<u8>,
    pub result: Result<Vec<SamplerResult>, RejectReason>,
    /// Per-model device/sample time in microseconds, reported in `SamplerMeta`.
    pub device_access_time_us: u64,
}

/// A backend that samples Ising problems for the miner harness.
///
/// Implementations own their device and algorithm. Only [`sample`](Sampler::sample)
/// is required; the other methods default to a no-governor, uncapped backend
/// (the CPU miner's shape).
pub trait Sampler: Send + Sync + 'static {
    /// Sample one job. Device errors map to a reject reason
    /// (`Overloaded`, `TooLarge`, …).
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason>;

    /// Stream-process jobs: pull from `jobs`, keep up to [`stream_width`] models
    /// in flight, emit each result to `out` in completion order. Blocks until
    /// `jobs` closes and all in-flight finish. Runs on a blocking thread (uses
    /// `blocking_recv`/`blocking_send`), so it must not be called from an async
    /// task directly. Default: serial loop over [`sample`](Sampler::sample).
    ///
    /// [`stream_width`]: Sampler::stream_width
    fn sample_stream(
        &self,
        mut jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
    ) {
        while let Some(j) = jobs.blocking_recv() {
            if self.should_throttle() {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            let t0 = std::time::Instant::now();
            let result = self.sample(&j.graph, &j.params);
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
    }

    /// Number of models the backend keeps in flight. Default 1 (serial).
    fn stream_width(&self) -> usize {
        1
    }

    /// Current utilization for Status messages. `0.0` when no governor.
    fn utilization(&self) -> f64 {
        0.0
    }

    /// Whether to briefly back off before the next job (governor backpressure).
    fn should_throttle(&self) -> bool {
        false
    }

    /// Largest `num_reads` this backend accepts. Defaults to no cap; a backend
    /// with a device-memory bound overrides it.
    fn max_reads(&self) -> u32 {
        u32::MAX
    }

    /// Apply this backend's configuration from `Configure.backend_toml` — the
    /// verbatim `config.toml` subsection the coordinator forwards. Called once
    /// when `Configure` arrives, before any job. Each backend parses against its
    /// own schema, applies recognized fields (config overrides CLI, see
    /// [`config::config_override`]), and warns on unknown fields
    /// ([`config::warn_unknown_fields`]). Default: no configurable settings (the
    /// CPU miner's shape).
    fn apply_config(&self, _backend_toml: &str) {}
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    struct OneResultSampler;
    impl Sampler for OneResultSampler {
        fn sample(
            &self,
            graph: &IsingGraph,
            _params: &SampleParams,
        ) -> Result<Vec<SamplerResult>, RejectReason> {
            Ok(vec![SamplerResult {
                spins: vec![1i8; graph.h.len()],
                energy_milli: 0,
            }])
        }
    }

    fn tiny_graph() -> IsingGraph {
        IsingGraph::new(vec![1.0, -1.0], vec![1.0], vec![(0, 1)])
    }

    #[test]
    fn default_sample_stream_returns_every_result_once() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("rt");
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(8);
        let (res_tx, mut res_rx) = tokio::sync::mpsc::channel::<StreamResult>(8);

        let worker = std::thread::spawn(move || {
            OneResultSampler.sample_stream(job_rx, res_tx);
        });

        rt.block_on(async {
            for i in 0u8..5 {
                job_tx
                    .send(StreamJob {
                        job_id: vec![i],
                        graph: tiny_graph(),
                        params: SampleParams::default(),
                    })
                    .await
                    .expect("send job");
            }
            drop(job_tx); // close the stream so the worker exits

            let mut seen: Vec<u8> = Vec::new();
            while let Some(r) = res_rx.recv().await {
                assert!(r.result.is_ok());
                seen.push(r.job_id[0]);
            }
            seen.sort_unstable();
            assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        });
        worker.join().expect("worker join");
    }
}
