//! Generic gRPC session loop over a UDS bidi stream.
//!
//! Modeled on `quip-mock-miner`: Hello → Welcome → Configure → Ready, job
//! handling with Reject reasons, Status on Ping/Cancel, clean drain on
//! Shutdown / idle timeout. Backends supply only a [`Sampler`].

use crate::cli::CommonArgs;
use crate::job::{
    finalize_result, miner, num_sweeps_from_toml, prepare_job, status_msg, Prepared, SessionTarget,
    TopologyCache, DEFAULT_NUM_SWEEPS,
};
use crate::{Sampler, StreamJob, StreamResult};
use quip_proto::v1::miner_service_client::MinerServiceClient;
use quip_proto::v1::{coord_msg, miner_msg, CoordMsg, JobKind, JobRequest, MinerMsg, Ready};
use quip_protocol::session::{build_hello, ExitCode, SessionConfig, SessionError};
use std::collections::HashMap;
use std::process::ExitCode as StdExitCode;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Endpoint, Uri};

/// Static backend metadata advertised in capabilities and Hello.
#[derive(Clone, Copy, Debug)]
pub struct BackendIdentity {
    pub backend: &'static str,
    pub algorithm: &'static str,
    pub max_nodes: u32,
    pub max_edges: u32,
    /// Sampling-parameter envelope used by `adapt::adapt_params`.
    pub adapt: crate::adapt::AdaptBounds,
}

/// Failure to open a device / build a sampler, surfaced as `EnvIncompatible`.
#[derive(Debug)]
pub struct OpenError(pub String);

fn print_capabilities(id: &BackendIdentity) {
    println!(
        r#"{{"backend":"{}","algorithm":"{}","supported_kinds":["ISING_SAMPLE"],"max_nodes":{},"max_edges":{}}}"#,
        id.backend, id.algorithm, id.max_nodes, id.max_edges
    );
}

/// Emit a miner progress line every N completed jobs (v0.2 `mine_work_item`
/// parity).
const PROGRESS_LOG_INTERVAL: u64 = 10;

#[expect(
    clippy::cast_precision_loss,
    reason = "job count and elapsed seconds are small; the f64 rate is display-only"
)]
fn log_progress(
    backend: &str,
    jobs_done: u64,
    elapsed: std::time::Duration,
    reads: u32,
    sweeps: u32,
    best_energy_milli: i64,
) {
    let secs = elapsed.as_secs_f64();
    let rate = if secs > 0.0 {
        jobs_done as f64 / secs
    } else {
        0.0
    };
    let best = if best_energy_milli == i64::MAX {
        "n/a".to_owned()
    } else {
        best_energy_milli.to_string()
    };
    eprintln!(
        "{backend} progress: {jobs_done} jobs | {rate:.1} jobs/s | reads={reads} sweeps={sweeps} | best={best} milli"
    );
}

async fn run_session<S: Sampler>(
    uri: &str,
    miner_id: &str,
    id: &BackendIdentity,
    sampler: Arc<S>,
    sweeps_per_beta: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve token before any network I/O so a missing QUIP_SESSION_TOKEN
    // always maps to exit 77 (never InternalFatal from a connect failure).
    let hello = build_hello(miner_id, id.backend, id.algorithm, &[JobKind::IsingSample])?;

    let path = uri.strip_prefix("unix://").unwrap_or(uri).to_string();
    let channel = Endpoint::try_from("http://[::]:50051")? // dummy authority for UDS
        .connect_with_connector(tower::service_fn(move |_: Uri| {
            let p = path.clone();
            async move {
                let s = tokio::net::UnixStream::connect(p).await?;
                Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(s))
            }
        }))
        .await?;
    let mut client = MinerServiceClient::new(channel);

    let (tx, rx) = mpsc::channel::<MinerMsg>(16);
    tx.send(miner(miner_msg::Msg::Hello(hello))).await?;

    let mut inbound = client.session(ReceiverStream::new(rx)).await?.into_inner();

    // Streaming sampler on a blocking thread: it pulls StreamJobs and emits
    // StreamResults in completion order, keeping `stream_width` models in flight.
    let width = sampler.stream_width().max(1);
    let cap = width.max(8);
    let (job_tx, job_rx) = mpsc::channel::<StreamJob>(cap);
    let (res_tx, mut res_rx) = mpsc::channel::<StreamResult>(cap);
    let sampler_thread = {
        let s = Arc::clone(&sampler);
        std::thread::spawn(move || s.sample_stream(job_rx, res_tx))
    };

    let mut grace_ms: u64 = 5000;
    let mut num_sweeps = DEFAULT_NUM_SWEEPS;
    let mut jobs_done: u64 = 0;
    let mut topology: Option<TopologyCache> = None;
    let mut target: Option<SessionTarget> = None;
    // job_id → (num_reads, num_sweeps) resolved at prepare, for the result meta.
    let mut pending: HashMap<Vec<u8>, (u32, u32)> = HashMap::new();
    // Progress logging (mirrors v0.2 mine_work_item's every-N-attempts line).
    let session_start = std::time::Instant::now();
    let mut best_energy_milli: i64 = i64::MAX;

    loop {
        tokio::select! {
            biased;
            // Drain completed results first so a busy sampler never backs up.
            Some(sr) = res_rx.recv() => {
                let (reads, sweeps) = pending.remove(&sr.job_id).unwrap_or((0, 0));
                if let Ok(samples) = &sr.result {
                    if let Some(e) = samples.iter().map(|r| r.energy_milli).min() {
                        best_energy_milli = best_energy_milli.min(e);
                    }
                }
                for reply in finalize_result(sr, reads, sweeps, &mut jobs_done) {
                    tx.send(reply).await?;
                }
                if jobs_done > 0 && jobs_done.is_multiple_of(PROGRESS_LOG_INTERVAL) {
                    log_progress(
                        id.backend,
                        jobs_done,
                        session_start.elapsed(),
                        reads,
                        sweeps,
                        best_energy_milli,
                    );
                }
            }
            // No wall-clock idle timeout: a miner grinding a hard nonce for
            // minutes-to-hours legitimately receives nothing from the
            // coordinator meanwhile — it is busy, not dead. Liveness of a
            // truly-gone peer surfaces here as a closed stream (`Ok(None)`) or
            // a transport error (`Err`); dead-peer detection over the network
            // belongs to HTTP/2 keepalive, not an application quiet-period.
            msg = inbound.message() => {
                let cm: CoordMsg = match msg {
                    Ok(Some(cm)) => cm,
                    Ok(None) => break,
                    Err(status) => return Err(status.into()),
                };
                match cm.msg {
                    Some(coord_msg::Msg::Welcome(w)) => {
                        if w.protocol_version != 1 {
                            return Err(SessionError::BadWelcome(w.protocol_version).into());
                        }
                    }
                    Some(coord_msg::Msg::Configure(c)) => {
                        // Hand the verbatim config subsection to the backend to
                        // parse against its own schema (overrides CLI, warns on
                        // unknown fields / overrides) before mining starts.
                        sampler.apply_config(&c.backend_toml);
                        num_sweeps = num_sweeps_from_toml(&c.backend_toml);
                        let config = SessionConfig::from_configure(miner_id.into(), &c);
                        tx.send(miner(miner_msg::Msg::Ready(Ready {}))).await?;
                        // Request enough credits to keep `width` models in flight.
                        let depth = config.queue_depth.max(width as u32);
                        tx.send(miner(miner_msg::Msg::JobRequest(JobRequest {
                            credits: depth,
                        })))
                        .await?;
                    }
                    Some(coord_msg::Msg::Topology(t)) => {
                        topology = Some(TopologyCache::from_proto(&t));
                    }
                    Some(coord_msg::Msg::SetTarget(s)) => {
                        target = Some(SessionTarget::from_proto(&s));
                    }
                    Some(coord_msg::Msg::Job(job)) => {
                        match prepare_job(
                            job,
                            &*sampler,
                            id,
                            num_sweeps,
                            sweeps_per_beta,
                            topology.as_ref(),
                            target.as_ref(),
                        ) {
                            Prepared::Reject(msg) => {
                                // Rejecting at prepare time is terminal for the
                                // job, so ask for a replacement credit — same as
                                // a completion — to keep the coordinator's
                                // consume-on-dispatch pool from leaking a slot.
                                tx.send(msg).await?;
                                tx.send(miner(miner_msg::Msg::JobRequest(JobRequest {
                                    credits: 1,
                                })))
                                .await?;
                            }
                            Prepared::Sample {
                                job,
                                num_reads,
                                num_sweeps: ns,
                            } => {
                                let _ = pending.insert(job.job_id.clone(), (num_reads, ns));
                                if job_tx.send(job).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Some(coord_msg::Msg::Cancel(_)) => {
                        tx.send(status_msg(miner_id, jobs_done, sampler.utilization()))
                            .await?;
                    }
                    Some(coord_msg::Msg::Ping(_)) => {
                        tx.send(status_msg(miner_id, jobs_done, sampler.utilization()))
                            .await?;
                    }
                    Some(coord_msg::Msg::Shutdown(s)) => {
                        grace_ms = if s.grace_ms == 0 { 5000 } else { s.grace_ms as u64 };
                        break;
                    }
                    None => {}
                }
            }
        }
    }

    // Stop feeding, then drain in-flight results within the grace window.
    drop(job_tx);
    let grace = Duration::from_millis(grace_ms);
    while let Ok(Some(sr)) = tokio::time::timeout(grace, res_rx.recv()).await {
        let (reads, sweeps) = pending.remove(&sr.job_id).unwrap_or((0, 0));
        for reply in finalize_result(sr, reads, sweeps, &mut jobs_done) {
            tx.send(reply).await?;
        }
    }
    let _ = sampler_thread.join();

    drop(tx);
    let drain = async {
        while inbound.message().await?.is_some() {}
        Ok::<(), tonic::Status>(())
    };
    let _ = tokio::time::timeout(grace, drain).await;
    Ok(())
}

fn map_err_to_exit(err: Box<dyn std::error::Error>, backend: &str) -> StdExitCode {
    // Prefer the canonical `SessionError -> ExitCode` mapping (quip-protocol)
    // over a hand-rolled match, so real miners exit the same code as the mock
    // reference (e.g. BadWelcome -> ConfigInvalid/64, not InternalFatal/70).
    let err = match err.downcast::<SessionError>() {
        Ok(se) => return StdExitCode::from(ExitCode::from(*se) as u8),
        Err(err) => err,
    };
    // Type-erased fallback: the error crossed a boundary that lost the
    // concrete `SessionError` (e.g. tonic::Status::into()). Recover only the
    // two documented exit codes we can identify from the message.
    let msg = err.to_string();
    if msg.contains("QUIP_SESSION_TOKEN") || msg.contains("session token") {
        return StdExitCode::from(ExitCode::TokenRejected as u8);
    }
    if msg.contains("unexpected protocol version") {
        return StdExitCode::from(ExitCode::ConfigInvalid as u8);
    }
    eprintln!("quip-miner-{backend} fatal: {err}");
    StdExitCode::from(ExitCode::InternalFatal as u8)
}

/// Miner entry point. Dispatches `--capabilities`/`--check`/session mode.
///
/// `open` opens the device and builds the [`Sampler`]; it runs for `--check`
/// (result discarded) and for session mode. `--capabilities` never calls it.
pub fn run<S: Sampler>(
    id: BackendIdentity,
    common: &CommonArgs,
    open: impl FnOnce() -> Result<S, OpenError>,
) -> StdExitCode {
    let _ = &common.log_level;

    if common.capabilities {
        print_capabilities(&id);
        return StdExitCode::SUCCESS;
    }
    if common.check {
        return match open() {
            Ok(_) => StdExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{} check failed: {}", id.backend, e.0);
                StdExitCode::from(ExitCode::EnvIncompatible as u8)
            }
        };
    }

    let uri = match &common.quip_coordinator {
        Some(u) => u.clone(),
        None => {
            eprintln!("error: --quip-coordinator required for session mode");
            return StdExitCode::from(ExitCode::ConfigInvalid as u8);
        }
    };
    let miner_id = common
        .miner_id
        .clone()
        .unwrap_or_else(|| format!("{}-0", id.backend));

    let sampler = match open() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to open {} device: {}", id.backend, e.0);
            return StdExitCode::from(ExitCode::EnvIncompatible as u8);
        }
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to start tokio runtime: {e}");
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    };

    match rt.block_on(run_session(
        &uri,
        &miner_id,
        &id,
        Arc::new(sampler),
        common.sweeps_per_beta,
    )) {
        Ok(()) => StdExitCode::SUCCESS,
        Err(e) => map_err_to_exit(e, id.backend),
    }
}
