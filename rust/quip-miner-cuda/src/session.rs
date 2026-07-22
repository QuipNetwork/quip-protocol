//! Shared gRPC session loop for `quip-cuda-sa` / `quip-cuda-gibbs`.
//!
//! Modeled on `quip-mock-miner` / `quip-miner-cpu`: UDS bidi stream, Hello →
//! Welcome → Configure → Ready, job handling with Reject reasons, Status on
//! Ping/Cancel, clean drain on Shutdown / idle timeout.
//!
//! One process per GPU: `--device N` binds CUDA device N and defaults
//! `--miner-id` to `cuda-N` (matching `[cuda.N]` config sections).

use crate::csr::IsingGraph;
use crate::cuda_device::CudaDevice;
use crate::nvml_gov::UtilGovernor;
use crate::sampler::{sample_ising, Algorithm, SampleParams};
use clap::Parser;
use quip_proto::v1::miner_service_client::MinerServiceClient;
use quip_proto::v1::{
    coord_msg, ising_problem, miner_msg, CoordMsg, IsingProblem, Job, JobKind, JobRequest,
    MinerMsg, Ready, Reject, RejectReason, Result as JobResult, SamplerMeta, Solution, Status,
};
use quip_protocol::session::{build_hello, ExitCode, SessionConfig, SessionError};
use quip_protocol::wire::{decode_i32_le, encode_spins, WireError};
use std::process::ExitCode as StdExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Endpoint, Uri};

/// CLI surface shared by both CUDA miner binaries.
#[derive(Parser, Debug)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
pub struct Cli {
    #[arg(long)]
    pub quip_coordinator: Option<String>,
    #[arg(long)]
    pub miner_id: Option<String>,
    /// CUDA device index (one process per GPU). Default 0 → miner id `cuda-0`.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Target GPU utilization ceiling percent (1–100). Used by NVML governor.
    #[arg(long, default_value_t = 100)]
    pub utilization: u32,
    /// Yield to other GPU users when NVML util exceeds 90%.
    #[arg(long, default_value_t = false)]
    pub yielding: bool,
    #[arg(long)]
    pub capabilities: bool,
    #[arg(long)]
    pub check: bool,
    #[arg(long, default_value = "info")]
    pub log_level: String,
}

/// Algorithm identity advertised in Hello / capabilities.
#[derive(Clone, Copy, Debug)]
pub struct AlgorithmIdentity {
    pub algorithm: &'static str,
    pub sampler: Algorithm,
}

const DEFAULT_NUM_SWEEPS: usize = 64;
const DEFAULT_MAX_NODES: u32 = 100_000;
const DEFAULT_MAX_EDGES: u32 = 1_000_000;

fn print_capabilities(id: AlgorithmIdentity) {
    println!(
        r#"{{"backend":"cuda","algorithm":"{}","supported_kinds":["ISING_SAMPLE"],"max_nodes":{},"max_edges":{}}}"#,
        id.algorithm, DEFAULT_MAX_NODES, DEFAULT_MAX_EDGES
    );
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn miner(msg: miner_msg::Msg) -> MinerMsg {
    MinerMsg { msg: Some(msg) }
}

fn status_msg(miner_id: &str, jobs_done: u64, utilization: f64) -> MinerMsg {
    miner(miner_msg::Msg::Status(Status {
        miner_id: miner_id.into(),
        utilization,
        jobs_done,
        abandoned_generation: 0,
        sampler_stats: Default::default(),
    }))
}

fn decode_milli_f64(bytes: &[u8]) -> Result<Vec<f64>, WireError> {
    Ok(decode_i32_le(bytes)?
        .iter()
        .map(|&v| v as f64 / 1000.0)
        .collect())
}

fn edges_of(ising: &IsingProblem) -> Result<Vec<(usize, usize)>, RejectReason> {
    match &ising.graph {
        Some(ising_problem::Graph::Edges(e)) => {
            if e.u.len() != e.v.len() {
                return Err(RejectReason::Malformed);
            }
            Ok(e.u
                .iter()
                .zip(&e.v)
                .map(|(&u, &v)| (u as usize, v as usize))
                .collect())
        }
        Some(ising_problem::Graph::TopologyHash(_)) => Err(RejectReason::Malformed),
        None => Ok(Vec::new()),
    }
}

fn parse_ising(ising: &IsingProblem) -> Result<IsingGraph, RejectReason> {
    let h = decode_milli_f64(&ising.h_milli_le32).map_err(|_| RejectReason::Malformed)?;
    let j = decode_milli_f64(&ising.j_milli_le32).map_err(|_| RejectReason::Malformed)?;
    let edges = edges_of(ising)?;
    if !edges.is_empty() && j.len() != edges.len() {
        return Err(RejectReason::Malformed);
    }
    let n = h.len();
    if n > DEFAULT_MAX_NODES as usize || edges.len() > DEFAULT_MAX_EDGES as usize {
        return Err(RejectReason::TooLarge);
    }
    for &(u, v) in &edges {
        if u >= n || v >= n {
            return Err(RejectReason::Malformed);
        }
    }
    Ok(IsingGraph::new(h, j, edges))
}

fn reject(job_id: Vec<u8>, reason: RejectReason) -> MinerMsg {
    miner(miner_msg::Msg::Reject(Reject {
        job_id,
        reason: reason as i32,
    }))
}

fn handle_job(
    job: Job,
    algorithm: Algorithm,
    num_sweeps: usize,
    device: &CudaDevice,
    jobs_done: &mut u64,
    gov: &UtilGovernor,
) -> Vec<MinerMsg> {
    let job_id = job.job_id.clone();

    if job.kind != JobKind::IsingSample as i32 {
        return vec![reject(job_id, RejectReason::UnsupportedKind)];
    }
    if job.deadline_ms < now_unix_ms() {
        return vec![reject(job_id, RejectReason::Expired)];
    }
    let ising = match job.ising {
        Some(i) => i,
        None => return vec![reject(job_id, RejectReason::Malformed)],
    };
    let graph = match parse_ising(&ising) {
        Ok(g) => g,
        Err(reason) => return vec![reject(job_id, reason)],
    };

    if gov.should_throttle() {
        std::thread::sleep(Duration::from_millis(50));
    }

    let num_reads = if ising.num_reads == 0 {
        1
    } else {
        ising.num_reads as usize
    };
    let params = SampleParams {
        num_reads,
        num_sweeps,
        seed: now_unix_ms(),
        ..Default::default()
    };

    let t0 = Instant::now();
    let samples = match sample_ising(device, &graph, &params, algorithm) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cuda sample failed: {e}");
            return vec![reject(job_id, RejectReason::Overloaded)];
        }
    };
    let elapsed_us = t0.elapsed().as_micros() as u64;

    let solutions: Vec<Solution> = samples
        .into_iter()
        .map(|r| Solution {
            spins_bytes: encode_spins(&r.spins),
            energy_milli: r.energy_milli,
        })
        .collect();

    *jobs_done = jobs_done.saturating_add(1);

    let result = JobResult {
        job_id,
        solutions,
        meta: Some(SamplerMeta {
            reads: num_reads as u32,
            sweeps: num_sweeps as u32,
            device_access_time_us: elapsed_us,
            qpu_access_us: 0,
            extra: Default::default(),
        }),
    };

    vec![
        miner(miner_msg::Msg::Result(result)),
        miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 })),
    ]
}

fn num_sweeps_from_toml(backend_toml: &str) -> usize {
    for line in backend_toml.lines() {
        let line = line.split('#').next().unwrap_or(line).trim();
        if let Some(rest) = line.strip_prefix("num_sweeps") {
            let rest = rest.trim().trim_start_matches('=').trim();
            if let Ok(n) = rest.parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
        // Also accept utilization / yielding from backend_toml for parity
        // with Python config; CLI flags remain authoritative if set.
    }
    DEFAULT_NUM_SWEEPS
}

async fn run_session(
    uri: &str,
    miner_id: &str,
    identity: AlgorithmIdentity,
    device: Arc<CudaDevice>,
    mut gov: UtilGovernor,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = uri.strip_prefix("unix://").unwrap_or(uri).to_string();
    let channel = Endpoint::try_from("http://[::]:50051")?
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
    let hello = build_hello(
        miner_id,
        "cuda",
        identity.algorithm,
        &[JobKind::IsingSample],
    )?;
    tx.send(miner(miner_msg::Msg::Hello(hello))).await?;

    let mut inbound = client.session(ReceiverStream::new(rx)).await?.into_inner();

    let mut config: Option<SessionConfig> = None;
    let mut grace_ms: u64 = 5000;
    let mut num_sweeps = DEFAULT_NUM_SWEEPS;
    let mut jobs_done: u64 = 0;

    loop {
        let idle = config.as_ref().map(|c| c.idle_timeout_s).unwrap_or(300) as u64;
        let next = tokio::time::timeout(Duration::from_secs(idle), inbound.message()).await;
        let cm: CoordMsg = match next {
            Err(_) => break,
            Ok(Ok(Some(cm))) => cm,
            Ok(Ok(None)) => break,
            Ok(Err(status)) => return Err(status.into()),
        };
        match cm.msg {
            Some(coord_msg::Msg::Welcome(w)) => {
                if w.protocol_version != 1 {
                    return Err(SessionError::BadWelcome(w.protocol_version).into());
                }
            }
            Some(coord_msg::Msg::Configure(c)) => {
                num_sweeps = num_sweeps_from_toml(&c.backend_toml);
                config = Some(SessionConfig::from_configure(miner_id.into(), &c));
                tx.send(miner(miner_msg::Msg::Ready(Ready {}))).await?;
                let depth = config.as_ref().map(|c| c.queue_depth).unwrap_or(3);
                tx.send(miner(miner_msg::Msg::JobRequest(JobRequest {
                    credits: depth,
                })))
                .await?;
            }
            Some(coord_msg::Msg::Topology(_)) => {}
            Some(coord_msg::Msg::Job(job)) => {
                for reply in handle_job(
                    job,
                    identity.sampler,
                    num_sweeps,
                    &device,
                    &mut jobs_done,
                    &gov,
                ) {
                    tx.send(reply).await?;
                }
            }
            Some(coord_msg::Msg::Cancel(_)) => {
                tx.send(status_msg(miner_id, jobs_done, gov.utilization() as f64))
                    .await?;
            }
            Some(coord_msg::Msg::Ping(_)) => {
                tx.send(status_msg(miner_id, jobs_done, gov.utilization() as f64))
                    .await?;
            }
            Some(coord_msg::Msg::Shutdown(s)) => {
                grace_ms = if s.grace_ms == 0 {
                    5000
                } else {
                    s.grace_ms as u64
                };
                break;
            }
            None => {}
        }
    }

    drop(tx);
    let drain = async {
        while inbound.message().await?.is_some() {}
        Ok::<(), tonic::Status>(())
    };
    let _ = tokio::time::timeout(Duration::from_millis(grace_ms), drain).await;
    gov.stop();
    Ok(())
}

fn map_err_to_exit(err: Box<dyn std::error::Error>) -> StdExitCode {
    if let Some(se) = err.downcast_ref::<SessionError>() {
        return match se {
            SessionError::MissingToken => StdExitCode::from(ExitCode::TokenRejected as u8),
            SessionError::BadWelcome(_) => StdExitCode::from(ExitCode::InternalFatal as u8),
        };
    }
    let msg = err.to_string();
    if msg.contains("QUIP_SESSION_TOKEN") || msg.contains("session token") {
        return StdExitCode::from(ExitCode::TokenRejected as u8);
    }
    if msg.contains("unexpected protocol version") {
        return StdExitCode::from(ExitCode::InternalFatal as u8);
    }
    eprintln!("quip-miner-cuda fatal: {err}");
    StdExitCode::from(ExitCode::InternalFatal as u8)
}

/// Entry point for both binaries. Returns a process exit code.
pub fn run_cli(identity: AlgorithmIdentity) -> StdExitCode {
    let cli = Cli::parse();
    let _ = cli.log_level;

    if cli.capabilities {
        print_capabilities(identity);
        return StdExitCode::SUCCESS;
    }
    if cli.check {
        match CudaDevice::check(cli.device) {
            Ok(()) => return StdExitCode::SUCCESS,
            Err(e) => {
                eprintln!("cuda check failed (device {}): {e}", cli.device);
                return StdExitCode::from(ExitCode::EnvIncompatible as u8);
            }
        }
    }

    let uri = match cli.quip_coordinator {
        Some(u) => u,
        None => {
            eprintln!("error: --quip-coordinator required for session mode");
            return StdExitCode::from(ExitCode::ConfigInvalid as u8);
        }
    };
    let miner_id = cli
        .miner_id
        .unwrap_or_else(|| format!("cuda-{}", cli.device));

    let device = match CudaDevice::open(cli.device) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("failed to open CUDA device {}: {e}", cli.device);
            return StdExitCode::from(ExitCode::EnvIncompatible as u8);
        }
    };
    let gov = UtilGovernor::start(cli.device as u32, cli.utilization, cli.yielding);

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to start tokio runtime: {e}");
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    };

    match rt.block_on(run_session(&uri, &miner_id, identity, device, gov)) {
        Ok(()) => StdExitCode::SUCCESS,
        Err(e) => map_err_to_exit(e),
    }
}
