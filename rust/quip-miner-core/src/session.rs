//! Generic gRPC session loop over a UDS bidi stream.
//!
//! Modeled on `quip-mock-miner`: Hello → Welcome → Configure → Ready, job
//! handling with Reject reasons, Status on Ping/Cancel, clean drain on
//! Shutdown / idle timeout. Backends supply only a [`Sampler`].

use crate::cli::CommonArgs;
use crate::job::{
    handle_job, miner, num_sweeps_from_toml, status_msg, TopologyCache, DEFAULT_NUM_SWEEPS,
};
use crate::Sampler;
use quip_proto::v1::miner_service_client::MinerServiceClient;
use quip_proto::v1::{coord_msg, miner_msg, CoordMsg, JobKind, JobRequest, MinerMsg, Ready};
use quip_protocol::session::{build_hello, ExitCode, SessionConfig, SessionError};
use std::process::ExitCode as StdExitCode;
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

async fn run_session<S: Sampler>(
    uri: &str,
    miner_id: &str,
    id: &BackendIdentity,
    sampler: &S,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let hello = build_hello(miner_id, id.backend, id.algorithm, &[JobKind::IsingSample])?;
    tx.send(miner(miner_msg::Msg::Hello(hello))).await?;

    let mut inbound = client.session(ReceiverStream::new(rx)).await?.into_inner();

    let mut config: Option<SessionConfig> = None;
    let mut grace_ms: u64 = 5000;
    let mut num_sweeps = DEFAULT_NUM_SWEEPS;
    let mut jobs_done: u64 = 0;
    let mut topology: Option<TopologyCache> = None;

    loop {
        let idle = config.as_ref().map(|c| c.idle_timeout_s).unwrap_or(300) as u64;
        let next = tokio::time::timeout(Duration::from_secs(idle), inbound.message()).await;
        let cm: CoordMsg = match next {
            Err(_) => break, // idle timeout
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
            Some(coord_msg::Msg::Topology(t)) => {
                topology = Some(TopologyCache::from_proto(&t));
            }
            Some(coord_msg::Msg::SetTarget(_)) => {} // T5: cache target + adapt
            Some(coord_msg::Msg::Job(job)) => {
                for reply in handle_job(job, sampler, id, num_sweeps, &mut jobs_done, topology.as_ref())
                {
                    tx.send(reply).await?;
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
    Ok(())
}

fn map_err_to_exit(err: Box<dyn std::error::Error>, backend: &str) -> StdExitCode {
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

    match rt.block_on(run_session(&uri, &miner_id, &id, &sampler)) {
        Ok(()) => StdExitCode::SUCCESS,
        Err(e) => map_err_to_exit(e, id.backend),
    }
}
