use clap::Parser;
use quip_proto::v1::miner_service_client::MinerServiceClient;
use quip_proto::v1::{
    coord_msg, ising_problem, miner_msg, CoordMsg, IsingProblem, Job, JobKind, JobRequest,
    MinerMsg, Ready, Reject, RejectReason, Result as JobResult, Solution, Status,
};
use quip_protocol::scoring::energy_milli;
use quip_protocol::session::{build_hello, SessionConfig};
use quip_protocol::wire::decode_i32_le;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Endpoint, Uri};

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[arg(long)]
    quip_coordinator: Option<String>,
    #[arg(long)]
    miner_id: Option<String>,
    #[arg(long)]
    capabilities: bool,
    #[arg(long)]
    check: bool,
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn print_capabilities() {
    // schema lives in quip-protocol; mock advertises a permissive envelope
    println!(
        r#"{{"backend":"mock","algorithm":"sa","supported_kinds":["ISING_SAMPLE"],"max_nodes":100000,"max_edges":1000000}}"#
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if cli.capabilities {
        print_capabilities();
        return Ok(());
    }
    if cli.check {
        return Ok(());
    } // mock is always runnable
    let uri = cli
        .quip_coordinator
        .expect("--quip-coordinator required for session mode");
    let miner_id = cli.miner_id.unwrap_or_else(|| "mock-0".into());
    run_session(&uri, &miner_id).await
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

fn status_msg(miner_id: &str) -> MinerMsg {
    miner(miner_msg::Msg::Status(Status {
        miner_id: miner_id.into(),
        utilization: 0.0,
        jobs_done: 0,
        abandoned_generation: 0,
        sampler_stats: Default::default(),
    }))
}

/// Milli-int-encoded field -> float vector; an empty field decodes to an empty vec.
fn decode_milli_f64(bytes: &[u8]) -> Result<Vec<f64>, quip_protocol::wire::WireError> {
    Ok(decode_i32_le(bytes)?
        .iter()
        .map(|&v| v as f64 / 1000.0)
        .collect())
}

fn edges_of(ising: &IsingProblem) -> Vec<(usize, usize)> {
    match &ising.graph {
        Some(ising_problem::Graph::Edges(e)) => {
            e.u.iter()
                .zip(&e.v)
                .map(|(&u, &v)| (u as usize, v as usize))
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Validate a job and produce the reply(s): a `Reject` on bad input, otherwise a
/// `Result` (all-+1 spins) followed by a `JobRequest` for one more credit.
fn handle_job(job: Job) -> Vec<MinerMsg> {
    let job_id = job.job_id.clone();
    let ising = job.ising.unwrap_or_default();

    // MALFORMED: h field is not a valid i32-LE array (length not a multiple of 4).
    let h = match decode_milli_f64(&ising.h_milli_le32) {
        Ok(h) => h,
        Err(_) => {
            return vec![miner(miner_msg::Msg::Reject(Reject {
                job_id,
                reason: RejectReason::Malformed as i32,
            }))];
        }
    };

    // EXPIRED: the deadline has already passed.
    if job.deadline_ms < now_unix_ms() {
        return vec![miner(miner_msg::Msg::Reject(Reject {
            job_id,
            reason: RejectReason::Expired as i32,
        }))];
    }

    let j = decode_milli_f64(&ising.j_milli_le32).unwrap_or_default();
    let edges = edges_of(&ising);
    let n = h.len();
    let spins = vec![1i8; n];
    let energy = energy_milli(&spins, &h, &j, &edges);
    let result = JobResult {
        job_id,
        solutions: vec![Solution {
            spins_bytes: vec![0x01u8; n],
            energy_milli: energy,
        }],
        meta: None,
    };
    vec![
        miner(miner_msg::Msg::Result(result)),
        miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 })),
    ]
}

async fn run_session(uri: &str, miner_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = uri.strip_prefix("unix://").unwrap_or(uri).to_string();
    let channel = Endpoint::try_from("http://[::]:50051")? // dummy authority, unused for UDS
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
    // Send Hello before opening the inbound stream so the coordinator's handshake
    // has something to read immediately.
    tx.send(miner(miner_msg::Msg::Hello(build_hello(
        miner_id,
        "mock",
        "sa",
        &[JobKind::IsingSample],
    )?)))
    .await?;

    let mut inbound = client.session(ReceiverStream::new(rx)).await?.into_inner();

    let mut config: Option<SessionConfig> = None;
    loop {
        let idle = config.as_ref().map(|c| c.idle_timeout_s).unwrap_or(300) as u64;
        let next = tokio::time::timeout(Duration::from_secs(idle), inbound.message()).await;
        let cm: CoordMsg = match next {
            Err(_) => break, // idle timeout with no job -> clean exit
            Ok(Ok(Some(cm))) => cm,
            Ok(Ok(None)) => break, // coordinator closed the stream
            Ok(Err(status)) => return Err(status.into()),
        };
        match cm.msg {
            Some(coord_msg::Msg::Welcome(_)) => {}
            Some(coord_msg::Msg::Configure(c)) => {
                config = Some(SessionConfig::from_configure(miner_id.into(), &c));
                tx.send(miner(miner_msg::Msg::Ready(Ready {}))).await?;
            }
            Some(coord_msg::Msg::Topology(_)) => {}
            Some(coord_msg::Msg::Job(job)) => {
                for reply in handle_job(job) {
                    tx.send(reply).await?;
                }
            }
            Some(coord_msg::Msg::Cancel(_)) => {
                // No jobs buffered in this mock; acknowledge via Status.
                tx.send(status_msg(miner_id)).await?;
            }
            Some(coord_msg::Msg::Ping(_)) => {
                tx.send(status_msg(miner_id)).await?;
            }
            Some(coord_msg::Msg::Shutdown(_)) => break,
            None => {}
        }
    }
    // Signal end-of-outbound so tonic flushes every buffered reply, then drain
    // inbound until the coordinator closes its side. Without this the process
    // (and its runtime) can tear down before queued Results reach the socket.
    drop(tx);
    while inbound.message().await?.is_some() {}
    Ok(())
}
