//! Mock coordinator: a tonic `MinerService` server that walks a miner binary
//! through a scripted protocol-conformance session over a Unix domain socket.

use quip_proto::v1::miner_service_server::{MinerService, MinerServiceServer};
use quip_proto::v1::{
    coord_msg, ising_problem, miner_msg, Configure, CoordMsg, EdgeList, IsingProblem, Job, JobKind,
    MinerMsg, Shutdown, Topology, Welcome,
};
use quip_protocol::wire::encode_i32_le;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

/// Outcome of driving one miner through the conformance session.
#[derive(Debug, Clone)]
pub struct DriverReport {
    pub handshake_ok: bool,
    pub results_received: usize,
    pub rejects: Vec<i32>,
    pub exit_code: i32,
}

/// Everything the scripted session observes, excluding the child's exit code.
#[derive(Debug, Default)]
struct SessionOutcome {
    handshake_ok: bool,
    results_received: usize,
    rejects: Vec<i32>,
}

struct MockCoordinator {
    outcome_tx: Mutex<Option<oneshot::Sender<SessionOutcome>>>,
}

#[tonic::async_trait]
impl MinerService for MockCoordinator {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<CoordMsg, Status>>(64);
        let outcome_tx = self.outcome_tx.lock().await.take();

        tokio::spawn(async move {
            let outcome = run_script(&mut inbound, &tx).await;
            if let Some(otx) = outcome_tx {
                let _ = otx.send(outcome);
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Build a `CoordMsg` from a oneof payload. Callers wrap it in `Ok` for the
/// channel, whose item type is `Result<_, Status>` per tonic's `SessionStream`;
/// this mock never emits a stream-level error.
fn coord(msg: coord_msg::Msg) -> CoordMsg {
    CoordMsg { msg: Some(msg) }
}

/// A minimal, well-formed two-spin Ising problem with an edge (0,1).
fn valid_ising() -> IsingProblem {
    IsingProblem {
        graph: Some(ising_problem::Graph::Edges(EdgeList {
            u: vec![0],
            v: vec![1],
        })),
        h_milli_le32: encode_i32_le(&[1000, -1000]),
        j_milli_le32: encode_i32_le(&[500]),
        num_reads: 1,
        gates: None,
    }
}

fn job(job_id: &[u8], deadline_ms: u64, ising: IsingProblem) -> Job {
    Job {
        job_id: job_id.to_vec(),
        kind: JobKind::IsingSample as i32,
        generation: 1,
        deadline_ms,
        ising: Some(ising),
        provenance: None,
    }
}

/// Drive the full scripted CoordMsg sequence and collect the miner's replies.
async fn run_script(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
) -> SessionOutcome {
    let mut outcome = SessionOutcome::default();

    // 1. First inbound message must be a valid Hello.
    match inbound.message().await {
        Ok(Some(MinerMsg {
            msg: Some(miner_msg::Msg::Hello(h)),
        })) => {
            outcome.handshake_ok = h.session_token == "test-token" && h.protocol_version == 1;
        }
        _ => return outcome,
    }

    // 2. Handshake response + topology.
    if tx
        .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 1,
        }))))
        .await
        .is_err()
    {
        return outcome;
    }
    let configure = Configure {
        queue_depth: 3,
        idle_timeout_s: 300,
        heartbeat_s: 15,
        reconnect_window_s: 60,
        backend_toml: String::new(),
    };
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Configure(configure))))
        .await;
    let topology = Topology {
        hash: vec![],
        nodes: vec![0, 1],
        edges: Some(EdgeList {
            u: vec![0],
            v: vec![1],
        }),
    };
    let _ = tx.send(Ok(coord(coord_msg::Msg::Topology(topology)))).await;

    // 3. Two valid jobs with far-future deadlines -> expect two Results.
    let future = now_unix_ms() + 3_600_000;
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-1",
            future,
            valid_ising(),
        )))))
        .await;
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-2",
            future,
            valid_ising(),
        )))))
        .await;

    // 4. Cancel (no buffered work here; miner reports via Status).
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Cancel(quip_proto::v1::Cancel {
            max_generation: 1,
        }))))
        .await;

    // 5. Malformed job: h byte length not a multiple of 4 -> expect Reject{MALFORMED}.
    let mut malformed = valid_ising();
    malformed.h_milli_le32 = vec![0x01, 0x02, 0x03]; // len 3, not a multiple of 4
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-bad", future, malformed,
        )))))
        .await;

    // 6. Expired job: valid problem, deadline in the past -> expect Reject{EXPIRED}.
    let past = now_unix_ms().saturating_sub(60_000);
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-old",
            past,
            valid_ising(),
        )))))
        .await;

    // 7. Shutdown -> miner flushes and exits 0, closing its send stream.
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Shutdown(Shutdown {
            grace_ms: 1000,
        }))))
        .await;

    // Drain the miner's replies until it closes the stream.
    while let Ok(Some(MinerMsg { msg: Some(m) })) = inbound.message().await {
        match m {
            miner_msg::Msg::Result(_) => outcome.results_received += 1,
            miner_msg::Msg::Reject(r) => outcome.rejects.push(r.reason),
            _ => {}
        }
    }

    outcome
}

/// Bind a UDS mock coordinator, spawn `bin_path` as a miner client against it,
/// run the scripted conformance session, and report what was observed.
///
/// `socket` is a `unix://<path>` URI; the same value is passed to the miner via
/// `--quip-coordinator`.
pub async fn drive_miner(bin_path: &str, socket: &str) -> DriverReport {
    let path = socket.strip_prefix("unix://").unwrap_or(socket).to_string();
    let _ = std::fs::remove_file(&path);
    let uds = UnixListener::bind(&path).expect("bind unix socket");
    let incoming = UnixListenerStream::new(uds);

    let (otx, orx) = oneshot::channel::<SessionOutcome>();
    let svc = MockCoordinator {
        outcome_tx: Mutex::new(Some(otx)),
    };
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
    });

    let mut child = Command::new(bin_path)
        .arg("--quip-coordinator")
        .arg(socket)
        .arg("--miner-id")
        .arg("mock-0")
        .env("QUIP_SESSION_TOKEN", "test-token")
        .spawn()
        .expect("spawn miner");

    let status = child.wait().await.expect("wait for miner");
    // The session handler sends the outcome as the miner closes its stream on
    // exit; the timeout guards a miner that dies before ever connecting.
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), orx)
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or_default();
    server.abort();
    let _ = std::fs::remove_file(&path);

    DriverReport {
        handshake_ok: outcome.handshake_ok,
        results_received: outcome.results_received,
        rejects: outcome.rejects,
        exit_code: status.code().unwrap_or(-1),
    }
}
