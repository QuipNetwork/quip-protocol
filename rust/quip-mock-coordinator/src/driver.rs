//! Mock coordinator: a tonic `MinerService` server that walks a miner binary
//! through a scripted protocol-conformance session over a Unix domain socket.

use quip_proto::v1::miner_service_server::{MinerService, MinerServiceServer};
use quip_proto::v1::{
    coord_msg, ising_problem, miner_msg, Configure, CoordMsg, EdgeList, IsingProblem, Job, JobKind,
    MinerMsg, RejectReason, Shutdown, Topology, Welcome,
};
use quip_protocol::wire::encode_i32_le;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

/// A reject observed during the scripted session, bound to its job_id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedReject {
    pub job_id: Vec<u8>,
    pub reason: i32,
}

/// Outcome of driving one miner through the conformance session.
#[derive(Debug, Clone)]
pub struct DriverReport {
    pub handshake_ok: bool,
    /// True when a `Ready` arrived after `Configure` was sent.
    pub ready_received: bool,
    /// Credits advertised on each `JobRequest` (must be non-empty for pass).
    pub job_request_credits: Vec<u32>,
    /// `job_id` of every `Result` received (order preserved).
    pub result_job_ids: Vec<Vec<u8>>,
    /// Every `Reject`, bound to its `job_id` (not just the reason code).
    pub rejects: Vec<ObservedReject>,
    /// True when a `Status` arrived after `Cancel` (cancel acknowledgement).
    pub cancel_acked: bool,
    pub exit_code: i32,
}

impl DriverReport {
    /// Full conformance verdict — every axis the harness grades.
    pub fn is_conformant(&self) -> bool {
        self.handshake_ok
            && self.ready_received
            && !self.job_request_credits.is_empty()
            && self.job_request_credits.iter().all(|&c| c > 0)
            && self.result_job_ids.len() == 2
            && self.result_job_ids.iter().any(|id| id == b"job-1")
            && self.result_job_ids.iter().any(|id| id == b"job-2")
            && self.has_reject(b"job-bad-h", RejectReason::Malformed)
            && self.has_reject(b"job-bad-j", RejectReason::Malformed)
            && self.has_reject(b"job-gate", RejectReason::UnsupportedKind)
            && self.has_reject(b"job-old", RejectReason::Expired)
            && self.cancel_acked
            && self.exit_code == 0
    }

    pub fn has_reject(&self, job_id: &[u8], reason: RejectReason) -> bool {
        self.rejects
            .iter()
            .any(|r| r.job_id == job_id && r.reason == reason as i32)
    }
}

/// Everything the scripted session observes, excluding the child's exit code.
#[derive(Debug, Default)]
struct SessionOutcome {
    handshake_ok: bool,
    ready_received: bool,
    job_request_credits: Vec<u32>,
    result_job_ids: Vec<Vec<u8>>,
    rejects: Vec<ObservedReject>,
    cancel_acked: bool,
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

fn job_kind(job_id: &[u8], deadline_ms: u64, kind: JobKind, ising: IsingProblem) -> Job {
    Job {
        job_id: job_id.to_vec(),
        kind: kind as i32,
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
    let mut malformed_h = valid_ising();
    malformed_h.h_milli_le32 = vec![0x01, 0x02, 0x03]; // len 3, not a multiple of 4
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-bad-h",
            future,
            malformed_h,
        )))))
        .await;

    // 6. Malformed job: j byte length not a multiple of 4 -> expect Reject{MALFORMED}.
    let mut malformed_j = valid_ising();
    malformed_j.j_milli_le32 = vec![0x01, 0x02, 0x03];
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-bad-j",
            future,
            malformed_j,
        )))))
        .await;

    // 7. Unsupported kind: GATE_CIRCUIT -> expect Reject{UNSUPPORTED_KIND}.
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job_kind(
            b"job-gate",
            future,
            JobKind::GateCircuit,
            valid_ising(),
        )))))
        .await;

    // 8. Expired job: valid problem, deadline in the past -> expect Reject{EXPIRED}.
    let past = now_unix_ms().saturating_sub(60_000);
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-old",
            past,
            valid_ising(),
        )))))
        .await;

    // 9. Shutdown -> miner flushes and exits 0, closing its send stream.
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Shutdown(Shutdown {
            grace_ms: 1000,
        }))))
        .await;

    // Drain the miner's replies until it closes the stream.
    while let Ok(Some(MinerMsg { msg: Some(m) })) = inbound.message().await {
        match m {
            miner_msg::Msg::Ready(_) => outcome.ready_received = true,
            miner_msg::Msg::JobRequest(jr) => outcome.job_request_credits.push(jr.credits),
            miner_msg::Msg::Result(r) => outcome.result_job_ids.push(r.job_id),
            miner_msg::Msg::Reject(r) => outcome.rejects.push(ObservedReject {
                job_id: r.job_id,
                reason: r.reason,
            }),
            // This script only emits Status as the Cancel acknowledgement
            // (no Ping), so any Status counts as the cancel ack.
            miner_msg::Msg::Status(_) => outcome.cancel_acked = true,
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

    // Bound the wait so a hung miner can't hang the test suite; on timeout,
    // kill the child and report a sentinel exit code so callers fail loudly
    // instead of blocking forever.
    let exit_code =
        match tokio::time::timeout(std::time::Duration::from_secs(30), child.wait()).await {
            Ok(status) => status.expect("wait for miner").code().unwrap_or(-1),
            Err(_) => {
                let _ = child.kill().await;
                -1
            }
        };
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
        ready_received: outcome.ready_received,
        job_request_credits: outcome.job_request_credits,
        result_job_ids: outcome.result_job_ids,
        rejects: outcome.rejects,
        cancel_acked: outcome.cancel_acked,
        exit_code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare_report() -> DriverReport {
        DriverReport {
            handshake_ok: true,
            ready_received: false,
            job_request_credits: vec![],
            result_job_ids: vec![],
            rejects: vec![],
            cancel_acked: false,
            exit_code: 0,
        }
    }

    #[test]
    fn under_asserted_report_is_not_conformant() {
        // Regression: handshake+exit alone used to green-light non-conformant miners.
        let mut r = bare_report();
        r.handshake_ok = true;
        r.exit_code = 0;
        assert!(
            !r.is_conformant(),
            "Ready / JobRequest / per-job rejects / Cancel ack are required"
        );
    }

    #[test]
    fn per_job_id_reject_binding() {
        let mut r = bare_report();
        r.rejects.push(ObservedReject {
            job_id: b"job-bad-h".to_vec(),
            reason: RejectReason::Malformed as i32,
        });
        assert!(r.has_reject(b"job-bad-h", RejectReason::Malformed));
        assert!(!r.has_reject(b"job-old", RejectReason::Malformed));
        assert!(!r.has_reject(b"job-bad-h", RejectReason::Expired));
    }
}
