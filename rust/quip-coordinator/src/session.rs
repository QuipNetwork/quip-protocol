//! tonic `MinerService` server: token verify, handshake, job dispatch, results.

use crate::chain::{ChainClient, Proof};
use crate::config::LaunchEntry;
use crate::router::{MinerCaps, Router};
use crate::topology::Topology;
use crate::validate::{beats_current, validate_result};
use quip_proto::v1::miner_service_server::{MinerService, MinerServiceServer};
use quip_proto::v1::{
    coord_msg, miner_msg, Configure, CoordMsg, MinerMsg, QualityGates, Shutdown, Welcome,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

/// Unguessable per-spawn session token (32 random bytes, hex-encoded).
pub fn gen_session_token() -> String {
    let mut buf = [0u8; 32];
    getrandom::getrandom(&mut buf).expect("os rng");
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

fn coord(msg: coord_msg::Msg) -> CoordMsg {
    CoordMsg { msg: Some(msg) }
}

/// Shared coordinator runtime state for one or more miner sessions.
pub struct CoordinatorState {
    pub expected_tokens: HashMap<String, String>,
    pub configure: HashMap<String, Configure>,
    pub topology: Option<Topology>,
    pub router: Router,
    /// job_id → Job, for validation context.
    pub inflight: HashMap<Vec<u8>, quip_proto::v1::Job>,
    pub current_best_milli: Option<i64>,
    pub results_validated: u64,
    pub last_abandoned_generation: u64,
}

impl CoordinatorState {
    pub fn new() -> Self {
        Self {
            expected_tokens: HashMap::new(),
            configure: HashMap::new(),
            topology: None,
            router: Router::new(),
            inflight: HashMap::new(),
            current_best_milli: None,
            results_validated: 0,
            last_abandoned_generation: 0,
        }
    }
}

impl Default for CoordinatorState {
    fn default() -> Self {
        Self::new()
    }
}

/// gRPC service holding shared state + chain client.
pub struct CoordinatorService<C: ChainClient + 'static> {
    pub state: Arc<Mutex<CoordinatorState>>,
    pub chain: Arc<C>,
    /// Notifies waiters when a proof is submitted successfully.
    pub submit_notify: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

#[tonic::async_trait]
impl<C: ChainClient + 'static> MinerService for CoordinatorService<C> {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<CoordMsg, Status>>(64);
        let state = Arc::clone(&self.state);
        let chain = Arc::clone(&self.chain);
        let submit_notify = Arc::clone(&self.submit_notify);

        tokio::spawn(async move {
            run_session(&mut inbound, &tx, state, chain, submit_notify).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

async fn run_session<C: ChainClient>(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    state: Arc<Mutex<CoordinatorState>>,
    chain: Arc<C>,
    submit_notify: Arc<Mutex<Option<oneshot::Sender<()>>>>,
) {
    // 1. Hello
    let hello = match inbound.message().await {
        Ok(Some(MinerMsg {
            msg: Some(miner_msg::Msg::Hello(h)),
        })) => h,
        _ => return,
    };

    let miner_id = hello.miner_id.clone();
    let (token_ok, protocol_ok, configure) = {
        let st = state.lock().await;
        let expected = st
            .expected_tokens
            .get(&miner_id)
            .cloned()
            .unwrap_or_default();
        let token_ok = !expected.is_empty() && hello.session_token == expected;
        let protocol_ok = hello.protocol_version == 1;
        let configure = st.configure.get(&miner_id).cloned().unwrap_or(Configure {
            queue_depth: 3,
            idle_timeout_s: 300,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        });
        (token_ok, protocol_ok, configure)
    };

    if !token_ok || !protocol_ok {
        // Drop the stream; well-behaved miners exit 77 on handshake failure.
        return;
    }

    {
        let mut st = state.lock().await;
        st.router.register_miner(
            miner_id.clone(),
            MinerCaps {
                backend: hello.backend.clone(),
                algorithm: hello.algorithm.clone(),
                supported_kinds: hello.supported_kinds.clone(),
                max_nodes: hello.max_nodes,
                max_edges: hello.max_edges,
            },
        );
    }

    // 2. Welcome + Configure (+ Topology if cached)
    if tx
        .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 1,
        }))))
        .await
        .is_err()
    {
        return;
    }
    let seed_credits = configure.queue_depth.max(1);
    if tx
        .send(Ok(coord(coord_msg::Msg::Configure(configure))))
        .await
        .is_err()
    {
        return;
    }
    {
        let st = state.lock().await;
        if let Some(topo) = st.topology.as_ref() {
            let _ = tx
                .send(Ok(coord(coord_msg::Msg::Topology(topo.to_proto()))))
                .await;
        }
    }

    // 3. Message loop
    loop {
        let msg = match inbound.message().await {
            Ok(Some(m)) => m,
            Ok(None) => break,
            Err(_) => break,
        };
        match msg.msg {
            Some(miner_msg::Msg::Ready(_)) => {
                // Seed credits from Configure so the first job can dispatch
                // before the miner sends JobRequest (mock-miner only requests
                // after completing a job).
                let credits = seed_credits;
                let mut st = state.lock().await;
                st.router.grant_credits(&miner_id, credits);
                while let Some(job) = st.router.next_job(&miner_id) {
                    st.inflight.insert(job.job_id.clone(), job.clone());
                    if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
                        return;
                    }
                }
            }
            Some(miner_msg::Msg::JobRequest(req)) => {
                let mut st = state.lock().await;
                st.router.grant_credits(&miner_id, req.credits);
                while let Some(job) = st.router.next_job(&miner_id) {
                    st.inflight.insert(job.job_id.clone(), job.clone());
                    if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
                        return;
                    }
                }
            }
            Some(miner_msg::Msg::Result(result)) => {
                let (job, topo_nodes, topo_edges, best) = {
                    let mut st = state.lock().await;
                    st.router.ack(&miner_id);
                    let job = st.inflight.remove(&result.job_id);
                    let (nodes, edges) = st
                        .topology
                        .as_ref()
                        .map(|t| (t.nodes.clone(), t.edge_pairs()))
                        .unwrap_or_default();
                    let best = st.current_best_milli;
                    (job, nodes, edges, best)
                };
                if let Some(job) = job {
                    if let Some(ising) = job.ising.as_ref() {
                        let gates = ising.gates.unwrap_or(QualityGates {
                            min_energy_milli: i64::MAX,
                            min_diversity_milli: 0,
                            min_solutions: 0,
                        });
                        let validated = validate_result(
                            ising,
                            &result.solutions,
                            &gates,
                            &topo_nodes,
                            &topo_edges,
                        );
                        {
                            let mut st = state.lock().await;
                            st.results_validated += 1;
                        }
                        if validated.accepted && beats_current(validated.best_energy_milli, best) {
                            let proof = Proof {
                                job_id: result.job_id.clone(),
                                best_energy_milli: validated.best_energy_milli,
                                diversity_milli: validated.diversity_milli,
                                n_valid: validated.n_valid,
                                solutions: result.solutions.clone(),
                                is_pow: job.provenance.as_ref().map(|p| p.is_pow).unwrap_or(false),
                                order_id: job
                                    .provenance
                                    .as_ref()
                                    .map(|p| p.order_id.clone())
                                    .unwrap_or_default(),
                                generation: job.generation,
                                // Salt is chosen when the PoW job is derived;
                                // session does not yet thread it through the
                                // Job message. Live RealChainClient submit
                                // requires proof.salt == 32 bytes.
                                salt: vec![],
                            };
                            if let Ok(crate::chain::SubmitAction::Success) =
                                chain.submit_proof(&proof).await
                            {
                                let mut st = state.lock().await;
                                st.current_best_milli = Some(validated.best_energy_milli);
                                if let Some(n) = submit_notify.lock().await.take() {
                                    let _ = n.send(());
                                }
                            }
                        }
                    }
                }
            }
            Some(miner_msg::Msg::Reject(rej)) => {
                let mut st = state.lock().await;
                if let Some(job) = st.inflight.remove(&rej.job_id) {
                    st.router.on_reject(&miner_id, job, rej.reason);
                } else {
                    st.router.ack(&miner_id);
                }
            }
            Some(miner_msg::Msg::Status(s)) => {
                let mut st = state.lock().await;
                if s.abandoned_generation > 0 {
                    st.last_abandoned_generation = s.abandoned_generation;
                }
            }
            Some(miner_msg::Msg::Fatal(_)) | Some(miner_msg::Msg::Hello(_)) | None => {}
        }
    }
}

/// Push a cancel for generations `<= max_generation` to a live session channel.
pub async fn send_cancel(
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    max_generation: u64,
) -> Result<(), mpsc::error::SendError<Result<CoordMsg, Status>>> {
    tx.send(Ok(coord(coord_msg::Msg::Cancel(quip_proto::v1::Cancel {
        max_generation,
    }))))
    .await
}

/// Result of a one-shot handshake harness run.
#[derive(Debug)]
pub struct SessionHarnessReport {
    pub caps: Option<MinerCaps>,
    pub miner_exit_code: i32,
    pub handshake_ok: bool,
}

/// Spawn `miner_bin` against a coordinator on `sock_path`, expecting `token`.
///
/// Used by integration tests. `sock_path` is a filesystem path (no `unix://`).
pub async fn serve_one_session(
    miner_bin: &str,
    sock_path: &str,
    miner_id: &str,
    token: &str,
) -> Result<MinerCaps, SessionHarnessError> {
    let report = serve_one_session_expecting(miner_bin, sock_path, miner_id, token, token).await;
    if report.handshake_ok {
        report.caps.ok_or(SessionHarnessError {
            miner_exit_code: report.miner_exit_code,
            reason: "handshake ok but no caps".into(),
        })
    } else {
        Err(SessionHarnessError {
            miner_exit_code: report.miner_exit_code,
            reason: "handshake failed".into(),
        })
    }
}

#[derive(Debug)]
pub struct SessionHarnessError {
    pub miner_exit_code: i32,
    pub reason: String,
}

impl std::fmt::Display for SessionHarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "session harness error (exit {}): {}",
            self.miner_exit_code, self.reason
        )
    }
}

impl std::error::Error for SessionHarnessError {}

/// Like `serve_one_session`, but the coordinator expects `coord_token` while
/// the miner is given `miner_token` (for mismatch tests).
pub async fn serve_one_session_expecting(
    miner_bin: &str,
    sock_path: &str,
    miner_id: &str,
    miner_token: &str,
    coord_token: &str,
) -> SessionHarnessReport {
    let _ = std::fs::remove_file(sock_path);
    if let Some(parent) = Path::new(sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let uds = UnixListener::bind(sock_path).expect("bind unix socket");
    let incoming = UnixListenerStream::new(uds);

    let mut st = CoordinatorState::new();
    st.expected_tokens
        .insert(miner_id.into(), coord_token.into());
    st.configure.insert(
        miner_id.into(),
        Configure {
            queue_depth: 3,
            idle_timeout_s: 5,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    );
    let state = Arc::new(Mutex::new(st));
    // Chain is unused for handshake-only; use a no-op fake with empty snapshot.
    let chain = Arc::new(crate::chain::FakeChain::new(
        crate::chain::MiningSnapshot {
            last_proof_block_hash: [0u8; 32],
            topology_hash: vec![0u8; 32],
            nodes: vec![],
            edges: vec![],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![0],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 0,
            max_energy_milli: 0,
            min_diversity_milli: 0,
            block_number: 0,
        },
        None,
    ));

    let svc = CoordinatorService {
        state: Arc::clone(&state),
        chain,
        submit_notify: Arc::new(Mutex::new(None)),
    };
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
    });

    let socket_uri = format!("unix://{sock_path}");
    let mut child = Command::new(miner_bin)
        .arg("--quip-coordinator")
        .arg(&socket_uri)
        .arg("--miner-id")
        .arg(miner_id)
        .env("QUIP_SESSION_TOKEN", miner_token)
        .spawn()
        .expect("spawn miner");

    // Give the miner a moment to handshake, then shut it down cleanly so the
    // idle path doesn't dominate: send nothing further; for a successful
    // handshake the miner waits on idle_timeout (5s). For a failed handshake
    // the stream is already dropped and the miner exits promptly.
    let exit_code = match tokio::time::timeout(Duration::from_secs(15), child.wait()).await {
        Ok(Ok(status)) => status.code().unwrap_or(-1),
        Ok(Err(_)) => -1,
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            -1
        }
    };

    // After Ready, send Shutdown so a successful session exits 0 quickly
    // rather than waiting the full idle timeout — only applies if still alive.
    // (If already exited, we're done.)
    let _ = exit_code;

    server.abort();
    let _ = std::fs::remove_file(sock_path);

    let st = state.lock().await;
    let caps = st.router.caps(miner_id).cloned();
    let handshake_ok = caps.is_some();
    SessionHarnessReport {
        caps,
        miner_exit_code: exit_code,
        handshake_ok,
    }
}

/// Inputs for the e2e PoW drive harness.
pub struct DrivePowParams<'a, C: ChainClient + 'static> {
    pub miner_bin: &'a str,
    pub sock_path: &'a str,
    pub miner_id: &'a str,
    pub token: &'a str,
    pub entry: &'a LaunchEntry,
    pub topology: Topology,
    pub job: quip_proto::v1::Job,
    pub chain: Arc<C>,
    /// After first submit, optionally cancel this generation and stage a second job.
    pub cancel_then_job: Option<(u64, quip_proto::v1::Job)>,
}

/// Full end-to-end drive: handshake, feed one PoW job, validate Result, submit.
pub async fn drive_pow_round<C: ChainClient + 'static>(p: DrivePowParams<'_, C>) -> DriveReport {
    let sock_path = p.sock_path;
    let miner_id = p.miner_id;
    let _ = std::fs::remove_file(sock_path);
    if let Some(parent) = Path::new(sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let uds = UnixListener::bind(sock_path).expect("bind");
    let incoming = UnixListenerStream::new(uds);

    let mut st = CoordinatorState::new();
    st.expected_tokens.insert(miner_id.into(), p.token.into());
    st.configure
        .insert(miner_id.into(), p.entry.configure.clone());
    st.topology = Some(p.topology);
    // Stage the PoW job before the miner connects so JobRequest can pull it.
    st.router.register_miner(
        miner_id,
        MinerCaps {
            backend: "pending".into(),
            algorithm: "pending".into(),
            supported_kinds: vec![quip_proto::v1::JobKind::IsingSample as i32],
            max_nodes: 0,
            max_edges: 0,
        },
    );
    // Route will re-register on Hello and overwrite caps; stage via direct insert
    // after Hello by using a channel. Instead: put job in a pre-stage list.
    let pre_jobs = Arc::new(Mutex::new(vec![p.job]));
    let cancel_then = Arc::new(Mutex::new(p.cancel_then_job));

    let state = Arc::new(Mutex::new(st));
    let (submit_tx, submit_rx) = oneshot::channel();
    let submit_notify = Arc::new(Mutex::new(Some(submit_tx)));

    // Wrap service with a custom session that injects pre-staged jobs on Ready.
    let svc = DriveService {
        inner_state: Arc::clone(&state),
        chain: Arc::clone(&p.chain),
        submit_notify: Arc::clone(&submit_notify),
        pre_jobs: Arc::clone(&pre_jobs),
        cancel_then: Arc::clone(&cancel_then),
        miner_id: miner_id.to_string(),
    };

    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
    });

    let socket_uri = format!("unix://{sock_path}");
    let mut child = Command::new(p.miner_bin)
        .arg("--quip-coordinator")
        .arg(&socket_uri)
        .arg("--miner-id")
        .arg(miner_id)
        .env("QUIP_SESSION_TOKEN", p.token)
        .spawn()
        .expect("spawn miner");

    // Wait for first successful submit (or timeout).
    let submitted = tokio::time::timeout(Duration::from_secs(20), submit_rx)
        .await
        .is_ok();

    // Allow cancel/second-job path a moment, then shut down.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Miner may still be running; kill after grace (shutdown is in-band but we
    // don't hold the session tx here — kill is fine for the harness).
    let _ = child.kill().await;
    let exit = child.wait().await.ok().and_then(|s| s.code()).unwrap_or(-1);
    server.abort();
    let _ = std::fs::remove_file(sock_path);

    let st = state.lock().await;
    DriveReport {
        handshake_ok: st.router.caps(miner_id).is_some(),
        results_validated: st.results_validated,
        submitted,
        abandoned_generation: st.last_abandoned_generation,
        miner_exit_code: exit,
    }
}

#[derive(Debug)]
pub struct DriveReport {
    pub handshake_ok: bool,
    pub results_validated: u64,
    pub submitted: bool,
    pub abandoned_generation: u64,
    pub miner_exit_code: i32,
}

/// Session service used by the e2e harness: injects pre-staged jobs on Ready.
struct DriveService<C: ChainClient + 'static> {
    inner_state: Arc<Mutex<CoordinatorState>>,
    chain: Arc<C>,
    submit_notify: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    pre_jobs: Arc<Mutex<Vec<quip_proto::v1::Job>>>,
    cancel_then: Arc<Mutex<Option<(u64, quip_proto::v1::Job)>>>,
    miner_id: String,
}

#[tonic::async_trait]
impl<C: ChainClient + 'static> MinerService for DriveService<C> {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<CoordMsg, Status>>(64);
        let state = Arc::clone(&self.inner_state);
        let chain = Arc::clone(&self.chain);
        let submit_notify = Arc::clone(&self.submit_notify);
        let pre_jobs = Arc::clone(&self.pre_jobs);
        let cancel_then = Arc::clone(&self.cancel_then);
        let miner_id = self.miner_id.clone();

        tokio::spawn(async move {
            // Reuse handshake logic by running a tailored loop.
            let hello = match inbound.message().await {
                Ok(Some(MinerMsg {
                    msg: Some(miner_msg::Msg::Hello(h)),
                })) => h,
                _ => return,
            };
            {
                let st = state.lock().await;
                let expected = st
                    .expected_tokens
                    .get(&miner_id)
                    .cloned()
                    .unwrap_or_default();
                if hello.session_token != expected || hello.protocol_version != 1 {
                    return;
                }
            }
            {
                let mut st = state.lock().await;
                st.router.register_miner(
                    miner_id.clone(),
                    MinerCaps {
                        backend: hello.backend,
                        algorithm: hello.algorithm,
                        supported_kinds: hello.supported_kinds,
                        max_nodes: hello.max_nodes,
                        max_edges: hello.max_edges,
                    },
                );
                // Stage pre-jobs now that the miner is registered.
                let jobs = pre_jobs.lock().await.drain(..).collect::<Vec<_>>();
                for j in jobs {
                    st.router.route(j);
                }
            }

            let configure = {
                let st = state.lock().await;
                st.configure.get(&miner_id).cloned().unwrap_or(Configure {
                    queue_depth: 3,
                    idle_timeout_s: 300,
                    heartbeat_s: 15,
                    reconnect_window_s: 60,
                    backend_toml: String::new(),
                })
            };

            let seed_credits = configure.queue_depth.max(1);
            let _ = tx
                .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
                    protocol_version: 1,
                }))))
                .await;
            let _ = tx
                .send(Ok(coord(coord_msg::Msg::Configure(configure))))
                .await;
            {
                let st = state.lock().await;
                if let Some(topo) = st.topology.as_ref() {
                    let _ = tx
                        .send(Ok(coord(coord_msg::Msg::Topology(topo.to_proto()))))
                        .await;
                }
            }

            let mut cancel_sent = false;

            loop {
                let msg = match inbound.message().await {
                    Ok(Some(m)) => m,
                    _ => break,
                };
                match msg.msg {
                    Some(miner_msg::Msg::Ready(_)) => {
                        let mut st = state.lock().await;
                        st.router.grant_credits(&miner_id, seed_credits);
                        while let Some(job) = st.router.next_job(&miner_id) {
                            st.inflight.insert(job.job_id.clone(), job.clone());
                            if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
                                return;
                            }
                        }
                    }
                    Some(miner_msg::Msg::JobRequest(req)) => {
                        let mut st = state.lock().await;
                        st.router.grant_credits(&miner_id, req.credits);
                        while let Some(job) = st.router.next_job(&miner_id) {
                            st.inflight.insert(job.job_id.clone(), job.clone());
                            if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
                                return;
                            }
                        }
                    }
                    Some(miner_msg::Msg::Result(result)) => {
                        let (job, topo_nodes, topo_edges, best) = {
                            let mut st = state.lock().await;
                            st.router.ack(&miner_id);
                            let job = st.inflight.remove(&result.job_id);
                            let (nodes, edges) = st
                                .topology
                                .as_ref()
                                .map(|t| (t.nodes.clone(), t.edge_pairs()))
                                .unwrap_or_default();
                            (job, nodes, edges, st.current_best_milli)
                        };
                        if let Some(job) = job {
                            if let Some(ising) = job.ising.as_ref() {
                                let gates = ising.gates.unwrap_or(QualityGates {
                                    min_energy_milli: i64::MAX,
                                    min_diversity_milli: 0,
                                    min_solutions: 0,
                                });
                                let validated = validate_result(
                                    ising,
                                    &result.solutions,
                                    &gates,
                                    &topo_nodes,
                                    &topo_edges,
                                );
                                {
                                    let mut st = state.lock().await;
                                    st.results_validated += 1;
                                }
                                if validated.accepted
                                    && beats_current(validated.best_energy_milli, best)
                                {
                                    let proof = Proof {
                                        job_id: result.job_id.clone(),
                                        best_energy_milli: validated.best_energy_milli,
                                        diversity_milli: validated.diversity_milli,
                                        n_valid: validated.n_valid,
                                        solutions: result.solutions.clone(),
                                        is_pow: job
                                            .provenance
                                            .as_ref()
                                            .map(|p| p.is_pow)
                                            .unwrap_or(false),
                                        order_id: job
                                            .provenance
                                            .as_ref()
                                            .map(|p| p.order_id.clone())
                                            .unwrap_or_default(),
                                        generation: job.generation,
                                        salt: vec![],
                                    };
                                    if let Ok(crate::chain::SubmitAction::Success) =
                                        chain.submit_proof(&proof).await
                                    {
                                        {
                                            let mut st = state.lock().await;
                                            st.current_best_milli =
                                                Some(validated.best_energy_milli);
                                        }
                                        if let Some(n) = submit_notify.lock().await.take() {
                                            let _ = n.send(());
                                        }
                                        // Optional cancel + re-stage path.
                                        if !cancel_sent {
                                            if let Some((max_gen, next_job)) =
                                                cancel_then.lock().await.take()
                                            {
                                                cancel_sent = true;
                                                {
                                                    let mut st = state.lock().await;
                                                    st.router.cancel(max_gen);
                                                }
                                                let _ = tx
                                                    .send(Ok(coord(coord_msg::Msg::Cancel(
                                                        quip_proto::v1::Cancel {
                                                            max_generation: max_gen,
                                                        },
                                                    ))))
                                                    .await;
                                                // Stage next generation job.
                                                {
                                                    let mut st = state.lock().await;
                                                    st.router.route(next_job);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some(miner_msg::Msg::Reject(rej)) => {
                        let mut st = state.lock().await;
                        if let Some(job) = st.inflight.remove(&rej.job_id) {
                            st.router.on_reject(&miner_id, job, rej.reason);
                        } else {
                            st.router.ack(&miner_id);
                        }
                    }
                    Some(miner_msg::Msg::Status(s)) => {
                        let mut st = state.lock().await;
                        if s.abandoned_generation > 0 {
                            st.last_abandoned_generation = s.abandoned_generation;
                        }
                    }
                    _ => {}
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// Issue a Shutdown on an outbound channel (for supervisor).
pub fn shutdown_msg(grace_ms: u32) -> CoordMsg {
    coord(coord_msg::Msg::Shutdown(Shutdown { grace_ms }))
}
