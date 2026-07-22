//! Generalized drive harness: spawn a miner, stage every job pulled from a
//! `JobSource` up front, validate each `Result` against its gates, and send a
//! clean `Shutdown` once every job has reached a terminal outcome.
//!
//! v1 is single-miner (scope guardrail), so a `Reject` is recorded as a
//! failing row rather than re-queued to the same miner: re-routing to the
//! only capable miner cannot resolve a structural reject (malformed/too
//! large) and risks an infinite reject loop. Production `run_session`
//! (`session.rs`) keeps the multi-miner reroute behavior; this is a
//! deliberate drive-mode-only deviation.

use crate::chain::{ChainClient, Proof};
use crate::config::LaunchEntry;
use crate::drive::report::JobRow;
use crate::router::MinerCaps;
use crate::session::CoordinatorState;
use crate::topology::Topology;
use crate::validate::{beats_current, validate_result};
use quip_proto::v1::miner_service_server::{MinerService, MinerServiceServer};
use quip_proto::v1::{
    coord_msg, miner_msg, Configure, CoordMsg, Job, MinerMsg, QualityGates, Reject,
    Result as JobResult, Shutdown, Welcome,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

/// Inputs for a full drive-many run.
pub struct DriveManyParams<'a, C: ChainClient + 'static> {
    pub miner_bin: &'a str,
    pub sock_path: &'a str,
    pub miner_id: &'a str,
    pub token: &'a str,
    pub entry: &'a LaunchEntry,
    pub topology: Option<Topology>,
    pub jobs: Vec<Job>,
    pub chain: Arc<C>,
    /// Hard ceiling on the whole run (bounds a stuck or rejecting miner).
    pub overall_timeout: Duration,
}

/// Result of a drive-many run: per-job rows plus handshake/exit status.
#[derive(Debug, Default)]
pub struct DriveManyReport {
    pub handshake_ok: bool,
    pub rows: Vec<JobRow>,
    /// Number of jobs handed to the run. `rows.len() < total` means the run was
    /// truncated (miner crash / dropped Result / timeout) — an incomplete run,
    /// not per-job data.
    pub total: usize,
    pub miner_exit_code: i32,
}

fn coord(msg: coord_msg::Msg) -> CoordMsg {
    CoordMsg { msg: Some(msg) }
}

fn shutdown_msg() -> CoordMsg {
    coord(coord_msg::Msg::Shutdown(Shutdown { grace_ms: 500 }))
}

/// A failing `JobRow` for a job that produced no valid `Result` — rejected,
/// unroutable (no capable miner), or cut off by a miner `Fatal`.
fn failing_row(job_id: Vec<u8>, is_pow: bool, wall_ms: u64) -> JobRow {
    JobRow {
        job_id,
        is_pow,
        n_solutions: 0,
        best_energy_milli: i64::MAX,
        diversity_milli: 0,
        passed: false,
        device_access_time_us: 0,
        wall_ms,
        rejected: true,
    }
}

fn job_is_pow(job: &Job) -> bool {
    job.provenance.as_ref().map(|p| p.is_pow).unwrap_or(false)
}

/// Shared per-run bookkeeping the session task mutates as jobs complete.
struct RunState {
    rows: StdMutex<Vec<JobRow>>,
    dispatch_at: StdMutex<HashMap<Vec<u8>, Instant>>,
    total: usize,
}

impl RunState {
    fn new(total: usize) -> Self {
        Self {
            rows: StdMutex::new(Vec::new()),
            dispatch_at: StdMutex::new(HashMap::new()),
            total,
        }
    }

    fn note_dispatch(&self, job_id: Vec<u8>) {
        self.dispatch_at
            .lock()
            .unwrap()
            .insert(job_id, Instant::now());
    }

    fn wall_ms_since_dispatch(&self, job_id: &[u8]) -> u64 {
        self.dispatch_at
            .lock()
            .unwrap()
            .remove(job_id)
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    fn record(&self, row: JobRow) {
        self.rows.lock().unwrap().push(row);
    }

    fn is_complete(&self) -> bool {
        self.rows.lock().unwrap().len() >= self.total
    }
}

/// tonic service used by `run_drive`: stages every job in `jobs` on
/// handshake, then drives the same Ready/JobRequest/Result/Reject flow as
/// production, recording a `JobRow` per terminal outcome.
struct DriveManyService<C: ChainClient + 'static> {
    state: Arc<Mutex<CoordinatorState>>,
    chain: Arc<C>,
    jobs: Arc<Mutex<Vec<Job>>>,
    run: Arc<RunState>,
    miner_id: String,
}

#[tonic::async_trait]
impl<C: ChainClient + 'static> MinerService for DriveManyService<C> {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<CoordMsg, Status>>(64);
        let state = Arc::clone(&self.state);
        let chain = Arc::clone(&self.chain);
        let jobs = Arc::clone(&self.jobs);
        let run = Arc::clone(&self.run);
        let miner_id = self.miner_id.clone();

        tokio::spawn(async move {
            run_drive_session(&mut inbound, &tx, state, chain, jobs, run, miner_id).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// Hello -> token/version check -> register + stage every job -> Welcome/
/// Configure/Topology. Returns the effective `Configure`, or `None` on a
/// failed handshake (stream is dropped; a well-behaved miner exits 77).
async fn handshake(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    state: &Arc<Mutex<CoordinatorState>>,
    jobs: &Arc<Mutex<Vec<Job>>>,
    run: &Arc<RunState>,
    miner_id: &str,
) -> Option<Configure> {
    let hello = match inbound.message().await {
        Ok(Some(MinerMsg {
            msg: Some(miner_msg::Msg::Hello(h)),
        })) => h,
        _ => return None,
    };
    {
        let st = state.lock().await;
        let expected = st
            .expected_tokens
            .get(miner_id)
            .cloned()
            .unwrap_or_default();
        // Mirror production `run_session`: an empty expected token must never
        // authenticate an empty-token miner.
        if expected.is_empty() || hello.session_token != expected || hello.protocol_version != 1 {
            return None;
        }
    }
    let configure = {
        let mut st = state.lock().await;
        st.router.register_miner(
            miner_id.to_string(),
            MinerCaps {
                backend: hello.backend,
                algorithm: hello.algorithm,
                supported_kinds: hello.supported_kinds,
                max_nodes: hello.max_nodes,
                max_edges: hello.max_edges,
            },
        );
        let staged = jobs.lock().await.drain(..).collect::<Vec<_>>();
        for j in staged {
            st.router.route(j);
        }
        // Reconcile jobs no registered miner can serve: record failing rows now
        // so the completion gate accounts for them instead of stalling to the
        // overall timeout.
        for j in std::mem::take(&mut st.router.unroutable) {
            run.record(failing_row(j.job_id.clone(), job_is_pow(&j), 0));
        }
        st.configure.get(miner_id).cloned().unwrap_or(Configure {
            queue_depth: 3,
            idle_timeout_s: 300,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        })
    };
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 1,
        }))))
        .await;
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Configure(configure.clone()))))
        .await;
    {
        let st = state.lock().await;
        if let Some(topo) = st.topology.as_ref() {
            let _ = tx
                .send(Ok(coord(coord_msg::Msg::Topology(topo.to_proto()))))
                .await;
        }
    }
    Some(configure)
}

/// Grant `credits` and dispatch every staged job the router allows.
/// Returns `false` when the outbound channel is closed (session should end).
async fn dispatch_jobs(
    state: &Arc<Mutex<CoordinatorState>>,
    miner_id: &str,
    credits: u32,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    run: &Arc<RunState>,
) -> bool {
    let mut st = state.lock().await;
    st.router.grant_credits(miner_id, credits);
    while let Some(job) = st.router.next_job(miner_id) {
        run.note_dispatch(job.job_id.clone());
        st.inflight.insert(job.job_id.clone(), job.clone());
        if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
            return false;
        }
    }
    true
}

/// Validate a `Result` against its job's gates, submit a would-be proof to
/// `chain` when it beats the current best, and record a `JobRow`.
async fn handle_result<C: ChainClient>(
    state: &Arc<Mutex<CoordinatorState>>,
    chain: &Arc<C>,
    miner_id: &str,
    result: JobResult,
    run: &Arc<RunState>,
) {
    let (job, topo_edges, best) = {
        let mut st = state.lock().await;
        st.router.ack(miner_id);
        let job = st.inflight.remove(&result.job_id);
        let edges = st
            .topology
            .as_ref()
            .map(|t| t.edge_pairs())
            .unwrap_or_default();
        (job, edges, st.current_best_milli)
    };
    let Some(job) = job else { return };
    let Some(ising) = job.ising.as_ref() else {
        return;
    };
    let gates = ising.gates.unwrap_or(QualityGates {
        min_energy_milli: i64::MAX,
        min_diversity_milli: 0,
        min_solutions: 0,
    });
    let validated = validate_result(ising, &result.solutions, &gates, &topo_edges);
    let wall_ms = run.wall_ms_since_dispatch(&result.job_id);
    let device_us = result
        .meta
        .as_ref()
        .map(|m| m.device_access_time_us)
        .unwrap_or(0);
    run.record(JobRow {
        job_id: result.job_id.clone(),
        is_pow: job.provenance.as_ref().map(|p| p.is_pow).unwrap_or(false),
        n_solutions: result.solutions.len(),
        best_energy_milli: validated.best_energy_milli,
        diversity_milli: validated.diversity_milli,
        passed: validated.accepted,
        device_access_time_us: device_us,
        wall_ms,
        rejected: false,
    });
    if validated.accepted && beats_current(validated.best_energy_milli, best) {
        let scored = ScoredResult {
            job: &job,
            result: &result,
            best_energy_milli: validated.best_energy_milli,
            diversity_milli: validated.diversity_milli,
            n_valid: validated.n_valid,
        };
        submit_if_winning(state, chain, scored).await;
    }
}

/// A validated `Result` that beat the current best and is ready to submit.
struct ScoredResult<'a> {
    job: &'a Job,
    result: &'a JobResult,
    best_energy_milli: i64,
    diversity_milli: u32,
    n_valid: u32,
}

async fn submit_if_winning<C: ChainClient>(
    state: &Arc<Mutex<CoordinatorState>>,
    chain: &Arc<C>,
    scored: ScoredResult<'_>,
) {
    let proof = Proof {
        job_id: scored.result.job_id.clone(),
        best_energy_milli: scored.best_energy_milli,
        diversity_milli: scored.diversity_milli,
        n_valid: scored.n_valid,
        solutions: scored.result.solutions.clone(),
        is_pow: scored
            .job
            .provenance
            .as_ref()
            .map(|p| p.is_pow)
            .unwrap_or(false),
        order_id: scored
            .job
            .provenance
            .as_ref()
            .map(|p| p.order_id.clone())
            .unwrap_or_default(),
        generation: scored.job.generation,
    };
    if let Ok(crate::chain::SubmitAction::Success) = chain.submit_proof(&proof).await {
        let mut st = state.lock().await;
        st.current_best_milli = Some(scored.best_energy_milli);
    }
}

/// Record a `Reject` as a failing row (see module docs for the no-reroute
/// deviation) and ack the router so its credit bookkeeping stays consistent.
async fn handle_reject(
    state: &Arc<Mutex<CoordinatorState>>,
    miner_id: &str,
    rej: Reject,
    run: &Arc<RunState>,
) {
    let is_pow = {
        let mut st = state.lock().await;
        st.router.ack(miner_id);
        st.inflight
            .remove(&rej.job_id)
            .map(|j| job_is_pow(&j))
            .unwrap_or(false)
    };
    let wall_ms = run.wall_ms_since_dispatch(&rej.job_id);
    run.record(failing_row(rej.job_id, is_pow, wall_ms));
}

/// A miner `Fatal` ends the session; record every still-inflight job as a
/// failing row so those jobs are reconciled rather than left to stall the run
/// to the overall timeout. Jobs never dispatched stay unrecorded on purpose so
/// the caller can see the run was truncated (`rows.len() < total`).
async fn handle_fatal(state: &Arc<Mutex<CoordinatorState>>, run: &Arc<RunState>) {
    let inflight: Vec<Job> = {
        let mut st = state.lock().await;
        st.inflight.drain().map(|(_, j)| j).collect()
    };
    for j in inflight {
        let wall_ms = run.wall_ms_since_dispatch(&j.job_id);
        run.record(failing_row(j.job_id.clone(), job_is_pow(&j), wall_ms));
    }
}

async fn handle_message<C: ChainClient + 'static>(
    msg: MinerMsg,
    state: &Arc<Mutex<CoordinatorState>>,
    chain: &Arc<C>,
    miner_id: &str,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    run: &Arc<RunState>,
    seed_credits: u32,
) -> bool {
    match msg.msg {
        Some(miner_msg::Msg::Ready(_)) => {
            dispatch_jobs(state, miner_id, seed_credits, tx, run).await
        }
        Some(miner_msg::Msg::JobRequest(req)) => {
            dispatch_jobs(state, miner_id, req.credits, tx, run).await
        }
        Some(miner_msg::Msg::Result(result)) => {
            handle_result(state, chain, miner_id, result, run).await;
            true
        }
        Some(miner_msg::Msg::Reject(rej)) => {
            handle_reject(state, miner_id, rej, run).await;
            true
        }
        Some(miner_msg::Msg::Fatal(_)) => {
            handle_fatal(state, run).await;
            false
        }
        _ => true,
    }
}

async fn run_drive_session<C: ChainClient + 'static>(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    state: Arc<Mutex<CoordinatorState>>,
    chain: Arc<C>,
    jobs: Arc<Mutex<Vec<Job>>>,
    run: Arc<RunState>,
    miner_id: String,
) {
    let Some(configure) = handshake(inbound, tx, &state, &jobs, &run, &miner_id).await else {
        return;
    };
    // Nothing left to run — either no jobs, or every job was unroutable and
    // already reconciled during handshake. Shut down without waiting for Ready.
    if run.total == 0 || run.is_complete() {
        let _ = tx.send(Ok(shutdown_msg())).await;
        return;
    }
    let seed_credits = configure.queue_depth.max(1);
    loop {
        let msg = match inbound.message().await {
            Ok(Some(m)) => m,
            _ => break,
        };
        let keep_going =
            handle_message(msg, &state, &chain, &miner_id, tx, &run, seed_credits).await;
        if !keep_going {
            break;
        }
        if run.is_complete() {
            let _ = tx.send(Ok(shutdown_msg())).await;
            break;
        }
    }
}

/// Full drive-many run: spawn the miner over UDS, stage every job, validate
/// every `Result`, and send a clean `Shutdown` once all jobs are terminal.
pub async fn run_drive<C: ChainClient + 'static>(p: DriveManyParams<'_, C>) -> DriveManyReport {
    let sock_path = p.sock_path;
    let miner_id = p.miner_id;
    let _ = std::fs::remove_file(sock_path);
    if let Some(parent) = Path::new(sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let uds = UnixListener::bind(sock_path).expect("bind unix socket");
    let incoming = UnixListenerStream::new(uds);

    let mut st = CoordinatorState::new();
    st.expected_tokens.insert(miner_id.into(), p.token.into());
    st.configure
        .insert(miner_id.into(), p.entry.configure.clone());
    st.topology = p.topology;
    let total = p.jobs.len();
    let state = Arc::new(Mutex::new(st));
    let jobs = Arc::new(Mutex::new(p.jobs));
    let run = Arc::new(RunState::new(total));

    let svc = DriveManyService {
        state: Arc::clone(&state),
        chain: Arc::clone(&p.chain),
        jobs,
        run: Arc::clone(&run),
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

    let exit_code = match tokio::time::timeout(p.overall_timeout, child.wait()).await {
        Ok(Ok(status)) => status.code().unwrap_or(-1),
        Ok(Err(_)) => -1,
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            -1
        }
    };

    server.abort();
    let _ = std::fs::remove_file(sock_path);

    let st = state.lock().await;
    let handshake_ok = st.router.caps(miner_id).is_some();
    let rows = run.rows.lock().unwrap().clone();
    DriveManyReport {
        handshake_ok,
        rows,
        total,
        miner_exit_code: exit_code,
    }
}
