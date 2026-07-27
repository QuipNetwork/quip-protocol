//! Production runtime: bind the miner session server, spawn + supervise every
//! configured miner, and shut down cleanly.
//!
//! This module owns process wiring and the graceful-shutdown fan-out. The live
//! job-production loop (chain-head following + continuous feeder + decay-ratchet
//! stash) is layered on top by the live-mining epic; a coordinator built from
//! this alone serves sessions and supervises miners but stages no work yet.

use crate::chain::{ChainClient, MiningSnapshot};
use crate::config::LaunchEntry;
use crate::decay::{build_decay_schedule, EnergyCurve};
use crate::producer::derive_pow_job;
use crate::session::{coord, CoordinatorService, CoordinatorState};
use crate::supervisor::{supervise_miner, BackoffPolicy};
use crate::topology::Topology;
use quip_proto::v1::miner_service_server::MinerServiceServer;
use quip_proto::v1::{coord_msg, SetTarget};
use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UnixListener;
use tokio::sync::{watch, Mutex};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;

/// Runtime knobs shared by every supervised miner.
pub struct RuntimeParams {
    /// Unix-domain socket the server binds and miners connect to.
    pub sock_path: String,
    /// Grace period (ms) between an in-band `Shutdown` and a hard kill.
    pub grace_ms: u32,
    /// Restart backoff + failure budget applied to each miner.
    pub backoff: BackoffPolicy,
    /// Canonical miner account (`blake2_256(SCALE(account))`) seeding nonce
    /// derivation for `PoW` jobs.
    pub miner_account: [u8; 32],
    /// Floor for the adaptive staging window: minimum staged jobs per miner
    /// (0 disables feeding). The window grows above this with each miner's
    /// observed drain rate; there is no ceiling. See [`feeder_loop`].
    pub buffer_depth: usize,
    /// How often the feeder polls the chain head and tops up buffers.
    pub poll_interval_ms: u64,
    /// Optional attempt-dashboard: `(listen_addr, data_dir)`. `None` disables
    /// recording + the REST endpoint.
    pub dashboard: Option<(String, std::path::PathBuf)>,
}

/// Inputs to the feeder loop.
pub struct FeederParams {
    /// Canonical miner account seeding `PoW` nonce derivation.
    pub miner_account: [u8; 32],
    /// Floor for the adaptive staging window: the minimum staged jobs kept per
    /// miner (0 disables feeding). The window grows above this from the miner's
    /// observed drain rate — see [`feeder_loop`] — with no ceiling.
    pub buffer_depth: usize,
    /// How often the feeder polls the chain head and tops up buffers.
    pub poll_interval: Duration,
}

/// EMA smoothing for the per-miner consumption signal. Lower reacts slower but
/// steadier; 0.3 favors stability over reactivity for a ~1s poll, so a single
/// slow or fast poll doesn't swing the window.
const CONSUMPTION_EMA_ALPHA: f64 = 0.3;

/// Staging headroom over smoothed consumption: keep ~2 poll-intervals of drain
/// staged so a fast miner never idles waiting for the next top-up.
const WINDOW_HEADROOM: f64 = 2.0;

/// How many decay steps (epochs) ahead the win-time stash projects viability.
/// A candidate needing more than this to clear is dropped as too far out.
const DECAY_HORIZON_STEPS: usize = 256;

/// Adaptive staging depth for one miner from its smoothed drain rate. The
/// `buffer_depth` floor keeps a small reserve for idle/slow miners. `ema` is
/// anchored to real completions, so depth self-bounds to ~headroom × actual
/// throughput; the feeder additionally clamps it to [`stage_ceiling`] so a fast
/// miner on a large topology can't stage an unbounded-memory reserve.
fn adaptive_depth(ema: f64, floor: usize) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "EMA*headroom is non-negative and small for operational drain rates"
    )]
    {
        floor.max((ema * WINDOW_HEADROOM).ceil() as usize)
    }
}

/// Per-miner budget on staged-job memory. The adaptive window sizes to drain
/// rate with no ceiling of its own, so a fast miner on a large topology could
/// stage an unbounded reserve — each staged `PoW` job holds a full materialized
/// Ising model (`h`/`j`), ~`4·(nodes+edges)` bytes. This bounds the peak
/// staging memory per miner regardless of topology size or drain rate.
const MAX_STAGE_BYTES_PER_MINER: usize = 64 * 1024 * 1024; // 64 MiB

/// Periodic liveness ping interval for [`crate::liveness::liveness_loop`].
const LIVENESS_PING_INTERVAL: Duration = Duration::from_secs(15);

/// Estimated materialized-model bytes for one staged job on this topology: one
/// i32 field per node (`h`) and one per edge (`j`). The 32-byte `topology_hash`
/// and `job_id` are negligible next to these. Floored at 1 to avoid a zero
/// divisor on an empty topology.
fn est_job_bytes(num_nodes: usize, num_edges: usize) -> usize {
    4usize
        .saturating_mul(num_nodes.saturating_add(num_edges))
        .max(1)
}

/// Memory-aware ceiling on the staging window: the most jobs that fit the
/// per-miner byte budget, but never below `floor` — a miner must not idle for
/// lack of a staged job even on a topology where the floor alone exceeds the
/// budget.
fn stage_ceiling(num_nodes: usize, num_edges: usize, floor: usize) -> usize {
    floor.max(MAX_STAGE_BYTES_PER_MINER / est_job_bytes(num_nodes, num_edges))
}

/// Difficulty gates advertised to miners, from the snapshot.
fn target_from_snapshot(snap: &MiningSnapshot) -> SetTarget {
    SetTarget {
        max_energy_milli: snap.max_energy_milli,
        min_solutions: snap.min_solutions,
        min_diversity_milli: snap.min_diversity_milli,
        num_reads: 0,
        num_sweeps: 0,
        anneal_time_us: 0,
    }
}

/// Broadcast `target` to every live miner off-lock: clone the outbound senders
/// under the state lock, drop it, then await the sends so the fan-out never
/// holds the state lock across I/O.
async fn broadcast_set_target(state: &Arc<Mutex<CoordinatorState>>, target: SetTarget) {
    let senders: Vec<_> = {
        let st = state.lock().await;
        st.outbound.values().cloned().collect()
    };
    for tx in senders {
        let _ = tx.send(Ok(coord(coord_msg::Msg::SetTarget(target)))).await;
    }
}

/// Push a refreshed topology to every live miner off-lock. Live sessions only
/// receive `Topology` at handshake, so a topology that first becomes available
/// (miner connected before the first snapshot) or changes on reseed must be
/// fanned out here — otherwise those miners reject subsequent hash-based jobs as
/// topology-missing/mismatch.
async fn broadcast_topology(state: &Arc<Mutex<CoordinatorState>>, topo: quip_proto::v1::Topology) {
    let senders: Vec<_> = {
        let st = state.lock().await;
        st.outbound.values().cloned().collect()
    };
    for tx in senders {
        let _ = tx
            .send(Ok(coord(coord_msg::Msg::Topology(topo.clone()))))
            .await;
    }
}

/// Fan out a protocol `Cancel(max_generation)` to every live miner on reseed so
/// they abandon stale-generation work promptly. Belt-and-suspenders with the
/// miner-side cooperative-cancel guard and the cleared salts (which already make
/// a late stale-generation result unsubmittable).
async fn broadcast_cancel(state: &Arc<Mutex<CoordinatorState>>, max_generation: u64) {
    let senders: Vec<_> = {
        let st = state.lock().await;
        st.outbound.values().cloned().collect()
    };
    for tx in senders {
        let _ = tx
            .send(Ok(coord(coord_msg::Msg::Cancel(quip_proto::v1::Cancel {
                max_generation,
            }))))
            .await;
    }
}

/// A unique 32-byte salt from a monotonic counter: distinct salts derive
/// distinct nonces (job ids), so each attempt is a fresh `PoW` draw.
fn salt_from_counter(ctr: u64) -> [u8; 32] {
    let mut salt = [0u8; 32];
    salt[..8].copy_from_slice(&ctr.to_le_bytes());
    salt
}

/// The replenished `PoW` feeder: follow the chain head, and keep each registered
/// miner's staged queue topped up to `buffer_depth` with fresh-salt jobs.
///
/// On a new `last_proof_block_hash` it reseeds — bump the generation, cancel the
/// prior generation's staged jobs, refresh topology + difficulty target, and
/// drop stale salts — then refills under the new seed. Runs until `stop` flips.
/// `pub` so it can be exercised directly in tests without a gRPC server.
#[expect(
    clippy::too_many_lines,
    reason = "single feeder loop: reseed, top-up, win-time submit"
)]
pub async fn feeder_loop<C: ChainClient>(
    chain: Arc<C>,
    state: Arc<Mutex<CoordinatorState>>,
    params: FeederParams,
    mut stop: watch::Receiver<bool>,
) {
    let mut current_head: Option<[u8; 32]> = None;
    let mut generation: u64 = 0;
    let mut salt_ctr: u64 = 0;
    // Monotonic anchor for block-time estimation (win-time submission).
    let start = std::time::Instant::now();
    // Per-miner smoothed jobs-consumed-per-poll, driving the adaptive window.
    let mut consumption_ema: HashMap<String, f64> = HashMap::new();
    // Last difficulty triple broadcast to miners, so we only re-push on change.
    let mut last_broadcast: Option<(i64, u32, u32)> = None;
    // Last topology hash pushed to miners, so we re-push only when it changes
    // (including first availability), not on every per-block reseed.
    let mut last_topology_hash: Option<Vec<u8>> = None;

    loop {
        let snap = match chain
            .fetch_mining_snapshot(None, params.miner_account, None)
            .await
        {
            Ok(Some(s)) => Some(s),
            Ok(None) => None,
            Err(e) => {
                tracing::warn!("feeder: snapshot fetch failed: {e}");
                None
            }
        };

        if let Some(snap) = snap.as_ref() {
            let head = snap.last_proof_block_hash;
            if current_head != Some(head) {
                // Reseed on a new head.
                current_head = Some(head);
                generation = generation.saturating_add(1);
                let topo = Topology::from_nodes_edges(
                    snap.nodes.clone(),
                    snap.edges.clone(),
                    &snap.allowed_h_milli,
                    &snap.allowed_j_milli,
                    &snap.allowed_spin_milli,
                );
                // Refresh the chain qblock id + decay-projection inputs for the
                // attempt logs and win-time stash (best-effort; fetched before
                // locking so we never await under the state lock).
                let qblock_id = match chain.fetch_latest_qblock_id().await {
                    Ok(id) => id,
                    Err(e) => {
                        tracing::warn!(error = %e, "feeder: qblock id read failed; attempt logs lose the qblock key");
                        None
                    }
                };
                let decay = match <[u8; 32]>::try_from(snap.topology_hash.as_slice()) {
                    Ok(h) => match chain.fetch_decay_params(h).await {
                        Ok(dp) => dp,
                        Err(e) => {
                            tracing::warn!(error = %e, "feeder: decay params read failed; running without decay projection");
                            None
                        }
                    },
                    Err(_) => None,
                };
                // Project the per-generation decay schedule for the stash.
                let (schedule, last_proof_block, epoch_length) = match &decay {
                    Some(dp) => {
                        let curve = EnergyCurve::from_topology(
                            snap.nodes.len() as u64,
                            snap.edges.len() as u64,
                            dp.c_easy_milli,
                            dp.c_knee_milli,
                            dp.c_hard_milli,
                            &snap.allowed_h_milli,
                            &snap.allowed_j_milli,
                        );
                        (
                            build_decay_schedule(
                                dp.base_max_energy_milli,
                                Some(&curve),
                                DECAY_HORIZON_STEPS,
                            ),
                            dp.last_proof_block,
                            dp.epoch_length,
                        )
                    }
                    None => (Vec::new(), 0, 0),
                };
                let topo_proto = topo.to_proto();
                let topology_changed =
                    last_topology_hash.as_deref() != Some(snap.topology_hash.as_slice());
                let mut st = state.lock().await;
                st.set_topology(Some(topo));
                st.target = Some(target_from_snapshot(snap));
                st.generation = generation;
                st.qblock_id = qblock_id;
                // Each generation is an independent PoW problem (new nonce/model),
                // so best energies are not comparable across generations. Reset
                // the best or later generations would have to beat a historical
                // minimum and never submit (the chain clears its block-best too).
                st.current_best_milli = None;
                st.stash
                    .reset(generation, schedule, last_proof_block, epoch_length);
                st.router.cancel(generation - 1); // drop the prior generation's staged jobs
                st.clear_salts();
                drop(st);
                // Fan out the reseed to live miners, off-lock. Cancel the prior
                // generation's work (skip the first reseed: generation 0 has none).
                if generation > 1 {
                    broadcast_cancel(&state, generation - 1).await;
                }
                // Push topology only when it changed (incl. first availability),
                // so a miner that connected before it was cached can resolve
                // hash-based jobs — without re-sending an identical graph each block.
                if topology_changed {
                    broadcast_topology(&state, topo_proto).await;
                    last_topology_hash = Some(snap.topology_hash.clone());
                }
            }

            // Push the refreshed difficulty to live miners when it changed, so
            // they adapt their sampling budget as the chain difficulty moves.
            let target = target_from_snapshot(snap);
            let key = (
                target.max_energy_milli,
                target.min_solutions,
                target.min_diversity_milli,
            );
            if last_broadcast != Some(key) {
                broadcast_set_target(&state, target).await;
                last_broadcast = Some(key);
            }

            // Top every registered miner up to its adaptive window with fresh
            // jobs. The window tracks each miner's drain rate: sample the
            // jobs-consumed counter (read-and-reset), smooth it, and size the
            // staged reserve to ~headroom poll-intervals of that rate.
            let mut st = state.lock().await;
            // Persist the current difficulty so session-path validation gates on
            // the same target the miners were just told. Within one proof-block
            // the decay ratchet eases `max_energy` each block without a reseed;
            // without this, `st.target` would stay pinned at the last reseed's
            // (harder) gate and under-accept solutions viable at the eased one.
            st.target = Some(target);
            for id in st.router.miner_ids() {
                let consumed = f64::from(st.router.take_consumed(&id));
                let ema = match consumption_ema.get(&id) {
                    Some(&prev) => {
                        CONSUMPTION_EMA_ALPHA * consumed + (1.0 - CONSUMPTION_EMA_ALPHA) * prev
                    }
                    // First sample seeds the EMA directly so a fast miner reaches
                    // full depth in one poll instead of ramping over several.
                    None => consumed,
                };
                let _ = consumption_ema.insert(id.clone(), ema);
                let depth = adaptive_depth(ema, params.buffer_depth).min(stage_ceiling(
                    snap.nodes.len(),
                    snap.edges.len(),
                    params.buffer_depth,
                ));
                while st.router.staged_len(&id) < depth {
                    salt_ctr = salt_ctr.saturating_add(1);
                    let salt = salt_from_counter(salt_ctr);
                    let job = match derive_pow_job(snap, params.miner_account, salt, generation, 0)
                    {
                        Ok(job) => job,
                        Err(e) => {
                            // The snapshot's allowed-value sets are validated at
                            // fetch (`require_set_values`), so an empty set here is
                            // an invariant violation, not routine. Stop topping up
                            // this miner rather than crash the feeder.
                            tracing::error!(error = %e, miner = %id, "feeder: cannot draw PoW job");
                            break;
                        }
                    };
                    let job_id = job.job_id.clone();
                    if st.router.stage_on(&id, job) {
                        st.record_salt(&job_id, salt);
                    } else {
                        break; // not capable for this shape — stop topping up
                    }
                }
            }

            // Win-time submission: observe the head for block estimation, pick
            // the best stashed candidate whose projected viability block has
            // arrived (and still improves on the current best), then submit it
            // off-lock. `mark_submitted` + the `beats_current` guard keep this
            // from double-submitting or regressing what the session path sent.
            let now_mono = start.elapsed().as_secs_f64();
            let now_wall = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0.0, |d| d.as_secs_f64());
            st.timing
                .observe_head(snap.block_number, now_wall, now_mono, now_wall);
            let current_block = st
                .timing
                .estimate_block(now_mono)
                .unwrap_or(snap.block_number);
            let best = st.current_best_milli;
            let due = st.stash.due_improving(current_block, best).cloned();
            drop(st);

            if let Some(cand) = due {
                use crate::chain::SubmitAction;

                let proof = crate::chain::Proof {
                    job_id: cand.job_id.clone(),
                    best_energy_milli: cand.best_energy_milli,
                    diversity_milli: cand.diversity_milli,
                    n_valid: cand.n_valid,
                    solutions: cand.solutions.clone(),
                    is_pow: cand.is_pow,
                    order_id: cand.order_id.clone(),
                    generation: cand.generation,
                    salt: cand.salt.map(|s| s.to_vec()).unwrap_or_default(),
                    device_access_time_us: cand.device_access_time_us,
                };
                let job_hex = crate::chain::extrinsic::hex_encode(&cand.job_id);
                match chain.submit_proof(&proof).await {
                    Ok(SubmitAction::Success) => {
                        let mut st = state.lock().await;
                        st.stash.mark_submitted(&cand.job_id);
                        st.current_best_milli = Some(cand.best_energy_milli);
                        let qblock_id = st.qblock_id;
                        let body = crate::attempt::summary_body(
                            qblock_id,
                            st.current_best_milli,
                            st.results_validated,
                            st.stash.summary(),
                        );
                        if let Some(tx) = st.attempt_tx.as_ref() {
                            let _ = tx.send(crate::attempt::WriterMsg::Summary { qblock_id, body });
                        }
                    }
                    Ok(SubmitAction::Retry) => {
                        // Retryable reject (e.g. InsufficientEnergy / ProofLimit):
                        // leave the candidate in the stash so a later due-window
                        // resubmits it; do NOT mark_submitted.
                        tracing::warn!(job = %job_hex, "win-time submit rejected (retryable); leaving stashed");
                    }
                    Ok(SubmitAction::StopRoundStale) => {
                        // Stale for this round (e.g. InvalidNonce / topology moved);
                        // the next reseed clears the stash. Stop resubmitting it now.
                        tracing::info!(job = %job_hex, "win-time submit stale for round; dropping candidate");
                        state.lock().await.stash.mark_submitted(&cand.job_id);
                    }
                    Ok(SubmitAction::StopFatal) => {
                        // Fatally rejected by the pallet (BadProof/BadSignature/...):
                        // never retry it, or the same candidate loops every window.
                        tracing::error!(job = %job_hex, "win-time submit fatally rejected by pallet; dropping candidate");
                        state.lock().await.stash.mark_submitted(&cand.job_id);
                    }
                    Err(e) => {
                        // Transient chain error (RPC down / unreachable): keep the
                        // winning candidate stashed for the next due window.
                        tracing::error!(job = %job_hex, error = %e, "win-time submit failed (transient); leaving stashed for retry");
                    }
                }
            }
        }

        tokio::select! {
            () = tokio::time::sleep(params.poll_interval) => {}
            _ = stop.changed() => break,
        }
    }
}

/// Serve the UDS session server, supervise every miner in `launch`, and return
/// once `shutdown` resolves — fanning an in-band `Shutdown` to each live miner
/// and killing after grace. Generic over [`ChainClient`] so tests drive it with
/// `FakeChain`; `main` passes `RealChainClient`. `state` is shared with the
/// caller so a live coordinator (and tests) can inspect routing/inflight.
///
/// # Errors
/// Returns an I/O error if the UDS socket cannot be bound.
pub async fn run_runtime<C, S>(
    launch: Vec<LaunchEntry>,
    chain: Arc<C>,
    state: Arc<Mutex<CoordinatorState>>,
    params: RuntimeParams,
    shutdown: S,
) -> std::io::Result<()>
where
    C: ChainClient + 'static,
    S: Future<Output = ()>,
{
    let _ = std::fs::remove_file(&params.sock_path);
    if let Some(parent) = Path::new(&params.sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let uds = UnixListener::bind(&params.sock_path)?;
    let incoming = UnixListenerStream::new(uds);

    // Seed each miner's Configure so the server can answer its handshake; the
    // per-spawn session token is set by the supervisor before each spawn.
    {
        let mut st = state.lock().await;
        for e in &launch {
            let _ = st.configure.insert(e.miner_id.clone(), e.configure.clone());
        }
    }

    // Optional mining-attempt dashboard: a single writer thread records every
    // solved model to `<data_dir>/<qblock_id>/attempts.jsonl`, and an HTTP task
    // serves those files statically.
    let dashboard_server = if let Some((listen, data_dir)) = params.dashboard.clone() {
        let tx = crate::attempt::spawn_writer(data_dir.clone());
        state.lock().await.attempt_tx = Some(tx);
        Some(tokio::spawn(crate::dashboard::serve(listen, data_dir)))
    } else {
        None
    };

    let svc = CoordinatorService {
        state: Arc::clone(&state),
        chain: Arc::clone(&chain),
        submit_notify: Arc::new(Mutex::new(None)),
    };
    let server = tokio::spawn(
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming),
    );

    let (stop_tx, stop_rx) = watch::channel(false);
    let sock_uri = format!("unix://{}", params.sock_path);
    let mut supervisors = Vec::with_capacity(launch.len());
    for entry in launch {
        supervisors.push(tokio::spawn(supervise_miner(
            entry,
            sock_uri.clone(),
            Arc::clone(&state),
            params.backoff,
            params.grace_ms,
            stop_rx.clone(),
        )));
    }

    // The replenished feeder follows the chain head and keeps buffers full.
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        FeederParams {
            miner_account: params.miner_account,
            buffer_depth: params.buffer_depth,
            poll_interval: Duration::from_millis(params.poll_interval_ms),
        },
        stop_rx.clone(),
    ));

    let liveness = tokio::spawn(crate::liveness::liveness_loop(
        Arc::clone(&state),
        LIVENESS_PING_INTERVAL,
        stop_rx.clone(),
    ));
    drop(stop_rx);

    // Run until asked to stop.
    shutdown.await;

    // Fan shutdown out; drain the feeder + supervisors before dropping the
    // server so in-flight Shutdown/kill completes.
    let _ = stop_tx.send(true);
    let _ = feeder.await;
    let _ = liveness.await;
    for h in supervisors {
        let _ = h.await;
    }
    server.abort();
    if let Some(h) = dashboard_server {
        h.abort();
    }
    let _ = std::fs::remove_file(&params.sock_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{adaptive_depth, stage_ceiling, MAX_STAGE_BYTES_PER_MINER, WINDOW_HEADROOM};

    #[test]
    fn adaptive_depth_holds_floor_when_idle() {
        // No consumption -> stay at the floor, never below it.
        assert_eq!(adaptive_depth(0.0, 8), 8);
        assert_eq!(adaptive_depth(0.0, 0), 0);
    }

    #[test]
    fn adaptive_depth_grows_with_consumption_no_ceiling() {
        // depth = ceil(ema * headroom), floor when that is smaller.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test fixture: exact integer headroom product"
        )]
        let expected = (10.0 * WINDOW_HEADROOM) as usize;
        assert_eq!(adaptive_depth(10.0, 8), expected);
        // A many-core miner grows far past the floor — no upper cap.
        assert_eq!(adaptive_depth(250.0, 8), 500);
    }

    #[test]
    fn adaptive_depth_ceils_fractional_ema_and_respects_floor() {
        // 2.5 * 2.0 = 5.0 exactly.
        assert_eq!(adaptive_depth(2.5, 2), 5);
        // Fractional below the floor rounds up but the floor still wins:
        // 0.4 * 2.0 = 0.8 -> ceil 1, floor 4 -> 4.
        assert_eq!(adaptive_depth(0.4, 4), 4);
    }

    #[test]
    fn stage_ceiling_never_binds_on_dev_topology() {
        // 2 nodes / 1 edge -> ~12 bytes/job -> the byte budget allows millions
        // of staged jobs, so the ceiling never clamps a real drain-driven depth.
        let cap = stage_ceiling(2, 1, 8);
        assert!(cap > 1_000_000, "dev topology must not be capped: {cap}");
        assert!(cap >= MAX_STAGE_BYTES_PER_MINER / 12);
    }

    #[test]
    fn stage_ceiling_caps_large_topology_below_uncapped_depth() {
        // Zephyr-scale: ~4577 nodes / 41515 edges -> ~184 KB/job. A fast miner
        // (ema 250 -> uncapped depth 500) is clamped to the byte budget.
        let cap = stage_ceiling(4577, 41515, 8);
        let uncapped = adaptive_depth(250.0, 8); // 500
        assert!(cap >= 8, "never below floor: {cap}");
        assert!(cap < uncapped, "large topology is memory-capped: {cap}");
    }

    #[test]
    fn stage_ceiling_holds_floor_when_budget_smaller_than_floor() {
        // Pathological topology where even one job exceeds the budget: the floor
        // still wins so the miner never idles for lack of a staged job.
        assert_eq!(stage_ceiling(usize::MAX / 8, 0, 8), 8);
    }
}
