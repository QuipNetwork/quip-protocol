//! Production runtime: bind the miner session server, spawn + supervise every
//! configured miner, and shut down cleanly.
//!
//! This module owns process wiring and the graceful-shutdown fan-out. The live
//! job-production loop (chain-head following + continuous feeder + decay-ratchet
//! stash) is layered on top by the live-mining epic; a coordinator built from
//! this alone serves sessions and supervises miners but stages no work yet.

use crate::chain::{ChainClient, MiningSnapshot};
use crate::config::LaunchEntry;
use crate::producer::derive_pow_job;
use crate::session::{CoordinatorService, CoordinatorState};
use crate::supervisor::{supervise_miner, BackoffPolicy};
use crate::topology::Topology;
use quip_proto::v1::miner_service_server::MinerServiceServer;
use quip_proto::v1::SetTarget;
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
    /// derivation for PoW jobs.
    pub miner_account: [u8; 32],
    /// Feeder buffer depth: staged jobs kept per miner (0 disables feeding).
    pub buffer_depth: usize,
    /// How often the feeder polls the chain head and tops up buffers.
    pub poll_interval_ms: u64,
}

/// Inputs to the feeder loop.
pub struct FeederParams {
    pub miner_account: [u8; 32],
    pub buffer_depth: usize,
    pub poll_interval: Duration,
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

/// A unique 32-byte salt from a monotonic counter: distinct salts derive
/// distinct nonces (job ids), so each attempt is a fresh PoW draw.
fn salt_from_counter(ctr: u64) -> [u8; 32] {
    let mut salt = [0u8; 32];
    salt[..8].copy_from_slice(&ctr.to_le_bytes());
    salt
}

/// The replenished PoW feeder: follow the chain head, and keep each registered
/// miner's staged queue topped up to `buffer_depth` with fresh-salt jobs.
///
/// On a new `last_proof_block_hash` it reseeds — bump the generation, cancel the
/// prior generation's staged jobs, refresh topology + difficulty target, and
/// drop stale salts — then refills under the new seed. Runs until `stop` flips.
/// `pub` so it can be exercised directly in tests without a gRPC server.
pub async fn feeder_loop<C: ChainClient>(
    chain: Arc<C>,
    state: Arc<Mutex<CoordinatorState>>,
    params: FeederParams,
    mut stop: watch::Receiver<bool>,
) {
    let mut current_head: Option<[u8; 32]> = None;
    let mut generation: u64 = 0;
    let mut salt_ctr: u64 = 0;

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
                let mut st = state.lock().await;
                st.set_topology(Some(topo));
                st.target = Some(target_from_snapshot(snap));
                st.router.cancel(generation - 1); // drop the prior generation
                st.clear_salts();
            }

            // Top every registered miner up to buffer depth with fresh jobs.
            let mut st = state.lock().await;
            for id in st.router.miner_ids() {
                while st.router.staged_len(&id) < params.buffer_depth {
                    salt_ctr = salt_ctr.saturating_add(1);
                    let salt = salt_from_counter(salt_ctr);
                    let job = derive_pow_job(snap, params.miner_account, salt, generation, 0);
                    let job_id = job.job_id.clone();
                    if st.router.stage_on(&id, job) {
                        st.record_salt(&job_id, salt);
                    } else {
                        break; // not capable for this shape — stop topping up
                    }
                }
            }
        }

        tokio::select! {
            _ = tokio::time::sleep(params.poll_interval) => {}
            _ = stop.changed() => break,
        }
    }
}

/// Serve the UDS session server, supervise every miner in `launch`, and return
/// once `shutdown` resolves — fanning an in-band `Shutdown` to each live miner
/// and killing after grace. Generic over `ChainClient` so tests drive it with
/// `FakeChain`; `main` passes `RealChainClient`. `state` is shared with the
/// caller so a live coordinator (and tests) can inspect routing/inflight.
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
    drop(stop_rx);

    // Run until asked to stop.
    shutdown.await;

    // Fan shutdown out; drain the feeder + supervisors before dropping the
    // server so in-flight Shutdown/kill completes.
    let _ = stop_tx.send(true);
    let _ = feeder.await;
    for h in supervisors {
        let _ = h.await;
    }
    server.abort();
    let _ = std::fs::remove_file(&params.sock_path);
    Ok(())
}
