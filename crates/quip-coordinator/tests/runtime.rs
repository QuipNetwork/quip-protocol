//! Runtime lifecycle: run_runtime binds the server, supervises a real
//! quip-mock-miner, and shuts down cleanly on signal. Job production (feeding
//! work, solve→submit) is exercised by tests/e2e.rs; this covers the process
//! wiring + graceful-shutdown seam.

use quip_coordinator::chain::{FakeChain, MiningSnapshot};
use quip_coordinator::config::LaunchEntry;
use quip_coordinator::router::MinerCaps;
use quip_coordinator::runtime::{feeder_loop, run_runtime, FeederParams, RuntimeParams};
use quip_coordinator::session::CoordinatorState;
use quip_coordinator::supervisor::BackoffPolicy;
use quip_proto::v1::{Configure, JobKind};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch, Mutex};

/// Build + resolve the sibling `quip-mock-miner` binary (not the same package,
/// so no `CARGO_BIN_EXE_*`).
fn mock_miner() -> String {
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-mock-miner"])
        .status()
        .expect("build quip-mock-miner");
    assert!(status.success(), "failed to build quip-mock-miner");
    let name = if cfg!(windows) {
        "quip-mock-miner.exe"
    } else {
        "quip-mock-miner"
    };
    let mut p = std::env::current_exe().expect("test exe path");
    p.pop();
    p.pop();
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// A no-work snapshot: the runtime under test stages no jobs, so the chain is
/// only held by the service and never queried.
fn trivial_snapshot() -> MiningSnapshot {
    MiningSnapshot {
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
    }
}

fn cpu_entry(binary: String) -> LaunchEntry {
    LaunchEntry {
        miner_id: "cpu-0".into(),
        binary,
        configure: Configure {
            // Long idle so the miner stays connected until we shut it down.
            queue_depth: 3,
            idle_timeout_s: 60,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    }
}

#[tokio::test]
async fn runtime_serves_supervises_and_shuts_down_clean() {
    let miner = mock_miner();
    let sock = format!("/tmp/quip-rt-{}.sock", std::process::id());
    let chain = Arc::new(FakeChain::new(trivial_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let params = RuntimeParams {
        sock_path: sock,
        grace_ms: 500,
        backoff: BackoffPolicy::default(),
        // Lifecycle test only: buffer_depth 0 disables feeding.
        miner_account: [0u8; 32],
        buffer_depth: 0,
        poll_interval_ms: 200,
        dashboard: None,
    };
    let (trigger_tx, trigger_rx) = oneshot::channel::<()>();

    let state_for_run = Arc::clone(&state);
    let run = tokio::spawn(async move {
        run_runtime(
            vec![cpu_entry(miner)],
            chain,
            state_for_run,
            params,
            async move {
                let _ = trigger_rx.await;
            },
        )
        .await
    });

    // The supervisor spawns the mock-miner, which handshakes and registers its
    // outbound channel — that is the observable "miner is live" signal.
    let mut handshook = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if state.lock().await.outbound.contains_key("cpu-0") {
            handshook = true;
            break;
        }
    }
    assert!(
        handshook,
        "mock-miner never handshook / registered an outbound"
    );

    // Graceful shutdown: run_runtime fans out Shutdown, drains supervisors, and
    // returns Ok promptly (well within the grace + kill margin).
    trigger_tx.send(()).expect("send shutdown trigger");
    let res = tokio::time::timeout(Duration::from_secs(6), run)
        .await
        .expect("run_runtime did not return after shutdown")
        .expect("run_runtime task panicked");
    res.expect("run_runtime returned an error");
}

/// A non-trivial snapshot (real nodes/edges) so derived PoW jobs are staged.
fn ising_snapshot() -> MiningSnapshot {
    let nodes = vec![0, 1, 2, 3];
    let edges = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
    let h = vec![-1000, 0, 1000];
    let j = vec![-1000, 1000];
    let spin = vec![-1000, 1000];
    let topology_hash =
        quip_coordinator::topology::topology_hash_sets(&nodes, &edges, &h, &j, &spin).to_vec();
    MiningSnapshot {
        last_proof_block_hash: [7u8; 32],
        topology_hash,
        nodes,
        edges,
        allowed_h_milli: h,
        allowed_j_milli: j,
        allowed_spin_milli: spin,
        min_solutions: 1,
        max_energy_milli: i64::MAX / 2,
        min_diversity_milli: 0,
        block_number: 42,
    }
}

fn ising_caps() -> MinerCaps {
    MinerCaps {
        backend: "mock".into(),
        algorithm: "sa".into(),
        supported_kinds: vec![JobKind::IsingSample as i32],
        max_nodes: 0,
        max_edges: 0,
    }
}

#[tokio::test]
async fn feeder_tops_up_to_buffer_depth_records_salts_and_sets_target() {
    let chain = Arc::new(FakeChain::new(ising_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    // Pre-register a miner, as if it had handshaked.
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        FeederParams {
            miner_account: [0u8; 32],
            buffer_depth: 4,
            poll_interval: Duration::from_millis(50),
        },
        stop_rx,
    ));

    let mut filled = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if state.lock().await.router.staged_len("cpu-0") == 4 {
            filled = true;
            break;
        }
    }
    assert!(filled, "feeder never filled the buffer to depth 4");

    {
        let st = state.lock().await;
        assert_eq!(st.router.staged_len("cpu-0"), 4);
        assert_eq!(st.salts.len(), 4, "one salt recorded per staged job");
        // Reseed set topology + difficulty target from the snapshot.
        assert!(st.topology.is_some());
        assert!(st.target.is_some());
    }

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

#[tokio::test]
async fn feeder_grows_window_for_drainer_and_holds_floor_for_idle() {
    let chain = Arc::new(FakeChain::new(ising_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    {
        let mut st = state.lock().await;
        st.router.register_miner("cpu-fast", ising_caps());
        st.router.register_miner("cpu-idle", ising_caps());
    }

    // Floor of 2; the fast miner drains a fixed ~8/interval so its adaptive
    // window converges to ~2x that (headroom), well above the floor. The idle
    // miner never consumes, so it stays pinned at the floor.
    const FLOOR: usize = 2;
    const DRAIN_RATE: usize = 8;

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        FeederParams {
            miner_account: [0u8; 32],
            buffer_depth: FLOOR,
            poll_interval: Duration::from_millis(30),
        },
        stop_rx,
    ));

    // Simulate a fixed-throughput miner: each interval, grant credits and pull
    // up to DRAIN_RATE staged jobs (min with what's available, so an unramped
    // buffer can't force runaway growth). Watch the window climb past the floor.
    let mut grew = false;
    for _ in 0..80 {
        tokio::time::sleep(Duration::from_millis(30)).await;
        {
            let mut st = state.lock().await;
            st.router.grant_credits("cpu-fast", DRAIN_RATE as u32);
            for _ in 0..DRAIN_RATE {
                if st.router.next_job("cpu-fast").is_none() {
                    break;
                }
            }
        }
        if state.lock().await.router.staged_len("cpu-fast") >= DRAIN_RATE {
            grew = true;
            break;
        }
    }
    assert!(
        grew,
        "adaptive window never grew to the drain rate under sustained consumption"
    );

    // The idle miner consumed nothing, so its window is still the floor.
    assert_eq!(
        state.lock().await.router.staged_len("cpu-idle"),
        FLOOR,
        "idle miner should stay pinned at the buffer_depth floor"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

#[tokio::test]
async fn feeder_broadcasts_set_target_once_per_difficulty() {
    use quip_proto::v1::{coord_msg, CoordMsg};

    let chain = Arc::new(FakeChain::new(ising_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let (tx, mut rx) = mpsc::channel::<Result<CoordMsg, tonic::Status>>(16);
    state.lock().await.register_outbound("cpu-0", tx);

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        FeederParams {
            miner_account: [0u8; 32],
            buffer_depth: 4,
            poll_interval: Duration::from_millis(50),
        },
        stop_rx,
    ));

    // First poll pushes the current difficulty to the live miner.
    let msg = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("no SetTarget within 2s")
        .expect("outbound channel closed")
        .expect("status error");
    let snap = ising_snapshot();
    assert!(
        matches!(&msg.msg, Some(coord_msg::Msg::SetTarget(_))),
        "first outbound message should be SetTarget"
    );
    if let Some(coord_msg::Msg::SetTarget(t)) = msg.msg {
        assert_eq!(t.max_energy_milli, snap.max_energy_milli);
        assert_eq!(t.min_solutions, snap.min_solutions);
        assert_eq!(t.min_diversity_milli, snap.min_diversity_milli);
    }

    // Unchanged difficulty across further polls must not re-broadcast.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        matches!(rx.try_recv(), Err(mpsc::error::TryRecvError::Empty)),
        "unchanged difficulty must not re-broadcast SetTarget"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}
