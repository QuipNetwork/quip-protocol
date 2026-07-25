//! Runtime lifecycle: run_runtime binds the server, supervises a real
//! quip-mock-miner, and shuts down cleanly on signal. Job production (feeding
//! work, solve→submit) is exercised by tests/e2e.rs; this covers the process
//! wiring + graceful-shutdown seam.

use quip_coordinator::chain::{FakeChain, MiningSnapshot};
use quip_coordinator::config::LaunchEntry;
use quip_coordinator::runtime::{run_runtime, RuntimeParams};
use quip_coordinator::session::CoordinatorState;
use quip_coordinator::supervisor::BackoffPolicy;
use quip_proto::v1::Configure;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{oneshot, Mutex};

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
