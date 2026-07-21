use quip_mock_coordinator::driver::drive_miner;

/// Path to the sibling `quip-mock-miner` binary.
///
/// `CARGO_BIN_EXE_*` is only defined for binaries in the *same* package, so a
/// cross-package test derives the path from its own executable location
/// (`target/<profile>/deps/<test>` -> `target/<profile>/quip-mock-miner`).
///
/// A bare `cargo test -p quip-mock-coordinator` does not (re)build the sibling
/// miner crate, so build it here first to guarantee the binary matches current
/// source rather than a stale artifact. The nested `cargo build` is a no-op when
/// already current, and the outer test run has released the build lock by the
/// time tests execute.
fn miner_bin() -> String {
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-mock-miner"])
        .status()
        .expect("build quip-mock-miner");
    assert!(status.success(), "failed to build quip-mock-miner");

    let name = if cfg!(windows) { "quip-mock-miner.exe" } else { "quip-mock-miner" };
    let mut p = std::env::current_exe().expect("test exe path");
    p.pop(); // deps/
    p.pop(); // <profile>/
    p.push(name);
    p.to_string_lossy().into_owned()
}

#[tokio::test]
async fn mock_miner_passes_conformance() {
    let miner = miner_bin();
    let socket = format!("/tmp/quip-conf-{}.sock", std::process::id());
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "handshake failed");
    assert_eq!(report.results_received, 2, "expected 2 job results");
    assert!(report.rejects.contains(&(quip_proto::v1::RejectReason::Malformed as i32)));
    assert!(report.rejects.contains(&(quip_proto::v1::RejectReason::Expired as i32)));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}
