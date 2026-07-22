use quip_mock_coordinator::driver::drive_miner;
use quip_proto::v1::RejectReason;

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

    let name = if cfg!(windows) {
        "quip-mock-miner.exe"
    } else {
        "quip-mock-miner"
    };
    let mut p = std::env::current_exe().expect("test exe path");
    p.pop(); // deps/
    p.pop(); // <profile>/
    p.push(name);
    p.to_string_lossy().into_owned()
}

#[tokio::test]
async fn mock_miner_passes_conformance() {
    let miner = miner_bin();
    let socket = format!(
        "/tmp/quip-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;

    assert!(report.handshake_ok, "handshake failed: {report:?}");
    assert!(
        report.ready_received,
        "Ready must arrive after Configure: {report:?}"
    );
    assert!(
        !report.job_request_credits.is_empty(),
        "expected at least one JobRequest{{credits}}: {report:?}"
    );
    assert!(
        report.job_request_credits.iter().all(|&c| c > 0),
        "JobRequest credits must be > 0: {report:?}"
    );
    assert_eq!(
        report.result_job_ids.len(),
        2,
        "expected 2 Results: {report:?}"
    );
    assert!(
        report.result_job_ids.iter().any(|id| id == b"job-1"),
        "missing Result for job-1: {report:?}"
    );
    assert!(
        report.result_job_ids.iter().any(|id| id == b"job-2"),
        "missing Result for job-2: {report:?}"
    );
    assert!(
        report.has_reject(b"job-bad-h", RejectReason::Malformed),
        "missing per-job_id Reject MALFORMED for job-bad-h: {report:?}"
    );
    assert!(
        report.has_reject(b"job-bad-j", RejectReason::Malformed),
        "missing per-job_id Reject MALFORMED for job-bad-j: {report:?}"
    );
    assert!(
        report.has_reject(b"job-gate", RejectReason::UnsupportedKind),
        "missing per-job_id Reject UNSUPPORTED_KIND for job-gate: {report:?}"
    );
    assert!(
        report.has_reject(b"job-old", RejectReason::Expired),
        "missing per-job_id Reject EXPIRED for job-old: {report:?}"
    );
    assert!(
        report.cancel_acked,
        "Cancel must be acknowledged via Status: {report:?}"
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected: {report:?}");
    assert!(
        report.is_conformant(),
        "full DriverReport must pass: {report:?}"
    );
}
