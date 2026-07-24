//! Protocol conformance: spawn SA and Gibbs miners against quip-mock-coordinator.
//!
//! Metal GPU tests — macOS only (file is empty on Linux CI).

#![cfg(target_os = "macos")]

use quip_mock_coordinator::driver::drive_miner;
use std::process::Command;

/// Cross-package binary path (deps/ → profile/ → bin).
fn profile_bin(name: &str) -> String {
    let name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let mut p = std::env::current_exe().expect("test exe path");
    p.pop(); // deps/
    p.pop(); // <profile>/
    p.push(&name);
    p.to_string_lossy().into_owned()
}

fn ensure_built(package_bins: &[&str]) {
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-miner-metal"])
        .status()
        .expect("cargo build quip-miner-metal");
    assert!(status.success(), "failed to build quip-miner-metal");
    for b in package_bins {
        assert!(
            std::path::Path::new(&profile_bin(b)).exists(),
            "missing binary {b} at {}",
            profile_bin(b)
        );
    }
}

#[tokio::test]
async fn quip_metal_sa_passes_conformance() {
    ensure_built(&["quip-metal-sa"]);
    let miner = profile_bin("quip-metal-sa");
    let socket = format!(
        "/tmp/quip-metal-sa-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "SA handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
    );
    assert!(
        report.has_reject(b"job-bad-h", quip_proto::v1::RejectReason::Malformed),
        "missing MALFORMED reject: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-old", quip_proto::v1::RejectReason::Expired),
        "missing EXPIRED reject: {:?}",
        report.rejects
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

#[tokio::test]
async fn quip_metal_gibbs_passes_conformance() {
    ensure_built(&["quip-metal-gibbs"]);
    let miner = profile_bin("quip-metal-gibbs");
    let socket = format!(
        "/tmp/quip-metal-gibbs-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "Gibbs handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
    );
    assert!(report.has_reject(b"job-bad-h", quip_proto::v1::RejectReason::Malformed));
    assert!(report.has_reject(b"job-old", quip_proto::v1::RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

#[test]
fn capabilities_and_version_and_check() {
    ensure_built(&["quip-metal-sa", "quip-metal-gibbs"]);

    for (bin, algo) in [("quip-metal-sa", "sa"), ("quip-metal-gibbs", "gibbs")] {
        let path = profile_bin(bin);

        let out = Command::new(&path).arg("--capabilities").output().unwrap();
        assert!(out.status.success(), "{bin} --capabilities failed");
        let s = String::from_utf8(out.stdout).unwrap();
        assert!(s.contains("\"backend\":\"metal\""), "{bin}: {s}");
        assert!(
            s.contains(&format!("\"algorithm\":\"{algo}\"")),
            "{bin}: {s}"
        );

        let out = Command::new(&path).arg("--version").output().unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8(out.stdout).unwrap().contains("protocol"));

        // --check opens the GPU and compiles kernels.
        let status = Command::new(&path)
            .arg("--check")
            .arg("--device")
            .arg("0")
            .status();
        assert!(
            status.unwrap().success(),
            "{bin} --check must succeed when a GPU is present"
        );
    }
}
