//! Protocol conformance: spawn SA and Gibbs miners against quip-mock-coordinator.

use quip_mock_coordinator::driver::drive_miner;
use quip_proto::v1::RejectReason;
use serial_test::serial;
use std::process::Command;

mod common;
use common::{cuda_available, ensure_built, profile_bin};

#[tokio::test]
#[serial]
async fn quip_cuda_sa_passes_conformance() {
    if !cuda_available() {
        eprintln!("SKIP quip_cuda_sa_passes_conformance: no usable CUDA device");
        return;
    }
    ensure_built(&["quip-cuda-sa"]);
    let miner = profile_bin("quip-cuda-sa");
    let socket = format!(
        "/tmp/quip-cuda-sa-conf-{}-{}.sock",
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
        report.result_job_ids().iter().any(|id| id == b"job-1"),
        "missing result for job-1: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-2"),
        "missing result for job-2: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-hash"),
        "missing result for topology-hash job-hash: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.has_reject(b"job-bad-h", RejectReason::Malformed),
        "missing MALFORMED reject for job-bad-h: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-gate", RejectReason::UnsupportedKind),
        "missing UNSUPPORTED_KIND reject for job-gate: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-old", RejectReason::Expired),
        "missing EXPIRED reject for job-old: {:?}",
        report.rejects
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

#[tokio::test]
#[serial]
async fn quip_cuda_gibbs_passes_conformance() {
    if !cuda_available() {
        eprintln!("SKIP quip_cuda_gibbs_passes_conformance: no usable CUDA device");
        return;
    }
    ensure_built(&["quip-cuda-gibbs"]);
    let miner = profile_bin("quip-cuda-gibbs");
    let socket = format!(
        "/tmp/quip-cuda-gibbs-conf-{}-{}.sock",
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
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-gate", RejectReason::UnsupportedKind));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

#[test]
#[serial]
fn capabilities_and_version_and_check() {
    ensure_built(&["quip-cuda-sa", "quip-cuda-gibbs"]);
    // --capabilities and --version are headless (no CUDA); --check needs a GPU.
    let gpu = cuda_available();

    for (bin, algo) in [("quip-cuda-sa", "sa"), ("quip-cuda-gibbs", "gibbs")] {
        let path = profile_bin(bin);

        let out = Command::new(&path).arg("--capabilities").output().unwrap();
        assert!(out.status.success(), "{bin} --capabilities failed");
        let s = String::from_utf8(out.stdout).unwrap();
        assert!(s.contains("\"backend\":\"cuda\""), "{bin}: {s}");
        assert!(
            s.contains(&format!("\"algorithm\":\"{algo}\"")),
            "{bin}: {s}"
        );

        let out = Command::new(&path).arg("--version").output().unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8(out.stdout).unwrap().contains("protocol"));

        // --check opens the GPU and compiles kernels; only meaningful with hardware.
        if gpu {
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
}
