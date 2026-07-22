//! Shared helpers for quip-miner-cuda integration tests.

use std::process::Command;

/// Cross-package binary path (deps/ → profile/ → bin).
pub fn profile_bin(name: &str) -> String {
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

pub fn ensure_built(package_bins: &[&str]) {
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-miner-cuda"])
        .status()
        .expect("cargo build quip-miner-cuda");
    assert!(status.success(), "failed to build quip-miner-cuda");
    for b in package_bins {
        assert!(
            std::path::Path::new(&profile_bin(b)).exists(),
            "missing binary {b} at {}",
            profile_bin(b)
        );
    }
}

/// Whether a usable CUDA device is present. Probes by running the built
/// `quip-cuda-sa --check` in a subprocess: a missing driver or NVRTC library
/// panics inside cudarc, crashing the child (captured here, not this test), so
/// a non-zero exit means "no usable GPU". CI runners have no GPU, so hardware
/// tests skip there while still running on GPU machines.
pub fn cuda_available() -> bool {
    ensure_built(&["quip-cuda-sa"]);
    Command::new(profile_bin("quip-cuda-sa"))
        .args(["--check", "--device", "0"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
