//! Host energy_milli golden vectors + live Metal SA/Gibbs consensus scoring.
//!
//! Requires a Metal device (Apple Silicon). There is no GPU energy kernel
//! (MSL has no `double`); production scores always use host `energy_milli`.
//!
//! Run with `cargo test -p quip-miner-metal`.

#![cfg(target_os = "macos")]

use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::sampler::sample_ising;
use quip_miner_metal::{Algorithm, IsingGraph, SampleParams};
use quip_protocol::scoring::energy_milli;
use quip_protocol::wire::{decode_spins, encode_spins};
use serde_json::Value;
use std::fs;

fn golden() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/golden_vectors.json"
    );
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

/// Open device 0 for each test. Each test owns its own `MetalDevice` rather
/// than sharing one from a `static OnceLock` — a simple per-test GPU context.
fn open_device() -> MetalDevice {
    MetalDevice::open(0).unwrap_or_else(|e| {
        panic!("Metal device 0 required for golden_parity tests: {e}");
    })
}

/// Golden Ising cases: host energy_milli bit-exact vs vectors (no GPU energy).
#[test]
fn host_energy_matches_golden_vectors() {
    for case in golden()["energy"].as_array().unwrap() {
        let spins: Vec<i8> = case["spins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect();
        let h: Vec<f64> = case["h_milli"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as f64 / 1000.0)
            .collect();
        let j: Vec<f64> = case["j_milli"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as f64 / 1000.0)
            .collect();
        let edges: Vec<(usize, usize)> = case["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e[0].as_u64().unwrap() as usize,
                    e[1].as_u64().unwrap() as usize,
                )
            })
            .collect();

        let expected = case["energy_milli"].as_i64().unwrap();

        assert_eq!(
            energy_milli(&spins, &h, &j, &edges),
            expected,
            "host energy_milli mismatch"
        );

        // Wire round-trip must preserve scoring.
        let bytes = encode_spins(&spins);
        let decoded = decode_spins(&bytes).unwrap();
        assert_eq!(energy_milli(&decoded, &h, &j, &edges), expected);
    }
}

/// Truncation cases from golden vectors (toward zero, not round-to-nearest).
#[test]
fn truncation_matches_golden() {
    for case in golden()["truncation"].as_array().unwrap() {
        let energy = case["energy"].as_f64().unwrap();
        assert_eq!(
            (energy * 1000.0) as i64,
            case["energy_milli"].as_i64().unwrap()
        );
    }
}

/// Live SA/Gibbs sample: every returned energy equals consensus scoring.
#[test]
fn live_sample_energies_match_energy_milli() {
    let dev = open_device();
    let graph = IsingGraph::new(
        vec![1.0, -0.5, 0.0, 0.25],
        vec![1.0, -1.0, -1.0, 1.0],
        vec![(0, 1), (1, 2), (2, 3), (0, 3)],
    );
    let params = SampleParams {
        num_reads: 16,
        num_sweeps: 64,
        seed: 12345,
        ..Default::default()
    };

    for algo in [Algorithm::Sa, Algorithm::Gibbs] {
        let results = sample_ising(&dev, &graph, &params, algo).expect("sample");
        assert_eq!(results.len(), 16);
        for r in &results {
            let expected = energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges);
            assert_eq!(
                r.energy_milli, expected,
                "{algo:?} reported energy_milli {} != consensus {}",
                r.energy_milli, expected
            );
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
            let bytes = encode_spins(&r.spins);
            assert_eq!(decode_spins(&bytes).unwrap(), r.spins);
        }
    }
}

/// Positive-sign convention pin (host consensus only — no GPU energy kernel).
#[test]
fn positive_sign_convention_host() {
    // spins [+1,-1]; h=[1.0, -0.5]; edge (0,1) J=2.0
    // E = 1 + 0.5 - 2.0 = -0.5 → -500 milli
    let spins = vec![1i8, -1i8];
    let h = vec![1.0, -0.5];
    let j = vec![2.0];
    let edges = vec![(0usize, 1usize)];
    assert_eq!(energy_milli(&spins, &h, &j, &edges), -500);
}

/// SA finds ferro ground state (sanity that the kernel actually anneals).
#[test]
fn sa_finds_ground_state_on_ferro() {
    let dev = open_device();
    let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
    let params = SampleParams {
        num_reads: 16,
        num_sweeps: 128,
        seed: 42,
        ..Default::default()
    };
    let results = sample_ising(&dev, &graph, &params, Algorithm::Sa).expect("sa");
    assert!(
        results.iter().any(|r| r.energy_milli == -1000),
        "SA failed to find ferro ground: {:?}",
        results.iter().map(|r| r.energy_milli).collect::<Vec<_>>()
    );
}
