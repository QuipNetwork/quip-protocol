//! Consensus: GPU energy kernel + host energy_milli match golden vectors.
//!
//! Requires a CUDA device. Run with `cargo test -p quip-miner-cuda`.

use quip_miner_cuda::cuda_device::CudaDevice;
use quip_miner_cuda::sampler::{gpu_energy_milli, sample_ising};
use quip_miner_cuda::{Algorithm, IsingGraph, SampleParams};
use quip_protocol::scoring::energy_milli;
use quip_protocol::wire::{decode_spins, encode_spins};
use serde_json::Value;
use std::fs;
use std::sync::OnceLock;

mod common;
use common::cuda_available;

fn golden() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/golden_vectors.json"
    );
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn device() -> &'static CudaDevice {
    static DEV: OnceLock<CudaDevice> = OnceLock::new();
    DEV.get_or_init(|| {
        CudaDevice::open(0).unwrap_or_else(|e| {
            panic!("CUDA device 0 required for golden_parity tests: {e}");
        })
    })
}

/// Golden Ising cases: run the energy kernel on the GPU and compare bit-exact.
#[test]
fn gpu_energy_kernel_matches_golden_vectors() {
    if !cuda_available() {
        eprintln!("SKIP gpu_energy_kernel_matches_golden_vectors: no usable CUDA device");
        return;
    }
    let dev = device();
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

        // Host consensus scorer.
        assert_eq!(
            energy_milli(&spins, &h, &j, &edges),
            expected,
            "host energy_milli mismatch"
        );

        // GPU energy kernel on the golden spins / problem.
        let gpu_e = gpu_energy_milli(dev, &spins, &h, &j, &edges).expect("gpu energy");
        assert_eq!(
            gpu_e, expected,
            "GPU energy kernel must match golden energy_milli bit-exactly"
        );

        // Wire round-trip must preserve scoring.
        let bytes = encode_spins(&spins);
        let decoded = decode_spins(&bytes).unwrap();
        assert_eq!(energy_milli(&decoded, &h, &j, &edges), expected);
        assert_eq!(
            gpu_energy_milli(dev, &decoded, &h, &j, &edges).unwrap(),
            expected
        );
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
    if !cuda_available() {
        eprintln!("SKIP live_sample_energies_match_energy_milli: no usable CUDA device");
        return;
    }
    let dev = device();
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
        let results = sample_ising(dev, &graph, &params, algo).expect("sample");
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

            // GPU energy kernel agrees with host for each sample.
            let gpu_e = gpu_energy_milli(dev, &r.spins, &graph.h, &graph.j, &graph.edges).unwrap();
            assert_eq!(gpu_e, expected, "GPU energy vs host for {algo:?}");
        }
    }
}

/// Positive-sign convention pin via GPU energy kernel.
#[test]
fn positive_sign_convention_on_gpu() {
    if !cuda_available() {
        eprintln!("SKIP positive_sign_convention_on_gpu: no usable CUDA device");
        return;
    }
    let dev = device();
    // spins [+1,-1]; h=[1.0, -0.5]; edge (0,1) J=2.0
    // E = 1 + 0.5 - 2.0 = -0.5 → -500 milli
    let spins = vec![1i8, -1i8];
    let h = vec![1.0, -0.5];
    let j = vec![2.0];
    let edges = vec![(0usize, 1usize)];
    assert_eq!(energy_milli(&spins, &h, &j, &edges), -500);
    assert_eq!(gpu_energy_milli(dev, &spins, &h, &j, &edges).unwrap(), -500);
}

/// SA finds ferro ground state (sanity that the kernel actually anneals).
#[test]
fn sa_finds_ground_state_on_ferro() {
    if !cuda_available() {
        eprintln!("SKIP sa_finds_ground_state_on_ferro: no usable CUDA device");
        return;
    }
    let dev = device();
    let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
    let params = SampleParams {
        num_reads: 16,
        num_sweeps: 128,
        seed: 42,
        ..Default::default()
    };
    let results = sample_ising(dev, &graph, &params, Algorithm::Sa).expect("sa");
    assert!(
        results.iter().any(|r| r.energy_milli == -1000),
        "SA failed to find ferro ground: {:?}",
        results.iter().map(|r| r.energy_milli).collect::<Vec<_>>()
    );
}
