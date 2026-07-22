//! Consensus: reported energies must match `energy_milli` and golden vectors.

use quip_miner_cpu::sampler_core::{
    sample_ising, Algorithm, IsingGraph, SampleParams, SamplerResult,
};
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

/// Golden energy cases: scoring through the same path miners use for Results.
#[test]
fn sampler_energy_matches_golden_vectors() {
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
        let graph = IsingGraph::new(h.clone(), j.clone(), edges.clone());

        // Direct scoring (what Result.energy_milli is set from).
        assert_eq!(energy_milli(&spins, &h, &j, &edges), expected);

        // Via the sampler result path (encode spins as on the wire, re-score).
        let scored = SamplerResult {
            spins: spins.clone(),
            energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
        };
        assert_eq!(scored.energy_milli, expected);

        // Wire round-trip must preserve spin values used for scoring.
        let bytes = encode_spins(&spins);
        let decoded = decode_spins(&bytes).unwrap();
        assert_eq!(
            energy_milli(&decoded, &h, &j, &edges),
            expected,
            "wire spin encoding must not change energy_milli"
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
        let results = sample_ising(&graph, &params, algo);
        assert_eq!(results.len(), 16);
        for r in &results {
            let expected = energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges);
            assert_eq!(
                r.energy_milli, expected,
                "{algo:?} reported energy_milli {} != consensus {}",
                r.energy_milli, expected
            );
            // Spins must be valid wire values.
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
            let bytes = encode_spins(&r.spins);
            assert_eq!(decode_spins(&bytes).unwrap(), r.spins);
        }
    }
}

/// Positive-sign convention pin from the SDK unit test, re-checked via graph path.
#[test]
fn positive_sign_convention() {
    // spins [+1,-1]; h=[1.0, -0.5]; edge (0,1) J=2.0
    // E = 1 + 0.5 - 2.0 = -0.5 → -500 milli
    let graph = IsingGraph::new(vec![1.0, -0.5], vec![2.0], vec![(0, 1)]);
    let spins = vec![1i8, -1i8];
    assert_eq!(energy_milli(&spins, &graph.h, &graph.j, &graph.edges), -500);
}
