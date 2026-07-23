//! Result revalidation via `quantum_validation` (authoritative, matches the
//! pallet) with `quip_protocol::scoring` as a golden regression check.

use quantum_validation::{
    calculate_diversity, energy_of_solution, select_diverse, MilliValue, ValidationError,
};
use quip_proto::v1::{ising_problem, IsingProblem, QualityGates, Solution};
use quip_protocol::scoring::{
    energy_milli as golden_energy_milli, set_diversity as golden_diversity,
};
use quip_protocol::wire::{decode_i32_le, decode_spins};

#[derive(Debug, Clone, PartialEq)]
pub struct Validated {
    pub best_energy_milli: i64,
    pub diversity_milli: u32,
    pub n_valid: u32,
    pub accepted: bool,
}

/// Recompute energies, apply quality gates, return validation summary.
///
/// A solution is energy-valid when `energy < gates.min_energy_milli` (strict).
/// The `min_energy_milli` field on wire gates carries the chain's
/// `max_energy_milli` ceiling (see `derive_pow_job`).
///
/// Set is accepted when `n_valid >= min_solutions` and
/// `diversity_milli >= min_diversity_milli`. Diversity uses the crate's
/// half-up `round_div_u64` (not golden truncation).
///
/// `topology_nodes` / `topology_edges` are used when the problem references a
/// topology hash; edge endpoints are **node ids** resolved via the nodes
/// slice (matching `energy_of_solution`), not bare spin indices.
pub fn validate_result(
    problem: &IsingProblem,
    solutions: &[Solution],
    gates: &QualityGates,
    topology_nodes: &[u32],
    topology_edges: &[(u32, u32)],
) -> Validated {
    let h_milli = decode_i32_le(&problem.h_milli_le32).unwrap_or_default();
    let j_milli = decode_i32_le(&problem.j_milli_le32).unwrap_or_default();
    let (nodes, edges) = resolve_graph(problem, &h_milli, topology_nodes, topology_edges);

    let mut spin_sets: Vec<Vec<i8>> = Vec::new();
    let mut energies: Vec<i64> = Vec::new();

    for sol in solutions {
        let spins = match decode_spins(&sol.spins_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        match energy_of_solution(&spins, &h_milli, &edges, &j_milli, &nodes) {
            Ok(e) => {
                energies.push(e);
                spin_sets.push(spins);
            }
            Err(ValidationError::InvalidSpinValue { .. })
            | Err(ValidationError::SolutionLengthMismatch { .. }) => continue,
            Err(_) => continue,
        }
    }

    // Golden regression: energy must match for index-aligned graphs.
    debug_assert_energies_match(&spin_sets, &h_milli, &j_milli, &edges, &nodes, &energies);

    let energy_valid_indices: Vec<usize> = energies
        .iter()
        .enumerate()
        .filter_map(|(i, &e)| (e < gates.min_energy_milli).then_some(i))
        .collect();
    let n_valid = energy_valid_indices.len() as u32;

    let (best_energy_milli, diversity_milli) = if energy_valid_indices.is_empty() {
        (i64::MAX, 0u32)
    } else {
        // Mirror pallet: select a diverse subset of energy-valid solutions,
        // then score diversity on that subset; best energy over selected.
        let energy_valid: Vec<&[i8]> = energy_valid_indices
            .iter()
            .map(|&i| spin_sets[i].as_slice())
            .collect();
        let target = energy_valid.len().min(gates.min_solutions.max(1) as usize);
        let selected = select_diverse(&energy_valid, target)
            .unwrap_or_else(|_| (0..energy_valid.len()).collect());
        let selected_spins: Vec<&[i8]> = selected.iter().map(|&i| energy_valid[i]).collect();
        let diversity = calculate_diversity(&selected_spins).unwrap_or(0);
        let best = selected
            .iter()
            .map(|&i| energies[energy_valid_indices[i]])
            .min()
            .unwrap_or(i64::MAX);
        (best, diversity)
    };

    // Golden diversity is truncation; accept-path uses the crate value.
    let _golden_div_milli = {
        let valid_spins: Vec<Vec<i8>> = energy_valid_indices
            .iter()
            .map(|&i| spin_sets[i].clone())
            .collect();
        (golden_diversity(&valid_spins) * 1000.0) as u32
    };

    let accepted = n_valid >= gates.min_solutions && diversity_milli >= gates.min_diversity_milli;

    Validated {
        best_energy_milli,
        diversity_milli,
        n_valid,
        accepted,
    }
}

/// Strict less-than tie-break: first accepted holds on equal energy.
pub fn beats_current(candidate_milli: i64, current_best_milli: Option<i64>) -> bool {
    match current_best_milli {
        None => true,
        Some(cur) => candidate_milli < cur,
    }
}

fn resolve_graph(
    problem: &IsingProblem,
    h_milli: &[MilliValue],
    topology_nodes: &[u32],
    topology_edges: &[(u32, u32)],
) -> (Vec<u32>, Vec<(u32, u32)>) {
    match &problem.graph {
        Some(ising_problem::Graph::Edges(e)) => {
            let edges: Vec<(u32, u32)> = e.u.iter().zip(&e.v).map(|(&u, &v)| (u, v)).collect();
            // Inline edge lists treat endpoints as spin indices; nodes are
            // the contiguous positions 0..len(h).
            let nodes: Vec<u32> = (0..h_milli.len() as u32).collect();
            (nodes, edges)
        }
        Some(ising_problem::Graph::TopologyHash(_)) => {
            (topology_nodes.to_vec(), topology_edges.to_vec())
        }
        None => ((0..h_milli.len() as u32).collect(), Vec::new()),
    }
}

#[cfg(debug_assertions)]
fn debug_assert_energies_match(
    spin_sets: &[Vec<i8>],
    h_milli: &[MilliValue],
    j_milli: &[MilliValue],
    edges: &[(u32, u32)],
    nodes: &[u32],
    energies: &[i64],
) {
    // Golden scoring treats edges as spin indices. Only compare when the
    // nodes slice is the identity map 0..n (inline EdgeList graphs).
    let identity =
        nodes.len() == h_milli.len() && nodes.iter().enumerate().all(|(i, &n)| n as usize == i);
    if !identity {
        return;
    }
    let h: Vec<f64> = h_milli.iter().map(|&v| v as f64 / 1000.0).collect();
    let j: Vec<f64> = j_milli.iter().map(|&v| v as f64 / 1000.0).collect();
    let idx_edges: Vec<(usize, usize)> = edges
        .iter()
        .map(|&(u, v)| (u as usize, v as usize))
        .collect();
    for (spins, &e) in spin_sets.iter().zip(energies) {
        let g = golden_energy_milli(spins, &h, &j, &idx_edges);
        debug_assert_eq!(
            g, e,
            "quantum_validation energy {e} diverged from golden {g}"
        );
    }
}

#[cfg(not(debug_assertions))]
fn debug_assert_energies_match(
    _spin_sets: &[Vec<i8>],
    _h_milli: &[MilliValue],
    _j_milli: &[MilliValue],
    _edges: &[(u32, u32)],
    _nodes: &[u32],
    _energies: &[i64],
) {
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_protocol::wire::encode_i32_le;
    use quip_protocol::wire::encode_spins;

    #[test]
    fn golden_energy_and_strict_tiebreak() {
        // E = 1*1 + (-0.5)*(-1) + 2.0*1*(-1) = 1 + 0.5 - 2 = -0.5 → -500 milli
        let problem = IsingProblem {
            graph: Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                u: vec![0],
                v: vec![1],
            })),
            h_milli_le32: encode_i32_le(&[1000, -500]),
            j_milli_le32: encode_i32_le(&[2000]),
            num_reads: 1,
            num_sweeps: 0,
            anneal_time_us: 0,
            gates: None,
        };
        let sol = Solution {
            spins_bytes: encode_spins(&[1, -1]),
            energy_milli: -500,
        };
        let gates = QualityGates {
            min_energy_milli: 0, // accept anything < 0
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let v = validate_result(&problem, &[sol], &gates, &[], &[]);
        assert_eq!(v.best_energy_milli, -500);
        assert!(v.accepted);
        assert_eq!(v.n_valid, 1);

        assert!(beats_current(-501, Some(-500)));
        assert!(!beats_current(-500, Some(-500))); // strict <
        assert!(!beats_current(-499, Some(-500)));
        assert!(beats_current(-1, None));
    }

    #[test]
    fn rejects_when_below_min_solutions() {
        let problem = IsingProblem {
            graph: None,
            h_milli_le32: encode_i32_le(&[1000]),
            j_milli_le32: vec![],
            num_reads: 1,
            num_sweeps: 0,
            anneal_time_us: 0,
            gates: None,
        };
        let sol = Solution {
            spins_bytes: encode_spins(&[1]),
            energy_milli: 1000,
        };
        let gates = QualityGates {
            min_energy_milli: 0, // 1000 is not < 0
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let v = validate_result(&problem, &[sol], &gates, &[], &[]);
        assert!(!v.accepted);
        assert_eq!(v.n_valid, 0);
    }

    #[test]
    fn topology_edges_use_node_ids_not_spin_indices() {
        // nodes [10, 20]; edge (10, 20); h/j aligned to node positions.
        // spins [1, -1] → E = 1000*1 + (-500)*(-1) + 2000*1*(-1) = -500
        let problem = IsingProblem {
            graph: Some(ising_problem::Graph::TopologyHash(vec![0u8; 32])),
            h_milli_le32: encode_i32_le(&[1000, -500]),
            j_milli_le32: encode_i32_le(&[2000]),
            num_reads: 1,
            num_sweeps: 0,
            anneal_time_us: 0,
            gates: None,
        };
        let sol = Solution {
            spins_bytes: encode_spins(&[1, -1]),
            energy_milli: -500,
        };
        let gates = QualityGates {
            min_energy_milli: 0,
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let v = validate_result(&problem, &[sol], &gates, &[10, 20], &[(10, 20)]);
        assert_eq!(v.best_energy_milli, -500);
        assert!(v.accepted);
    }

    #[test]
    fn diversity_uses_quantum_validation_rounding() {
        // Two fully opposite solutions on 3 spins: hamming=3, symmetric=0
        // under flip → diversity 0. Two solutions that differ in 1 of 2:
        // dist=1, pairs=1, n=2 → round(1000*1/(1*2)) = 500.
        let problem = IsingProblem {
            graph: None,
            h_milli_le32: encode_i32_le(&[0, 0]),
            j_milli_le32: vec![],
            num_reads: 2,
            num_sweeps: 0,
            anneal_time_us: 0,
            gates: None,
        };
        let sols = [
            Solution {
                spins_bytes: encode_spins(&[1, 1]),
                energy_milli: 0,
            },
            Solution {
                spins_bytes: encode_spins(&[1, -1]),
                energy_milli: 0,
            },
        ];
        let gates = QualityGates {
            min_energy_milli: 1, // 0 < 1 → both valid
            min_diversity_milli: 0,
            min_solutions: 2,
        };
        let v = validate_result(&problem, &sols, &gates, &[], &[]);
        assert_eq!(v.n_valid, 2);
        assert_eq!(v.diversity_milli, 500);
        assert!(v.accepted);
    }
}
