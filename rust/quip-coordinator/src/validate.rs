//! Result revalidation via foundation `scoring` + strict energy tie-break.
//!
//! Uses golden-matched `quip_protocol::scoring`. External `quantum-validation`
//! (when available) must match the same golden vectors.

use quip_proto::v1::{ising_problem, IsingProblem, QualityGates, Solution};
use quip_protocol::scoring::{energy_milli, set_diversity};
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
/// A solution is "valid" when `energy_milli < gates.min_energy_milli` (strict).
/// Set is accepted when `n_valid >= min_solutions` and
/// `diversity_milli >= min_diversity_milli`.
pub fn validate_result(
    problem: &IsingProblem,
    solutions: &[Solution],
    gates: &QualityGates,
    // Topology edges used when the problem references `topology_hash`.
    topology_edges: &[(u32, u32)],
) -> Validated {
    let h_milli = decode_i32_le(&problem.h_milli_le32).unwrap_or_default();
    let j_milli = decode_i32_le(&problem.j_milli_le32).unwrap_or_default();
    let h: Vec<f64> = h_milli.iter().map(|&v| v as f64 / 1000.0).collect();
    let j: Vec<f64> = j_milli.iter().map(|&v| v as f64 / 1000.0).collect();
    let edges = resolve_edges(problem, topology_edges);

    let mut spin_sets: Vec<Vec<i8>> = Vec::new();
    let mut best: Option<i64> = None;
    let mut n_valid = 0u32;

    for sol in solutions {
        let spins = match decode_spins(&sol.spins_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let e = energy_milli(&spins, &h, &j, &edges);
        if e < gates.min_energy_milli {
            n_valid += 1;
            best = Some(match best {
                Some(b) if b <= e => b,
                _ => e,
            });
        }
        spin_sets.push(spins);
    }

    let diversity = set_diversity(&spin_sets);
    // Truncation toward zero, same convention as energy milli.
    let diversity_milli = (diversity * 1000.0) as u32;

    let accepted = n_valid >= gates.min_solutions && diversity_milli >= gates.min_diversity_milli;

    Validated {
        best_energy_milli: best.unwrap_or(i64::MAX),
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

fn resolve_edges(problem: &IsingProblem, topology_edges: &[(u32, u32)]) -> Vec<(usize, usize)> {
    match &problem.graph {
        Some(ising_problem::Graph::Edges(e)) => {
            e.u.iter()
                .zip(&e.v)
                .map(|(&u, &v)| (u as usize, v as usize))
                .collect()
        }
        Some(ising_problem::Graph::TopologyHash(_)) => topology_edges
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_protocol::wire::encode_i32_le;
    use quip_protocol::wire::encode_spins;

    #[test]
    fn golden_energy_and_strict_tiebreak() {
        // E = 1*1 + (-0.5)*(-1) + 2.0*1*(-1) = 1 + 0.5 - 2 = -0.5 → -500 milli
        // (matches foundation scoring unit test / golden energy convention)
        let problem = IsingProblem {
            graph: Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                u: vec![0],
                v: vec![1],
            })),
            h_milli_le32: encode_i32_le(&[1000, -500]),
            j_milli_le32: encode_i32_le(&[2000]),
            num_reads: 1,
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
        let v = validate_result(&problem, &[sol], &gates, &[]);
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
        let v = validate_result(&problem, &[sol], &gates, &[]);
        assert!(!v.accepted);
        assert_eq!(v.n_valid, 0);
    }
}
