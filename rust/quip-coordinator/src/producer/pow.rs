//! Derive PoW `Job`s from mining snapshots via golden-pinned ChaCha8 + derive_nonce.

use crate::chain::snapshot::MiningSnapshot;
use quip_proto::v1::{ising_problem, IsingProblem, Job, JobKind, Provenance, QualityGates};
use quip_protocol::chacha8::draw_ising_milli;
use quip_protocol::derive::derive_nonce;
use quip_protocol::wire::encode_i32_le;

/// Build an `ISING_SAMPLE` PoW job from a snapshot, miner account, and salt.
///
/// Uses `derive_nonce` + `draw_ising_milli` (golden-pinned). Graph is a
/// `topology_hash` reference; gates come from the snapshot difficulty.
pub fn derive_pow_job(
    snap: &MiningSnapshot,
    miner_account: [u8; 32],
    salt: [u8; 32],
    generation: u64,
    deadline_ms: u64,
) -> Job {
    let nonce = derive_nonce(snap.last_proof_block_hash, miner_account, salt);
    let (h, j) = draw_ising_milli(
        nonce,
        snap.nodes.len(),
        snap.edges.len(),
        &snap.allowed_h_milli,
        &snap.allowed_j_milli,
    );
    Job {
        job_id: nonce.to_vec(),
        kind: JobKind::IsingSample as i32,
        generation,
        deadline_ms,
        ising: Some(IsingProblem {
            graph: Some(ising_problem::Graph::TopologyHash(
                snap.topology_hash.clone(),
            )),
            h_milli_le32: encode_i32_le(&h),
            j_milli_le32: encode_i32_le(&j),
            num_reads: 0,
            gates: Some(QualityGates {
                min_energy_milli: snap.max_energy_milli,
                min_diversity_milli: snap.min_diversity_milli,
                min_solutions: snap.min_solutions,
            }),
        }),
        provenance: Some(Provenance {
            is_pow: true,
            order_id: vec![],
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::{ising_problem, JobKind};
    use quip_protocol::wire::decode_i32_le;

    fn fixture() -> MiningSnapshot {
        MiningSnapshot {
            last_proof_block_hash: [7u8; 32],
            topology_hash: vec![9u8; 32],
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            min_solutions: 5,
            max_energy_milli: -14_000_000,
            min_diversity_milli: 200,
            block_number: 42,
        }
    }

    #[test]
    fn derives_job_with_correct_shape() {
        let snap = fixture();
        let job = derive_pow_job(&snap, [1u8; 32], [2u8; 32], 42, 9999);
        assert_eq!(job.kind, JobKind::IsingSample as i32);
        assert_eq!(job.generation, 42);
        assert!(job.provenance.as_ref().unwrap().is_pow);
        let ising = job.ising.unwrap();
        assert!(matches!(
            ising.graph,
            Some(ising_problem::Graph::TopologyHash(_))
        ));
        let h = decode_i32_le(&ising.h_milli_le32).unwrap();
        let j = decode_i32_le(&ising.j_milli_le32).unwrap();
        assert_eq!(h.len(), 4);
        assert_eq!(j.len(), 4);
        assert!(h.iter().all(|v| [-1000, 0, 1000].contains(v)));
        assert!(j.iter().all(|v| [-1000, 1000].contains(v)));
        let gates = ising.gates.unwrap();
        assert_eq!(gates.min_solutions, 5);
        assert_eq!(gates.min_energy_milli, -14_000_000);
    }

    #[test]
    fn derivation_is_deterministic_for_same_inputs() {
        let snap = fixture();
        let a = derive_pow_job(&snap, [1u8; 32], [2u8; 32], 1, 1);
        let b = derive_pow_job(&snap, [1u8; 32], [2u8; 32], 1, 1);
        assert_eq!(a.ising.unwrap().h_milli_le32, b.ising.unwrap().h_milli_le32);
    }
}
