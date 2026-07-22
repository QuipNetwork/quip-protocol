//! Mining snapshot types and head-change generation key.

use crate::topology::{topology_hash_sets, DEFAULT_SPIN_SET};

/// Chain mining snapshot consumed by the PoW producer.
#[derive(Debug, Clone)]
pub struct MiningSnapshot {
    pub last_proof_block_hash: [u8; 32],
    pub topology_hash: Vec<u8>,
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub allowed_h_milli: Vec<i32>,
    pub allowed_j_milli: Vec<i32>,
    /// Allowed spin milli values (Set); default binary `±1000` when empty.
    pub allowed_spin_milli: Vec<i32>,
    pub min_solutions: u32,
    /// Energy ceiling: solutions must be strictly below this (milli).
    pub max_energy_milli: i64,
    pub min_diversity_milli: u32,
    pub block_number: u64,
}

/// Change-detection key over the round-identifying snapshot fields.
///
/// Mirrors `substrate/event_manager.py:_state_key`: a new key means a new
/// PoW generation. Pure blake3 over identifying fields (groundable).
pub fn head_state_key(snap: &MiningSnapshot) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&snap.last_proof_block_hash);
    h.update(&snap.topology_hash);
    h.update(&snap.min_solutions.to_le_bytes());
    h.update(&snap.max_energy_milli.to_le_bytes());
    h.update(&snap.min_diversity_milli.to_le_bytes());
    h.update(&snap.block_number.to_le_bytes());
    *h.finalize().as_bytes()
}

/// Build topology hash for a snapshot's graph when not provided by the chain.
pub fn snapshot_topology_hash(snap: &MiningSnapshot) -> Vec<u8> {
    if snap.topology_hash.len() == 32 {
        snap.topology_hash.clone()
    } else {
        let spin = if snap.allowed_spin_milli.is_empty() {
            DEFAULT_SPIN_SET.as_slice()
        } else {
            snap.allowed_spin_milli.as_slice()
        };
        topology_hash_sets(
            &snap.nodes,
            &snap.edges,
            &snap.allowed_h_milli,
            &snap.allowed_j_milli,
            spin,
        )
        .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> MiningSnapshot {
        MiningSnapshot {
            last_proof_block_hash: [7u8; 32],
            topology_hash: vec![9u8; 32],
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 5,
            max_energy_milli: -14_000_000,
            min_diversity_milli: 200,
            block_number: 42,
        }
    }

    #[test]
    fn head_key_stable_for_same_snapshot() {
        let a = head_state_key(&sample());
        let b = head_state_key(&sample());
        assert_eq!(a, b);
    }

    #[test]
    fn head_key_changes_when_last_proof_changes() {
        let mut s = sample();
        let k1 = head_state_key(&s);
        s.last_proof_block_hash = [8u8; 32];
        let k2 = head_state_key(&s);
        assert_ne!(k1, k2);
    }
}
