//! Session-cached topology and deterministic topology hash.
//!
//! > **CONFIRM crate API:** the exact chain `topology_hash` construction.
//! > This local hash must equal the chain's H256, or PoW jobs reference a
//! > hash miners cannot resolve. Confirm against `quip-protocol-rs`.

/// A hardware (or logical) graph identified by a deterministic hash.
#[derive(Debug, Clone)]
pub struct Topology {
    pub hash: Vec<u8>,
    pub nodes: Vec<u32>,
    /// Parallel edge endpoint arrays `(u, v)`.
    pub edges: (Vec<u32>, Vec<u32>),
}

impl Topology {
    pub fn from_nodes_edges(nodes: Vec<u32>, edges: Vec<(u32, u32)>) -> Self {
        let hash = topology_hash(&nodes, &edges);
        let (u, v): (Vec<u32>, Vec<u32>) = edges.into_iter().unzip();
        Self {
            hash,
            nodes,
            edges: (u, v),
        }
    }

    pub fn edge_pairs(&self) -> Vec<(u32, u32)> {
        self.edges
            .0
            .iter()
            .zip(&self.edges.1)
            .map(|(&u, &v)| (u, v))
            .collect()
    }

    pub fn to_proto(&self) -> quip_proto::v1::Topology {
        quip_proto::v1::Topology {
            hash: self.hash.clone(),
            nodes: self.nodes.clone(),
            edges: Some(quip_proto::v1::EdgeList {
                u: self.edges.0.clone(),
                v: self.edges.1.clone(),
            }),
        }
    }
}

/// Deterministic BLAKE3 hash of nodes then edges (u,v pairs in order).
///
/// Placeholder until `quip-protocol-rs` confirms the canonical chain hash.
pub fn topology_hash(nodes: &[u32], edges: &[(u32, u32)]) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    for n in nodes {
        h.update(&n.to_le_bytes());
    }
    for (u, v) in edges {
        h.update(&u.to_le_bytes());
        h.update(&v.to_le_bytes());
    }
    h.finalize().as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_and_order_sensitive() {
        let h1 = topology_hash(&[0, 1, 2], &[(0, 1), (1, 2)]);
        let h2 = topology_hash(&[0, 1, 2], &[(0, 1), (1, 2)]);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 32);
        assert_ne!(h1, topology_hash(&[0, 1, 2], &[(1, 2), (0, 1)]));
    }
}
