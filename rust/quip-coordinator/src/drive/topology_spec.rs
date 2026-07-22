//! Drive-mode topology provider: parses a JSON topology-spec file into a
//! `Topology` + synthetic `MiningSnapshot` (chain-only fields defaulted).
//!
//! Sub-project #1 ships this file-spec provider only. Topology-by-hash from
//! the network is sub-project #2, behind the same seam (a `TopologySpec`).

use crate::chain::MiningSnapshot;
use crate::topology::Topology;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GatesJson {
    #[serde(default)]
    min_solutions: u32,
    #[serde(default = "default_max_energy_milli")]
    max_energy_milli: i64,
    #[serde(default)]
    min_diversity_milli: u32,
}

fn default_max_energy_milli() -> i64 {
    i64::MAX
}

impl Default for GatesJson {
    fn default() -> Self {
        Self {
            min_solutions: 0,
            max_energy_milli: default_max_energy_milli(),
            min_diversity_milli: 0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TopologySpecJson {
    nodes: Vec<u32>,
    #[serde(default)]
    edges: Vec<(u32, u32)>,
    allowed_h_milli: Vec<i32>,
    allowed_j_milli: Vec<i32>,
    #[serde(default)]
    gates: Option<GatesJson>,
}

/// Errors parsing/validating a drive-mode topology-spec file.
#[derive(Debug, PartialEq)]
pub enum TopologySpecError {
    Json(String),
    EmptyNodes,
    EmptyAllowedH,
    EmptyAllowedJ,
    /// `edges` reference positions in `nodes` (0-based), not arbitrary node
    /// IDs; `endpoint` is out of range for `node_count`.
    EdgeOutOfRange {
        edge_index: usize,
        endpoint: u32,
        node_count: usize,
    },
}

impl std::fmt::Display for TopologySpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopologySpecError::Json(e) => write!(f, "invalid topology spec JSON: {e}"),
            TopologySpecError::EmptyNodes => write!(f, "topology spec: nodes must be non-empty"),
            TopologySpecError::EmptyAllowedH => {
                write!(f, "topology spec: allowed_h_milli must be non-empty")
            }
            TopologySpecError::EmptyAllowedJ => {
                write!(f, "topology spec: allowed_j_milli must be non-empty")
            }
            TopologySpecError::EdgeOutOfRange {
                edge_index,
                endpoint,
                node_count,
            } => write!(
                f,
                "topology spec: edge {edge_index} references node position {endpoint}, \
                 but only {node_count} nodes are defined"
            ),
        }
    }
}

impl std::error::Error for TopologySpecError {}

/// Parsed drive-mode topology spec: a `Topology` (for the wire message) plus
/// the golden-draw + gate parameters needed to build a synthetic
/// `MiningSnapshot`.
#[derive(Debug, Clone)]
pub struct TopologySpec {
    pub topology: Topology,
    pub allowed_h_milli: Vec<i32>,
    pub allowed_j_milli: Vec<i32>,
    pub min_solutions: u32,
    pub max_energy_milli: i64,
    pub min_diversity_milli: u32,
}

impl TopologySpec {
    /// Build a synthetic `MiningSnapshot`: chain-only fields (last proof
    /// block hash, block number) are zeroed since drive mode has no chain.
    pub fn to_snapshot(&self) -> MiningSnapshot {
        MiningSnapshot {
            last_proof_block_hash: [0u8; 32],
            topology_hash: self.topology.hash.clone(),
            nodes: self.topology.nodes.clone(),
            edges: self.topology.edge_pairs(),
            allowed_h_milli: self.allowed_h_milli.clone(),
            allowed_j_milli: self.allowed_j_milli.clone(),
            min_solutions: self.min_solutions,
            max_energy_milli: self.max_energy_milli,
            min_diversity_milli: self.min_diversity_milli,
            block_number: 0,
        }
    }
}

fn validate_edges(nodes: &[u32], edges: &[(u32, u32)]) -> Result<(), TopologySpecError> {
    let node_count = nodes.len();
    for (edge_index, &(u, v)) in edges.iter().enumerate() {
        for endpoint in [u, v] {
            if endpoint as usize >= node_count {
                return Err(TopologySpecError::EdgeOutOfRange {
                    edge_index,
                    endpoint,
                    node_count,
                });
            }
        }
    }
    Ok(())
}

/// Parse and validate a drive-mode topology-spec JSON document.
pub fn parse_topology_spec(text: &str) -> Result<TopologySpec, TopologySpecError> {
    let raw: TopologySpecJson =
        serde_json::from_str(text).map_err(|e| TopologySpecError::Json(e.to_string()))?;
    if raw.nodes.is_empty() {
        return Err(TopologySpecError::EmptyNodes);
    }
    if raw.allowed_h_milli.is_empty() {
        return Err(TopologySpecError::EmptyAllowedH);
    }
    if raw.allowed_j_milli.is_empty() {
        return Err(TopologySpecError::EmptyAllowedJ);
    }
    validate_edges(&raw.nodes, &raw.edges)?;

    let topology = Topology::from_nodes_edges(raw.nodes, raw.edges);
    let gates = raw.gates.unwrap_or_default();
    Ok(TopologySpec {
        topology,
        allowed_h_milli: raw.allowed_h_milli,
        allowed_j_milli: raw.allowed_j_milli,
        min_solutions: gates.min_solutions,
        max_energy_milli: gates.max_energy_milli,
        min_diversity_milli: gates.min_diversity_milli,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"{
        "nodes": [0, 1, 2, 3],
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "allowed_h_milli": [-1000, 0, 1000],
        "allowed_j_milli": [-1000, 1000]
    }"#;

    #[test]
    fn parses_valid_spec_with_default_gates() {
        let spec = parse_topology_spec(VALID).unwrap();
        assert_eq!(spec.topology.nodes.len(), 4);
        assert_eq!(spec.topology.edge_pairs().len(), 4);
        assert_eq!(spec.min_solutions, 0);
        assert_eq!(spec.max_energy_milli, i64::MAX);
        assert_eq!(spec.min_diversity_milli, 0);
    }

    #[test]
    fn parses_explicit_gates() {
        let text = r#"{
            "nodes": [0, 1],
            "edges": [[0, 1]],
            "allowed_h_milli": [1000],
            "allowed_j_milli": [1000],
            "gates": { "min_solutions": 5, "max_energy_milli": -100, "min_diversity_milli": 200 }
        }"#;
        let spec = parse_topology_spec(text).unwrap();
        assert_eq!(spec.min_solutions, 5);
        assert_eq!(spec.max_energy_milli, -100);
        assert_eq!(spec.min_diversity_milli, 200);
    }

    #[test]
    fn snapshot_reuses_topology_hash() {
        let spec = parse_topology_spec(VALID).unwrap();
        let snap = spec.to_snapshot();
        assert_eq!(snap.topology_hash, spec.topology.hash);
        assert_eq!(snap.nodes, spec.topology.nodes);
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(matches!(
            parse_topology_spec("not json"),
            Err(TopologySpecError::Json(_))
        ));
    }

    #[test]
    fn rejects_empty_nodes() {
        let text = r#"{"nodes":[],"edges":[],"allowed_h_milli":[1],"allowed_j_milli":[1]}"#;
        assert_eq!(
            parse_topology_spec(text).unwrap_err(),
            TopologySpecError::EmptyNodes
        );
    }

    #[test]
    fn rejects_empty_allowed_sets() {
        let no_h = r#"{"nodes":[0],"edges":[],"allowed_h_milli":[],"allowed_j_milli":[1]}"#;
        assert_eq!(
            parse_topology_spec(no_h).unwrap_err(),
            TopologySpecError::EmptyAllowedH
        );
        let no_j = r#"{"nodes":[0],"edges":[],"allowed_h_milli":[1],"allowed_j_milli":[]}"#;
        assert_eq!(
            parse_topology_spec(no_j).unwrap_err(),
            TopologySpecError::EmptyAllowedJ
        );
    }

    #[test]
    fn rejects_out_of_range_edge() {
        let text = r#"{
            "nodes": [0, 1],
            "edges": [[0, 5]],
            "allowed_h_milli": [1000],
            "allowed_j_milli": [1000]
        }"#;
        assert_eq!(
            parse_topology_spec(text).unwrap_err(),
            TopologySpecError::EdgeOutOfRange {
                edge_index: 0,
                endpoint: 5,
                node_count: 2,
            }
        );
    }
}
