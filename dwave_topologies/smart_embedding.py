"""
Smart Zephyr-to-Zephyr embedding using graph structure knowledge.

For Zephyr(m1,t1) → Zephyr(m2,t2) with defects, use a greedy algorithm
that's much faster than generic minorminer.
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import random


def greedy_zephyr_embedding(
    source: nx.Graph,
    target: nx.Graph,
    max_chain_length: int = 3,
    random_seed: Optional[int] = None
) -> Optional[Dict[int, List[int]]]:
    """
    Fast greedy embedding for Zephyr graphs with defects.

    Strategy:
    1. Try identity mapping first for nodes that exist
    2. For missing nodes, find nearby available qubits
    3. Build chains only when necessary

    Args:
        source: Source Zephyr graph
        target: Target Zephyr graph (may have defects)
        max_chain_length: Maximum allowed chain length
        random_seed: Random seed for reproducibility

    Returns:
        Embedding dict {source_node: [target_qubits]} or None if failed
    """
    if random_seed is not None:
        random.seed(random_seed)

    embedding = {}
    used_qubits = set()
    source_nodes = set(source.nodes())
    target_nodes = set(target.nodes())

    # Phase 1: Identity mapping for nodes that exist
    available_for_identity = source_nodes & target_nodes

    for node in available_for_identity:
        embedding[node] = [node]
        used_qubits.add(node)

    # Phase 2: Find embeddings for missing nodes
    missing_nodes = source_nodes - target_nodes

    for missing_node in missing_nodes:
        # Get neighbors of this node in source graph
        neighbors_in_source = set(source.neighbors(missing_node))

        # Find candidate qubits: nearby qubits that connect to neighbor embeddings
        candidates = set()
        for neighbor in neighbors_in_source:
            if neighbor in embedding:
                # Look at qubits in neighbor's chain and their neighbors
                for qubit in embedding[neighbor]:
                    if qubit not in used_qubits:
                        candidates.add(qubit)
                    # Add neighbors of this qubit
                    for q_neighbor in target.neighbors(qubit):
                        if q_neighbor not in used_qubits:
                            candidates.add(q_neighbor)

        if not candidates:
            # Failed to find embedding for this node
            return None

        # Pick best candidate (prefer lower node numbers for determinism)
        best_qubit = min(candidates)
        embedding[missing_node] = [best_qubit]
        used_qubits.add(best_qubit)

    # Phase 3: Verify all edges are covered
    # For each edge in source, ensure there's a path in target
    for u, v in source.edges():
        u_qubits = embedding[u]
        v_qubits = embedding[v]

        # Check if any qubit in u's chain connects to any qubit in v's chain
        edge_covered = False
        for u_q in u_qubits:
            for v_q in v_qubits:
                if target.has_edge(u_q, v_q):
                    edge_covered = True
                    break
            if edge_covered:
                break

        if not edge_covered:
            # Need to extend chains - for now, fail
            # (A full implementation would extend chains here)
            return None

    return embedding


def create_fast_zephyr_embedding(
    source_m: int,
    source_t: int,
    target_graph: nx.Graph,
    max_attempts: int = 10
) -> Optional[Dict[int, List[int]]]:
    """
    Create fast Zephyr embedding with multiple attempts.

    Args:
        source_m: Source Zephyr m parameter
        source_t: Source Zephyr t parameter
        target_graph: Target graph (e.g., Advantage2 with defects)
        max_attempts: Number of attempts with different random seeds

    Returns:
        Embedding dict or None
    """
    import dwave_networkx as dnx

    source_graph = dnx.zephyr_graph(source_m, source_t)

    # Try with different random seeds
    for attempt in range(max_attempts):
        embedding = greedy_zephyr_embedding(
            source_graph,
            target_graph,
            random_seed=attempt
        )
        if embedding:
            return embedding

    return None
