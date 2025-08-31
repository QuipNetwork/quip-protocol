"""Utility functions for quantum proof-of-work (diversity and distances).

Extracted from BaseMiner to be reusable and stateless.
"""
from __future__ import annotations

from blake3 import blake3
from typing import Tuple, Dict, Optional
import numpy as np
from typing import List
import dwave_networkx as dnx

# DWave Topology Configurations
# These match real DWave systems for consistent energy scales

# D-Wave 2000Q (Legacy) - Chimera topology
CHIMERA_TOPOLOGY = {
    'type': 'chimera',
    'params': {'m': 16, 'n': 16, 't': 4},  # 16x16x4 Chimera
    'graph_func': lambda: dnx.chimera_graph(16, 16, 4),
    'chip_id': 'DW_2000Q_VFYC_1',
    'description': 'Legacy D-Wave 2000Q Chimera topology (~2048 qubits)'
}

# D-Wave Advantage - Pegasus topology (DEFAULT)
PEGASUS_TOPOLOGY = {
    'type': 'pegasus',
    'params': {'m': 16},  # P16 Pegasus
    'graph_func': lambda: dnx.pegasus_graph(16),
    'chip_id': 'Advantage_system6.4',
    'description': 'D-Wave Advantage Pegasus topology (~5000 qubits)'
}

# D-Wave Advantage2 - Zephyr topology
ZEPHYR_TOPOLOGY = {
    'type': 'zephyr',
    'params': {'m': 16, 't': 4},  # Z16,4 Zephyr
    'graph_func': lambda: dnx.zephyr_graph(16, 4),
    'chip_id': 'Advantage2_prototype',
    'description': 'D-Wave Advantage2 Zephyr topology (~1000+ qubits)'
}

# Default topology for all miners
DEFAULT_TOPOLOGY = PEGASUS_TOPOLOGY

def get_topology_config(topology_name: Optional[str] = None):
    """Get topology configuration by name, defaults to DEFAULT_TOPOLOGY."""
    if topology_name is None:
        return DEFAULT_TOPOLOGY

    topologies = {
        'chimera': CHIMERA_TOPOLOGY,
        'pegasus': PEGASUS_TOPOLOGY,
        'zephyr': ZEPHYR_TOPOLOGY
    }

    return topologies.get(topology_name.lower(), DEFAULT_TOPOLOGY)

def create_topology_graph(topology_name: Optional[str] = None):
    """Create a topology graph using the specified topology configuration."""
    config = get_topology_config(topology_name)
    return config['graph_func']()

def get_topology_properties(topology_name: Optional[str] = None):
    """Get mock DWave properties for the specified topology."""
    config = get_topology_config(topology_name)
    graph = create_topology_graph(topology_name)

    # Format properties to match MockDWaveSampler expectations
    if config['type'] == 'chimera':
        topology_props = {
            'type': 'chimera',
            'shape': [config['params']['m'], config['params']['n'], config['params']['t']]
        }
    elif config['type'] == 'pegasus':
        topology_props = {
            'type': 'pegasus',
            'shape': [config['params']['m']]
        }
    elif config['type'] == 'zephyr':
        topology_props = {
            'type': 'zephyr',
            'shape': [config['params']['m'], config['params']['t']]
        }
    else:
        # Fallback
        topology_props = {
            'type': config['type'],
            'shape': []
        }

    return {
        'topology': topology_props,
        'num_qubits': len(graph.nodes()),
        'num_couplers': len(graph.edges()),
        'chip_id': config['chip_id'],
        'supported_problem_types': ['qubo', 'ising'],
        'description': config['description']
    }

def ising_seed_from_block(prev_hash: bytes, miner_id: str, cur_index: int, nonce: int) -> int:
    """Generate deterministic seed for Ising model from block parameters.

    Uses miner_id instead of timestamp to ensure reproducible seeds between
    mining and validation phases.
    """
    seed_string = f"{prev_hash.hex()}{miner_id}{cur_index}{nonce}"
    return int(blake3(seed_string.encode()).hexdigest()[:8], 16)


def generate_ising_model_from_seed(seed: int, nodes: List[int], edges: List[Tuple[int, int]]) -> Tuple[Dict[int, int], Dict[tuple, int]]:
    """Generate (h, J) Ising parameters deterministically from a block.

    Deterministic given seed, node list and edge list. We assign h=0 and J in {-1,+1} per edge.
    """
    np.random.seed(seed)

    h = {int(i): 0.0 for i in nodes}
    J = { (int(u), int(v)) if isinstance(u, (int, np.integer)) and isinstance(v, (int, np.integer)) else (int(u), int(v)) : float(2*np.random.randint(2)-1) for (u, v) in edges }

    return h, J


def energy_of_solution(solution: List[int], h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int]) -> float:
    """Compute Ising energy for a solution vector respecting node order.

    - solution values are mapped to spins in {-1,+1}
    - h, J dictionaries are keyed by node ids and node-id pairs respectively
    - nodes defines the variable ordering used in the sampler
    """
    # Map values to spins in {-1, +1}
    spins = [1 if v > 0 else -1 for v in solution]
    e = 0.0
    # Map node id -> position
    node_pos = {int(node_id): pos for pos, node_id in enumerate(nodes)}
    # Local fields
    for pos, node_id in enumerate(nodes[:len(spins)]):
        e += float(h.get(int(node_id), 0.0)) * spins[pos]
    # Couplers
    for (u, v), Jij in J.items():
        pu = node_pos.get(int(u))
        pv = node_pos.get(int(v))
        if pu is not None and pv is not None and pu < len(spins) and pv < len(spins):
            e += float(Jij) * spins[pu] * spins[pv]
    return float(e)


def energies_for_solutions(solutions: List[List[int]], h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int]) -> List[float]:
    """Compute energies for a list of solutions using energy_of_solution."""
    return [energy_of_solution(sol, h, J, nodes) for sol in solutions]

def calculate_hamming_distance(s1: List[int], s2: List[int]) -> int:
    """Calculate symmetric Hamming distance between two binary strings.

    Uses bitwise operations for efficiency:
    - XOR to find differences
    - Population count (bit counting) for distance
    - Compares both normal and inverted to handle symmetry
    """
    # Convert sequences to bit representations
    def to_bits(seq):
        """Convert sequence to integer bit representation."""
        bits = 0
        for i, val in enumerate(seq):
            if val == 1:
                bits |= (1 << i)
        return bits, len(seq)

    bits1, len1 = to_bits(s1)
    bits2, len2 = to_bits(s2)

    # XOR gives us the positions where bits differ
    diff = bits1 ^ bits2

    # Count the number of set bits (Hamming distance)
    distance = bin(diff).count("1")

    # For symmetric Hamming distance, also check inverted
    inverted_bits2 = (1 << len2) - 1 - bits2
    diff_inverted = bits1 ^ inverted_bits2
    distance_inverted = bin(diff_inverted).count("1")

    # Return minimum of normal and inverted distance
    return min(distance, distance_inverted)


def calculate_diversity(solutions: List[List[int]]) -> float:
    """Calculate average normalized Hamming distance between all pairs of solutions."""
    if len(solutions) < 2:
        return 0.0

    distances = []
    n = len(solutions[0])

    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            dist = calculate_hamming_distance(solutions[i], solutions[j])
            distances.append(dist / n)

    return float(np.mean(distances)) if distances else 0.0


def _calculate_set_diversity(indices: List[int], dist_matrix: np.ndarray) -> float:
    """Calculate average pairwise distance for a set of solutions."""
    if len(indices) < 2:
        return 0.0

    total_dist = 0
    count = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            total_dist += dist_matrix[indices[i], indices[j]]
            count += 1

    return total_dist / count if count > 0 else 0.0


def filter_diverse_solutions(solutions: List[List[int]], target_count: int) -> List[List[int]]:
    """Filter solutions to maintain maximum diversity using farthest point sampling.

    Uses farthest point sampling with local search refinement.
    This method provides better diversity than pure greedy selection.
    """
    if len(solutions) <= target_count:
        return solutions

    n_solutions = len(solutions)

    # Pre-compute distance matrix for efficiency
    dist_matrix = np.zeros((n_solutions, n_solutions))
    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            dist = calculate_hamming_distance(solutions[i], solutions[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Farthest Point Sampling
    # Start with the two most distant points
    max_dist = 0
    start_pair = (0, 1)
    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            if dist_matrix[i, j] > max_dist:
                max_dist = dist_matrix[i, j]
                start_pair = (i, j)

    selected_indices = list(start_pair)

    # Iteratively add the farthest point from the current set
    while len(selected_indices) < target_count:
        best_idx = -1
        best_min_dist = -1

        for i in range(n_solutions):
            if i in selected_indices:
                continue

            # Find minimum distance to selected set
            min_dist = min(dist_matrix[i, j] for j in selected_indices)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx != -1:
            selected_indices.append(best_idx)

    # Optional: Local search refinement (can be disabled for performance)
    # Try swapping elements to improve total diversity
    improved = True
    iterations = 0
    max_iterations = 10

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i, sel_idx in enumerate(selected_indices):
            for cand_idx in range(n_solutions):
                if cand_idx in selected_indices:
                    continue

                # Try swapping
                test_indices = selected_indices.copy()
                test_indices[i] = cand_idx

                # Calculate diversity for both sets
                current_div = _calculate_set_diversity(selected_indices, dist_matrix)
                test_div = _calculate_set_diversity(test_indices, dist_matrix)

                if test_div > current_div:
                    selected_indices[i] = cand_idx
                    improved = True
                    break

            if improved:
                break

    return [solutions[i] for i in selected_indices]

