"""Utility functions for quantum proof-of-work (diversity and distances).

Extracted from BaseMiner to be reusable and stateless.
"""
from __future__ import annotations

from blake3 import blake3
from shared.logging_config import get_logger
from typing import Tuple, Dict, Optional
import numpy as np
from typing import List
import dwave_networkx as dnx

logger = get_logger('quantum_proof_of_work')

# Import the default topology from the new topology system
from dwave_topologies import DEFAULT_TOPOLOGY





def ising_nonce_from_block(prev_hash: bytes, miner_id: str, cur_index: int, salt: bytes) -> int:
    """Generate deterministic seed for Ising model from block parameters.

    Uses miner_id instead of timestamp to ensure reproducible seeds between
    mining and validation phases.
    """
    seed = f"{prev_hash.hex()}{miner_id}{cur_index}".encode() + salt
    nonce_bytes = blake3(seed).digest()
    nonce = int.from_bytes(nonce_bytes[:4], 'big')
    logger.debug(f"ising_nonce_from_block: prev_hash={prev_hash.hex()[:8]}, miner_id={miner_id}, cur_index={cur_index}, salt={salt.hex()[:8]}, nonce={nonce}")
    return nonce


def generate_ising_model_from_nonce(nonce: int, nodes: List[int], edges: List[Tuple[int, int]]) -> Tuple[Dict[int, float], Dict[tuple, float]]:
    """Generate (h, J) Ising parameters deterministically from a block.

    Deterministic given seed, node list and edge list. We assign h=0 and J in {-1,+1} per edge.
    """
    np.random.seed(nonce)

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


def select_diverse_solutions(solutions: List[List[int]], target_count: int) -> List[int]:
    """Filter solutions to maintain maximum diversity using farthest point sampling.

    Uses farthest point sampling with local search refinement.
    This method provides better diversity than pure greedy selection.
    """
    if len(solutions) <= target_count:
        return list(range(0, len(solutions)))

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

    return selected_indices


def evaluate_sampleset(sampleset, requirements, nodes: List[int], edges: List[Tuple[int, int]],
                      nonce: int, salt: bytes, prev_timestamp: int, start_time: float,
                      miner_id: str, miner_type: str):
    """Convert a sample set into a mining result if it meets requirements, otherwise return None.

    Args:
        sampleset: dimod.SampleSet from the sampler
        requirements: BlockRequirements object with difficulty settings
        nodes: List of node indices for the topology
        edges: List of edge tuples for the topology
        nonce: Nonce used for this mining attempt
        salt: Salt bytes used for this mining attempt
        prev_timestamp: Timestamp from previous block
        start_time: Start time of mining attempt
        miner_id: ID of the miner
        miner_type: Type of the miner (CPU, GPU, QPU)

    Returns:
        MiningResult if successful, None if requirements not met
    """
    import time
    from shared.miner_types import MiningResult

    difficulty_energy = requirements.difficulty_energy
    min_diversity = requirements.min_diversity
    min_solutions = requirements.min_solutions
    best_energy = float('inf')
    valid_solutions = []
    diversity = 0.0
    result = None

    try:
        # Best Energy
        all_energies = sampleset.record.energy
        if len(all_energies) == 0:
            raise ValueError("No samples in sampleset")

        best_energy = float(np.min(all_energies))

        # NOTE: we use the same energy function to ensure consistency. Unfortunately
        # it disagrees with energies created by different sampler impls, but not significantly.
        solutions = list(sampleset.record.sample)
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)
        best_energy = min(energies_for_solutions(solutions, h, J, nodes))

        if best_energy > difficulty_energy:
            raise ValueError(f"Best energy {best_energy} exceeds difficulty energy {difficulty_energy}")

        # Process results from this mining attempt
        # Find all solutions meeting energy threshold
        valid_indices = np.where(all_energies < difficulty_energy)[0]
        # Get unique solutions that meet energy threshold
        valid_solutions = []
        seen = set()
        for idx in valid_indices:
            solution = tuple(sampleset.record.sample[idx])
            if solution not in seen:
                seen.add(solution)
                valid_solutions.append(list(solution))
        if len(valid_solutions) < min_solutions:
            raise ValueError(f"Insufficient valid solutions: {len(valid_solutions)} < {min_solutions}")

        # Filter solutions if we have too many
        filtered_solutions = valid_solutions
        final_diversity = diversity
        if len(valid_solutions) >= min_solutions:
            selected_solutions_indices = select_diverse_solutions(valid_solutions, min_solutions)
            filtered_solutions = [valid_solutions[i] for i in selected_solutions_indices]
            final_diversity = calculate_diversity(filtered_solutions)

        # Recalculate best energy from filtered solutions
        best_energy = min(energies_for_solutions(filtered_solutions, h, J, nodes))

        if final_diversity < min_diversity:
            raise ValueError(f"Insufficient diversity: {final_diversity} < {min_diversity}")

        # Create mining result for this attempt
        mining_time = time.time() - start_time

        # Create result for this attempt
        result = MiningResult(
            miner_id=miner_id,
            miner_type=miner_type,
            nonce=nonce,
            salt=salt,
            timestamp=int(time.time()),
            prev_timestamp=prev_timestamp,
            solutions=filtered_solutions,
            energy=best_energy,
            diversity=final_diversity,
            num_valid=len(valid_solutions),
            mining_time=int(mining_time),
            node_list=nodes,
            edge_list=edges,
            variable_order=nodes
        )
    except ValueError as e:
        # Use a local logger since we don't have access to the miner's logger
        import logging
        local_logger = logging.getLogger(__name__)
        local_logger.debug(f"Failed to meet requirements: {e}")
    finally:
        # Use a local logger since we don't have access to the miner's logger
        import logging
        local_logger = logging.getLogger(__name__)
        local_logger.info(f"Mining attempt - Energy: {best_energy:.2f}, Valid: {len(valid_solutions)}, Diversity: {diversity:.3f} (requirements: energy<={difficulty_energy:.2f}, valid>={min_solutions}, diversity>={min_diversity:.3f})")
    return result
