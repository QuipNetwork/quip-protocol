"""Utility functions for quantum proof-of-work (diversity and distances).

Extracted from BaseMiner to be reusable and stateless.
"""
from __future__ import annotations

from blake3 import blake3
from shared.logging_config import get_logger
from typing import Any, Tuple, Dict
import numpy as np
from typing import List

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


def _validate_topology_consistency(h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int]) -> List[str]:
    """Validate that h, J parameters match expected topology and constraints.
    
    Args:
        h: Field parameters dictionary
        J: Coupling parameters dictionary
        nodes: List of node indices for the topology
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Get expected topology
    expected_nodes = set(DEFAULT_TOPOLOGY.graph.nodes())
    expected_edges = set(DEFAULT_TOPOLOGY.graph.edges())
    
    # 1. Validate nodes match topology
    provided_nodes = set(nodes)
    if provided_nodes != expected_nodes:
        missing_nodes = expected_nodes - provided_nodes
        extra_nodes = provided_nodes - expected_nodes
        if missing_nodes:
            errors.append(f"Missing topology nodes: {sorted(missing_nodes)}")
        if extra_nodes:
            errors.append(f"Extra nodes not in topology: {sorted(extra_nodes)}")
    
    # 2. Validate h parameters (should be h=0 for all nodes)
    h_nodes = set(h.keys())
    for node_id in h_nodes:
        if node_id not in expected_nodes:
            errors.append(f"h parameter for invalid node: {node_id}")
        elif h[node_id] != 0.0:
            errors.append(f"Non-zero field value h[{node_id}] = {h[node_id]} (expected 0.0)")
    
    # Check for missing h parameters
    missing_h = expected_nodes - h_nodes
    if missing_h:
        errors.append(f"Missing h parameters for nodes: {sorted(missing_h)}")
    
    # 3. Validate J parameters (couplings)
    j_edges = set()
    for (u, v) in J.keys():
        # Normalize edge order for comparison
        edge = (min(u, v), max(u, v))
        j_edges.add(edge)
        
        # Check edge exists in topology
        if edge not in expected_edges and (edge[1], edge[0]) not in expected_edges:
            errors.append(f"J parameter for invalid edge: ({u}, {v})")
        
        # Check J values are ±1
        j_val = J[(u, v)]
        if j_val not in [-1.0, 1.0]:
            errors.append(f"Invalid J value J[({u}, {v})] = {j_val} (expected ±1.0)")
    
    # Normalize expected edges for comparison  
    normalized_expected = set()
    for (u, v) in expected_edges:
        normalized_expected.add((min(u, v), max(u, v)))
    
    # Check for missing J parameters
    missing_j = normalized_expected - j_edges
    if missing_j:
        errors.append(f"Missing J parameters for edges: {sorted(missing_j)}")
    
    return errors


def validate_quantum_proof(quantum_proof, miner_id: str, requirements, block_index: int, previous_hash: bytes) -> bool:
    """Validate quantum proof against requirements and compute metrics.
    
    Args:
        quantum_proof: QuantumProof object containing solutions and metadata
        miner_id: ID of the miner who created the proof
        requirements: BlockRequirements object with difficulty settings
        block_index: Index of the block being validated
        previous_hash: Hash of the previous block
        
    Returns:
        bool: True if quantum proof is valid, False otherwise
    """
    if not quantum_proof:
        logger.error(f"Block {block_index} rejected: no quantum proof")
        return False

    solutions = quantum_proof.solutions
    if not solutions:
        logger.error(f"Block {block_index} rejected: no solutions in quantum proof")
        return False

    # For block validation, use the miner_id from the quantum proof
    nonce = ising_nonce_from_block(previous_hash, miner_id, block_index, quantum_proof.salt)
    if quantum_proof.nonce != nonce:
        logger.error(f"Block {block_index} rejected: invalid nonce {quantum_proof.nonce} != {nonce}")
        return False

    h, J = generate_ising_model_from_nonce(nonce, quantum_proof.nodes, quantum_proof.edges)

    # Validate each solution for correctness
    valid_solutions = []
    invalid_count = 0
    
    for solution in solutions:
        validation_result = validate_solution(solution, h, J, quantum_proof.nodes)
        if validation_result["valid"]:
            valid_solutions.append(solution)
        else:
            invalid_count += 1
            logger.warning(f"Invalid solution found in quantum proof: {validation_result['errors']}")
    
    if invalid_count > 0:
        logger.error(f"Block {block_index} rejected: {invalid_count} invalid solutions found")
        return False

    # Compute energies respecting variable order (quantum_proof.nodes)
    energies = energies_for_solutions(valid_solutions, h, J, quantum_proof.nodes)

    # Find solutions meeting energy threshold
    energy_valid_indices = [i for i, e in enumerate(energies) if e < requirements.difficulty_energy]
    energy_valid_solutions = [valid_solutions[i] for i in energy_valid_indices]

    if len(energy_valid_solutions) < requirements.min_solutions:
        logger.error(f"Block {block_index} rejected: insufficient valid solutions ({len(energy_valid_solutions)} < {requirements.min_solutions})")
        logger.error(f"Solutions presented in result: {len(solutions)} - energies: {energies}")
        return False

    # Calculate diversity using shared utility
    diversity = calculate_diversity(energy_valid_solutions)
    if diversity < requirements.min_diversity:
        logger.error(f"Block {block_index} rejected: insufficient diversity ({diversity} < {requirements.min_diversity})")
        return False

    return True


def validate_solution(spins: List[int], h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int]) -> Dict[str, Any]:
    """Validate an Ising model solution for correctness.
    
    Args:
        spins: Spin configuration as list of {-1, +1} values
        h: Field parameters dictionary
        J: Coupling parameters dictionary  
        nodes: List of node indices for the topology
        
    Returns:
        Dictionary with validation results including validity status and energy
    """
    from typing import Any
    
    n = len(nodes)
    node_to_pos = {node_id: pos for pos, node_id in enumerate(nodes)}
    
    result = {
        "valid": True,
        "errors": [],
        "energy": 0.0,
        "satisfaction_rate": 0.0
    }
    
    # 1. Basic format validation
    if len(spins) != n:
        result["valid"] = False
        result["errors"].append(f"Wrong solution length: {len(spins)} != {n}")
        return result
    
    # 2. Check values are {-1, +1}
    unique_values = set(spins)
    if not unique_values.issubset({-1, 1}):
        invalid_values = unique_values - {-1, 1}
        result["valid"] = False  
        result["errors"].append(f"Invalid spin values: {invalid_values} (must be -1 or +1)")
        return result
    
    # 3. Validate topology consistency
    topology_errors = _validate_topology_consistency(h, J, nodes)
    if topology_errors:
        result["valid"] = False
        result["errors"].extend(topology_errors)
        return result
    
    # Calculate energy using existing function
    result["energy"] = energy_of_solution(spins, h, J, nodes)
    
    # Calculate coupling satisfaction rate
    satisfied_couplings = 0
    total_couplings = len(J)
    
    for (node_i, node_j), val in J.items():
        pos_i = node_to_pos.get(int(node_i))
        pos_j = node_to_pos.get(int(node_j))
        
        if pos_i is not None and pos_j is not None:
            spin_i = spins[pos_i]
            spin_j = spins[pos_j]
            coupling_energy = val * spin_i * spin_j
            
            if coupling_energy < 0:  # Satisfied coupling
                satisfied_couplings += 1
    
    result["satisfaction_rate"] = satisfied_couplings / total_couplings if total_couplings > 0 else 0
    
    return result


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
        invalid_solutions = []
        
        for idx in valid_indices:
            solution = tuple(sampleset.record.sample[idx])
            if solution not in seen:
                seen.add(solution)
                solution_list = list(solution)
                
                # Validate solution format and correctness
                validation_result = validate_solution(solution_list, h, J, nodes)
                if validation_result["valid"]:
                    valid_solutions.append(solution_list)
                else:
                    invalid_solutions.append({
                        "solution": solution_list,
                        "errors": validation_result["errors"]
                    })
        
        # Log any invalid solutions found
        if invalid_solutions:
            import logging
            local_logger = logging.getLogger(__name__)
            local_logger.warning(f"Found {len(invalid_solutions)} invalid solutions with errors: {[s['errors'] for s in invalid_solutions[:3]]}")
        if len(valid_solutions) < min_solutions:
            raise ValueError(f"Insufficient valid solutions: {len(valid_solutions)} < {min_solutions}")

        # Filter solutions if we have too many
        filtered_solutions = valid_solutions
        if len(valid_solutions) >= min_solutions:
            selected_solutions_indices = select_diverse_solutions(valid_solutions, min_solutions)
            filtered_solutions = [valid_solutions[i] for i in selected_solutions_indices]
            diversity = calculate_diversity(filtered_solutions)

        # Recalculate best energy from filtered solutions
        best_energy = min(energies_for_solutions(filtered_solutions, h, J, nodes))

        if diversity < min_diversity:
            raise ValueError(f"Insufficient diversity: {diversity} < {min_diversity}")

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
            diversity=diversity,
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
