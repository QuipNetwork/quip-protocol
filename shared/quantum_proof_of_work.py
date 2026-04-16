"""Utility functions for quantum proof-of-work (diversity and distances).

Extracted from BaseMiner to be reusable and stateless.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Tuple, Dict, Optional, List

from blake3 import blake3
import numpy as np

from shared.chacha8 import ChaCha8Rng
from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from dwave_topologies import DEFAULT_TOPOLOGY

logger = get_logger('quantum_proof_of_work')


def ising_nonce_from_block(prev_hash: bytes, miner_id: str, cur_index: int, salt: bytes) -> int:
    """Generate deterministic nonce from block parameters using BLAKE3.

    Matches Rust's derive_nonce() in quip-protocol-rs:
      - Hashes raw bytes (not hex-encoded strings)
      - Uses u32 big-endian for block index
      - Returns u64 (8 bytes)
    """
    if not (0 <= cur_index < 2**32):
        raise ValueError(
            f"cur_index must be a u32 (0..2^32-1), got {cur_index}"
        )
    hasher = blake3()
    hasher.update(prev_hash)
    hasher.update(miner_id.encode())
    hasher.update(cur_index.to_bytes(4, 'big'))
    hasher.update(salt)
    digest = hasher.digest()
    return int.from_bytes(digest[:8], 'big')


def generate_ising_model_from_nonce(
    nonce: int,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    h_values: Optional[List[float]] = None,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """Generate (h, J) Ising parameters using ChaCha8Rng.

    Matches Rust's generate_ising_model() in quip-protocol-rs:
      - Uses ChaCha8Rng (not numpy PCG64)
      - Generates h FIRST, then J
      - Uses next_u32() % len for h (modulo selection, matches Rust)
      - Uses next_u32() & 1 for J sign
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]
    if not h_values:
        raise ValueError("h_values must be non-empty")
    if not nodes:
        raise ValueError("nodes must be non-empty for Ising model generation")

    rng = ChaCha8Rng.seed_from_u64(nonce)
    n_h = len(h_values)

    # h FIRST: one next_u32() per node
    h: Dict[int, float] = {}
    for node_id in nodes:
        index = rng.next_u32() % n_h
        h[int(node_id)] = h_values[index]

    # J SECOND: one next_u32() per edge
    J: Dict[Tuple[int, int], float] = {}
    for (u, v) in edges:
        sign = -1.0 if (rng.next_u32() & 1) == 0 else 1.0
        J[(int(u), int(v))] = sign

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
    """Compute Ising energies for multiple solutions using vectorized numpy.

    Converts h and J to arrays and computes all energies in one pass.
    ~10x faster than calling energy_of_solution() in a loop for large
    solution counts.
    """
    if not solutions:
        return []

    n = len(nodes)
    node_pos = {int(nid): pos for pos, nid in enumerate(nodes)}

    # Build h_arr: shape (n,)
    h_arr = np.zeros(n, dtype=np.float64)
    for nid, val in h.items():
        pos = node_pos.get(int(nid))
        if pos is not None:
            h_arr[pos] = val

    # Build J arrays: edge endpoints + values
    edge_u = []
    edge_v = []
    j_vals = []
    for (u, v), val in J.items():
        pu = node_pos.get(int(u))
        pv = node_pos.get(int(v))
        if pu is not None and pv is not None:
            edge_u.append(pu)
            edge_v.append(pv)
            j_vals.append(val)
    edge_u = np.array(edge_u, dtype=np.intp)
    edge_v = np.array(edge_v, dtype=np.intp)
    j_arr = np.array(j_vals, dtype=np.float64)

    # Build spin matrix: shape (n_solutions, n)
    # Fall back to per-solution if lengths are inconsistent
    try:
        spin_matrix = np.array(solutions, dtype=np.float64)
    except ValueError:
        return [energy_of_solution(sol, h, J, nodes) for sol in solutions]
    # Map to {-1, +1}
    spin_matrix = np.where(spin_matrix > 0, 1.0, -1.0)

    # h contribution: sum(h_i * s_i) for each solution
    h_energies = spin_matrix @ h_arr  # (n_solutions,)

    # J contribution: sum(J_ij * s_i * s_j) for each solution
    # Vectorized: s_u * s_v for all edges, then dot with J values
    s_u = spin_matrix[:, edge_u]  # (n_solutions, n_edges)
    s_v = spin_matrix[:, edge_v]  # (n_solutions, n_edges)
    j_energies = (s_u * s_v) @ j_arr  # (n_solutions,)

    return (h_energies + j_energies).tolist()

def calculate_hamming_distance(s1: List[int], s2: List[int]) -> int:
    """Calculate symmetric Hamming distance between two spin arrays.

    For Ising spin variables {-1, +1}, symmetric distance accounts for
    global spin flip symmetry: distance(s, -s) = 0.

    Uses numpy for vectorized operations - much faster than Python loops.
    """
    a1 = np.asarray(s1, dtype=np.int8)
    a2 = np.asarray(s2, dtype=np.int8)

    # Count mismatches (where spins differ)
    distance = np.count_nonzero(a1 != a2)

    # Symmetric: also check inverted (global spin flip)
    distance_inverted = np.count_nonzero(a1 != -a2)

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


def _compute_distance_matrix_vectorized(solutions: List[List[int]]) -> np.ndarray:
    """Compute symmetric Hamming distance matrix using vectorized operations.

    Uses PyTorch MPS/CUDA when available for large matrices (5x speedup on
    GPU at 500+ solutions). Falls back to numpy for small matrices or when
    no GPU is available.
    """
    arr = np.array(solutions, dtype=np.int8)
    n_solutions = arr.shape[0]

    # GPU acceleration for large matrices (amortizes transfer overhead)
    if n_solutions >= 200:
        try:
            import torch
            device = None
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'

            if device is not None:
                t = torch.from_numpy(arr).to(torch.int8).to(device)
                a1 = t.unsqueeze(1)
                a2 = t.unsqueeze(0)
                dist_normal = (a1 != a2).sum(dim=2)
                dist_inverted = (a1 != -a2).sum(dim=2)
                return torch.minimum(
                    dist_normal, dist_inverted
                ).cpu().numpy().astype(np.float64)
        except Exception:
            pass  # Fall through to numpy

    # Numpy path (fast for small matrices, no GPU needed)
    a1 = arr[:, np.newaxis, :]
    a2 = arr[np.newaxis, :, :]
    dist_normal = np.count_nonzero(a1 != a2, axis=2)
    dist_inverted = np.count_nonzero(a1 != -a2, axis=2)
    return np.minimum(dist_normal, dist_inverted).astype(np.float64)


def select_diverse_solutions(solutions: List[List[int]], target_count: int) -> List[int]:
    """Filter solutions to maintain maximum diversity using farthest point sampling.

    Uses farthest point sampling with local search refinement.
    This method provides better diversity than pure greedy selection.
    """
    if len(solutions) <= target_count:
        return list(range(0, len(solutions)))

    n_solutions = len(solutions)

    # Pre-compute distance matrix using vectorized operations (MUCH faster)
    dist_matrix = _compute_distance_matrix_vectorized(solutions)

    # Farthest Point Sampling
    # Start with the two most distant points (use numpy to find max)
    # Only look at upper triangle to avoid duplicates
    upper_tri = np.triu(dist_matrix, k=1)
    max_idx = np.unravel_index(np.argmax(upper_tri), upper_tri.shape)
    selected_indices = list(max_idx)

    # Convert to set for O(1) lookup
    selected_set = set(selected_indices)

    # Iteratively add the farthest point from the current set
    while len(selected_indices) < target_count:
        # Get distances to all selected points
        selected_arr = np.array(selected_indices)
        min_dists = np.min(dist_matrix[:, selected_arr], axis=1)

        # Mask already selected
        min_dists[selected_arr] = -1

        # Find point with maximum minimum distance
        best_idx = np.argmax(min_dists)
        selected_indices.append(best_idx)
        selected_set.add(best_idx)

    # Optional: Local search refinement (limited iterations for performance)
    # Try swapping elements to improve total diversity
    improved = True
    iterations = 0
    max_iterations = 5  # Reduced from 10 for performance

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        current_div = _calculate_set_diversity(selected_indices, dist_matrix)

        for i in range(len(selected_indices)):
            sel_idx = selected_indices[i]

            # Check a subset of candidates (not all) for performance
            # Sample up to 50 random candidates if n_solutions is large
            if n_solutions > 100:
                candidates = np.random.choice(n_solutions, min(50, n_solutions), replace=False)
            else:
                candidates = range(n_solutions)

            for cand_idx in candidates:
                if cand_idx in selected_set:
                    continue

                # Try swapping
                test_indices = selected_indices.copy()
                test_indices[i] = cand_idx
                test_div = _calculate_set_diversity(test_indices, dist_matrix)

                if test_div > current_div:
                    selected_set.remove(sel_idx)
                    selected_set.add(cand_idx)
                    selected_indices[i] = cand_idx
                    current_div = test_div
                    improved = True
                    break

            if improved:
                break

    return selected_indices


def _validate_topology_consistency(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    nodes: List[int],
    edges: Optional[List[Tuple[int, int]]] = None,
    allowed_h_values: Optional[List[float]] = None
) -> List[str]:
    """Validate that h, J parameters match expected topology and constraints.

    Args:
        h: Field parameters dictionary
        J: Coupling parameters dictionary
        nodes: List of node indices for the topology
        edges: List of edges in the topology (if None, uses DEFAULT_TOPOLOGY edges)
        allowed_h_values: List of valid h values (default: any float)
                         Set to validate h values are in allowed set

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Use provided nodes
    expected_nodes = set(nodes)

    # Use provided edges or fall back to DEFAULT_TOPOLOGY
    if edges is not None:
        expected_edges = set(edges)
    else:
        expected_edges = set(DEFAULT_TOPOLOGY.graph.edges())

    # 2. Validate h parameters
    h_nodes = set(h.keys())

    # Validate h values are in allowed set (if specified)
    if allowed_h_values is not None:
        allowed_set = set(allowed_h_values)
        for node_id in h_nodes:
            if node_id not in expected_nodes:
                errors.append(f"h parameter for invalid node: {node_id}")
            elif h[node_id] not in allowed_set:
                errors.append(
                    f"Invalid h[{node_id}] = {h[node_id]}, "
                    f"expected one of {allowed_h_values}"
                )
    else:
        # No allowed_h_values specified, just check nodes are valid
        for node_id in h_nodes:
            if node_id not in expected_nodes:
                errors.append(f"h parameter for invalid node: {node_id}")

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

    # Get h_values from requirements
    h_values = getattr(requirements, 'h_values', None)

    h, J = generate_ising_model_from_nonce(
        nonce,
        quantum_proof.nodes,
        quantum_proof.edges,
        h_values=h_values,
    )

    # Validate each solution for correctness
    valid_solutions = []
    invalid_count = 0
    
    for solution in solutions:
        validation_result = validate_solution(solution, h, J, quantum_proof.nodes, quantum_proof.edges)
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
        logger.error(f"Energy threshold: {requirements.difficulty_energy:.2f} (solutions must be < this value)")
        # Show which solutions failed and by how much
        for i, e in enumerate(energies):
            status = "PASS" if e < requirements.difficulty_energy else f"FAIL (gap: {e - requirements.difficulty_energy:.2f})"
            logger.error(f"  Solution {i}: energy={e:.2f} - {status}")
        return False

    # Select most diverse subset of min_solutions and check diversity
    # This ensures we find AT LEAST min_solutions with AT LEAST min_diversity
    selected_solution_indices = select_diverse_solutions(energy_valid_solutions, requirements.min_solutions)
    selected_solutions = [energy_valid_solutions[i] for i in selected_solution_indices]
    diversity = calculate_diversity(selected_solutions)

    if diversity < requirements.min_diversity:
        logger.error(f"Block {block_index} rejected: insufficient diversity in best {requirements.min_solutions} solutions ({diversity:.3f} < {requirements.min_diversity})")
        return False

    return True


def validate_solution(spins: List[int], h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int], edges: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """Validate an Ising model solution for correctness.

    Args:
        spins: Spin configuration as list of {-1, +1} values
        h: Field parameters dictionary
        J: Coupling parameters dictionary
        nodes: List of node indices for the topology
        edges: List of edges in the topology (optional, for validation)

    Returns:
        Dictionary with validation results including validity status and energy
    """
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
    topology_errors = _validate_topology_consistency(h, J, nodes, edges)
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
                      miner_id: str, miner_type: str,
                      h: Optional[Dict[int, float]] = None,
                      J: Optional[Dict[Tuple[int, int], float]] = None,
                      skip_validation: bool = True):
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
        h: Optional pre-computed field parameters (avoids regeneration)
        J: Optional pre-computed coupling parameters (avoids regeneration)
        skip_validation: If True, skip per-solution validation (faster for mining).
                        Set to False for block validation from other miners.

    Returns:
        MiningResult if successful, None if requirements not met
    """
    difficulty_energy = requirements.difficulty_energy
    min_diversity = requirements.min_diversity
    min_solutions = requirements.min_solutions
    best_energy = float('inf')
    valid_count = 0
    valid_solutions = []
    diversity = 0.0
    result = None

    try:
        # Best Energy - use sampler-reported energy for fast early exit
        all_energies = sampleset.record.energy
        if len(all_energies) == 0:
            raise ValueError("No samples in sampleset")

        best_energy = float(np.min(all_energies))

        # FAST PATH: Early exit if best energy doesn't meet threshold
        # This avoids expensive Ising model regeneration and energy recalculation
        if best_energy > difficulty_energy:
            raise ValueError(f"Best energy {best_energy} exceeds difficulty energy {difficulty_energy}")

        # Count how many samples meet threshold before expensive operations
        valid_count = np.sum(all_energies <= difficulty_energy)
        if valid_count < min_solutions:
            raise ValueError(f"Insufficient valid solutions: {valid_count} < {min_solutions}")

        # Process results from this mining attempt
        # Find all solutions meeting energy threshold
        valid_indices = np.where(all_energies < difficulty_energy)[0]

        # Get unique solutions that meet energy threshold
        # Track best energy among valid solutions
        valid_solutions = []
        valid_energies = []
        seen = set()

        if skip_validation:
            # FAST PATH: Trust sampler output, skip per-solution validation
            # This is safe during mining since we control the sampler
            for idx in valid_indices:
                solution = tuple(sampleset.record.sample[idx])
                if solution not in seen:
                    seen.add(solution)
                    valid_solutions.append(list(solution))
                    valid_energies.append(all_energies[idx])
        else:
            # SLOW PATH: Full validation for untrusted sources (block validation)
            # Use pre-computed Ising model if provided, otherwise regenerate
            if h is None or J is None:
                h_values = getattr(requirements, 'h_values', None)
                h, J = generate_ising_model_from_nonce(nonce, nodes, edges, h_values=h_values)

            invalid_solutions = []
            for idx in valid_indices:
                solution = tuple(sampleset.record.sample[idx])
                if solution not in seen:
                    seen.add(solution)
                    solution_list = list(solution)

                    # Validate solution format and correctness
                    validation_result = validate_solution(solution_list, h, J, nodes, edges)
                    if validation_result["valid"]:
                        valid_solutions.append(solution_list)
                        valid_energies.append(all_energies[idx])
                    else:
                        invalid_solutions.append({
                            "solution": solution_list,
                            "errors": validation_result["errors"]
                        })

            # Log any invalid solutions found
            if invalid_solutions:
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
            # Use tracked energy for best of filtered solutions
            best_energy = min(valid_energies[i] for i in selected_solutions_indices)
        elif valid_energies:
            best_energy = min(valid_energies)

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
        # Use module logger for consistency
        logger.debug(f"Failed to meet requirements: {e}")
    finally:
        # Log every mining attempt (successful or not) for analysis
        logger.info(f"[{miner_id}] Mining attempt - Energy: {best_energy:.0f}, Valid: {valid_count} (best {min_solutions} diversity: {diversity:.3f}) (requirements: energy<={difficulty_energy:.0f}, valid>={min_solutions}, diversity>={min_diversity:.3f})")
    return result
