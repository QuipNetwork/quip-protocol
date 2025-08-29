"""Abstract base miner for quantum blockchain mining.

Contains core mining logic and defines abstract methods for miner-specific implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from blake3 import blake3
import multiprocessing
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


@dataclass
class MiningResult:
    miner_id: str
    miner_type: str
    nonce: int
    solutions: List[List[int]]
    energy: float
    diversity: float
    num_valid: int
    mining_time: float


class BaseMiner(ABC):
    """Abstract base class for concrete miners.

    Subclasses must implement:
      - mine_block(): miner-specific mining logic
      - set self.miner_type and self.sampler in __init__
    """

    def __init__(
        self,
        miner_id: str,
    ) -> None:
        if type(self) is BaseMiner:
            raise TypeError("BaseMiner is abstract; instantiate a concrete subclass")
        self.miner_id = miner_id
        self.miner_type: str = "UNKNOWN"
        self.sampler = None
        self.mining = False
        self.blocks_won = 0
        self.total_rewards = 0

        print(f"{miner_id} initialized ({self.miner_type})")
        
        # Initialize timing statistics
        self.timing_stats = {
            'preprocessing': [],
            'sampling': [],
            'postprocessing': [],
            'quantum_annealing_time': [],
            'per_sample_overhead': [],
            'total_samples': 0,
            'blocks_attempted': 0
        }
        
        # Track timing history for graphing (block_number, timing_value)
        self.timing_history = {
            'block_numbers': [],
            'preprocessing_times': [],
            'sampling_times': [],
            'postprocessing_times': [],
            'total_times': [],
            'win_rates': [],
            'adaptive_params_history': []  # Track adaptive params over time
        }
        
        # Track participation in current round
        self.current_round_attempted = False
        
        # Track current stage timing
        self.current_stage: Optional[str] = None
        self.current_stage_start: Optional[float] = None
        
        # Adaptive parameters for performance tuning
        # Initialize num_sweeps based on miner ID for SA miners
        initial_sweeps = 512
        if miner_id and miner_id[-1].isdigit():
            initial_sweeps = pow(2, 6 + int(miner_id[-1]))
        
        self.adaptive_params = {
            'quantum_annealing_time': 20.0,  # microseconds for QPU
            'beta_range': [0.1, 10.0],  # for SA
            'beta_schedule': 'geometric',  # or 'linear'
            'num_sweeps': initial_sweeps  # for SA
        }

    def calculate_hamming_distance(self, s1: List[int], s2: List[int]) -> int:
        """Calculate symmetric Hamming distance between two binary strings.
        
        Uses bitwise operations for efficiency:
        - XOR to find differences
        - Population count (bit counting) for distance
        - Compares both normal and inverted to handle symmetry
        """
        # Convert sequences to bit representations
        # Map -1 to 0, and 1 to 1 for bit operations
        def to_bits(seq):
            """Convert sequence to integer bit representation."""
            bits = 0
            for i, val in enumerate(seq):
                if val == 1 or val == -1:
                    # Set bit i to 1 if val is 1, 0 if val is -1
                    if val == 1:
                        bits |= (1 << i)
            return bits, len(seq)
        
        bits1, len1 = to_bits(s1)
        bits2, len2 = to_bits(s2)
        
        # Create mask for valid bits
        max_len = max(len1, len2)
        mask = (1 << max_len) - 1
        
        # Calculate normal Hamming distance using XOR and popcount
        xor_normal = bits1 ^ bits2
        normal_dist = bin(xor_normal & mask).count('1')
        
        # Calculate symmetric distance (with bits2 inverted)
        bits2_inv = (~bits2) & mask
        xor_inv = bits1 ^ bits2_inv
        inv_dist = bin(xor_inv & mask).count('1')
        
        # Return minimum for symmetric property
        return min(normal_dist, inv_dist)

    def calculate_diversity(self, solutions: List[List[int]]) -> float:
        """Calculate average normalized Hamming distance between all pairs of solutions."""
        if len(solutions) < 2:
            return 0.0

        distances = []
        n = len(solutions[0])

        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                dist = self.calculate_hamming_distance(solutions[i], solutions[j])
                distances.append(dist / n)

        return float(np.mean(distances)) if distances else 0.0
    
    def filter_diverse_solutions(self, solutions: List[List[int]], target_count: int) -> List[List[int]]:
        """Filter solutions to maintain maximum diversity while reducing to target count.
        
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
                dist = self.calculate_hamming_distance(solutions[i], solutions[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # Method 1: Farthest Point Sampling
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
        
        # Method 2: Local search refinement
        # Try swapping elements to improve total diversity
        def calculate_total_diversity(indices):
            """Calculate sum of all pairwise distances."""
            total = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    total += dist_matrix[indices[i], indices[j]]
            return total
        
        current_diversity = calculate_total_diversity(selected_indices)
        improved = True
        max_iterations = 10
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try swapping each selected with each unselected
            for i, selected_idx in enumerate(selected_indices):
                for unselected_idx in range(n_solutions):
                    if unselected_idx in selected_indices:
                        continue
                    
                    # Try swap
                    test_indices = selected_indices.copy()
                    test_indices[i] = unselected_idx
                    test_diversity = calculate_total_diversity(test_indices)
                    
                    if test_diversity > current_diversity:
                        selected_indices[i] = unselected_idx
                        current_diversity = test_diversity
                        improved = True
                        break
                
                if improved:
                    break
        
        return [solutions[i] for i in selected_indices]
    
    def capture_partial_timing(self):
        """Capture timing for current mining attempt, including partial progress."""
        current_time = time.time()
        
        # Initialize with zeros
        preprocessing_time = 0
        sampling_time = 0
        postprocessing_time = 0
        
        # If we have completed preprocessing
        if len(self.timing_stats['preprocessing']) > len(self.timing_stats['sampling']):
            # Preprocessing was completed
            preprocessing_time = self.timing_stats['preprocessing'][-1]
            
            # Check if sampling was started
            if self.current_stage == 'sampling' and self.current_stage_start:
                # Sampling was in progress
                sampling_time = (current_time - self.current_stage_start) * 1e6
                postprocessing_time = 0  # Not started
            elif self.current_stage == 'postprocessing' and self.current_stage_start:
                # Sampling was completed, postprocessing in progress
                if self.timing_stats['sampling']:
                    sampling_time = self.timing_stats['sampling'][-1]
                postprocessing_time = (current_time - self.current_stage_start) * 1e6
        elif self.current_stage == 'preprocessing' and self.current_stage_start:
            # Still in preprocessing
            preprocessing_time = (current_time - self.current_stage_start) * 1e6
            sampling_time = 0
            postprocessing_time = 0
        
        return preprocessing_time, sampling_time, postprocessing_time
    
    def get_timing_summary(self) -> str:
        """Generate a summary of timing statistics for this miner."""
        summary_lines = [f"\nTiming Statistics for {self.miner_id}:"]
        
        if self.timing_stats['blocks_attempted'] > 0:
            summary_lines.append(f"  Blocks Attempted: {self.timing_stats['blocks_attempted']}")
            summary_lines.append(f"  Total Samples: {self.timing_stats['total_samples']}")
            summary_lines.append(f"  Blocks Won: {self.blocks_won}")
            summary_lines.append(f"  Win Rate: {self.blocks_won / self.timing_stats['blocks_attempted'] * 100:.1f}%")
        
        # Calculate averages for each timing component
        for component in ['preprocessing', 'sampling', 'postprocessing']:
            if self.timing_stats[component]:
                avg_time = np.mean(self.timing_stats[component])
                std_time = np.std(self.timing_stats[component])
                summary_lines.append(f"  {component.capitalize()} Time: {avg_time:.2f} ± {std_time:.2f} μs")
        
        # QPU-specific timing
        if self.timing_stats['quantum_annealing_time']:
            avg_anneal = np.mean(self.timing_stats['quantum_annealing_time'])
            summary_lines.append(f"  Quantum Annealing Time: {avg_anneal:.2f} μs")
        
        # Show adaptive parameters
        if self.miner_type == "QPU":
            summary_lines.append(f"  Current Annealing Time: {self.adaptive_params['quantum_annealing_time']:.2f} μs")
        else:
            summary_lines.append(f"  Current Num Sweeps: {self.adaptive_params['num_sweeps']}")
            summary_lines.append(f"  Beta Range: {self.adaptive_params['beta_range']}")
            summary_lines.append(f"  Beta Schedule: {self.adaptive_params['beta_schedule']}")
        
        return "\n".join(summary_lines)
    
    def adapt_parameters(self, network_stats: dict):
        """Adapt miner parameters based on performance relative to network.
        
        Args:
            network_stats: Dict containing total_blocks, total_miners, avg_win_rate
        """
        if self.timing_stats['blocks_attempted'] < 5:
            return  # Need enough data before adapting
        
        # Calculate expected win rate (fair share)
        expected_win_rate = 1.0 / network_stats['total_miners']
        actual_win_rate = self.blocks_won / self.timing_stats['blocks_attempted']
        
        # If winning less than expected, improve parameters
        if actual_win_rate < expected_win_rate * 0.8:  # 20% below expected
            if self.miner_type == "QPU":
                # Increase annealing time for better solutions
                self.adaptive_params['quantum_annealing_time'] *= 1.2
                print(f"{self.miner_id} increasing annealing time to {self.adaptive_params['quantum_annealing_time']:.2f} μs")
            else:
                # For SA, increase sweeps or adjust beta range
                self.adaptive_params['num_sweeps'] = int(self.adaptive_params['num_sweeps'] * 1.1)
                # Widen beta range for better exploration
                self.adaptive_params['beta_range'][0] *= 0.9
                self.adaptive_params['beta_range'][1] *= 1.1
                print(f"{self.miner_id} adapting: sweeps={self.adaptive_params['num_sweeps']}, beta_range={self.adaptive_params['beta_range']}")
        
        # If winning too much, can reduce parameters to save resources
        elif actual_win_rate > expected_win_rate * 1.5:  # 50% above expected
            if self.miner_type == "QPU":
                # Reduce annealing time to save QPU resources
                self.adaptive_params['quantum_annealing_time'] *= 0.9
                print(f"{self.miner_id} reducing annealing time to {self.adaptive_params['quantum_annealing_time']:.2f} μs")
            else:
                # For SA, reduce sweeps for faster mining
                self.adaptive_params['num_sweeps'] = int(self.adaptive_params['num_sweeps'] * 0.95)
                print(f"{self.miner_id} reducing sweeps to {self.adaptive_params['num_sweeps']}")

    def generate_quantum_model(self, block_header: str, nonce: int) -> Tuple[dict, dict]:
        """Generate Ising model parameters based on block header and nonce."""
        if self.sampler is None:
            raise RuntimeError("Sampler not initialized")
            
        seed_string = f"{block_header}{nonce}"
        seed = int(blake3(seed_string.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)

        h = {i: 0 for i in self.sampler.nodelist}
        J = {edge: 2*np.random.randint(2)-1 for edge in self.sampler.edgelist}

        return h, J

    @abstractmethod
    def mine_block(
        self,
        block_header: str,
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
    ) -> Optional[MiningResult]:
        """Abstract method for miner-specific mining implementation.
        
        Args:
            block_header: Format is f"{previous_hash}{index}{timestamp}{data}"
            result_queue: Multiprocessing queue for results
            stop_event: Multiprocessing event to signal stop
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return machine-readable stats for this miner."""
        stats = dict(self.timing_stats)
        stats.update({
            "miner_id": self.miner_id,
            "miner_type": self.miner_type,
        })
        return stats

    def stop_mining(self) -> None:
        """No-op: cancellation is handled by stop_event provided to mine_block."""
        return None

    def shutdown(self) -> None:
        """Close underlying sampler if it supports close()."""
        try:
            if self.sampler and hasattr(self.sampler, "close"):
                self.sampler.close()
        except Exception:
            pass

