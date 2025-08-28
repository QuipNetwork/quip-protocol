"""Miner class for quantum blockchain mining."""

import hashlib
import multiprocessing
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from shared.block_signer import BlockSigner, HashSigsWrapper


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


class Miner:
    def __init__(self, miner_id: str, miner_type: str, sampler, difficulty_energy: float,
                 min_diversity: float, min_solutions: int):
        """
        Initialize a miner with unique ID

        Note: For GPU miners, integrate gpu_benchmark_modal.py which uses Modal Labs
        for cost-effective GPU acceleration. Modal provides $30/month free credits.
        """
        self.miner_id = miner_id
        self.miner_type = miner_type
        self.sampler = sampler
        self.difficulty_energy = difficulty_energy
        self.min_diversity = min_diversity
        self.min_solutions = min_solutions
        self.mining = False
        self.blocks_won = 0
        self.total_rewards = 0
        
        # Initialize crypto manager
        self.crypto = BlockSigner()
        self.ecdsa_public_key_hex = self.crypto.ecdsa_public_key_hex
        self.wots_plus_public_key_hex = self.crypto.wots_plus_public_key_hex
        
        # Keep backward compatibility references
        self.ecdsa_private_key = self.crypto.ecdsa_private_key
        self.ecdsa_public_key = self.crypto.ecdsa_public_key
        self.wots_plus = self.crypto.wots_plus
        
        print(f"{miner_id} initialized with:")
        print(f"  ECDSA Public Key: {self.ecdsa_public_key_hex[:16]}...")
        print(f"  WOTS+ Public Key: {self.wots_plus_public_key_hex[:16]}...")
        
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
        
        # Track last block received time for difficulty adjustment
        self.last_block_received_time = time.time()
        self.no_block_timeout = 1800  # 30 minutes in seconds
        self.difficulty_reduction_factor = 0.1  # Reduce difficulty by 10% per timeout
        
        # Adaptive parameters for performance tuning
        # Initialize num_sweeps based on miner ID for SA miners
        initial_sweeps = 512
        if miner_type != "QPU" and miner_id and miner_id[-1].isdigit():
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

        return np.mean(distances) if distances else 0.0
    
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
    
    def generate_new_wots_key(self):
        """Generate a new WOTS+ key pair after using the current one."""
        self.crypto.wots_plus = HashSigsWrapper()
        self.wots_plus = self.crypto.wots_plus
        self.wots_plus_public_key_hex = self.crypto.wots_plus.get_public_key_hex()
        self.crypto.wots_plus_public_key_hex = self.wots_plus_public_key_hex
        print(f"{self.miner_id} generated new WOTS+ key: {self.wots_plus_public_key_hex[:16]}...")
    
    def sign_block_data(self, block_data: str) -> Tuple[str, str]:
        """
        Sign block data with WOTS+ and then sign that signature with ECDSA.
        
        Returns:
            Tuple of (combined_signature_hex, next_wots_public_key_hex)
        """
        # Use crypto manager to sign
        signature_hex, next_wots_key_hex = self.crypto.sign_block_data(block_data)
        
        # Update local references
        self.wots_plus_public_key_hex = next_wots_key_hex
        self.wots_plus = self.crypto.wots_plus
        
        return signature_hex, next_wots_key_hex

    def generate_quantum_model(self, block_header: str, nonce: int) -> Tuple[dict, dict]:
        """Generate Ising model parameters based on block header and nonce."""
        seed_string = f"{block_header}{nonce}"
        seed = int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)

        # QPU sampler
        h = {i: 0 for i in self.sampler.nodelist}
        J = {edge: 2*np.random.randint(2)-1 for edge in self.sampler.edgelist}

        return h, J

    def check_and_adjust_difficulty_for_timeout(self):
        """Check if no block has been received for 30 minutes and adjust difficulty."""
        time_since_last_block = time.time() - self.last_block_received_time
        
        if time_since_last_block > self.no_block_timeout:
            # Calculate how many 30-minute periods have passed
            timeout_periods = int(time_since_last_block / self.no_block_timeout)
            
            # Reduce difficulty for each timeout period
            original_difficulty = self.difficulty_energy
            self.difficulty_energy = min(
                -13000,  # Cap at easier difficulty
                self.difficulty_energy * (1 + self.difficulty_reduction_factor * timeout_periods)
            )
            
            # Also relax diversity and solution requirements
            self.min_diversity = max(0.15, self.min_diversity * (1 - self.difficulty_reduction_factor * timeout_periods))
            self.min_solutions = max(5, int(self.min_solutions * (1 - self.difficulty_reduction_factor * timeout_periods)))
            
            if original_difficulty != self.difficulty_energy:
                print(f"{self.miner_id} adjusting difficulty due to {time_since_last_block/60:.1f} minutes without new block:")
                print(f"  Energy: {original_difficulty:.2f} -> {self.difficulty_energy:.2f}")
                print(f"  Diversity: {self.min_diversity:.3f}, Solutions: {self.min_solutions}")
            
            return True
        return False
    
    def reset_block_received_time(self):
        """Reset the last block received time when a new block is received."""
        self.last_block_received_time = time.time()

    def mine_block(self, block_header: str, result_queue: multiprocessing.Queue, stop_event: multiprocessing.Event):
        """Mine a block in a separate process.
        
        Args:
            block_header: Format is f"{previous_hash}{index}{timestamp}{data}"
            result_queue: Multiprocessing queue for results
            stop_event: Multiprocessing event to signal stop
        """
        self.mining = True
        progress = 0  # Progress counter for logging
        start_time = time.time()
        
        # Check for timeout-based difficulty adjustment
        self.check_and_adjust_difficulty_for_timeout()
        
        # Track current stage timing
        self.current_stage = None
        self.current_stage_start = None
        
        # Mark that this miner is attempting this round
        self.current_round_attempted = True

        print(f"{self.miner_id} started...")

        while self.mining and not stop_event.is_set():
            # Check if we should stop before generating model
            if stop_event.is_set():
                print(f"{self.miner_id} interrupted")
                return

            # Generate random nonce for each attempt
            nonce = random.randint(0, sys.maxsize)
            
            # Generate quantum model
            h, J = self.generate_quantum_model(block_header, nonce)

            # Check again before sampling
            if stop_event.is_set():
                print(f"{self.miner_id} interrupted")
                return

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start
            
            # Sample from quantum/simulated annealer
            try:
                if self.miner_type == "QPU":
                    sampleset = self.sampler.sample_ising(
                        h, J, 
                        num_reads=100, 
                        answer_mode='raw',
                        annealing_time=self.adaptive_params.get('quantum_annealing_time', 20.0)
                    )
                    # Extract QPU timing information if available
                    if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
                        timing = sampleset.info['timing']
                        if 'qpu_anneal_time_per_sample' in timing:
                            self.timing_stats['quantum_annealing_time'].append(
                                timing['qpu_anneal_time_per_sample']
                            )
                        if 'qpu_sampling_time' in timing:
                            self.timing_stats['sampling'].append(timing['qpu_sampling_time'])
                        if 'qpu_programming_time' in timing:
                            self.timing_stats['preprocessing'].append(timing['qpu_programming_time'])
                else:
                    # For SA, use adaptive parameters
                    num_sweeps = self.adaptive_params.get('num_sweeps', 512)
                    
                    sample_start = time.time()
                    self.current_stage = 'sampling'
                    self.current_stage_start = sample_start
                    
                    # Build sampling parameters based on sampler type
                    sampling_params = {
                        'h': h,
                        'J': J,
                        'num_reads': 100,
                        'num_sweeps': num_sweeps
                    }
                    
                    # Only add beta parameters for actual SimulatedAnnealingSampler
                    # MockDWaveSampler doesn't support these parameters
                    if hasattr(self.sampler, 'sampler_type') and self.sampler.sampler_type == 'mock':
                        # MockDWaveSampler - don't add beta parameters
                        pass
                    else:
                        # Actual SimulatedAnnealingSampler - add beta parameters
                        sampling_params['beta_range'] = self.adaptive_params.get('beta_range', [0.1, 10.0])
                        sampling_params['beta_schedule_type'] = self.adaptive_params.get('beta_schedule', 'geometric')
                    
                    sampleset = self.sampler.sample_ising(**sampling_params)
                    sample_time = time.time() - sample_start
                    
                    # Estimate SA timing components
                    self.timing_stats['sampling'].append(sample_time * 1e6)  # Convert to microseconds
                    self.timing_stats['preprocessing'].append((time.time() - preprocess_start) * 1e6)
            except Exception as e:
                if stop_event.is_set():
                    print(f"{self.miner_id} interrupted during sampling")
                    return
                print(f"{self.miner_id} sampling error: {e}")
                nonce += 1
                continue

            # Check if interrupted before processing results
            if stop_event.is_set():
                print(f"{self.miner_id} interrupted")
                return

            # Track postprocessing time
            postprocess_start = time.time()
            self.current_stage = 'postprocessing'
            self.current_stage_start = postprocess_start
            
            # Find all solutions meeting energy threshold
            valid_indices = np.where(sampleset.record.energy < self.difficulty_energy)[0]
            
            # Update sample counts
            self.timing_stats['total_samples'] += len(sampleset.record.energy)
            self.timing_stats['blocks_attempted'] += 1

            if len(valid_indices) >= self.min_solutions:
                # Get unique solutions
                valid_solutions = []
                seen = set()

                for idx in valid_indices:
                    solution = tuple(sampleset.record.sample[idx])
                    if solution not in seen:
                        seen.add(solution)
                        valid_solutions.append(list(solution))

                # Calculate diversity
                diversity = self.calculate_diversity(valid_solutions)
                min_energy = float(np.min(sampleset.record.energy))

                # Filter excess solutions to maintain diversity
                filtered_solutions = self.filter_diverse_solutions(valid_solutions, self.min_solutions)

                # Recalculate diversity after filtering
                final_diversity = self.calculate_diversity(filtered_solutions)
                print(f"{self.miner_id} found sufficient solutions! Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}, Diversity: {diversity:.3f}, Final Diversity: {final_diversity:.3f}")

                # Track postprocessing time
                self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)
                
                # Check if diversity requirement is met
                if final_diversity >= self.min_diversity and len(valid_solutions) >= self.min_solutions:
                    mining_time = time.time() - start_time
                    min_energy = float(np.min(sampleset.record.energy[valid_indices]))

                    result = MiningResult(
                        miner_id=self.miner_id,
                        miner_type=self.miner_type,
                        nonce=nonce,
                        solutions=filtered_solutions,
                        energy=min_energy,
                        diversity=final_diversity,
                        num_valid=len(valid_solutions),
                        mining_time=mining_time
                    )

                    result_queue.put(result)
                    print(f"{self.miner_id} found valid block! Nonce: {nonce}, Energy: {min_energy:.2f}, Time: {mining_time:.2f}s")
                    return

            progress += 1

            # Progress update
            if progress % 10 == 0 and len(sampleset.record.energy) > 0:
                min_energy = float(np.min(sampleset.record.energy))
                print(f"{self.miner_id} - Progress: {progress}, Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}")

        # If we exit the loop due to stop event
        if stop_event.is_set():
            print(f"{self.miner_id} stopped")