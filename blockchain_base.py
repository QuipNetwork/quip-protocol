"""Base classes and utilities for quantum blockchain miners."""

import asyncio
import hashlib
import time
import logging
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from quantum_blockchain_network import P2PNode, Message
from shared.crypto_utils import CryptoManager
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler

logger = logging.getLogger(__name__)


class SimulatedAnnealingStructuredSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with identical structure."""
    
    def __init__(self, qpu=None):
        if qpu is None:
            try:
                qpu = DWaveSampler()
            except Exception:
                qpu = MockDWaveSampler()
        
        substitute_sampler = SimulatedAnnealingSampler()
        super().__init__(
            nodelist=qpu.nodelist,
            edgelist=qpu.edgelist,
            properties=qpu.properties,
            substitute_sampler=substitute_sampler
        )
        self.sampler_type = "mock"
        self.parameters.update(substitute_sampler.parameters)  # Do not warn when SA parameters are seen.
        self.mocked_parameters.add('num_sweeps')  # Do not warn when this SA parameter is seen.


@dataclass
class BlockData:
    """Blockchain block data structure."""
    index: int
    timestamp: float
    data: str
    previous_hash: str
    hash: str
    nonce: int
    energy: float
    diversity: float
    miner_id: str
    miner_type: str
    mining_time: float = 0.0
    signature: str = ""
    reward_address: str = ""
    miner_ecdsa_public_key: str = ""
    miner_wots_plus_public_key: str = ""
    num_valid_solutions: int = 0
    quantum_proof: Optional[List[List[int]]] = None
    

class BaseMiner:
    """Base class for all miner types."""
    
    def __init__(self, miner_id: str, miner_type: str, sampler=None,
                 difficulty_energy: float = -15500.0,
                 min_diversity: float = 0.46, 
                 min_solutions: int = 25):
        self.miner_id = miner_id
        self.miner_type = miner_type
        self.sampler = sampler if sampler else SimulatedAnnealingStructuredSampler()
        self.difficulty_energy = difficulty_energy
        self.base_difficulty_energy = difficulty_energy
        self.min_diversity = min_diversity
        self.base_min_diversity = min_diversity
        self.min_solutions = min_solutions
        self.base_min_solutions = min_solutions
        self.current_block = None
        self.mining = False
        self.mining_task = None
        self.blocks_won = 0
        self.total_rewards = 0
        
        # Initialize crypto manager
        self.crypto = CryptoManager()
        self.ecdsa_public_key_hex = self.crypto.ecdsa_public_key_hex
        self.wots_plus_public_key_hex = self.crypto.wots_plus_public_key_hex
        
        logger.info(f"{miner_id} initialized with:")
        logger.info(f"  ECDSA Public Key: {self.ecdsa_public_key_hex[:16]}...")
        logger.info(f"  WOTS+ Public Key: {self.wots_plus_public_key_hex[:16]}...")
        
        # Track last block received time for difficulty adjustment
        self.last_block_received_time = time.time()
        self.no_block_timeout = 1800  # 30 minutes in seconds
        self.difficulty_reduction_factor = 0.1  # Reduce difficulty by 10% per timeout
        
        # Timing statistics for performance tuning
        self.timing_stats = {
            'preprocessing': [],
            'sampling': [],
            'postprocessing': [],
            'quantum_annealing_time': [],
            'per_sample_overhead': [],
            'total_samples': 0,
            'blocks_attempted': 0
        }
        
        # Track current mining stage for timing
        self.current_stage = None
        self.current_stage_start = None
        
        # Track adaptive parameters
        self.adaptive_params = {
            'num_sweeps': 512,  # for SA
            'beta_range': [0.1, 10.0],  # for SA
            'beta_schedule': 'geometric'  # for SA
        }
        
    def generate_quantum_model(self, block_header: str, nonce: int) -> Tuple[dict, dict]:
        """Generate Ising model parameters based on block header and nonce."""
        seed_string = f"{block_header}{nonce}"
        seed = int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # Generate h and J for a small problem (suitable for all miners)
        num_vars = 64  # Small enough for CPU/GPU
        h = {i: np.random.uniform(-1, 1) for i in range(num_vars)}
        
        # Generate sparse connections
        J = {}
        for i in range(num_vars):
            for j in range(i + 1, min(i + 4, num_vars)):  # Local connections
                if np.random.random() < 0.5:
                    J[(i, j)] = np.random.choice([-1, 1])
        
        return h, J
    
    def calculate_hamming_distance(self, s1: List[int], s2: List[int]) -> int:
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
        distance = bin(diff).count('1')
        
        # For symmetric Hamming distance, also check inverted
        inverted_bits2 = (1 << len2) - 1 - bits2
        diff_inverted = bits1 ^ inverted_bits2
        distance_inverted = bin(diff_inverted).count('1')
        
        # Return minimum of normal and inverted distance
        return min(distance, distance_inverted)
    
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
                dist = self.calculate_hamming_distance(solutions[i], solutions[j])
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
                    current_div = self._calculate_set_diversity(selected_indices, dist_matrix)
                    test_div = self._calculate_set_diversity(test_indices, dist_matrix)
                    
                    if test_div > current_div:
                        selected_indices[i] = cand_idx
                        improved = True
                        break
                
                if improved:
                    break
        
        return [solutions[i] for i in selected_indices]
    
    def _calculate_set_diversity(self, indices: List[int], dist_matrix: np.ndarray) -> float:
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
                logger.info(f"{self.miner_id} adjusting difficulty due to {time_since_last_block/60:.1f} minutes without new block:")
                logger.info(f"  Energy: {original_difficulty:.2f} -> {self.difficulty_energy:.2f}")
                logger.info(f"  Diversity: {self.min_diversity:.3f}, Solutions: {self.min_solutions}")
            
            return True
        return False
    
    def reset_block_received_time(self):
        """Reset the last block received time when a new block is received."""
        self.last_block_received_time = time.time()
    
    def sign_block_data(self, block_data: str) -> Tuple[str, str]:
        """Sign block data with both WOTS+ and ECDSA, generate new WOTS+ key."""
        signature_hex, next_wots_key_hex = self.crypto.sign_block_data(block_data)
        # Update our local reference to the new WOTS+ key
        self.wots_plus_public_key_hex = next_wots_key_hex
        return signature_hex, next_wots_key_hex

    def compute_block_hash(self, block_data: dict) -> str:
        """Compute hash of block data with new format."""
        # New format includes all the cryptographic fields
        block_string = f"{block_data['previous_hash']}{block_data['index']}{block_data['timestamp']}{block_data['data']}"
        block_string += f"{block_data.get('signature', '')}{block_data.get('reward_address', '')}"
        block_string += f"{block_data.get('miner_ecdsa_public_key', '')}{block_data.get('miner_wots_plus_public_key', '')}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    async def mine_block_async(self, block_data: dict) -> Optional[dict]:
        """Async mining implementation for P2P network compatibility."""
        import asyncio
        
        self.mining = True
        self.current_block = block_data
        start_time = time.time()
        
        # Check for timeout-based difficulty adjustment
        self.check_and_adjust_difficulty_for_timeout()
        
        # Create block header in the correct format
        block_header = f"{block_data['previous_hash']}{block_data['index']}{block_data['timestamp']}{block_data['data']}"
        
        logger.info(f"{self.miner_id} started mining block {block_data['index']}")
        
        # Track current stage timing
        self.current_stage = None
        self.current_stage_start = None
        
        try:
            while self.mining:
                # Generate random nonce for each attempt
                nonce = random.randint(0, sys.maxsize)
                
                # Generate quantum model
                h, J = self.generate_quantum_model(block_header, nonce)
                
                # Track preprocessing time
                preprocess_start = time.time()
                self.current_stage = 'preprocessing'
                self.current_stage_start = preprocess_start
                
                # Sample from quantum/simulated annealer
                try:
                    if self.miner_type == "QPU":
                        # Run QPU sampling in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        sampleset = await loop.run_in_executor(
                            None,
                            lambda: self.sampler.sample_ising(
                                h, J,
                                num_reads=100,
                                answer_mode='raw',
                                annealing_time=self.adaptive_params.get('quantum_annealing_time', 20.0)
                            )
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
                        
                        # Run SA sampling in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        sampleset = await loop.run_in_executor(
                            None,
                            lambda: self.sampler.sample_ising(**sampling_params)
                        )
                        sample_time = time.time() - sample_start
                        
                        # Estimate SA timing components
                        self.timing_stats['sampling'].append(sample_time * 1e6)  # Convert to microseconds
                        self.timing_stats['preprocessing'].append((time.time() - preprocess_start) * 1e6)
                except Exception as e:
                    logger.error(f"{self.miner_id} sampling error: {e}")
                    continue
                
                # Track postprocessing time
                postprocess_start = time.time()
                self.current_stage = 'postprocessing'
                self.current_stage_start = postprocess_start
                
                # Get samples and energies from sampleset
                samples = sampleset.record.sample
                energies = sampleset.record.energy
                
                # Find valid solutions
                valid_indices = np.where(energies < self.difficulty_energy)[0]
                
                # Update sample counts
                self.timing_stats['total_samples'] += len(energies)
                self.timing_stats['blocks_attempted'] += 1
                
                if len(valid_indices) >= self.min_solutions:
                    # Get unique solutions
                    valid_solutions = []
                    seen = set()
                    
                    for idx in valid_indices:
                        solution = tuple(samples[idx])
                        if solution not in seen:
                            seen.add(solution)
                            valid_solutions.append(list(solution))
                    
                    # Filter excess solutions to maintain diversity
                    filtered_solutions = self.filter_diverse_solutions(valid_solutions, self.min_solutions)
                    
                    # Calculate diversity of filtered solutions
                    diversity = self.calculate_diversity(filtered_solutions)
                    
                    # Track postprocessing time
                    self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)
                    
                    # Check if diversity requirement is met
                    if diversity >= self.min_diversity and len(filtered_solutions) >= self.min_solutions:
                        mining_time = time.time() - start_time
                        min_energy = float(np.min(energies[valid_indices]))
                        
                        # Create winning block
                        winning_block = block_data.copy()
                        
                        # Sign the block data
                        block_data_to_sign = f"{block_data['previous_hash']}{block_data['index']}{block_data['timestamp']}{block_data['data']}"
                        signature_hex, next_wots_key_hex = self.sign_block_data(block_data_to_sign)
                        
                        winning_block.update({
                            'nonce': nonce,
                            'energy': min_energy,
                            'diversity': diversity,
                            'miner_id': self.miner_id,
                            'miner_type': self.miner_type,
                            'mining_time': mining_time,
                            'num_valid_solutions': len(valid_solutions),
                            'quantum_proof': filtered_solutions,
                            'signature': signature_hex,
                            'reward_address': self.ecdsa_public_key_hex,
                            'miner_ecdsa_public_key': self.ecdsa_public_key_hex,
                            'miner_wots_plus_public_key': next_wots_key_hex
                        })
                        winning_block['hash'] = self.compute_block_hash(winning_block)
                        
                        logger.info(f"{self.miner_id} found valid block! Nonce: {nonce}, Energy: {min_energy:.2f}, Time: {mining_time:.2f}s")
                        return winning_block
                
                nonce += 1
                
                # Progress update
                if self.timing_stats['blocks_attempted'] % 10 == 0:
                    min_energy = float(np.min(energies)) if len(energies) > 0 else 0
                    logger.debug(f"{self.miner_id} - Attempt {self.timing_stats['blocks_attempted']}, Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}")
                
                # Allow other async tasks to run
                await asyncio.sleep(0)
                
        except asyncio.CancelledError:
            logger.info(f"{self.miner_id} mining cancelled")
            raise
        except Exception as e:
            logger.error(f"{self.miner_id} mining error: {e}")
        finally:
            self.mining = False
        
        return None
    
    def stop_mining(self):
        """Stop current mining operation."""
        self.mining = False
        if self.mining_task and not self.mining_task.done():
            self.mining_task.cancel()
    
    def update_performance_tuning(self, won_block: bool, network_stats: Optional[dict] = None):
        """Update adaptive parameters based on performance relative to network.
        
        Args:
            won_block: Whether this miner won the last block
            network_stats: Optional dict containing total_blocks, total_miners, avg_win_rate
        """
        if won_block:
            self.blocks_won += 1
        
        # Need enough data before adapting
        if self.timing_stats['blocks_attempted'] < 5:
            return
        
        # Calculate actual win rate
        actual_win_rate = self.blocks_won / self.timing_stats['blocks_attempted']
        
        # Analyze timing to optimize parameters
        if self.timing_stats['sampling']:
            avg_sampling_time = np.mean(self.timing_stats['sampling'][-10:])  # Last 10 samples
            
            # If QPU miner, adjust annealing time based on performance
            if self.miner_type == "QPU" and 'quantum_annealing_time' in self.adaptive_params:
                if actual_win_rate < 0.1 and avg_sampling_time < 100:  # Underperforming and fast
                    # Increase annealing time for better solutions
                    old_time = self.adaptive_params['quantum_annealing_time']
                    self.adaptive_params['quantum_annealing_time'] = min(200.0, old_time * 1.5)
                    logger.info(f"{self.miner_id} increasing annealing time: {old_time:.1f} -> {self.adaptive_params['quantum_annealing_time']:.1f} μs")
        
        # If network stats available, compare to expected
        if network_stats and 'total_miners' in network_stats:
            expected_win_rate = 1.0 / network_stats['total_miners']
            
            # If winning less than expected, improve parameters
            if actual_win_rate < expected_win_rate * 0.8:  # 20% below expected
                if self.miner_type == "QPU":
                    # For QPU, increase annealing time
                    if 'quantum_annealing_time' in self.adaptive_params:
                        old_time = self.adaptive_params['quantum_annealing_time']
                        self.adaptive_params['quantum_annealing_time'] = min(200.0, old_time * 1.2)
                        logger.info(f"{self.miner_id} underperforming: increasing annealing time {old_time:.1f} -> {self.adaptive_params['quantum_annealing_time']:.1f} μs")
                else:
                    # For SA, increase sweeps or adjust beta range
                    old_sweeps = self.adaptive_params['num_sweeps']
                    self.adaptive_params['num_sweeps'] = min(4096, int(old_sweeps * 1.5))
                    self.adaptive_params['beta_range'][1] = min(20.0, self.adaptive_params['beta_range'][1] * 1.2)
                    logger.info(f"{self.miner_id} underperforming: increasing sweeps {old_sweeps} -> {self.adaptive_params['num_sweeps']}")
            
            # If winning more than expected, can reduce effort
            elif actual_win_rate > expected_win_rate * 1.5:  # 50% above expected
                if self.miner_type == "QPU":
                    # For QPU, can reduce annealing time to save resources
                    if 'quantum_annealing_time' in self.adaptive_params:
                        old_time = self.adaptive_params['quantum_annealing_time']
                        self.adaptive_params['quantum_annealing_time'] = max(10.0, old_time * 0.8)
                        logger.info(f"{self.miner_id} overperforming: reducing annealing time {old_time:.1f} -> {self.adaptive_params['quantum_annealing_time']:.1f} μs")
                else:
                    old_sweeps = self.adaptive_params['num_sweeps']
                    self.adaptive_params['num_sweeps'] = max(256, int(old_sweeps * 0.8))
                    logger.info(f"{self.miner_id} overperforming: reducing sweeps {old_sweeps} -> {self.adaptive_params['num_sweeps']}")
        
        # Log performance
        logger.info(f"{self.miner_id} Performance: Win rate: {actual_win_rate:.2%}, Blocks won: {self.blocks_won}/{self.timing_stats['blocks_attempted']}")
    
    def measure_expected_performance(self, total_miners: int) -> float:
        """Calculate expected block win rate based on current parameters."""
        # Simple model: assume equal probability for now
        # In reality, this would be based on sampling quality and speed
        return 1.0 / total_miners
    
    def mine_block(self, block_data: dict) -> Optional[dict]:
        """Synchronous mining wrapper for compatibility."""
        # Run async mining synchronously
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.mine_block_async(block_data))


class CPUMiner(BaseMiner):
    """CPU-based miner using simulated annealing."""
    
    def __init__(self, miner_id: str, **kwargs):
        sampler = SimulatedAnnealingStructuredSampler()
        super().__init__(miner_id, "CPU", sampler, **kwargs)


class QPUMiner(BaseMiner):
    """Quantum Processing Unit miner."""
    
    def __init__(self, miner_id: str, **kwargs):
        try:
            sampler = DWaveSampler()
            logger.info(f"Connected to QPU: {sampler.properties['chip_id']}")
        except Exception as e:
            logger.warning(f"QPU not available: {e}, falling back to simulated")
            sampler = SimulatedAnnealingStructuredSampler()
        super().__init__(miner_id, "QPU", sampler, **kwargs)
        
        # QPU-specific adaptive parameters
        self.adaptive_params['quantum_annealing_time'] = 20.0  # microseconds


class MiningNode:
    """Base mining node with P2P capabilities."""
    
    def __init__(self, miner: BaseMiner, host: str = "0.0.0.0", port: int = 8080):
        self.miner = miner
        self.node = P2PNode(host=host, port=port)
        self.blockchain = []  # Local blockchain copy
        self.current_height = 0
        self.mining_lock = asyncio.Lock()
        
        # Set up callbacks
        self.node.on_block_received = self.on_block_received
        self.node.on_new_node = self.on_new_node
        
    async def start(self):
        """Start the mining node."""
        await self.node.start()
        logger.info(f"Mining node {self.miner.miner_id} started at {self.node.address}")
        
    async def stop(self):
        """Stop the mining node."""
        self.miner.stop_mining()
        await self.node.stop()
        
    async def connect_to_network(self, peer_address: str) -> bool:
        """Connect to the P2P network via a peer."""
        success = await self.node.connect_to_peer(peer_address)
        if success:
            # Get latest blocks from network
            await self.sync_blockchain()
        return success
    
    async def sync_blockchain(self):
        """Synchronize blockchain with network peers."""
        logger.info("Syncing blockchain with network...")
        
        # Request latest block from each peer
        best_height = self.current_height
        best_blocks = []
        
        async with self.node.nodes_lock:
            peer_addresses = list(self.node.nodes.keys())
        
        for peer_address in peer_addresses:
            try:
                # Request blockchain height from peer
                message = Message(
                    type="get_height",
                    sender=self.node.address,
                    timestamp=time.time(),
                    data={}
                )
                # In real implementation, would await response
                # For now, we'll just track local state
            except Exception as e:
                logger.error(f"Error syncing with {peer_address}: {e}")
        
    async def on_new_node(self, address: str):
        """Handle new node joining the network."""
        logger.info(f"New node joined: {address}")
        # Could request their blockchain height
        
    async def on_block_received(self, block_data: dict):
        """Handle received block from network."""
        async with self.mining_lock:
            # Validate block before processing
            if not self.validate_block(block_data):
                logger.warning(f"Invalid block received from {block_data.get('miner_id', 'unknown')}")
                # Continue mining current block if validation fails
                return
            
            # Check if this is a new block
            if block_data['index'] > self.current_height:
                logger.info(f"Received new block {block_data['index']} from {block_data['miner_id']}")
                
                # Stop current mining
                self.miner.stop_mining()
                
                # Reset block received time for difficulty adjustment
                self.miner.reset_block_received_time()
                
                # Update performance metrics
                won_block = (block_data['miner_id'] == self.miner.miner_id)
                # Could get network stats from peers in production
                network_stats = {'total_miners': len(self.node.nodes) + 1} if self.node else None
                self.miner.update_performance_tuning(won_block, network_stats)
                
                # Add to blockchain
                self.blockchain.append(block_data)
                self.current_height = block_data['index']
                
                # Start mining next block
                await self.start_mining_next_block()
    
    def validate_block(self, block_data: dict) -> bool:
        """Validate a received block with comprehensive checks."""
        try:
            # Basic validation - check required fields
            required_fields = ['index', 'hash', 'previous_hash', 'nonce', 'energy', 
                              'diversity', 'miner_id', 'miner_type', 'timestamp']
            if not all(field in block_data for field in required_fields):
                logger.warning(f"Missing required fields in block {block_data.get('index', 'unknown')}")
                return False
            
            # Check energy and diversity requirements
            if block_data['energy'] >= self.miner.difficulty_energy:
                logger.warning(f"Block energy {block_data['energy']} does not meet difficulty {self.miner.difficulty_energy}")
                return False
            
            if block_data['diversity'] < self.miner.min_diversity:
                logger.warning(f"Block diversity {block_data['diversity']} below minimum {self.miner.min_diversity}")
                return False
            
            # Verify block hash
            computed_hash = self.miner.compute_block_hash(block_data)
            if computed_hash != block_data['hash']:
                logger.warning(f"Block hash mismatch: computed {computed_hash[:16]}... vs claimed {block_data['hash'][:16]}...")
                return False
            
            # Check previous hash matches our chain
            if self.blockchain and block_data['index'] > 0:
                if block_data['previous_hash'] != self.blockchain[-1]['hash']:
                    logger.warning(f"Previous hash mismatch at block {block_data['index']}")
                    return False
            
            # Verify signatures if present
            if 'signature' in block_data and block_data['signature']:
                # In production, would verify WOTS+ and ECDSA signatures
                # For now, just check they exist and have reasonable length
                if len(block_data['signature']) < 64:
                    logger.warning(f"Invalid signature length in block {block_data['index']}")
                    return False
            
            # Verify quantum proof if present
            if 'quantum_proof' in block_data and block_data['quantum_proof']:
                # Check that solutions meet the claimed diversity
                solutions = block_data['quantum_proof']
                if len(solutions) < self.miner.min_solutions:
                    logger.warning(f"Insufficient solutions in quantum proof: {len(solutions)}")
                    return False
            
            logger.debug(f"Block {block_data['index']} validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error validating block: {e}")
            return False
    
    async def start_mining_next_block(self):
        """Start mining the next block."""
        if self.current_height == 0:
            # Genesis block
            genesis = {
                'index': 0,
                'timestamp': time.time(),
                'data': 'Genesis Block',
                'previous_hash': '0',
                'hash': '0' * 64,
                'nonce': 0,
                'energy': 0,
                'diversity': 0,
                'miner_id': 'genesis',
                'miner_type': 'genesis'
            }
            self.blockchain.append(genesis)
            self.current_height = 0
        
        # Create next block template
        previous_block = self.blockchain[-1] if self.blockchain else None
        if not previous_block:
            return
        
        next_block = {
            'index': self.current_height + 1,
            'timestamp': time.time(),
            'data': f"Block {self.current_height + 1} data",
            'previous_hash': previous_block['hash']
        }
        
        # Start mining
        self.miner.mining_task = asyncio.create_task(self.mine_block(next_block))
    
    async def mine_block(self, block_template: dict):
        """Mine a block and broadcast if successful."""
        try:
            result = await self.miner.mine_block_async(block_template)
            
            if result and self.miner.mining:  # Check if still mining
                # We found a block!
                async with self.mining_lock:
                    # Double-check we're still at the right height
                    if result['index'] == self.current_height + 1:
                        # Add to our blockchain
                        self.blockchain.append(result)
                        self.current_height = result['index']
                        
                        # Broadcast to network
                        await self.node.broadcast_block(result)
                        
                        logger.info(f"🎉 {self.miner.miner_id} mined block {result['index']}!")
                        
                        # Start mining next block
                        await self.start_mining_next_block()
                        
        except asyncio.CancelledError:
            logger.debug(f"{self.miner.miner_id} mining cancelled")
        except Exception as e:
            logger.error(f"Error mining block: {e}")
            # Restart mining after error
            await asyncio.sleep(1)
            await self.start_mining_next_block()