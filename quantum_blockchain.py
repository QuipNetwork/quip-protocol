import hashlib
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler

# Optional GPU support via Modal Labs
try:
    import modal
    GPU_AVAILABLE = True
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"Modal not installed. GPU mining disabled. Install with: pip install modal. Error: {e}")

# Load environment variables
load_dotenv()


# Define Modal app globally
if GPU_AVAILABLE:
    gpu_app = modal.App("quantum-blockchain-gpu-miner")
    
    # GPU container image - simplified without CuPy for faster startup
    gpu_image = modal.Image.debian_slim().pip_install(
        "numpy",
        "numba",
    )
    
    # Define GPU functions for each type
    @gpu_app.function(
        image=gpu_image,
        gpu="t4",
        timeout=300,
    )
    def gpu_sample_t4(h_dict, J_dict, num_reads, num_sweeps):
        """GPU sampling on T4 using Numba acceleration."""
        import time

        import numpy as np
        from numba import cuda, jit
        
        start_time = time.time()
        
        # Convert to arrays
        num_vars = max(max(h_dict.keys()), max(max(j) for j in J_dict.keys())) + 1
        h = np.zeros(num_vars)
        for i, val in h_dict.items():
            h[i] = val
        
        # Create coupling matrix
        J_matrix = np.zeros((num_vars, num_vars))
        for (i, j), val in J_dict.items():
            J_matrix[i, j] = val
            J_matrix[j, i] = val
        
        # Numba-accelerated annealing
        @jit(nopython=True)
        def anneal(h, J_matrix, num_sweeps):
            state = np.random.choice(np.array([-1, 1]), size=num_vars)
            betas = np.linspace(0.1, 10.0, num_sweeps)
            
            for beta in betas:
                for _ in range(num_vars):
                    i = np.random.randint(0, num_vars)
                    neighbors_sum = np.dot(J_matrix[i], state)
                    delta_e = 2 * state[i] * (h[i] + neighbors_sum)
                    if delta_e < 0 or np.random.random() < np.exp(-beta * delta_e):
                        state[i] *= -1
            
            energy = -np.dot(state, h) - 0.5 * np.dot(state, np.dot(J_matrix, state))
            return state, energy
        
        # Run parallel simulated annealing
        samples = []
        energies = []
        
        for read in range(num_reads):
            state, energy = anneal(h, J_matrix, num_sweeps)
            samples.append(state.tolist())
            energies.append(float(energy))
        
        return {
            "samples": samples,
            "energies": energies,
            "timing": {"total": time.time() - start_time}
        }
    
    @gpu_app.function(
        image=gpu_image,
        gpu="a10g",
        timeout=300,
    )
    def gpu_sample_a10g(h_dict, J_dict, num_reads, num_sweeps):
        """GPU sampling on A10G - same implementation, different GPU."""
        # Reuse T4 implementation
        return gpu_sample_t4(h_dict, J_dict, num_reads, num_sweeps)
    
    @gpu_app.function(
        image=gpu_image,
        gpu="a100",
        timeout=300,
    )
    def gpu_sample_a100(h_dict, J_dict, num_reads, num_sweeps):
        """GPU sampling on A100 - same implementation, different GPU."""
        # Reuse T4 implementation
        return gpu_sample_t4(h_dict, J_dict, num_reads, num_sweeps)


class GPUSampler:
    """GPU-accelerated sampler using Modal Labs."""
    
    def __init__(self, gpu_type: str = "t4"):
        """
        Initialize GPU sampler.
        
        Args:
            gpu_type: GPU type to use ('t4', 'a10g', 'a100')
                     t4: ~$0.10/hour (budget option)
                     a10g: ~$0.30/hour (balanced)
                     a100: ~$1.00/hour (performance)
        """
        if not GPU_AVAILABLE:
            raise ImportError("Modal not installed. Run: pip install modal")
            
        self.gpu_type = gpu_type
        
        # Map GPU type to function
        self.gpu_functions = {
            "t4": gpu_sample_t4,
            "a10g": gpu_sample_a10g,
            "a100": gpu_sample_a100
        }
        
        if gpu_type not in self.gpu_functions:
            raise ValueError(f"Invalid GPU type: {gpu_type}. Choose from: t4, a10g, a100")
        
        self._gpu_sample_func = self.gpu_functions[gpu_type]
    
    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs):
        """Sample from Ising model using GPU acceleration."""
        # Convert h and J to dictionaries if needed
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J
        
        # Run on GPU via Modal (without context manager to avoid nested app.run)
        result = self._gpu_sample_func.remote(h_dict, J_dict, num_reads, num_sweeps)
        
        # Format result to match D-Wave interface
        class SampleSet:
            def __init__(self, samples, energies):
                self.record = type('Record', (), {
                    'sample': np.array(samples),
                    'energy': np.array(energies)  # Convert to numpy array
                })()
        
        return SampleSet(result["samples"], result["energies"])

class SimulatedAnnealingStructuredSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with identical structure

    """
    def __init__(
        self, qpu=None
    ):
        if qpu is None:
            qpu = DWaveSampler()
        
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
class MiningResult:
    miner_id: str
    miner_type: str
    nonce: int
    solutions: List[List[int]]
    energy: float
    diversity: float
    num_valid: int
    mining_time: float


@dataclass
class Block:
    index: int
    timestamp: float
    data: str
    previous_hash: str
    nonce: int
    quantum_proof: Optional[List[List[int]]] = None  # Multiple solutions
    energy: Optional[float] = None
    diversity: Optional[float] = None  # Average Hamming distance
    num_valid_solutions: Optional[int] = None
    miner_id: Optional[str] = None  # e.g., QPU-1, SA-2
    miner_type: Optional[str] = None  # QPU or SA
    mining_time: Optional[float] = None
    hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute the hash of the block."""
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def __post_init__(self):
        """Compute hash after initialization."""
        if self.hash is None:
            self.hash = self.compute_hash()


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
        
    def calculate_hamming_distance(self, s1: List[int], s2: List[int]) -> int:
        """Calculate Hamming distance between two binary strings."""
        return sum(a != b for a, b in zip(s1, s2))
    
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
    
    def generate_quantum_model(self, block_header: str, nonce: int) -> Tuple[dict, dict]:
        """Generate Ising model parameters based on block header and nonce."""
        seed_string = f"{block_header}{nonce}"
        seed = int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # QPU sampler
        h = {i: 0 for i in self.sampler.nodelist}
        J = {edge: 2*np.random.randint(2)-1 for edge in self.sampler.edgelist}
        
        return h, J
    
    def mine_block(self, block_header: str, result_queue: queue.Queue, stop_event: threading.Event):
        """Mine a block in a separate thread."""
        self.mining = True
        nonce = 0
        start_time = time.time()
        
        print(f"{self.miner_id} started...")
        
        while self.mining and not stop_event.is_set():
            # Check if we should stop before generating model
            if stop_event.is_set():
                print(f"{self.miner_id} interrupted")
                return
                
            # Generate quantum model
            h, J = self.generate_quantum_model(block_header, nonce)
            
            # Check again before sampling
            if stop_event.is_set():
                print(f"{self.miner_id} interrupted")
                return
            
            # Sample from quantum/simulated annealer
            try:
                if self.miner_type == "QPU":
                    sampleset = self.sampler.sample_ising(h, J, num_reads=100, answer_mode='raw')
                else:
                    sampleset = self.sampler.sample_ising(h, J, num_reads=100, num_sweeps=pow(2,(6 + int(self.miner_id[-1]))))
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
                
            # Find all solutions meeting energy threshold
            valid_indices = np.where(sampleset.record.energy < self.difficulty_energy)[0]
            
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
                
                # Check if diversity requirement is met
                if diversity >= self.min_diversity and len(valid_solutions) >= self.min_solutions:
                    mining_time = time.time() - start_time
                    min_energy = float(np.min(sampleset.record.energy[valid_indices]))
                    
                    result = MiningResult(
                        miner_id=self.miner_id,
                        miner_type=self.miner_type,
                        nonce=nonce,
                        solutions=valid_solutions[:self.min_solutions],
                        energy=min_energy,
                        diversity=diversity,
                        num_valid=len(valid_solutions),
                        mining_time=mining_time
                    )
                    
                    result_queue.put(result)
                    print(f"{self.miner_id} found valid block! Nonce: {nonce}, Energy: {min_energy:.2f}, Time: {mining_time:.2f}s")
                    return
            
            nonce += 1
            
            # Progress update
            if nonce % 10 == 0 and len(sampleset.record.energy) > 0:
                min_energy = float(np.min(sampleset.record.energy))
                print(f"{self.miner_id} - Nonce: {nonce}, Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}")
        
        # If we exit the loop due to stop event
        if stop_event.is_set():
            print(f"{self.miner_id} stopped")


class QuantumBlockchain:
    def __init__(self, competitive: bool = False, 
                 base_difficulty_energy: float = -15500.0,
                 base_min_diversity: float = 0.46,
                 base_min_solutions: int = 25,
                 num_qpu_miners: int = 1,
                 num_sa_miners: int = 1,
                 num_gpu_miners: int = 0,
                 gpu_types: List[str] = None):
        """
        Initialize the quantum blockchain.
        
        Args:
            competitive: Whether to use competitive mining (QPU vs SA)
            base_difficulty_energy: Base energy threshold for all miners
            base_min_diversity: Base diversity requirement for all miners
            base_min_solutions: Base minimum solutions requirement for all miners
            num_qpu_miners: Number of QPU miners to create (competitive mode)
            num_sa_miners: Number of SA miners to create (competitive mode)
            num_gpu_miners: Number of GPU miners to create (runs alongside SA miners)
            gpu_types: List of GPU types for each GPU miner ['t4', 'a10g', 'a100']
        """
        self.chain: List[Block] = []
        self.competitive = competitive
        self.mining_stats = {}  # Will track all miners
        self.num_qpu_miners = num_qpu_miners
        self.num_sa_miners = num_sa_miners  # Keep SA miners even with GPU miners
        self.num_gpu_miners = num_gpu_miners
        self.gpu_types = gpu_types or ['t4'] * num_gpu_miners
        
        # Base difficulty parameters
        self.base_difficulty_energy = base_difficulty_energy
        self.base_min_diversity = base_min_diversity
        self.base_min_solutions = base_min_solutions
        
        # Current difficulty (starts at base)
        self.difficulty_energy = base_difficulty_energy
        self.min_diversity = base_min_diversity
        self.min_solutions = base_min_solutions
        
        # Streak tracking
        self.last_winner = None
        self.win_streak = 0
        self.streak_multiplier = 1.0  # Block reward multiplier
        
        # Difficulty adjustment parameters
        self.energy_adjustment_rate = 0.01  # 1% easier per consecutive win
        self.diversity_adjustment_rate = 0.01  # 2% easier per consecutive win
        self.solutions_adjustment_rate = 0.1  # 10% fewer solutions required
        
        if competitive:
            # Initialize competitive miners with SAME parameters
            self.miners = []
            self.miners_by_id = {}  # Track miners by ID
            
            # Try to initialize QPU miners
            try:
                qpu_sampler = DWaveSampler()
                print(f"QPU connected to: {qpu_sampler.properties['chip_id']}")
                
                # Create multiple QPU miners sharing the same sampler
                for i in range(self.num_qpu_miners):
                    miner_id = f"QPU-{i+1}"
                    qpu_miner = Miner(
                        miner_id,
                        "QPU", 
                        qpu_sampler,
                        difficulty_energy=self.difficulty_energy,
                        min_diversity=self.min_diversity,
                        min_solutions=self.min_solutions
                    )
                    self.miners.append(qpu_miner)
                    self.miners_by_id[miner_id] = qpu_miner
                print(f"✓ Initialized {self.num_qpu_miners} QPU miner(s)")
            except Exception as e:
                print(f"QPU not available: {e}")
                
            # Initialize SA miners with SAME parameters
            for i in range(self.num_sa_miners):
                miner_id = f"CPU-{i+1}"
                sa_sampler = SimulatedAnnealingStructuredSampler()
                sa_miner = Miner(
                    miner_id,
                    "SA",
                    sa_sampler,
                    difficulty_energy=self.difficulty_energy,
                    min_diversity=self.min_diversity,
                    min_solutions=self.min_solutions
                )
                self.miners.append(sa_miner)
                self.miners_by_id[miner_id] = sa_miner
            if self.num_sa_miners > 0:
                print(f"✓ Initialized {self.num_sa_miners} SA miner(s)")
            
            # Initialize GPU miners if requested
            if self.num_gpu_miners > 0 and GPU_AVAILABLE:
                for i in range(self.num_gpu_miners):
                    miner_id = f"GPU-{i+1}"
                    gpu_type = self.gpu_types[i] if i < len(self.gpu_types) else 't4'
                    try:
                        gpu_sampler = GPUSampler(gpu_type)
                        gpu_miner = Miner(
                            miner_id,
                            f"GPU-{gpu_type.upper()}",
                            gpu_sampler,
                            difficulty_energy=self.difficulty_energy,
                            min_diversity=self.min_diversity,
                            min_solutions=self.min_solutions
                        )
                        self.miners.append(gpu_miner)
                        self.miners_by_id[miner_id] = gpu_miner
                        print(f"✓ Initialized GPU-{i+1} ({gpu_type.upper()}) miner")
                    except Exception as e:
                        print(f"Failed to initialize GPU-{i+1}: {e}")
                        # Fall back to SA miner
                        sa_sampler = SimulatedAnnealingStructuredSampler()
                        sa_miner = Miner(
                            f"CPU-{self.num_sa_miners + i + 1}",
                            "SA",
                            sa_sampler,
                            difficulty_energy=self.difficulty_energy,
                            min_diversity=self.min_diversity,
                            min_solutions=self.min_solutions
                        )
                        self.miners.append(sa_miner)
                        self.miners_by_id[sa_miner.miner_id] = sa_miner
                        print(f"  Falling back to SA miner")
            elif self.num_gpu_miners > 0:
                print("GPU mining requested but Modal not installed. Using SA miners instead.")
                # Create SA miners as fallback
                for i in range(self.num_gpu_miners):
                    miner_id = f"CPU-{self.num_sa_miners + i + 1}"
                    sa_sampler = SimulatedAnnealingStructuredSampler()
                    sa_miner = Miner(
                        miner_id,
                        "SA",
                        sa_sampler,
                        difficulty_energy=self.difficulty_energy,
                        min_diversity=self.min_diversity,
                        min_solutions=self.min_solutions
                    )
                    self.miners.append(sa_miner)
                    self.miners_by_id[miner_id] = sa_miner
                print(f"✓ Initialized {self.num_gpu_miners} SA miner(s) as GPU fallback")
            
            # Initialize mining stats for all miners
            for miner in self.miners:
                self.mining_stats[miner.miner_id] = 0
        else:
            # Single miner mode (legacy)
            self.difficulty_energy = -1000.0
            self.min_diversity = 0.25
            self.min_solutions = 10
            self.sampler = SimulatedAnnealingStructuredSampler()
            
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self) -> Block:
        """Create the first block in the chain."""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data="Genesis Block",
            previous_hash="0",
            nonce=0
        )
        self.chain.append(genesis)
        return genesis
    
    def get_latest_block(self) -> Block:
        """Get the most recent block in the chain."""
        return self.chain[-1]
    
    def adjust_difficulty(self, winner: str):
        """
        Adjust difficulty based on mining patterns.
        Inverted: Consecutive wins make it EASIER, new winners make it HARDER.
        
        Args:
            winner: The miner type that won the last block
        """
        if winner == self.last_winner:
            # Same miner won again - increase streak and make it EASIER
            self.win_streak += 1
            self.streak_multiplier = 1.0 + (0.5 * self.win_streak)  # 50% bonus per streak
            
            # Make it EASIER (higher energy threshold, lower diversity/solutions)
            self.difficulty_energy = min(-13500, self.base_difficulty_energy * (1 - (self.energy_adjustment_rate * self.win_streak)))  # Cap at -13500 energy 
            self.min_diversity = max(0.2, self.base_min_diversity - self.diversity_adjustment_rate * self.win_streak)
            self.min_solutions = max(10, int(self.base_min_solutions * (1 - self.solutions_adjustment_rate * self.win_streak)))
            
            print(f"\n🔥 {winner} win streak: {self.win_streak} (Reward multiplier: {self.streak_multiplier}x)")
            print(f"   Difficulty EASED to level {self.win_streak} - Energy: {self.difficulty_energy:.1f}, Diversity: {self.min_diversity:.2f}, Solutions: {self.min_solutions}")
        else:
            # Different miner won - make it HARDER
            if self.last_winner and self.win_streak > 0:
                print(f"\n⚡ {winner} broke {self.last_winner}'s {self.win_streak}-block streak!")
            
            self.last_winner = winner
            old_streak = self.win_streak
            self.win_streak = 1
            self.streak_multiplier = 1.0
            
            # Make it HARDER by incrementing difficulty
            self.difficulty_energy = max(-15600, self.base_difficulty_energy * (1 + self.energy_adjustment_rate))
            self.min_diversity = min(0.48, self.base_min_diversity + self.diversity_adjustment_rate)
            self.min_solutions = min(50, int(self.base_min_solutions * (1 + self.solutions_adjustment_rate)))
            
            print(f"   Difficulty HARDENED to level {old_streak - 1} - Energy: {self.difficulty_energy:.1f}, Diversity: {self.min_diversity:.2f}, Solutions: {self.min_solutions}")
            
        # Update all miners with new difficulty
        if self.competitive:
            for miner in self.miners:
                miner.difficulty_energy = self.difficulty_energy
                miner.min_diversity = self.min_diversity
                miner.min_solutions = self.min_solutions
    
    def generate_quantum_model(self, block_header: str, nonce: int) -> Tuple[dict, dict]:
        """
        Generate Ising model parameters based on block header and nonce.
        
        Args:
            block_header: String representation of block header
            nonce: Current nonce value
            
        Returns:
            h: Linear terms (biases)
            J: Quadratic terms (couplers)
        """
        # Create seed from block header and nonce
        seed_string = f"{block_header}{nonce}"
        seed = int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # QPU sampler
        h = {i: 0 for i in self.sampler.nodelist}
        J = {edge: 2*np.random.randint(2)-1 for edge in self.sampler.edgelist}
        
        return h, J
    
    def calculate_hamming_distance(self, s1: List[int], s2: List[int]) -> int:
        """Calculate Hamming distance between two binary strings."""
        return sum(a != b for a, b in zip(s1, s2))
    
    def calculate_diversity(self, solutions: List[List[int]]) -> float:
        """
        Calculate average normalized Hamming distance between all pairs of solutions.
        
        Returns:
            Average normalized Hamming distance (0 to 1)
        """
        if len(solutions) < 2:
            return 0.0
            
        distances = []
        n = len(solutions[0])  # Length of each solution
        
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                dist = self.calculate_hamming_distance(solutions[i], solutions[j])
                distances.append(dist / n)  # Normalize by length
                
        return np.mean(distances) if distances else 0.0
    
    def quantum_proof_of_work(self, block: Block) -> Tuple[int, List[List[int]], float, float, int]:
        """
        Perform quantum proof of work to find valid nonce.
        
        Args:
            block: Block to mine
            
        Returns:
            nonce: Valid nonce
            solutions: List of diverse quantum states
            min_energy: Minimum energy found
            diversity: Average Hamming distance between solutions
            num_valid: Number of valid solutions
        """
        block_header = f"{block.index}{block.timestamp}{block.data}{block.previous_hash}"
        nonce = 0
        
        while True:
            # Generate quantum model
            h, J = self.generate_quantum_model(block_header, nonce)
            
            # Sample from quantum/simulated annealer
            if self.use_qpu:
                sampleset = self.sampler.sample_ising(h, J, num_reads=100, answer_mode='raw')
            else:
                sampleset = self.sampler.sample_ising(h, J, num_reads=100, num_sweeps=512)
            
            # Find all solutions meeting energy threshold
            valid_indices = np.where(sampleset.record.energy < self.difficulty_energy)[0]
            
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
                
                # Check if diversity requirement is met
                if diversity >= self.min_diversity and len(valid_solutions) >= self.min_solutions:
                    min_energy = float(np.min(sampleset.record.energy[valid_indices]))
                    return nonce, valid_solutions[:self.min_solutions], min_energy, diversity, len(valid_solutions)
            
            nonce += 1
            
            # Print progress
            if nonce % 5 == 0:
                min_energy = float(np.min(sampleset.record.energy))
                num_valid = len(valid_indices)
                if num_valid > 0:
                    sample_solutions = [list(sampleset.record.sample[idx]) for idx in valid_indices[:10]]
                    diversity = self.calculate_diversity(sample_solutions)
                else:
                    diversity = 0.0
                print(f"Nonce: {nonce}, Min energy: {min_energy:.2f}, Valid: {num_valid}, Diversity: {diversity:.3f}")
    
    def competitive_mine(self, block_header: str) -> MiningResult:
        """
        Run competitive mining between available miners.
        
        Args:
            block_header: Block header string for mining
            
        Returns:
            Mining result from the winning miner
        """
        print("\nStarting competitive mining...")
        
        # Create result queue and stop event
        result_queue = queue.Queue()
        stop_event = threading.Event()
        
        # Start miners in separate threads
        threads = []
        for miner in self.miners:
            thread = threading.Thread(
                target=miner.mine_block,
                args=(block_header, result_queue, stop_event)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for first valid result
        winning_result = result_queue.get()
        
        # Stop all miners immediately
        stop_event.set()
        
        # Wait for threads to stop with timeout
        for thread in threads:
            thread.join(timeout=2.0)
            if thread.is_alive():
                print(f"Warning: Thread {thread.name} did not stop cleanly")
        
        # Update stats
        self.mining_stats[winning_result.miner_id] += 1
        
        # Update miner's personal stats
        winning_miner = self.miners_by_id[winning_result.miner_id]
        winning_miner.blocks_won += 1
        
        # Calculate block reward
        base_reward = 50  # Base QUIP tokens
        actual_reward = base_reward * self.streak_multiplier
        
        print(f"\n🏆 WINNER: {winning_result.miner_id} ({winning_result.miner_type})")
        print(f"Energy: {winning_result.energy:.2f}, Diversity: {winning_result.diversity:.3f}")
        print(f"Time: {winning_result.mining_time:.2f}s")
        print(f"💰 Block Reward: {actual_reward:.1f} QUIP (Base: {base_reward}, Multiplier: {self.streak_multiplier}x)")
        
        # Update miner's rewards
        winning_miner.total_rewards += actual_reward
        
        # Adjust difficulty for next block
        self.adjust_difficulty(winning_result.miner_id)
        
        return winning_result
    
    def add_block(self, data: str) -> Block:
        """
        Mine and add a new block to the chain.
        
        Args:
            data: Data to include in the block
            
        Returns:
            The newly created block
        """
        previous_block = self.get_latest_block()
        new_block = Block(
            index=previous_block.index + 1,
            timestamp=time.time(),
            data=data,
            previous_hash=previous_block.hash,
            nonce=0
        )
        
        if self.competitive:
            # Competitive mining
            print(f"\n{'='*60}")
            print(f"COMPETITIVE MINING - Block {new_block.index}")
            print(f"{'='*60}")
            print(f"Current Difficulty: Energy < {self.difficulty_energy:.1f}, Diversity >= {self.min_diversity:.2f}, Solutions >= {self.min_solutions}")
            if self.last_winner and self.win_streak > 1:
                print(f"Current Leader: {self.last_winner} (Streak: {self.win_streak-1})")
            
            block_header = f"{new_block.index}{new_block.timestamp}{new_block.data}{new_block.previous_hash}"
            result = self.competitive_mine(block_header)
            
            # Update block with result
            new_block.nonce = result.nonce
            new_block.quantum_proof = result.solutions
            new_block.energy = result.energy
            new_block.diversity = result.diversity
            new_block.num_valid_solutions = result.num_valid
            new_block.miner_id = result.miner_id
            new_block.miner_type = result.miner_type
            new_block.mining_time = result.mining_time
        else:
            # Single miner mode (legacy)
            print(f"\nMining block {new_block.index}...")
            print(f"Difficulty: Energy < {self.difficulty_energy}, Diversity >= {self.min_diversity}, Solutions >= {self.min_solutions}")
            start_time = time.time()
            
            # Perform quantum proof of work
            nonce, quantum_proof, energy, diversity, num_valid = self.quantum_proof_of_work(new_block)
            
            # Update block with proof
            new_block.nonce = nonce
            new_block.quantum_proof = quantum_proof
            new_block.energy = energy
            new_block.diversity = diversity
            new_block.num_valid_solutions = num_valid
            new_block.miner_type = "SA"  # Legacy mode uses SA
            new_block.mining_time = time.time() - start_time
            
            print(f"Block mined! Nonce: {nonce}, Energy: {energy:.2f}, Diversity: {diversity:.3f}, Valid solutions: {num_valid}, Time: {new_block.mining_time:.2f}s")
        
        new_block.hash = new_block.compute_hash()
        self.chain.append(new_block)
        return new_block
    
    def validate_chain(self) -> bool:
        """
        Validate the entire blockchain.
        
        Returns:
            True if chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check hash linkage
            if current_block.previous_hash != previous_block.hash:
                return False
            
            # Check block hash
            if current_block.hash != current_block.compute_hash():
                return False
            
            # Verify quantum proof (optional - can be expensive)
            # This would involve regenerating the quantum model and checking energy
            
        return True
    
    def print_chain(self):
        """Print the blockchain."""
        for block in self.chain:
            print(f"\nBlock {block.index}:")
            print(f"  Timestamp: {block.timestamp}")
            print(f"  Data: {block.data}")
            print(f"  Previous Hash: {block.previous_hash[:16]}...")
            print(f"  Hash: {block.hash[:16]}...")
            print(f"  Nonce: {block.nonce}")
            if block.energy is not None:
                print(f"  Quantum Energy: {block.energy:.2f}")
                print(f"  Diversity: {block.diversity:.3f}")
                print(f"  Valid Solutions: {block.num_valid_solutions}")
                if block.miner_id:
                    print(f"  Miner: {block.miner_id} ({block.miner_type})")
                    print(f"  Mining Time: {block.mining_time:.2f}s")
                elif block.miner_type:
                    print(f"  Miner: {block.miner_type}")
                    print(f"  Mining Time: {block.mining_time:.2f}s")
    
    def print_competitive_summary(self):
        """Print competitive mining summary."""
        if not self.competitive:
            return
            
        print("\n" + "="*60)
        print("COMPETITIVE MINING SUMMARY")
        print("="*60)
        
        # Individual miner stats
        print("\nIndividual Miner Performance:")
        total_blocks = sum(self.mining_stats.values())
        
        for miner_id, wins in sorted(self.mining_stats.items()):
            if wins > 0:
                percentage = (wins / total_blocks * 100) if total_blocks > 0 else 0
                miner = self.miners_by_id.get(miner_id)
                if miner:
                    print(f"  {miner_id}: {wins} blocks ({percentage:.1f}%), {miner.total_rewards:.1f} QUIP earned")
                else:
                    print(f"  {miner_id}: {wins} blocks ({percentage:.1f}%)")
        
        # Type summary
        print("\nSummary by Miner Type:")
        type_stats = {}
        for miner_id, wins in self.mining_stats.items():
            miner_type = miner_id.split('-')[0]
            type_stats[miner_type] = type_stats.get(miner_type, 0) + wins
        
        for miner_type, wins in sorted(type_stats.items()):
            percentage = (wins / total_blocks * 100) if total_blocks > 0 else 0
            print(f"  {miner_type}: {wins} blocks ({percentage:.1f}%)")
        
        print(f"\nTotal blocks mined: {total_blocks}")
        
        # Analyze streaks by individual miner
        current_winner = None
        current_streak = 0
        max_streaks_by_id = {}
        max_streaks_by_type = {}
        
        for block in self.chain[1:]:  # Skip genesis
            # Track by individual miner ID
            if block.miner_id == current_winner:
                current_streak += 1
            else:
                if current_winner:
                    max_streaks_by_id[current_winner] = max(max_streaks_by_id.get(current_winner, 0), current_streak)
                current_winner = block.miner_id
                current_streak = 1
            
            # Also track by type for summary
            base_type = block.miner_type.split('-')[0] if block.miner_type else None
            if base_type not in max_streaks_by_type:
                max_streaks_by_type[base_type] = 0
        
        if current_winner:
            max_streaks_by_id[current_winner] = max(max_streaks_by_id.get(current_winner, 0), current_streak)
        
        # Update max streaks by type from individual streaks
        for miner_id, streak in max_streaks_by_id.items():
            miner_type = miner_id.split('-')[0]
            max_streaks_by_type[miner_type] = max(max_streaks_by_type.get(miner_type, 0), streak)
        
        print(f"\nLongest streaks by individual miner:")
        for miner, streak in sorted(max_streaks_by_id.items()):
            print(f"  {miner}: {streak} blocks")
        
        print(f"\nLongest streaks by type:")
        for miner, streak in sorted(max_streaks_by_type.items()):
            print(f"  {miner}: {streak} blocks")
        
        # Analyze mining times by type
        mining_times_by_type = {}
        
        for block in self.chain[1:]:  # Skip genesis
            base_type = block.miner_type.split('-')[0] if block.miner_type else block.miner_type
            if base_type not in mining_times_by_type:
                mining_times_by_type[base_type] = []
            mining_times_by_type[base_type].append(block.mining_time)
        
        for miner_type, times in sorted(mining_times_by_type.items()):
            if times:
                print(f"\n{miner_type} Mining Times:")
                print(f"  Average: {np.mean(times):.2f}s")
                print(f"  Min: {np.min(times):.2f}s")
                print(f"  Max: {np.max(times):.2f}s")
    
    def generate_benchmark_plots(self, output_prefix: str = "benchmarks/blockchain_benchmark"):
        """Generate comprehensive benchmark plots."""
        if not self.competitive or len(self.chain) < 2:
            print("Not enough data for benchmarking")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create color mapping for individual miners
        def get_miner_color(miner_id: str) -> str:
            """Get unique color shade for each miner based on their ID."""
            base_qpu_color = '#4285F4'  # Google blue
            base_sa_color = '#FF8C00'    # Dark orange
            base_gpu_color = '#00C853'   # Green for GPU miners
            
            if miner_id.startswith('QPU'):
                # Extract miner number (e.g., QPU-1 -> 1)
                miner_num = int(miner_id.split('-')[1])
                # Create shades of blue - lighter for higher numbers
                # Adjust lightness: 0% = base color, 40% = much lighter
                lightness_factor = min(0.4, (miner_num - 1) * 0.20)
                # Convert hex to RGB, lighten, then back to hex
                r, g, b = int(base_qpu_color[1:3], 16), int(base_qpu_color[3:5], 16), int(base_qpu_color[5:7], 16)
                r = int(r + (255 - r) * lightness_factor)
                g = int(g + (255 - g) * lightness_factor)
                b = int(b + (255 - b) * lightness_factor)
                return f'#{r:02x}{g:02x}{b:02x}'
            elif miner_id.startswith('GPU'):
                # Extract miner number (e.g., GPU-1 -> 1)
                miner_num = int(miner_id.split('-')[1])
                # Create shades of green - lighter for higher numbers
                lightness_factor = min(0.4, (miner_num - 1) * 0.15)
                r, g, b = int(base_gpu_color[1:3], 16), int(base_gpu_color[3:5], 16), int(base_gpu_color[5:7], 16)
                r = int(r + (255 - r) * lightness_factor)
                g = int(g + (255 - g) * lightness_factor)
                b = int(b + (255 - b) * lightness_factor)
                return f'#{r:02x}{g:02x}{b:02x}'
            elif miner_id.startswith('CPU'):  # SA/CPU miner
                miner_num = int(miner_id.split('-')[1])
                # Create shades of orange - darker for higher numbers
                lightness_factor = min(0.4, (miner_num - 1) * 0.15)
                r, g, b = int(base_sa_color[1:3], 16), int(base_sa_color[3:5], 16), int(base_sa_color[5:7], 16)
                r = int(r + (255 - r) * lightness_factor)
                g = int(g + (255 - g) * lightness_factor)
                b = int(b + (255 - b) * lightness_factor)
                return f'#{r:02x}{g:02x}{b:02x}'
            else:  # Unknown miner type - use gray
                return '#808080'
        
        # Collect data
        blocks = self.chain[1:]  # Skip genesis
        block_numbers = []
        energies = []
        diversities = []
        mining_times = []
        miner_types = []
        miner_ids = []  # Track individual miner IDs
        difficulties = []
        rewards = []
        
        for i, block in enumerate(blocks, 1):
            block_numbers.append(i)
            energies.append(block.energy)
            diversities.append(block.diversity)
            mining_times.append(block.mining_time)
            miner_types.append(block.miner_type)
            miner_ids.append(block.miner_id)  # Track individual miner
            
            # Calculate difficulty at time of mining
            if i == 1:
                difficulties.append(self.base_difficulty_energy)
            else:
                # Approximate based on streak pattern
                prev_blocks = blocks[:i]
                current_streak = 1
                for j in range(len(prev_blocks)-1, 0, -1):
                    if prev_blocks[j].miner_type == prev_blocks[j-1].miner_type:
                        current_streak += 1
                    else:
                        break
                if block.miner_type == prev_blocks[-2].miner_type if i > 1 else None:
                    diff = self.base_difficulty_energy * (1 + self.energy_adjustment_rate * (current_streak-1))
                else:
                    diff = self.base_difficulty_energy
                difficulties.append(diff)
            
            # Calculate reward based on individual miner streak
            base_reward = 50
            if i == 1:
                rewards.append(base_reward)
            else:
                streak = 1
                for j in range(i-2, -1, -1):
                    if blocks[j].miner_id == block.miner_id:  # Compare by miner_id, not type
                        streak += 1
                    else:
                        break
                rewards.append(base_reward * (1.0 + 0.5 * (streak-1)))
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define base colors for backward compatibility
        qpu_color = '#4285F4'  # Google blue - softer than pure blue
        sa_color = 'orange'
        
        # 1. Mining time by individual miner over blocks
        ax1 = plt.subplot(3, 3, 1)
        for miner_id in sorted(set(miner_ids)):
            miner_blocks = [i for i, m in enumerate(miner_ids) if m == miner_id]
            miner_times_subset = [mining_times[i] for i in miner_blocks]
            color = get_miner_color(miner_id)
            ax1.plot([block_numbers[i] for i in miner_blocks], miner_times_subset, 
                    'o-', label=miner_id, markersize=8, linewidth=2, color=color)
        ax1.set_xlabel('Block Number')
        ax1.set_ylabel('Mining Time (s)')
        ax1.set_title('Mining Time Evolution by Individual Miner')
        ax1.legend()
        
        # 2. Energy achieved by each miner
        ax2 = plt.subplot(3, 3, 2)
        qpu_energies = [e for e, m in zip(energies, miner_types) if m == 'QPU']
        sa_energies = [e for e, m in zip(energies, miner_types) if m == 'SA']
        gpu_energies = [e for e, m in zip(energies, miner_types) if m and m.startswith('GPU')]
        
        data_for_violin = []
        labels = []
        if qpu_energies:
            data_for_violin.append(qpu_energies)
            labels.append('QPU')
        if sa_energies:
            data_for_violin.append(sa_energies)
            labels.append('SA')
        if gpu_energies:
            data_for_violin.append(gpu_energies)
            labels.append('GPU')
        
        if data_for_violin:
            parts = ax2.violinplot(data_for_violin, showmeans=True, showmedians=True)
            ax2.set_xticks(range(1, len(labels) + 1))
            ax2.set_xticklabels(labels)
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy Distribution by Miner')
        
        # 3. Difficulty evolution
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(block_numbers, difficulties, 'k-', linewidth=2)
        # Color points by individual winner
        colors = [get_miner_color(m_id) for m_id in miner_ids]
        ax3.scatter(block_numbers, difficulties, c=colors, s=50, alpha=0.7)
        ax3.set_xlabel('Block Number')
        ax3.set_ylabel('Difficulty (Energy Threshold)')
        ax3.set_title('Difficulty Adjustment Over Time')
        # Create custom legend with miner colors
        from matplotlib.patches import Patch
        legend_elements = [plt.Line2D([0], [0], color='k', linewidth=2, label='Difficulty')]
        # Add legend entries for each unique miner
        for miner_id in sorted(set(miner_ids)):
            legend_elements.append(Patch(facecolor=get_miner_color(miner_id), label=miner_id))
        ax3.legend(handles=legend_elements)
        
        # 4. Diversity scores by individual miner
        ax4 = plt.subplot(3, 3, 4)
        for miner_id in sorted(set(miner_ids)):
            miner_blocks = [i for i, m in enumerate(miner_ids) if m == miner_id]
            miner_diversities = [diversities[i] for i in miner_blocks]
            color = get_miner_color(miner_id)
            ax4.scatter([block_numbers[i] for i in miner_blocks], miner_diversities, 
                       label=miner_id, s=50, alpha=0.7, color=color)
        ax4.set_xlabel('Block Number')
        ax4.set_ylabel('Diversity Score')
        ax4.set_title('Solution Diversity by Individual Miner')
        ax4.legend()
        
        # 5. Block rewards over time
        ax5 = plt.subplot(3, 3, 5)
        ax5.bar(block_numbers, rewards, color=colors, alpha=0.7)
        ax5.set_xlabel('Block Number')
        ax5.set_ylabel('Block Reward (QUIP)')
        ax5.set_title('Block Rewards with Streak Multipliers')
        
        # 6. Win distribution pie chart by individual miner
        ax6 = plt.subplot(3, 3, 6)
        # Get individual miner wins
        individual_wins = []
        individual_labels = []
        individual_colors = []
        
        for miner_id in sorted(self.mining_stats.keys()):
            wins = self.mining_stats[miner_id]
            if wins > 0:
                individual_wins.append(wins)
                individual_labels.append(f"{miner_id} ({wins})")
                individual_colors.append(get_miner_color(miner_id))
        
        # Only create pie chart if there are wins
        if individual_wins:
            ax6.pie(individual_wins, labels=individual_labels, autopct='%1.1f%%', 
                   startangle=90, colors=individual_colors)
        else:
            ax6.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Win Distribution by Individual Miner')
        
        # 7. Streak analysis by individual miner
        ax7 = plt.subplot(3, 3, 7)
        streaks_by_id = []
        current_miner = miner_ids[0] if miner_ids else None
        current_streak = 1
        
        for i in range(1, len(miner_ids)):
            if miner_ids[i] == current_miner:
                current_streak += 1
            else:
                streaks_by_id.append((current_miner, current_streak))
                current_miner = miner_ids[i]
                current_streak = 1
        if current_miner:
            streaks_by_id.append((current_miner, current_streak))
        
        # Group streaks by miner ID
        miner_streak_data = {}
        for miner_id, streak in streaks_by_id:
            if miner_id not in miner_streak_data:
                miner_streak_data[miner_id] = []
            miner_streak_data[miner_id].append(streak)
        
        # Calculate stats for each miner
        miner_labels = sorted(miner_streak_data.keys())
        y_mean = [np.mean(miner_streak_data[m]) if miner_streak_data[m] else 0 for m in miner_labels]
        y_max = [max(miner_streak_data[m]) if miner_streak_data[m] else 0 for m in miner_labels]
        
        x_pos = np.arange(len(miner_labels))
        width = 0.35
        
        # Use miner colors
        mean_colors = [get_miner_color(m) for m in miner_labels]
        max_colors = [get_miner_color(m) for m in miner_labels]
        
        ax7.bar(x_pos - width/2, y_mean, width, label='Average Streak', alpha=0.7, color=mean_colors)
        ax7.bar(x_pos + width/2, y_max, width, label='Max Streak', alpha=0.9, color=max_colors)
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(miner_labels, rotation=45, ha='right')
        ax7.set_ylabel('Streak Length')
        ax7.set_title('Mining Streak Analysis by Individual Miner')
        ax7.legend()
        
        # 8. Mining efficiency (energy per second) by individual miner
        ax8 = plt.subplot(3, 3, 8)
        efficiency = [abs(e)/t for e, t in zip(energies, mining_times)]
        for miner_id in sorted(set(miner_ids)):
            miner_blocks = [i for i, m in enumerate(miner_ids) if m == miner_id]
            miner_efficiency = [efficiency[i] for i in miner_blocks]
            color = get_miner_color(miner_id)
            ax8.scatter([block_numbers[i] for i in miner_blocks], miner_efficiency, 
                       label=miner_id, s=50, alpha=0.7, color=color)
        ax8.set_xlabel('Block Number')
        ax8.set_ylabel('|Energy| / Time')
        ax8.set_title('Mining Efficiency by Individual Miner')
        ax8.legend()
        
        # 9. Cumulative rewards by individual miner
        ax9 = plt.subplot(3, 3, 9)
        # Track cumulative rewards for each miner
        miner_cumulative = {miner_id: [] for miner_id in set(miner_ids)}
        miner_totals = {miner_id: 0 for miner_id in set(miner_ids)}
        
        for i, (miner_id, reward) in enumerate(zip(miner_ids, rewards)):
            miner_totals[miner_id] += reward
            # Update cumulative for all miners at this point
            for m_id in miner_cumulative:
                miner_cumulative[m_id].append(miner_totals[m_id])
        
        # Plot cumulative rewards for each miner
        for miner_id in sorted(set(miner_ids)):
            color = get_miner_color(miner_id)
            ax9.plot(block_numbers, miner_cumulative[miner_id], color=color, 
                    linestyle='-', label=miner_id, linewidth=2)
        
        ax9.set_xlabel('Block Number')
        ax9.set_ylabel('Cumulative Rewards (QUIP)')
        ax9.set_title('Cumulative Earnings by Individual Miner')
        ax9.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive benchmark plot to {output_prefix}_comprehensive.png")
        
        # Additional detailed timing plot
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Mining times with moving average
        window_size = 3
        if len(mining_times) >= window_size:
            moving_avg = np.convolve(mining_times, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(block_numbers[:len(moving_avg)], moving_avg, 'g-', 
                    linewidth=2, label=f'{window_size}-block moving average')
        
        ax1.scatter(block_numbers, mining_times, c=colors, s=50, alpha=0.7)
        ax1.set_xlabel('Block Number')
        ax1.set_ylabel('Mining Time (s)')
        ax1.set_title('Mining Time Analysis')
        ax1.legend()
        
        # Time distribution histograms by individual miner
        for miner_id in sorted(set(miner_ids)):
            miner_times = [t for t, m_id in zip(mining_times, miner_ids) if m_id == miner_id]
            if miner_times:
                color = get_miner_color(miner_id)
                ax2.hist(miner_times, bins=10, alpha=0.5, label=miner_id, color=color)
        ax2.set_xlabel('Mining Time (s)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Mining Time Distribution by Individual Miner')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_timing.png', dpi=300, bbox_inches='tight')
        print(f"Saved timing analysis plot to {output_prefix}_timing.png")


def run_blockchain(args):
    """Run the blockchain demo."""
    competitive = args.competitive
    
    if competitive:
        print("Competitive Quantum Mining Demo")
        print(f"QPU Miners: {args.num_qpu}, SA Miners: {args.num_sa}")
        print("=" * 60)
        
        # Inverted difficulty: starts HARD (QPU-favored) and eases with streaks
        blockchain = QuantumBlockchain(
            competitive=True,
            base_difficulty_energy=-15500.0,   # Very challenging for SA, easy for QPU
            base_min_diversity=0.46,          # High diversity requirement
            base_min_solutions=25,            # High solution count requirement
            num_qpu_miners=args.num_qpu,
            num_sa_miners=args.num_sa,
            num_gpu_miners=args.num_gpu,
            gpu_types=args.gpu_types
        )
        
        # Names in alphabetical order
        names = [
            "Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Henry", 
            "Iris", "Jack", "Kate", "Liam", "Maya", "Noah", "Olivia", "Paul",
            "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
            "Yvonne", "Zachary", "Aaron", "Bella", "Carlos", "Diana", "Ethan",
            "Fiona", "Gabriel", "Hannah", "Isaac", "Julia", "Kevin", "Luna",
            "Marcus", "Nora", "Oscar", "Petra", "Quincy", "Rita", "Stefan",
            "Tina", "Ulrich", "Vera", "Walter", "Xena", "Yuri", "Zara"
        ]
        
        # Quantum-themed actions
        quantum_actions = [
            "initializes quantum wallet",
            "creates entangled transaction",
            "measures quantum state",
            "collapses superposition",
            "observes quantum channel",
            "teleports QUIP tokens",
            "implements BB84 protocol",
            "verifies quantum signature",
            "broadcasts quantum proof",
            "finalizes consensus",
            "entangles wallet states",
            "measures Bell inequality",
            "performs quantum swap",
            "validates superposition",
            "completes quantum circuit",
            "initiates phase kickback",
            "observes decoherence",
            "applies Hadamard gate",
            "executes CNOT operation",
            "performs Grover search",
            "applies Shor's algorithm",
            "creates GHZ state",
            "performs quantum error correction",
            "implements quantum key distribution",
            "executes quantum Fourier transform",
            "creates cat state",
            "performs amplitude amplification",
            "implements variational quantum eigensolver",
            "executes quantum phase estimation",
            "creates cluster state",
            "performs quantum annealing",
            "implements QAOA circuit",
            "validates quantum supremacy",
            "performs quantum tomography",
            "creates magic state",
            "implements surface code",
            "performs quantum bootstrapping",
            "creates topological qubit",
            "implements quantum machine learning",
            "performs quantum sensing"
        ]
        
        # Generate transactions for requested number of blocks
        transactions = []
        for i in range(args.blocks):
            name = names[i % len(names)]
            action = random.choice(quantum_actions)
            transactions.append(f"{name} {action}")
        
        # Mine blocks with generated transactions
        for i, tx in enumerate(transactions):
            print(f"\n📝 Transaction {i+1}/{args.blocks}: {tx}")
            blockchain.add_block(tx)
            time.sleep(0.5)  # Brief pause between blocks
        
        # Print the chain
        print("\nFinal Blockchain:")
        blockchain.print_chain()
        
        # Print competitive summary
        blockchain.print_competitive_summary()
        
        # Generate benchmark plots
        print("\nGenerating benchmark plots...")
        blockchain.generate_benchmark_plots()
    else:
        print("Quantum Blockchain Demo")
        print("=" * 50)
        print("Run with --competitive flag for QPU vs SA competition")
        
        # Create blockchain with diversity requirements
        blockchain = QuantumBlockchain(competitive=False)
        
        # Add some blocks
        transactions = [
            "Alice sends 10 QUIP to Bob",
            "Bob sends 5 QUIP to Charlie",
            "Charlie sends 2 QUIP to Alice"
        ]
        
        for tx in transactions:
            blockchain.add_block(tx)
        
        # Print the chain
        print("\nFinal Blockchain:")
        blockchain.print_chain()
    
    # Validate
    print(f"\nChain valid: {blockchain.validate_chain()}")


def main():
    """Demonstrate quantum blockchain."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Quantum Blockchain Demo')
    parser.add_argument('--competitive', action='store_true', 
                       help='Run competitive mining between QPU and SA miners')
    parser.add_argument('--num-qpu', type=int, default=1,
                       help='Number of QPU miners (default: 1)')
    parser.add_argument('--num-sa', type=int, default=1,
                       help='Number of SA miners (default: 1)')
    parser.add_argument('--num-gpu', type=int, default=0,
                       help='Number of GPU miners (runs concurrently with SA miners)')
    parser.add_argument('--gpu-types', type=str, nargs='+', 
                       default=['t4'],
                       help='GPU types for each GPU miner: t4, a10g, a100')
    parser.add_argument('--blocks', type=int, default=20,
                       help='Number of blocks to mine (default: 20)')
    
    args = parser.parse_args()
    
    # If GPU miners are requested and Modal is available, run with Modal context
    if args.num_gpu > 0 and GPU_AVAILABLE:
        with gpu_app.run():
            run_blockchain(args)
    else:
        run_blockchain(args)


if __name__ == "__main__":
    main()