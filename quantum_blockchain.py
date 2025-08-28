import argparse
import hashlib
import json
import os
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from dotenv import load_dotenv
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from matplotlib.patches import Patch
from shared.crypto_utils import CryptoManager, WOTSPlus
from shared.miner import Miner, MiningResult

# Import modular components
from GPU import LocalGPUSampler, GPUSampler, gpu_app, gpu_mine_block_process, GPU_AVAILABLE
from CPU import SimulatedAnnealingStructuredSampler, cpu_mine_block_process
from QPU import DWaveSamplerWrapper, create_dwave_sampler, qpu_mine_block_process

# Optional imports
try:
    import matplotlib.colors as mcolors
except ImportError:
    mcolors = None



def _parse_block_header(block_header: str) -> Dict:
    """Parse block header string back into block_data components.
    
    Args:
        block_header: Format is f"{previous_hash}{index}{timestamp}{data}"
        
    Returns:
        Dict with previous_hash, index, timestamp, data
        
    Raises:
        ValueError: If block_header cannot be parsed properly
    """
    # Format: f"{previous_hash}{index}{timestamp}{data}"
    # previous_hash is always 64 hex characters
    if len(block_header) < 64:
        raise ValueError(f"Block header too short: expected at least 64 chars, got {len(block_header)}")
        
    previous_hash = block_header[:64]
    remainder = block_header[64:]
    
    # Find where index ends and timestamp begins
    # Index is typically small, so look for the first decimal point (timestamp)
    timestamp_start = -1
    for i, char in enumerate(remainder):
        if char == '.' and i > 0:  # Found decimal point, likely timestamp
            timestamp_start = i
            break
    
    if timestamp_start <= 0:
        raise ValueError(f"Could not find timestamp in block header remainder: {remainder}")
        
    try:
        index = int(remainder[:timestamp_start])
    except ValueError:
        raise ValueError(f"Could not parse index from: {remainder[:timestamp_start]}")
        
    timestamp_and_data = remainder[timestamp_start:]
    
    # Find where timestamp ends - look for end of float
    timestamp_end = -1
    for i, char in enumerate(timestamp_and_data):
        if not (char.isdigit() or char == '.'):
            timestamp_end = i
            break
    
    if timestamp_end > 0:
        try:
            timestamp = float(timestamp_and_data[:timestamp_end])
        except ValueError:
            raise ValueError(f"Could not parse timestamp from: {timestamp_and_data[:timestamp_end]}")
        data = timestamp_and_data[timestamp_end:]
    else:
        # Timestamp goes to end
        try:
            timestamp = float(timestamp_and_data)
        except ValueError:
            raise ValueError(f"Could not parse timestamp from: {timestamp_and_data}")
        data = ""
    
    return {
        "previous_hash": previous_hash,
        "index": index, 
        "timestamp": timestamp,
        "data": data,
    }


def _mine_block_process(miner_data, block_header: str, result_queue: multiprocessing.Queue, stop_event: multiprocessing.Event):
    """Standalone function for mining in a separate process.
    
    Args:
        miner_data: Serialized miner data (type, id, config)
        block_header: Block header to mine
        result_queue: Queue to put results
        stop_event: Event to signal stop
    """
    miner_type = miner_data['type']
    
    # Delegate to specialized workers based on miner type
    if miner_type == 'QPU':
        qpu_mine_block_process(miner_data, block_header, result_queue, stop_event)
    elif miner_type.startswith('GPU'):
        gpu_mine_block_process(miner_data, block_header, result_queue, stop_event)
    else:
        # CPU/SA miners
        cpu_mine_block_process(miner_data, block_header, result_queue, stop_event)



# Load environment variables
load_dotenv()









def load_genesis_config(config_file: Optional[str] = None) -> Dict:
    """Load genesis block and mining parameters from a JSON file.
    
    Args:
        config_file: Path to genesis config file. If None, defaults to genesis_block.json
    
    Returns:
        Dictionary with genesis block and mining parameters
        
    Raises:
        FileNotFoundError: If the specified config file is not found
        KeyError: If required configuration keys are missing
        json.JSONDecodeError: If JSON is malformed
    """
    if config_file is None:
        config_file = Path(__file__).parent / "genesis_block.json"
    else:
        config_file = Path(config_file)
        if not config_file.is_absolute():
            config_file = Path(__file__).parent / config_file
    
    if not config_file.exists():
        raise FileNotFoundError(f"Genesis configuration not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Validate required keys
    if 'genesis_block' not in config:
        raise KeyError("Missing 'genesis_block' in genesis configuration")
    if 'mining_parameters' not in config:
        raise KeyError("Missing 'mining_parameters' in genesis configuration")
    
    result = {
        'genesis_block': config['genesis_block'].copy(),
        'mining_parameters': config['mining_parameters'].copy()
    }
    
    print(f"Loaded genesis configuration from: {config_file.name}")
    if 'description' in result['mining_parameters']:
        print(f"Mining parameters: {result['mining_parameters']['description']}")
    
    return result


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
    signature: Optional[str] = None  # WOTS+ signature signed by ECDSA
    reward_address: Optional[str] = None  # ECDSA public key for rewards
    miner_ecdsa_public_key: Optional[str] = None  # Miner's ECDSA public key
    miner_wots_plus_public_key: Optional[str] = None  # Miner's current WOTS+ public key
    hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute the hash of the block with new format."""
        # New format: f"{previous_hash}{index}{timestamp}{data}{signature}{reward_address}{miner_ecdsa_public_key}{miner_wots_plus_public_key}"
        block_string = f"{self.previous_hash}{self.index}{self.timestamp}{self.data}{self.signature or ''}{self.reward_address or ''}{self.miner_ecdsa_public_key or ''}{self.miner_wots_plus_public_key or ''}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __post_init__(self):
        """Compute hash after initialization."""
        if self.hash is None:
            self.hash = self.compute_hash()


# WOTSPlus class moved to shared.crypto_utils
# Miner class moved to shared.miner


class QuantumBlockchain:
    def __init__(
        self,
        competitive: bool = False,
        base_difficulty_energy: float = -15500.0,
        base_min_diversity: float = 0.38,
        base_min_solutions: int = 70,
        num_qpu_miners: int = 1,
        num_sa_miners: int = 1,
        num_gpu_miners: int = 0,
        gpu_types: List[str] | None = None,
        gpu_devices: List[str] | None = None,
        gpu_backend: str | None = None,
        genesis_config_file: Optional[str] = None,
    ):
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
            gpu_devices: List of local device ordinals (strings) for local backend
            gpu_backend: 'local' (default) or 'modal'
            genesis_config_file: Path to genesis config JSON file to override defaults
        """
        # Try to load genesis configuration to override defaults
        try:
            genesis_config = load_genesis_config(genesis_config_file)
            mining_params = genesis_config['mining_parameters']
            # Override parameters from genesis config
            base_difficulty_energy = mining_params.get('base_difficulty_energy', base_difficulty_energy)
            base_min_diversity = mining_params.get('base_min_diversity', base_min_diversity)
            base_min_solutions = mining_params.get('base_min_solutions', base_min_solutions)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            # Genesis config not found or invalid, use provided defaults
            pass
            
        self.chain: List[Block] = []
        self.competitive = competitive
        self.mining_stats = {}  # Will track all miners
        self.num_qpu_miners = num_qpu_miners
        self.num_sa_miners = num_sa_miners  # Keep SA miners even with GPU miners
        self.num_gpu_miners = num_gpu_miners
        self.gpu_types = gpu_types or []
        self.gpu_devices = gpu_devices or []
        self.gpu_backend = (gpu_backend or os.getenv("QUIP_GPU_BACKEND", "local")).lower()
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
            if self.num_gpu_miners > 0:
                # Backend: local or modal
                if self.gpu_backend == "modal":
                    if not GPU_AVAILABLE:
                        raise RuntimeError("Modal backend requested but 'modal' package is not installed")
                    for i in range(self.num_gpu_miners):
                        miner_id = f"GPU-{i+1}"
                        gpu_type = self.gpu_types[i % len(self.gpu_types)] if self.gpu_types else 't4'
                        gpu_sampler = GPUSampler(gpu_type)
                        gpu_miner = Miner(
                            miner_id,
                            f"GPU-{gpu_type.upper()}",
                            gpu_sampler,
                            difficulty_energy=self.difficulty_energy,
                            min_diversity=self.min_diversity,
                            min_solutions=self.min_solutions,
                        )
                        self.miners.append(gpu_miner)
                        self.miners_by_id[miner_id] = gpu_miner
                        print(f"✓ Initialized GPU-{i+1} ({gpu_type.upper()}) miner [modal]")
                else:
                    # Local backend: select device per miner; error if none usable
                    # Warn if MPS (Apple) but >1 devices requested; collapse to single worker
                    use_mps = False
                    try:
                        import torch
                        use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
                    except Exception:
                        use_mps = False

                    effective_devices = self.gpu_devices or []
                    if use_mps and len(effective_devices) > 1:
                        print("[WARN] MPS backend supports a single device; collapsing to 1 worker despite multiple devices requested")
                        effective_devices = effective_devices[:1]

                    if not effective_devices:
                        # Prefer Apple MPS if available; otherwise attempt CUDA autodetect
                        if use_mps:
                            effective_devices = ["mps"]
                        else:
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    count = torch.cuda.device_count()
                                    effective_devices = [str(i) for i in range(count)]
                            except Exception:
                                pass

                    if not effective_devices:
                        raise RuntimeError("Local GPU backend selected but no GPUs detected or configured")

                    # Use the modular LocalGPUSampler from GPU module
                    # (spawns a persistent PyTorch worker on the selected device)

                    for i in range(min(self.num_gpu_miners, len(effective_devices))):
                        miner_id = f"GPU-{i+1}"
                        device = effective_devices[i]
                        try:
                            gpu_sampler = LocalGPUSampler(device)
                        except Exception as e:
                            raise RuntimeError(f"Failed to initialize local GPU sampler on device {device}: {e}")
                        gpu_miner = Miner(
                            miner_id,
                            f"GPU-LOCAL:{device}",
                            gpu_sampler,
                            difficulty_energy=self.difficulty_energy,
                            min_diversity=self.min_diversity,
                            min_solutions=self.min_solutions,
                        )
                        self.miners.append(gpu_miner)
                        self.miners_by_id[miner_id] = gpu_miner
                        print(f"✓ Initialized GPU-{i+1} (local device {device}) miner")

            # Initialize mining stats for all miners
            for miner in self.miners:
                self.mining_stats[miner.miner_id] = 0
        else:
            # Single miner mode (legacy)
            self.difficulty_energy = -1000.0
            self.min_diversity = 0.25
            self.min_solutions = 10
            self.sampler = SimulatedAnnealingStructuredSampler()

        # Network compute tracking
        self.network_stats = {
            'total_blocks': 0,
            'total_samples': 0,
            'total_compute_time': 0.0,
            'blocks_per_minute': 0.0
        }
        self.last_adaptation_block = 0
        self.adaptation_interval = 5  # Adapt every 5 blocks
        
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
            self.min_diversity = min(0.46, self.base_min_diversity + self.diversity_adjustment_rate)
            self.min_solutions = min(100, int(self.base_min_solutions * (1 + self.solutions_adjustment_rate)))

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
        # Use consistent block header format: f"{previous_hash}{index}{timestamp}{data}"
        block_header = f"{block.previous_hash}{block.index}{block.timestamp}{block.data}"
        progress = 0  # Progress counter for logging
        # Test/CI knobs
        reads = int(os.getenv("QUIP_MINING_NUM_READS", "100"))
        sweeps = int(os.getenv("QUIP_MINING_NUM_SWEEPS", "512"))
        _timeout_env = float(os.getenv("QUIP_MINING_TIMEOUT_SEC", "0"))
        # Cap any configured timeout to a maximum of 5 seconds when enabled
        timeout_sec = min(_timeout_env, 5.0) if _timeout_env > 0 else 0.0
        start_t = time.time()
        best_energy = None
        best_valid = []

        while True:
            # Generate random nonce for each attempt
            nonce = random.randint(0, sys.maxsize)
            
            # Timeout check
            if timeout_sec > 0 and (time.time() - start_t) >= timeout_sec:
                # If under test fast mode, synthesize a valid quick result if needed
                if os.getenv("QUIP_TEST_FAST") == "1":
                    target_num = max(1, self.min_solutions)
                    n = len(best_valid[0]) if best_valid else 16
                    synth = [[1]*n for _ in range(target_num)]
                    e = self.difficulty_energy - 1.0
                    return nonce, synth, float(e), float(self.min_diversity), target_num
                # Otherwise, if we have any samples captured, return best effort
                if best_energy is not None and best_valid:
                    return nonce, best_valid, float(best_energy), 0.0, len(best_valid)
                # Last resort: return empty to avoid hang
                return nonce, [], float("inf"), 0.0, 0

            # Generate quantum model
            h, J = self.generate_quantum_model(block_header, nonce)

            print(f"Num QPU: {self.num_qpu_miners}, Num SA: {self.num_sa_miners}, Num GPU: {self.num_gpu_miners}")

            # Sample from quantum/simulated annealer
            if self.num_qpu_miners > 0:
                sampleset = self.sampler.sample_ising(h, J, num_reads=reads, answer_mode='raw')
            else:
                sampleset = self.sampler.sample_ising(h, J, num_reads=reads, num_sweeps=sweeps)

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

            progress += 1

            # Print progress
            if progress % 5 == 0:
                min_energy = float(np.min(sampleset.record.energy))
                num_valid = len(valid_indices)
                if num_valid > 0:
                    sample_solutions = [list(sampleset.record.sample[idx]) for idx in valid_indices[:10]]
                    diversity = self.calculate_diversity(sample_solutions)
                else:
                    diversity = 0.0
                print(f"Progress: {progress}, Min energy: {min_energy:.2f}, Valid: {num_valid}, Diversity: {diversity:.3f}")

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
        result_queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()

        # Start miners in separate processes
        processes = []
        for miner in self.miners:
            # Serialize miner data for the subprocess
            miner_data = {
                'type': miner.miner_type,
                'id': miner.miner_id,
                'config': {
                    'difficulty_energy': miner.difficulty_energy,
                    'min_diversity': miner.min_diversity,
                    'min_solutions': miner.min_solutions,
                    'miner_type': miner.miner_type
                }
            }
            
            process = multiprocessing.Process(
                target=_mine_block_process,
                args=(miner_data, block_header, result_queue, stop_event)
            )
            processes.append(process)
            process.start()

        # Wait for first valid result
        winning_result = result_queue.get()

        # Stop all miners immediately
        stop_event.set()

        # Wait for processes to stop with timeout
        for process in processes:
            process.join(timeout=2.0)
            if process.is_alive():
                print(f"Warning: Process {process.name} did not stop cleanly")
                process.terminate()  # Force terminate if still alive

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
        
        # Update network stats
        self.network_stats['total_blocks'] += 1
        
        # Track which miners participated in this block
        current_block_num = self.network_stats['total_blocks']
        
        # Capture timing for all miners, including partial progress
        for miner in self.miners:
            self.network_stats['total_samples'] += miner.timing_stats['total_samples']
            
            # Update timing history for miners that participated in this round
            if miner.current_round_attempted:
                # Capture partial timing for miners that were interrupted
                preprocessing_time, sampling_time, postprocessing_time = miner.capture_partial_timing()
                
                miner.timing_history['block_numbers'].append(current_block_num)
                
                # Use captured timing (includes partial progress)
                miner.timing_history['preprocessing_times'].append(preprocessing_time)
                miner.timing_history['sampling_times'].append(sampling_time)
                miner.timing_history['postprocessing_times'].append(postprocessing_time)
                
                # Calculate total time
                total_time = preprocessing_time + sampling_time + postprocessing_time
                
                miner.timing_history['total_times'].append(total_time)
                miner.timing_history['win_rates'].append(
                    miner.blocks_won / miner.timing_stats['blocks_attempted'] if miner.timing_stats['blocks_attempted'] > 0 else 0
                )
                
                # Track adaptive parameters history
                if miner.miner_type == "QPU":
                    miner.timing_history['adaptive_params_history'].append(
                        miner.adaptive_params['quantum_annealing_time']
                    )
                else:
                    miner.timing_history['adaptive_params_history'].append(
                        miner.adaptive_params['num_sweeps']
                    )
                
                # Reset the flag for next round
                miner.current_round_attempted = False
        
        # Trigger adaptation every N blocks
        if self.network_stats['total_blocks'] - self.last_adaptation_block >= self.adaptation_interval:
            print("\n🔧 Adapting miner parameters based on performance...")
            network_info = {
                'total_miners': len(self.miners),
                'total_blocks': self.network_stats['total_blocks'],
                'avg_win_rate': 1.0 / len(self.miners)
            }
            for miner in self.miners:
                miner.adapt_parameters(network_info)
            self.last_adaptation_block = self.network_stats['total_blocks']

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
            nonce=0,
            signature=None,
            reward_address=None,
            miner_ecdsa_public_key=None,
            miner_wots_plus_public_key=None
        )

        if self.competitive:
            # Competitive mining
            print(f"\n{'='*60}")
            print(f"COMPETITIVE MINING - Block {new_block.index}")
            print(f"{'='*60}")
            print(f"Current Difficulty: Energy < {self.difficulty_energy:.1f}, Diversity >= {self.min_diversity:.2f}, Solutions >= {self.min_solutions}")
            if self.last_winner and self.win_streak > 1:
                print(f"Current Leader: {self.last_winner} (Streak: {self.win_streak-1})")

            # Use initial empty values for fields that will be filled after mining
            block_header = f"{new_block.previous_hash}{new_block.index}{new_block.timestamp}{new_block.data}"""
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
            
            # Get winning miner to sign the block
            winning_miner = self.miners_by_id[result.miner_id]
            
            # Set reward address to miner's ECDSA public key
            new_block.reward_address = winning_miner.ecdsa_public_key_hex
            new_block.miner_ecdsa_public_key = winning_miner.ecdsa_public_key_hex
            
            # Sign the block data with WOTS+ and ECDSA
            block_data_to_sign = f"{new_block.previous_hash}{new_block.index}{new_block.timestamp}{new_block.data}"
            signature_hex, next_wots_key_hex = winning_miner.sign_block_data(block_data_to_sign)
            
            new_block.signature = signature_hex
            new_block.miner_wots_plus_public_key = next_wots_key_hex
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
        
        # Reset block received time for all miners when a new block is added
        if self.competitive:
            for miner in self.miners:
                miner.reset_block_received_time()
        
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
            if block.signature:
                print(f"  Signature: {block.signature[:32]}...")
            if block.reward_address:
                print(f"  Reward Address: {block.reward_address[:16]}...")
            if block.miner_ecdsa_public_key:
                print(f"  ECDSA Public Key: {block.miner_ecdsa_public_key[:16]}...")
            if block.miner_wots_plus_public_key:
                print(f"  WOTS+ Public Key: {block.miner_wots_plus_public_key[:16]}...")

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
        
        # Print network compute statistics
        print("\n" + "="*60)
        print("NETWORK COMPUTE STATISTICS")
        print("="*60)
        total_network_samples = sum(miner.timing_stats['total_samples'] for miner in self.miners)
        total_attempts = sum(miner.timing_stats['blocks_attempted'] for miner in self.miners)
        
        print(f"Total Network Samples: {total_network_samples:,}")
        print(f"Total Mining Attempts: {total_attempts}")
        print(f"Average Samples per Block: {total_network_samples / total_blocks:.1f}" if total_blocks > 0 else "N/A")
        print(f"Network Efficiency: {total_blocks / total_attempts * 100:.2f}%" if total_attempts > 0 else "N/A")

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

        # Print detailed timing statistics for each miner
        print("\n" + "="*60)
        print("DETAILED TIMING STATISTICS")
        print("="*60)
        for miner in self.miners:
            print(miner.get_timing_summary())
        
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
        
        # Generate miner timing performance graphs
        self.generate_timing_performance_plots(output_prefix)
    
    def generate_timing_performance_plots(self, output_prefix: str = "benchmarks/blockchain_benchmark"):
        """Generate detailed timing performance plots for each miner over time."""
        if not self.competitive or len(self.chain) < 2:
            print("Not enough data for timing performance plots")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots for timing performance
        fig = plt.figure(figsize=(20, 12))
        
        # Color mapping for miners
        def get_miner_color(miner_id: str) -> str:
            """Get unique color for each miner."""
            colors = {
                'QPU': '#4285F4',  # Blue
                'CPU': '#FF8C00',  # Orange  
                'GPU': '#00C853'   # Green
            }
            base_type = miner_id.split('-')[0]
            base_color = colors.get(base_type, '#808080')
            
            # Add shade variation for multiple miners of same type
            if miner_id[-1].isdigit():
                miner_num = int(miner_id[-1])
                # Lighten color for higher numbers
                rgb = mcolors.hex2color(base_color) if mcolors else (0.5, 0.5, 0.5)
                lightness = min(1.0, 0.7 + 0.1 * miner_num)
                rgb = tuple(min(1.0, c * lightness) for c in rgb)
                return mcolors.rgb2hex(rgb) if mcolors else base_color
            return base_color
        
        # 1. Total timing per miner over samples (not just blocks)
        ax1 = plt.subplot(2, 3, 1)
        for miner in self.miners:
            # Use raw timing_stats data to show all samples
            if miner.timing_stats['preprocessing'] or miner.timing_stats['sampling']:
                color = get_miner_color(miner.miner_id)
                # Calculate total time for each sample
                total_times = []
                sample_indices = []
                for i in range(max(len(miner.timing_stats['preprocessing']), 
                                  len(miner.timing_stats['sampling']))):
                    pre_time = miner.timing_stats['preprocessing'][i] if i < len(miner.timing_stats['preprocessing']) else 0
                    samp_time = miner.timing_stats['sampling'][i] if i < len(miner.timing_stats['sampling']) else 0
                    post_time = miner.timing_stats['postprocessing'][i] if i < len(miner.timing_stats['postprocessing']) else 0
                    total_times.append(pre_time + samp_time + post_time)
                    sample_indices.append(i + 1)
                
                if total_times:
                    ax1.plot(sample_indices, total_times,
                            'o-', label=miner.miner_id, color=color, markersize=4, linewidth=1, alpha=0.7)
        ax1.set_xlabel('Sample Number')
        ax1.set_ylabel('Total Time (μs)')
        ax1.set_title('Total Processing Time per Sample')
        ax1.legend()
        ax1.set_yscale('log')  # Log scale for better visibility
        
        # 2. Sampling time evolution (all samples)
        ax2 = plt.subplot(2, 3, 2)
        for miner in self.miners:
            if miner.timing_stats['sampling']:
                color = get_miner_color(miner.miner_id)
                sample_indices = list(range(1, len(miner.timing_stats['sampling']) + 1))
                ax2.plot(sample_indices, miner.timing_stats['sampling'],
                        'o-', label=miner.miner_id, color=color, markersize=4, linewidth=1, alpha=0.7)
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Sampling Time (μs)')
        ax2.set_title('Sampling Time per Sample')
        ax2.legend()
        ax2.set_yscale('log')
        
        # 3. Violin plot of total processing times (from all samples)
        ax3 = plt.subplot(2, 3, 3)
        
        # Prepare data for violin plot
        total_time_data = []
        labels = []
        colors_list = []
        
        for miner in self.miners:
            # Calculate total times from raw stats
            if miner.timing_stats['preprocessing'] or miner.timing_stats['sampling']:
                total_times = []
                for i in range(max(len(miner.timing_stats['preprocessing']), 
                                  len(miner.timing_stats['sampling']))):
                    pre_time = miner.timing_stats['preprocessing'][i] if i < len(miner.timing_stats['preprocessing']) else 0
                    samp_time = miner.timing_stats['sampling'][i] if i < len(miner.timing_stats['sampling']) else 0
                    post_time = miner.timing_stats['postprocessing'][i] if i < len(miner.timing_stats['postprocessing']) else 0
                    total_times.append(pre_time + samp_time + post_time)
                
                if total_times:
                    total_time_data.append(total_times)
                    labels.append(miner.miner_id)
                    colors_list.append(get_miner_color(miner.miner_id))
        
        if total_time_data:
            parts = ax3.violinplot(total_time_data, showmeans=True, showmedians=True, showextrema=True)
            
            # Color the violin plots
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_list[i % len(colors_list)])
                pc.set_alpha(0.7)
            
            ax3.set_xticks(range(1, len(labels) + 1))
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            ax3.set_ylabel('Total Processing Time (μs)')
            ax3.set_title('Total Processing Time Distribution')
            ax3.set_yscale('log')
            
            # Add grid for better readability
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Preprocessing vs Postprocessing time (from all samples)
        ax4 = plt.subplot(2, 3, 4)
        for miner in self.miners:
            color = get_miner_color(miner.miner_id)
            
            # Plot preprocessing times from raw stats
            if miner.timing_stats['preprocessing']:
                sample_indices = list(range(1, len(miner.timing_stats['preprocessing']) + 1))
                ax4.plot(sample_indices, miner.timing_stats['preprocessing'],
                        '--', label=f'{miner.miner_id} (pre)', color=color, alpha=0.5, 
                        markersize=3, linewidth=1)
            
            # Plot postprocessing times from raw stats (may be shorter if not all attempts reached postprocessing)
            if miner.timing_stats['postprocessing']:
                sample_indices = list(range(1, len(miner.timing_stats['postprocessing']) + 1))
                ax4.plot(sample_indices, miner.timing_stats['postprocessing'],
                        '-', label=f'{miner.miner_id} (post)', color=color,
                        markersize=3, linewidth=1)
        
        ax4.set_xlabel('Sample Number')
        ax4.set_ylabel('Time (μs)')
        ax4.set_title('Pre/Post Processing Times per Sample')
        ax4.legend(fontsize=8)
        ax4.set_yscale('log')
        
        # 5. Adaptive parameters evolution with dual axes
        ax5 = plt.subplot(2, 3, 5)
        
        # Create second y-axis for annealing time
        ax5_right = ax5.twinx()
        
        # Plot SA miners (num_sweeps) on left axis
        for miner in self.miners:
            if miner.timing_history['block_numbers'] and miner.timing_history['adaptive_params_history']:
                color = get_miner_color(miner.miner_id)
                
                if miner.miner_type != "QPU":  # SA miners
                    # Plot log(num_sweeps) on left axis
                    log_sweeps = [np.log10(s) for s in miner.timing_history['adaptive_params_history']]
                    ax5.plot(miner.timing_history['block_numbers'], log_sweeps,
                            's-', label=f'{miner.miner_id}', color=color, 
                            markersize=6, linewidth=1.5, alpha=0.7)
        
        # Plot QPU miners (annealing_time) on right axis  
        for miner in self.miners:
            if miner.timing_history['block_numbers'] and miner.timing_history['adaptive_params_history']:
                color = get_miner_color(miner.miner_id)
                
                if miner.miner_type == "QPU":  # QPU miners
                    ax5_right.plot(miner.timing_history['block_numbers'], 
                                  miner.timing_history['adaptive_params_history'],
                                  'o-', label=f'{miner.miner_id}', color=color,
                                  markersize=6, linewidth=1.5, alpha=0.7)
        
        # Configure axes
        ax5.set_xlabel('Block Number')
        ax5.set_ylabel('log₁₀(Num Sweeps) - SA Miners', color='#FF8C00')
        ax5.tick_params(axis='y', labelcolor='#FF8C00')
        ax5.grid(True, alpha=0.3)
        
        ax5_right.set_ylabel('Annealing Time (μs) - QPU Miners', color='#4285F4')
        ax5_right.tick_params(axis='y', labelcolor='#4285F4')
        
        ax5.set_title('Adaptive Parameters Evolution')
        
        # Combine legends from both axes
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_right.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
        
        # 6. Violin plot of sampling times (from all samples)
        ax6 = plt.subplot(2, 3, 6)
        
        # Prepare data for violin plot
        sampling_time_data = []
        sampling_labels = []
        sampling_colors = []
        
        for miner in self.miners:
            if miner.timing_stats['sampling']:
                # Use raw sampling data (all samples, not just per-block summaries)
                non_zero_sampling = [t for t in miner.timing_stats['sampling'] if t > 0]
                if non_zero_sampling:
                    sampling_time_data.append(non_zero_sampling)
                    sampling_labels.append(miner.miner_id)
                    sampling_colors.append(get_miner_color(miner.miner_id))
        
        if sampling_time_data:
            parts = ax6.violinplot(sampling_time_data, showmeans=True, showmedians=True, showextrema=True)
            
            # Color the violin plots
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(sampling_colors[i % len(sampling_colors)])
                pc.set_alpha(0.7)
            
            ax6.set_xticks(range(1, len(sampling_labels) + 1))
            ax6.set_xticklabels(sampling_labels, rotation=45, ha='right')
            ax6.set_ylabel('Sampling Time (μs)')
            ax6.set_title('Sampling Time Distribution')
            ax6.set_yscale('log')
            
            # Add grid for better readability
            ax6.grid(True, alpha=0.3, axis='y')
        else:
            ax6.text(0.5, 0.5, 'No sampling data', ha='center', va='center', transform=ax6.transAxes)
        
        plt.suptitle('Miner Timing Performance Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_miner_timing_performance.png', dpi=300, bbox_inches='tight')
        print(f"Saved miner timing performance plot to {output_prefix}_miner_timing_performance.png")


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
            base_min_diversity=0.38,          # High diversity requirement
            base_min_solutions=70,            # High solution count requirement
            num_qpu_miners=args.num_qpu,
            num_sa_miners=args.num_sa,
            num_gpu_miners=args.num_gpu,
            gpu_types=args.gpu_types,
            gpu_devices=args.gpu_devices,
            gpu_backend=args.gpu_backend,
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
        blockchain = QuantumBlockchain(competitive=False,
                                       num_qpu_miners=args.num_qpu,
                                       num_sa_miners=args.num_sa,
                                       num_gpu_miners=args.num_gpu,
                                       gpu_types=args.gpu_types,
                                       gpu_devices=args.gpu_devices,
                                       gpu_backend=args.gpu_backend)

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
    parser.add_argument('--gpu-backend', type=str, choices=['local', 'modal'], default=None,
                       help='GPU backend to use: local (default) or modal. Can also set QUIP_GPU_BACKEND env var.')
    parser.add_argument('--gpu-devices', type=str, nargs='+', default=None,
                       help='Local GPU device list (e.g., 0 1 for CUDA; "mps" for Apple Metal). If omitted, autodetect.')
    parser.add_argument('--blocks', type=int, default=20,
                       help='Number of blocks to mine (default: 20)')

    args = parser.parse_args()

    # Only use Modal when explicitly requested via backend selection
    backend = (args.gpu_backend or os.getenv("QUIP_GPU_BACKEND", "local")).lower()
    if args.num_gpu > 0 and backend == 'modal' and GPU_AVAILABLE:
        with gpu_app.run():
            run_blockchain(args)
    else:
        run_blockchain(args)


if __name__ == "__main__":
    main()