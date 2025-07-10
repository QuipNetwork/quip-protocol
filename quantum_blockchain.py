import hashlib
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler

# Load environment variables
load_dotenv()


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
        Initialize a miner with unique ID.
        
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
        
        if hasattr(self.sampler, 'nodelist'):
            # QPU sampler
            h = {i: 0 for i in self.sampler.nodelist}
            J = {edge: 2*np.random.randint(2)-1 for edge in self.sampler.edgelist}
        else:
            # Simulated annealing - scale problem size based on miner
            num_vars = 200 if self.miner_type == "SA" else 64
            h = {i: 0 for i in range(num_vars)}
            J = {}
            for i in range(num_vars):
                for j in range(i+1, num_vars):
                    if np.random.random() < 0.3:
                        J[(i, j)] = 2*np.random.randint(2)-1
        
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
                    # SA gets more reads to compensate
                    sampleset = self.sampler.sample_ising(h, J, num_reads=200, num_sweeps=2048)
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
                 base_difficulty_energy: float = -1200.0,
                 base_min_diversity: float = 0.45,
                 base_min_solutions: int = 20,
                 num_qpu_miners: int = 1,
                 num_sa_miners: int = 1):
        """
        Initialize the quantum blockchain.
        
        Args:
            competitive: Whether to use competitive mining (QPU vs SA)
            base_difficulty_energy: Base energy threshold for all miners
            base_min_diversity: Base diversity requirement for all miners
            base_min_solutions: Base minimum solutions requirement for all miners
            num_qpu_miners: Number of QPU miners to create (competitive mode)
            num_sa_miners: Number of SA miners to create (competitive mode)
        """
        self.chain: List[Block] = []
        self.competitive = competitive
        self.mining_stats = {}  # Will track all miners
        self.num_qpu_miners = num_qpu_miners
        self.num_sa_miners = num_sa_miners
        
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
        self.energy_adjustment_rate = 0.05  # 5% easier per consecutive win
        self.diversity_adjustment_rate = 0.02  # 2% easier per consecutive win
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
                miner_id = f"SA-{i+1}"
                sa_sampler = SimulatedAnnealingSampler()
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
            print(f"✓ Initialized {self.num_sa_miners} SA miner(s)")
            
            # Initialize mining stats for all miners
            for miner in self.miners:
                self.mining_stats[miner.miner_id] = 0
        else:
            # Single miner mode (legacy)
            self.difficulty_energy = -1000.0
            self.min_diversity = 0.25
            self.min_solutions = 10
            self.sampler = SimulatedAnnealingSampler()
            
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
            easiness_factor = 1 - (self.energy_adjustment_rate * self.win_streak)
            self.difficulty_energy = self.base_difficulty_energy * max(0.5, easiness_factor)  # Cap at 50% easier
            self.min_diversity = max(0.2, self.base_min_diversity * (1 - self.diversity_adjustment_rate * self.win_streak * 2))
            self.min_solutions = max(3, int(self.base_min_solutions * (1 - self.solutions_adjustment_rate * self.win_streak * 2)))
            
            print(f"\n🔥 {winner} win streak: {self.win_streak} (Reward multiplier: {self.streak_multiplier}x)")
            print(f"   Difficulty EASED - Energy: {self.difficulty_energy:.1f}, Diversity: {self.min_diversity:.2f}, Solutions: {self.min_solutions}")
        else:
            # Different miner won - make it HARDER
            if self.last_winner and self.win_streak > 0:
                print(f"\n⚡ {winner} broke {self.last_winner}'s {self.win_streak}-block streak!")
            
            self.last_winner = winner
            old_streak = self.win_streak
            self.win_streak = 1
            self.streak_multiplier = 1.0
            
            # Make it HARDER by incrementing difficulty
            hardness_level = min(5, old_streak)  # Cap at 5 levels harder
            hardness_factor = 1 + (self.energy_adjustment_rate * hardness_level)
            self.difficulty_energy = self.base_difficulty_energy * hardness_factor
            self.min_diversity = min(0.7, self.base_min_diversity * (1 + self.diversity_adjustment_rate * hardness_level))
            self.min_solutions = min(50, int(self.base_min_solutions * (1 + self.solutions_adjustment_rate * hardness_level)))
            
            print(f"   Difficulty HARDENED to level {hardness_level} - Energy: {self.difficulty_energy:.1f}, Diversity: {self.min_diversity:.2f}, Solutions: {self.min_solutions}")
            
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
        
        # Get sampler properties
        if hasattr(self.sampler, 'nodelist'):
            # QPU sampler
            h = {i: 0 for i in self.sampler.nodelist}
            J = {edge: 2*np.random.randint(2)-1 for edge in self.sampler.edgelist}
        else:
            # Simulated annealing - create a small fully connected graph
            num_vars = 64  # Small problem size for demonstration
            h = {i: 0 for i in range(num_vars)}
            J = {}
            for i in range(num_vars):
                for j in range(i+1, num_vars):
                    if np.random.random() < 0.3:  # 30% connectivity
                        J[(i, j)] = 2*np.random.randint(2)-1
        
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
                sampleset = self.sampler.sample_ising(h, J, num_reads=100, num_sweeps=4096)
            
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
        self.adjust_difficulty(winning_result.miner_type)
        
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
        type_stats = {"QPU": 0, "SA": 0}
        for miner_id, wins in self.mining_stats.items():
            miner_type = miner_id.split('-')[0]
            type_stats[miner_type] = type_stats.get(miner_type, 0) + wins
        
        for miner_type, wins in type_stats.items():
            percentage = (wins / total_blocks * 100) if total_blocks > 0 else 0
            print(f"  {miner_type}: {wins} blocks ({percentage:.1f}%)")
        
        print(f"\nTotal blocks mined: {total_blocks}")
        
        # Analyze streaks
        current_winner = None
        current_streak = 0
        max_streaks = {"QPU": 0, "SA": 0}
        
        for block in self.chain[1:]:  # Skip genesis
            if block.miner_type == current_winner:
                current_streak += 1
            else:
                if current_winner:
                    max_streaks[current_winner] = max(max_streaks[current_winner], current_streak)
                current_winner = block.miner_type
                current_streak = 1
        
        if current_winner:
            max_streaks[current_winner] = max(max_streaks[current_winner], current_streak)
        
        print(f"\nLongest streaks:")
        for miner, streak in max_streaks.items():
            print(f"  {miner}: {streak} blocks")
        
        # Analyze mining times
        qpu_times = []
        sa_times = []
        
        for block in self.chain[1:]:  # Skip genesis
            if block.miner_type == "QPU":
                qpu_times.append(block.mining_time)
            elif block.miner_type == "SA":
                sa_times.append(block.mining_time)
        
        if qpu_times:
            print(f"\nQPU Mining Times:")
            print(f"  Average: {np.mean(qpu_times):.2f}s")
            print(f"  Min: {np.min(qpu_times):.2f}s")
            print(f"  Max: {np.max(qpu_times):.2f}s")
        
        if sa_times:
            print(f"\nSA Mining Times:")
            print(f"  Average: {np.mean(sa_times):.2f}s")
            print(f"  Min: {np.min(sa_times):.2f}s")
            print(f"  Max: {np.max(sa_times):.2f}s")
    
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
            
            if miner_id.startswith('QPU'):
                # Extract miner number (e.g., QPU-1 -> 1)
                miner_num = int(miner_id.split('-')[1])
                # Create shades of blue - lighter for higher numbers
                # Adjust lightness: 0% = base color, 40% = much lighter
                lightness_factor = min(0.4, (miner_num - 1) * 0.15)
                # Convert hex to RGB, lighten, then back to hex
                r, g, b = int(base_qpu_color[1:3], 16), int(base_qpu_color[3:5], 16), int(base_qpu_color[5:7], 16)
                r = int(r + (255 - r) * lightness_factor)
                g = int(g + (255 - g) * lightness_factor)
                b = int(b + (255 - b) * lightness_factor)
                return f'#{r:02x}{g:02x}{b:02x}'
            else:  # SA miner
                miner_num = int(miner_id.split('-')[1])
                # Create shades of orange - darker for higher numbers
                darkness_factor = min(0.4, (miner_num - 1) * 0.15)
                r, g, b = int(base_sa_color[1:3], 16), int(base_sa_color[3:5], 16), int(base_sa_color[5:7], 16)
                r = int(r * (1 - darkness_factor))
                g = int(g * (1 - darkness_factor))
                b = int(b * (1 - darkness_factor))
                return f'#{r:02x}{g:02x}{b:02x}'
        
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
            
            # Calculate reward
            base_reward = 50
            if i == 1:
                rewards.append(base_reward)
            else:
                streak = 1
                for j in range(i-2, -1, -1):
                    if blocks[j].miner_type == block.miner_type:
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
        
        data_for_violin = []
        labels = []
        if qpu_energies:
            data_for_violin.append(qpu_energies)
            labels.append('QPU')
        if sa_energies:
            data_for_violin.append(sa_energies)
            labels.append('SA')
        
        parts = ax2.violinplot(data_for_violin, showmeans=True, showmedians=True)
        ax2.set_xticks(range(1, len(labels) + 1))
        ax2.set_xticklabels(labels)
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
        
        # 7. Streak analysis
        ax7 = plt.subplot(3, 3, 7)
        streaks = []
        current_miner = miner_types[0]
        current_streak = 1
        
        for i in range(1, len(miner_types)):
            if miner_types[i] == current_miner:
                current_streak += 1
            else:
                streaks.append((current_miner, current_streak))
                current_miner = miner_types[i]
                current_streak = 1
        streaks.append((current_miner, current_streak))
        
        qpu_streaks = [s[1] for s in streaks if s[0] == 'QPU']
        sa_streaks = [s[1] for s in streaks if s[0] == 'SA']
        
        x = ['QPU', 'SA']
        y_mean = [np.mean(qpu_streaks) if qpu_streaks else 0, 
                  np.mean(sa_streaks) if sa_streaks else 0]
        y_max = [max(qpu_streaks) if qpu_streaks else 0, 
                 max(sa_streaks) if sa_streaks else 0]
        
        x_pos = np.arange(len(x))
        ax7.bar(x_pos - 0.2, y_mean, 0.4, label='Average Streak', alpha=0.7)
        ax7.bar(x_pos + 0.2, y_max, 0.4, label='Max Streak', alpha=0.7)
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(x)
        ax7.set_ylabel('Streak Length')
        ax7.set_title('Mining Streak Analysis')
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
    parser.add_argument('--blocks', type=int, default=20,
                       help='Number of blocks to mine (default: 20)')
    
    args = parser.parse_args()
    competitive = args.competitive
    
    if competitive:
        print("Competitive Quantum Mining Demo")
        print(f"QPU Miners: {args.num_qpu}, SA Miners: {args.num_sa}")
        print("=" * 60)
        
        # Inverted difficulty: starts HARD (QPU-favored) and eases with streaks
        blockchain = QuantumBlockchain(
            competitive=True,
            base_difficulty_energy=-1200.0,   # Very challenging for SA, easy for QPU
            base_min_diversity=0.45,          # High diversity requirement
            base_min_solutions=20,            # High solution count requirement
            num_qpu_miners=args.num_qpu,
            num_sa_miners=args.num_sa
        )
        
        # Mine several blocks competitively
        transactions = [
            "Alice initializes quantum wallet",
            "Bob creates entangled transaction",
            "Charlie measures quantum state",
            "Dave collapses superposition",
            "Eve observes quantum channel",
            "Frank teleports QUIP tokens",
            "Grace implements BB84 protocol",
            "Henry verifies quantum signature",
            "Iris broadcasts quantum proof",
            "Jack finalizes consensus",
            "Kate entangles wallet states",
            "Liam measures Bell inequality",
            "Maya performs quantum swap",
            "Noah validates superposition",
            "Olivia completes quantum circuit",
            "Paul initiates phase kickback",
            "Quinn observes decoherence",
            "Rachel applies Hadamard gate",
            "Sam executes CNOT operation",
            "Tara finalizes quantum consensus"
        ]
        
        # Limit transactions to requested number of blocks
        for i, tx in enumerate(transactions[:args.blocks]):
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


if __name__ == "__main__":
    main()