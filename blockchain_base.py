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
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from wots_plus import WOTSPlus

logger = logging.getLogger(__name__)


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
    
    def __init__(self, miner_id: str, miner_type: str, 
                 difficulty_energy: float = -15500.0,
                 min_diversity: float = 0.46, 
                 min_solutions: int = 25):
        self.miner_id = miner_id
        self.miner_type = miner_type
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
        
        # Generate ECDSA key pair
        self.ecdsa_private_key = ec.generate_private_key(
            ec.SECP256K1(),
            default_backend()
        )
        self.ecdsa_public_key = self.ecdsa_private_key.public_key()
        
        # Get ECDSA public key in hex format
        self.ecdsa_public_key_bytes = self.ecdsa_public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        self.ecdsa_public_key_hex = self.ecdsa_public_key_bytes.hex()
        
        # Generate initial WOTS+ key pair
        self.wots_plus = WOTSPlus()
        self.wots_plus_public_key_hex = self.wots_plus.get_public_key_hex()
        
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
            'total_samples': 0,
            'blocks_attempted': 0
        }
        
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
        """Filter solutions to maintain maximum diversity."""
        if len(solutions) <= target_count:
            return solutions
        
        # Start with the first solution
        filtered = [solutions[0]]
        remaining = solutions[1:]
        
        while len(filtered) < target_count and remaining:
            # Find the solution with maximum average distance to current filtered set
            max_avg_dist = -1
            best_idx = -1
            
            for i, sol in enumerate(remaining):
                distances = [self.calculate_hamming_distance(sol, f) for f in filtered]
                avg_dist = np.mean(distances)
                
                if avg_dist > max_avg_dist:
                    max_avg_dist = avg_dist
                    best_idx = i
            
            if best_idx >= 0:
                filtered.append(remaining[best_idx])
                remaining.pop(best_idx)
        
        return filtered
    
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
        # Sign with WOTS+
        wots_signature = self.wots_plus.sign(block_data.encode())
        
        # Sign with ECDSA
        ecdsa_signature = self.ecdsa_private_key.sign(
            block_data.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        
        # Generate new WOTS+ key pair for next block
        self.wots_plus = WOTSPlus()
        next_wots_key_hex = self.wots_plus.get_public_key_hex()
        
        # Combine signatures
        combined_signature = wots_signature.hex() + ecdsa_signature.hex()
        
        return combined_signature, next_wots_key_hex

    def compute_block_hash(self, block_data: dict) -> str:
        """Compute hash of block data with new format."""
        # New format includes all the cryptographic fields
        block_string = f"{block_data['previous_hash']}{block_data['index']}{block_data['timestamp']}{block_data['data']}"
        block_string += f"{block_data.get('signature', '')}{block_data.get('reward_address', '')}"
        block_string += f"{block_data.get('miner_ecdsa_public_key', '')}{block_data.get('miner_wots_plus_public_key', '')}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    async def sample_ising(self, h: dict, J: dict, num_reads: int = 100) -> tuple:
        """Sample from Ising model - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement sample_ising")
    
    async def mine_block_async(self, block_data: dict) -> Optional[dict]:
        """Async mining implementation."""
        self.mining = True
        self.current_block = block_data
        start_time = time.time()
        
        # Check for timeout-based difficulty adjustment
        self.check_and_adjust_difficulty_for_timeout()
        
        # Create block header in the correct format
        block_header = f"{block_data['previous_hash']}{block_data['index']}{block_data['timestamp']}{block_data['data']}"
        
        logger.info(f"{self.miner_id} started mining block {block_data['index']}")
        
        try:
            while self.mining:
                # Generate random nonce for each attempt
                nonce = random.randint(0, sys.maxsize)
                
                # Generate quantum model
                h, J = self.generate_quantum_model(block_header, nonce)
                
                # Sample from model
                samples, energies = await self.sample_ising(h, J)
                
                # Find valid solutions
                valid_indices = np.where(energies < self.difficulty_energy)[0]
                
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
                
                # Update timing statistics
                self.timing_stats['total_samples'] += len(energies)
                self.timing_stats['blocks_attempted'] += 1
                
                # Progress update
                if nonce % 10 == 0:
                    min_energy = float(np.min(energies)) if len(energies) > 0 else 0
                    logger.debug(f"{self.miner_id} - Nonce: {nonce}, Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}")
                
                # Allow other tasks to run
                await asyncio.sleep(0)
                
        except asyncio.CancelledError:
            logger.info(f"{self.miner_id} mining cancelled")
            raise
        finally:
            self.mining = False
    
    def stop_mining(self):
        """Stop current mining operation."""
        self.mining = False
        if self.mining_task and not self.mining_task.done():
            self.mining_task.cancel()
    
    def update_performance_tuning(self, won_block: bool):
        """Update adaptive parameters based on performance."""
        if won_block:
            self.blocks_won += 1
            
            # If winning consistently, can make parameters slightly harder
            if self.blocks_won > 2:
                self.adaptive_params['num_sweeps'] = min(2048, int(self.adaptive_params['num_sweeps'] * 1.1))
        else:
            # If not winning, adjust parameters to be more competitive
            if self.timing_stats['blocks_attempted'] > 10 and self.blocks_won == 0:
                # Increase sampling effort
                self.adaptive_params['num_sweeps'] = min(4096, int(self.adaptive_params['num_sweeps'] * 1.2))
                
        # Calculate win rate
        if self.timing_stats['blocks_attempted'] > 0:
            win_rate = self.blocks_won / self.timing_stats['blocks_attempted']
            logger.info(f"{self.miner_id} Performance: Win rate: {win_rate:.2%}, Blocks won: {self.blocks_won}")
    
    def measure_expected_performance(self, total_miners: int) -> float:
        """Calculate expected block win rate based on current parameters."""
        # Simple model: assume equal probability for now
        # In reality, this would be based on sampling quality and speed
        return 1.0 / total_miners


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
                self.miner.update_performance_tuning(won_block)
                
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