"""Base classes and utilities for quantum blockchain miners."""

import asyncio
import hashlib
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from quantum_blockchain_network import P2PNode, Message

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
    

class BaseMiner:
    """Base class for all miner types."""
    
    def __init__(self, miner_id: str, miner_type: str, 
                 difficulty_energy: float = -15500.0,
                 min_diversity: float = 0.46, 
                 min_solutions: int = 25):
        self.miner_id = miner_id
        self.miner_type = miner_type
        self.difficulty_energy = difficulty_energy
        self.min_diversity = min_diversity
        self.min_solutions = min_solutions
        self.current_block = None
        self.mining = False
        self.mining_task = None
        
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
    
    def compute_block_hash(self, block_data: dict) -> str:
        """Compute hash of block data."""
        block_string = f"{block_data['index']}{block_data['timestamp']}{block_data['data']}{block_data['previous_hash']}{block_data['nonce']}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    async def sample_ising(self, h: dict, J: dict, num_reads: int = 100) -> tuple:
        """Sample from Ising model - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement sample_ising")
    
    async def mine_block_async(self, block_data: dict) -> Optional[dict]:
        """Async mining implementation."""
        self.mining = True
        self.current_block = block_data
        nonce = 0
        start_time = time.time()
        
        block_header = f"{block_data['index']}{block_data['timestamp']}{block_data['data']}{block_data['previous_hash']}"
        
        logger.info(f"{self.miner_id} started mining block {block_data['index']}")
        
        try:
            while self.mining:
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
                    
                    # Calculate diversity
                    diversity = self.calculate_diversity(valid_solutions)
                    
                    # Check if diversity requirement is met
                    if diversity >= self.min_diversity and len(valid_solutions) >= self.min_solutions:
                        mining_time = time.time() - start_time
                        min_energy = float(np.min(energies[valid_indices]))
                        
                        # Create winning block
                        winning_block = block_data.copy()
                        winning_block.update({
                            'nonce': nonce,
                            'energy': min_energy,
                            'diversity': diversity,
                            'miner_id': self.miner_id,
                            'miner_type': self.miner_type,
                            'mining_time': mining_time,
                            'solutions': valid_solutions[:self.min_solutions]
                        })
                        winning_block['hash'] = self.compute_block_hash(winning_block)
                        
                        logger.info(f"{self.miner_id} found valid block! Nonce: {nonce}, Energy: {min_energy:.2f}, Time: {mining_time:.2f}s")
                        return winning_block
                
                nonce += 1
                
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
            # Validate block
            if not self.validate_block(block_data):
                return
            
            # Check if this is a new block
            if block_data['index'] > self.current_height:
                logger.info(f"Received new block {block_data['index']} from {block_data['miner_id']}")
                
                # Stop current mining
                self.miner.stop_mining()
                
                # Add to blockchain
                self.blockchain.append(block_data)
                self.current_height = block_data['index']
                
                # Start mining next block
                await self.start_mining_next_block()
    
    def validate_block(self, block_data: dict) -> bool:
        """Validate a received block."""
        # Basic validation
        required_fields = ['index', 'hash', 'previous_hash', 'nonce', 'energy', 'diversity']
        if not all(field in block_data for field in required_fields):
            return False
        
        # Check energy and diversity requirements
        if block_data['energy'] >= self.miner.difficulty_energy:
            return False
        
        if block_data['diversity'] < self.miner.min_diversity:
            return False
        
        # In real implementation, would verify the quantum proof
        return True
    
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