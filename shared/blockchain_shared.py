"""Shared blockchain functionality for modular miners."""

import asyncio
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
import aiohttp

import sys
sys.path.append('..')

from quantum_blockchain import QuantumBlockchain, Block
from CPU import SimulatedAnnealingStructuredSampler
from dwave.system import DWaveSampler
try:
    from GPU import GPUSampler
except ImportError:
    GPUSampler = None
from quantum_blockchain_network import P2PNode, Message
from aiohttp import web

logger = logging.getLogger(__name__)


class NetworkedQuantumBlockchain(QuantumBlockchain):
    """Extended QuantumBlockchain with P2P networking capabilities."""
    
    def __init__(self, node: P2PNode, miner_type: str = "CPU", *args, **kwargs):
        # Don't use competitive mode for individual miners
        kwargs['competitive'] = False
        super().__init__(*args, **kwargs)
        
        self.node = node
        self.miner_type = miner_type
        self.miner_id = f"{miner_type}-{node.port}"
        
        # Override sampler based on miner type
        if miner_type == "CPU":
            self.sampler = SimulatedAnnealingStructuredSampler()
        elif miner_type == "GPU":
            # GPU uses same sampler as CPU but with GPU acceleration
            self.sampler = SimulatedAnnealingStructuredSampler()
        elif miner_type == "QPU":
            try:
                self.sampler = DWaveSampler()
                logger.info(f"Connected to QPU: {self.sampler.properties['chip_id']}")
            except Exception as e:
                logger.warning(f"QPU not available: {e}, falling back to SA")
                self.sampler = SimulatedAnnealingStructuredSampler()
    
    async def sync_with_network(self) -> bool:
        """Synchronize blockchain with network peers."""
        if not self.node.nodes:
            logger.info("No peers found, starting with genesis block")
            return True
        
        logger.info(f"Syncing with {len(self.node.nodes)} peers...")
        
        # Get latest block from each peer
        latest_blocks = []
        
        async with self.node.nodes_lock:
            peer_addresses = list(self.node.nodes.keys())
        
        for peer_address in peer_addresses:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{peer_address}/latest_block",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            block_data = await resp.json()
                            latest_blocks.append(block_data)
            except Exception as e:
                logger.debug(f"Failed to get block from {peer_address}: {e}")
        
        if not latest_blocks:
            logger.warning("Could not sync with any peers")
            return False
        
        # Find the highest block
        highest_block = max(latest_blocks, key=lambda b: b['index'])
        
        # Request full chain from peer with highest block
        # For now, just accept the highest block as latest
        # In production, would validate full chain
        
        # Clear our chain and rebuild
        self.chain = []
        
        # Add blocks up to highest
        # This is simplified - in production would get full chain
        for i in range(highest_block['index'] + 1):
            if i == 0:
                self.create_genesis_block()
            else:
                # For now, create placeholder blocks
                # In production, would get actual blocks from peers
                block = Block(
                    index=i,
                    timestamp=time.time(),
                    data=f"Synced block {i}",
                    previous_hash=self.chain[-1].hash if self.chain else "0",
                    nonce=0
                )
                block.hash = block.compute_hash()
                self.chain.append(block)
        
        logger.info(f"Synced to block height {len(self.chain) - 1}")
        return True
    
    def add_block_from_network(self, block_data: dict) -> bool:
        """Add a block received from the network."""
        # Validate block
        if block_data['index'] != len(self.chain):
            return False
        
        if block_data['index'] > 0 and block_data['previous_hash'] != self.chain[-1].hash:
            return False
        
        # Create block object
        block = Block(
            index=block_data['index'],
            timestamp=block_data['timestamp'],
            data=block_data['data'],
            previous_hash=block_data['previous_hash'],
            nonce=block_data['nonce'],
            quantum_proof=block_data.get('quantum_proof'),
            energy=block_data.get('energy'),
            diversity=block_data.get('diversity'),
            num_valid_solutions=block_data.get('num_valid_solutions'),
            miner_id=block_data.get('miner_id'),
            miner_type=block_data.get('miner_type'),
            mining_time=block_data.get('mining_time'),
            hash=block_data['hash']
        )
        
        self.chain.append(block)
        return True


class SharedMiningNode:
    """Shared mining node implementation using QuantumBlockchain."""
    
    def __init__(self, miner_type: str, miner_id: int, host: str = "0.0.0.0", 
                 port: int = 8080, num_sweeps: int = None, gpu_type: str = None):
        self.miner_type = miner_type
        self.miner_id = f"{miner_type}-{miner_id}"
        self.host = host
        self.port = port
        self.num_sweeps = num_sweeps
        self.gpu_type = gpu_type
        
        # Create P2P node
        self.node = P2PNode(host=host, port=port)
        
        # Create blockchain
        self.blockchain = NetworkedQuantumBlockchain(
            node=self.node,
            miner_type=miner_type
        )
        
        # Mining state
        self.mining = False
        self.mining_task = None
        
        # Set up P2P callbacks
        self.node.on_block_received = self.on_block_received
        self.setup_routes()
    
    def setup_routes(self):
        """Add additional routes for blockchain operations."""
        self.node.app.router.add_get('/latest_block', self.handle_get_latest_block)
        self.node.app.router.add_get('/blockchain', self.handle_get_blockchain)
    
    async def handle_get_latest_block(self, request):
        """Return the latest block."""
        if not self.blockchain.chain:
            return web.json_response({"error": "No blocks"}, status=404)
        
        latest = self.blockchain.get_latest_block()
        return web.json_response({
            'index': latest.index,
            'timestamp': latest.timestamp,
            'data': latest.data,
            'previous_hash': latest.previous_hash,
            'hash': latest.hash,
            'nonce': latest.nonce,
            'energy': latest.energy,
            'diversity': latest.diversity,
            'miner_id': latest.miner_id,
            'miner_type': latest.miner_type
        })
    
    async def handle_get_blockchain(self, request):
        """Return the full blockchain."""
        chain_data = []
        for block in self.blockchain.chain:
            chain_data.append({
                'index': block.index,
                'timestamp': block.timestamp,
                'data': block.data,
                'previous_hash': block.previous_hash,
                'hash': block.hash,
                'nonce': block.nonce,
                'energy': block.energy,
                'diversity': block.diversity,
                'miner_id': block.miner_id,
                'miner_type': block.miner_type
            })
        return web.json_response({'chain': chain_data})
    
    async def start(self):
        """Start the mining node."""
        await self.node.start()
        logger.info(f"{self.miner_id} started at {self.host}:{self.port}")
    
    async def stop(self):
        """Stop the mining node."""
        self.stop_mining()
        await self.node.stop()
    
    async def connect_to_network(self, peer_address: str) -> bool:
        """Connect to network and sync blockchain."""
        success = await self.node.connect_to_peer(peer_address)
        if success:
            await self.blockchain.sync_with_network()
            # Start mining after sync
            await self.start_mining()
        return success
    
    async def on_block_received(self, block_data: dict):
        """Handle new block from network."""
        logger.info(f"Received block {block_data['index']} from network")
        
        # Stop current mining
        self.stop_mining()
        
        # Add block to chain
        if self.blockchain.add_block_from_network(block_data):
            logger.info(f"Added block {block_data['index']} to chain")
            # Start mining next block
            await self.start_mining()
        else:
            logger.warning(f"Failed to add block {block_data['index']}")
    
    def stop_mining(self):
        """Stop current mining operation."""
        self.mining = False
        if self.mining_task and not self.mining_task.done():
            self.mining_task.cancel()
    
    async def start_mining(self):
        """Start mining the next block."""
        if self.mining:
            return
        
        self.mining = True
        self.mining_task = asyncio.create_task(self.mine_next_block())
    
    def quantum_proof_of_work_with_sweeps(self, block: Block, num_sweeps: int):
        """Run quantum_proof_of_work with custom num_sweeps."""
        block_header = f"{block.index}{block.timestamp}{block.data}{block.previous_hash}"
        nonce = 0
        
        while True:
            # Generate quantum model
            h, J = self.blockchain.generate_quantum_model(block_header, nonce)
            
            # Sample with custom num_sweeps
            sampleset = self.blockchain.sampler.sample_ising(h, J, num_reads=100, num_sweeps=num_sweeps)
            
            # Find all solutions meeting energy threshold
            valid_indices = np.where(sampleset.record.energy < self.blockchain.difficulty_energy)[0]
            
            if len(valid_indices) >= self.blockchain.min_solutions:
                # Get unique solutions
                valid_solutions = []
                seen = set()
                
                for idx in valid_indices:
                    solution = tuple(sampleset.record.sample[idx])
                    if solution not in seen:
                        seen.add(solution)
                        valid_solutions.append(list(solution))
                
                # Calculate diversity
                diversity = self.blockchain.calculate_diversity(valid_solutions)
                
                # Check if diversity requirement is met
                if diversity >= self.blockchain.min_diversity and len(valid_solutions) >= self.blockchain.min_solutions:
                    min_energy = float(np.min(sampleset.record.energy[valid_indices]))
                    return nonce, valid_solutions[:self.blockchain.min_solutions], min_energy, diversity, len(valid_solutions)
            
            nonce += 1
            
            # Print progress
            if nonce % 5 == 0:
                min_energy = float(np.min(sampleset.record.energy))
                num_valid = len(valid_indices)
                if num_valid > 0:
                    sample_solutions = [list(sampleset.record.sample[idx]) for idx in valid_indices[:10]]
                    diversity = self.blockchain.calculate_diversity(sample_solutions)
                else:
                    diversity = 0.0
                logger.info(f"Nonce: {nonce}, Min energy: {min_energy:.2f}, Valid: {num_valid}, Diversity: {diversity:.3f}")
    
    async def mine_next_block(self):
        """Mine the next block."""
        try:
            # Get next block data
            latest = self.blockchain.get_latest_block()
            next_index = latest.index + 1 if latest else 0
            
            block = Block(
                index=next_index,
                timestamp=time.time(),
                data=f"Block {next_index} mined by {self.miner_id}",
                previous_hash=latest.hash if latest else "0",
                nonce=0
            )
            
            logger.info(f"{self.miner_id} starting to mine block {next_index}")
            
            # Use appropriate mining method based on miner type
            if self.miner_type == "GPU" and self.gpu_type:
                # Use GPU sampler
                result = await self.gpu_mine_block(block)
            else:
                # Use quantum_proof_of_work from QuantumBlockchain
                # Override num_sweeps if specified
                if self.num_sweeps and hasattr(self.blockchain.sampler, 'parameters'):
                    original_params = {}
                    if 'num_sweeps' in self.blockchain.sampler.parameters:
                        # Temporarily set num_sweeps
                        result = await asyncio.to_thread(
                            self.quantum_proof_of_work_with_sweeps,
                            block,
                            self.num_sweeps
                        )
                    else:
                        result = await asyncio.to_thread(
                            self.blockchain.quantum_proof_of_work,
                            block
                        )
                else:
                    result = await asyncio.to_thread(
                        self.blockchain.quantum_proof_of_work,
                        block
                    )
            
            if result and self.mining:
                nonce, solutions, energy, diversity, num_valid = result
                
                # Update block with mining results
                block.nonce = nonce
                block.quantum_proof = solutions
                block.energy = energy
                block.diversity = diversity
                block.num_valid_solutions = num_valid
                block.miner_id = self.miner_id
                block.miner_type = self.miner_type
                block.hash = block.compute_hash()
                
                # Add to our chain
                self.blockchain.chain.append(block)
                
                logger.info(f"🎉 {self.miner_id} mined block {block.index}!")
                logger.info(f"   Energy: {energy:.2f}, Diversity: {diversity:.3f}")
                
                # Broadcast to network
                await self.node.broadcast_block({
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'data': block.data,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash,
                    'nonce': block.nonce,
                    'energy': block.energy,
                    'diversity': block.diversity,
                    'num_valid_solutions': block.num_valid_solutions,
                    'miner_id': block.miner_id,
                    'miner_type': block.miner_type,
                    'quantum_proof': block.quantum_proof
                })
                
                # Start mining next block
                self.mining = False
                await self.start_mining()
                
        except asyncio.CancelledError:
            logger.debug(f"{self.miner_id} mining cancelled")
        except Exception as e:
            logger.error(f"Mining error: {e}")
            self.mining = False
            # Retry after delay
            await asyncio.sleep(5)
            await self.start_mining()
    
    async def gpu_mine_block(self, block: Block) -> Optional[Tuple]:
        """GPU-specific mining using Modal."""
        if GPUSampler is None:
            logger.warning("GPUSampler not available, falling back to CPU mining")
            return await asyncio.to_thread(
                self.blockchain.quantum_proof_of_work,
                block
            )
            
        try:
            gpu_sampler = GPUSampler(self.gpu_type, logger=logger)
            
            # Override blockchain sampler temporarily
            original_sampler = self.blockchain.sampler
            self.blockchain.sampler = gpu_sampler
            
            # Use quantum_proof_of_work with GPU sampler
            result = await asyncio.to_thread(
                self.blockchain.quantum_proof_of_work,
                block
            )
            
            # Restore original sampler
            self.blockchain.sampler = original_sampler
            
            return result
            
        except Exception as e:
            logger.error(f"GPU mining error: {e}")
            # Fall back to CPU mining
            return await asyncio.to_thread(
                self.blockchain.quantum_proof_of_work,
                block
            )
