import asyncio
import json
import sys
import threading
from typing import Optional
from quantum_blockchain import QuantumBlockchain, Block
from quantum_blockchain_network import P2PNode
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkedQuantumBlockchain(QuantumBlockchain):
    """Quantum blockchain with P2P networking capabilities."""
    
    def __init__(self, node: P2PNode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node
        self.pending_blocks = asyncio.Queue()
        
        # Set up callbacks
        self.node.on_block_received = self.on_block_received
    
    async def on_block_received(self, block_data: dict):
        """Handle received block from network."""
        try:
            # Validate block data
            if not all(key in block_data for key in ['index', 'hash', 'previous_hash']):
                logger.warning("Received invalid block data")
                return
            
            # Check if we already have this block
            if block_data['index'] < len(self.chain):
                if self.chain[block_data['index']].hash == block_data['hash']:
                    return  # Already have this block
            
            # Add to pending blocks for validation
            await self.pending_blocks.put(block_data)
            logger.info(f"Received block {block_data['index']} from network")
            
        except Exception as e:
            logger.error(f"Error handling received block: {e}")
    
    def add_block(self, data: str) -> Block:
        """Override to broadcast new blocks to network."""
        # Mine the block
        block = super().add_block(data)
        
        # Broadcast to network
        block_data = {
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
        }
        
        # Run async broadcast in sync context
        asyncio.run_coroutine_threadsafe(
            self.node.broadcast_block(block_data),
            asyncio.get_event_loop()
        )
        
        return block


class P2PBlockchainNode:
    """Combined P2P node and quantum blockchain."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, 
                 competitive: bool = True, num_qpu: int = 0, 
                 num_sa: int = 2, num_gpu: int = 0):
        self.host = host
        self.port = port
        
        # Create P2P node
        self.node = P2PNode(host=host, port=port)
        
        # Create blockchain
        self.blockchain = NetworkedQuantumBlockchain(
            node=self.node,
            competitive=competitive,
            num_qpu_miners=num_qpu,
            num_sa_miners=num_sa,
            num_gpu_miners=num_gpu
        )
        
        self.event_loop = None
        self.async_thread = None
        self.running = False
    
    def start(self):
        """Start the P2P blockchain node."""
        self.running = True
        
        # Start async event loop in separate thread
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()
        
        # Wait for event loop to be ready
        while self.event_loop is None:
            threading.Event().wait(0.1)
        
        logger.info(f"P2P Blockchain node started at {self.host}:{self.port}")
    
    def _run_async_loop(self):
        """Run async event loop in separate thread."""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        # Start P2P node
        self.event_loop.run_until_complete(self.node.start())
        
        # Run event loop
        self.event_loop.run_forever()
    
    def stop(self):
        """Stop the P2P blockchain node."""
        self.running = False
        
        if self.event_loop:
            # Stop P2P node
            asyncio.run_coroutine_threadsafe(
                self.node.stop(),
                self.event_loop
            ).result()
            
            # Stop event loop
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            
        if self.async_thread:
            self.async_thread.join(timeout=5)
        
        logger.info("P2P Blockchain node stopped")
    
    def connect_to_peer(self, peer_address: str) -> bool:
        """Connect to a peer node."""
        if not self.event_loop:
            return False
        
        future = asyncio.run_coroutine_threadsafe(
            self.node.connect_to_peer(peer_address),
            self.event_loop
        )
        
        return future.result(timeout=10)
    
    def mine_block(self, data: str) -> Block:
        """Mine a new block."""
        return self.blockchain.add_block(data)
    
    def get_node_status(self) -> dict:
        """Get current node status."""
        if not self.event_loop:
            return {"error": "Node not started"}
        
        future = asyncio.run_coroutine_threadsafe(
            self._get_node_status_async(),
            self.event_loop
        )
        
        return future.result(timeout=5)
    
    async def _get_node_status_async(self) -> dict:
        """Async helper to get node status."""
        async with self.node.nodes_lock:
            active_nodes = sum(1 for node in self.node.nodes.values() 
                             if node.is_alive(self.node.node_timeout))
        
        return {
            "address": f"{self.host}:{self.port}",
            "blockchain_height": len(self.blockchain.chain),
            "total_nodes": len(self.node.nodes),
            "active_nodes": active_nodes,
            "miners": {
                "total": len(self.blockchain.miners) if self.blockchain.competitive else 1,
                "stats": self.blockchain.mining_stats if self.blockchain.competitive else {}
            }
        }


def interactive_menu(node: P2PBlockchainNode):
    """Interactive menu for P2P blockchain node."""
    while True:
        print("\n" + "="*50)
        print("Quantum Blockchain P2P Node")
        print("="*50)
        print("1. Mine a block")
        print("2. View blockchain")
        print("3. View network status")
        print("4. Connect to peer")
        print("5. View mining stats")
        print("6. Exit")
        print("-"*50)
        
        try:
            choice = input("Select option: ").strip()
            
            if choice == "1":
                data = input("Enter block data: ").strip()
                if data:
                    print("\nMining block...")
                    block = node.mine_block(data)
                    print(f"✅ Block {block.index} mined successfully!")
                    print(f"   Hash: {block.hash[:16]}...")
                    print(f"   Miner: {block.miner_id}")
                    print(f"   Energy: {block.energy:.2f}")
                
            elif choice == "2":
                node.blockchain.print_chain()
                
            elif choice == "3":
                status = node.get_node_status()
                print(f"\nNode Status:")
                print(f"  Address: {status['address']}")
                print(f"  Blockchain Height: {status['blockchain_height']}")
                print(f"  Connected Nodes: {status['active_nodes']}/{status['total_nodes']}")
                print(f"  Active Miners: {status['miners']['total']}")
                
            elif choice == "4":
                peer = input("Enter peer address (host:port): ").strip()
                if peer:
                    print(f"\nConnecting to {peer}...")
                    if node.connect_to_peer(peer):
                        print("✅ Successfully connected!")
                    else:
                        print("❌ Connection failed!")
                        
            elif choice == "5":
                if node.blockchain.competitive:
                    node.blockchain.print_competitive_summary()
                else:
                    print("Mining stats only available in competitive mode")
                    
            elif choice == "6":
                print("\nShutting down...")
                break
                
            else:
                print("Invalid option!")
                
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point for P2P quantum blockchain."""
    import argparse
    
    parser = argparse.ArgumentParser(description='P2P Quantum Blockchain Node')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--peer', help='Peer address to connect to (host:port)')
    parser.add_argument('--competitive', action='store_true', 
                       help='Enable competitive mining')
    parser.add_argument('--num-qpu', type=int, default=0,
                       help='Number of QPU miners')
    parser.add_argument('--num-sa', type=int, default=2,
                       help='Number of SA miners')
    parser.add_argument('--num-gpu', type=int, default=0,
                       help='Number of GPU miners')
    parser.add_argument('--auto-mine', type=int, default=0,
                       help='Automatically mine N blocks')
    
    args = parser.parse_args()
    
    # Create and start node
    node = P2PBlockchainNode(
        host=args.host,
        port=args.port,
        competitive=args.competitive,
        num_qpu=args.num_qpu,
        num_sa=args.num_sa,
        num_gpu=args.num_gpu
    )
    
    try:
        node.start()
        
        # Connect to peer if specified
        if args.peer:
            print(f"Connecting to peer: {args.peer}")
            if node.connect_to_peer(args.peer):
                print("✅ Successfully joined network!")
            else:
                print("❌ Failed to join network!")
        
        # Auto-mine blocks if requested
        if args.auto_mine > 0:
            print(f"\nAuto-mining {args.auto_mine} blocks...")
            for i in range(args.auto_mine):
                data = f"Auto-mined block {i+1}"
                block = node.mine_block(data)
                print(f"  Block {block.index} mined by {block.miner_id}")
        
        # Run interactive menu
        print(f"\nNode running at {args.host}:{args.port}")
        interactive_menu(node)
        
    finally:
        node.stop()


if __name__ == "__main__":
    main()