import argparse
import asyncio
import json
import os
import sys
import threading
import time
from typing import Optional, List
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

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        competitive: bool = True,
        num_qpu: int = 0,
        num_sa: int = 2,
        num_gpu: int = 0,
        gpu_backend: Optional[str] = None,
        gpu_devices: Optional[List[str]] = None,
        gpu_types: Optional[List[str]] = None,
        base_difficulty_energy: Optional[float] = None,
        base_min_diversity: Optional[float] = None,
        base_min_solutions: Optional[int] = None,
        genesis_config_file: Optional[str] = None,
    ):
        self.host = host
        self.port = port

        # Create P2P node
        self.node = P2PNode(host=host, port=port)

        # Build kwargs for blockchain
        qb_kwargs = dict(
            node=self.node,
            competitive=competitive,
            num_qpu_miners=num_qpu,
            num_sa_miners=num_sa,
            num_gpu_miners=num_gpu,
            gpu_backend=gpu_backend,
            gpu_devices=gpu_devices,
            gpu_types=gpu_types,
            genesis_config_file=genesis_config_file,
        )
        if base_difficulty_energy is not None:
            qb_kwargs["base_difficulty_energy"] = base_difficulty_energy
        if base_min_diversity is not None:
            qb_kwargs["base_min_diversity"] = base_min_diversity
        if base_min_solutions is not None:
            qb_kwargs["base_min_solutions"] = base_min_solutions

        # Create blockchain
        self.blockchain = NetworkedQuantumBlockchain(**qb_kwargs)

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


def get_latest_block_from_network(node: P2PBlockchainNode) -> Optional[Block]:
    """Attempt to get the latest block from connected peers."""
    if not node.node.nodes:
        logger.info("No peers connected - will start from genesis block")
        return None
    
    # Try to get latest block info from peers
    # For now, this is a placeholder - in a full implementation,
    # this would query peers for their latest block
    logger.info(f"Connected to {len(node.node.nodes)} peers")
    
    # Check if we have any blocks beyond genesis
    if len(node.blockchain.chain) > 1:
        return node.blockchain.chain[-1]
    
    return None


def start_mining_process(node: P2PBlockchainNode):
    """Start the mining process - check network, get latest block, then start mining."""
    latest_block = get_latest_block_from_network(node)
    
    if latest_block is None:
        logger.info("Starting mining from genesis block")
    else:
        logger.info(f"Latest block: {latest_block.index}, starting mining on block {latest_block.index + 1}")
    
    # Start continuous mining in a background thread
    def mining_loop():
        block_count = 1
        while True:
            try:
                data = f"Mined block {block_count}"
                logger.info(f"🔨 Starting mining attempt for block {len(node.blockchain.chain)}...")
                logger.info(f"Mining parameters: Energy < {node.blockchain.difficulty_energy}, Diversity >= {node.blockchain.min_diversity}, Solutions >= {node.blockchain.min_solutions}")
                
                start_time = time.time()
                block = node.mine_block(data)
                mining_time = time.time() - start_time
                
                logger.info(f"✅ Block {block.index} successfully mined by {block.miner_id} in {mining_time:.2f}s")
                logger.info(f"   Winning Block Details: {{ \"Energy\": {block.energy:.1f}, \"Diversity\": {block.diversity:.3f}, \"Solutions\": {block.num_valid_solutions} }}")
                block_count += 1
                # Small delay between mining attempts
                time.sleep(1)
            except Exception as e:
                logger.error(f"❌ Mining failed: {e}")
                time.sleep(5)
    
    # Start mining in background thread
    mining_thread = threading.Thread(target=mining_loop, daemon=True)
    mining_thread.start()
    logger.info("Mining process started in background")


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


def run_cli_node(
    host: str = "0.0.0.0",
    port: int = 8080,
    peer: Optional[str] = None,
    competitive: bool = True,
    num_qpu: int = 0,
    num_sa: int = 1,
    num_gpu: int = 0,
    genesis_config_file: Optional[str] = None
):
    """Run a node for CLI usage - starts mining and keeps process alive without interactive menu."""
    
    # Configure GPU settings from environment
    backend = os.getenv("QUIP_GPU_BACKEND", "local").lower()
    devices_csv = os.getenv("QUIP_GPU_DEVICES")
    types_csv = os.getenv("QUIP_GPU_TYPES")

    gpu_devices: List[str] = [d.strip() for d in devices_csv.split(",")] if devices_csv else []
    gpu_types: List[str] = [t.strip() for t in types_csv.split(",")] if types_csv else []

    if backend == "local":
        if not gpu_devices:
            gpu_devices = ["0"]  # Default device
    elif backend == "modal":
        if not gpu_types:
            gpu_types = ["t4"]  # Default type

    # Create P2P node
    node = P2PBlockchainNode(
        host=host,
        port=port,
        competitive=competitive,
        num_qpu=num_qpu,
        num_sa=num_sa,
        num_gpu=num_gpu,
        gpu_backend=backend,
        gpu_devices=gpu_devices,
        gpu_types=gpu_types,
        genesis_config_file=genesis_config_file,
    )

    # Attach gpu config onto blockchain for downstream use
    node.blockchain.gpu_backend = backend
    node.blockchain.gpu_devices = gpu_devices
    node.blockchain.gpu_types = gpu_types

    try:
        node.start()

        # Connect to peer if specified
        if peer:
            logger.info(f"Connecting to peer: {peer}")
            if node.connect_to_peer(peer):
                logger.info("✅ Successfully joined network!")
            else:
                logger.info("❌ Failed to join network!")

        # Start mining process
        start_mining_process(node)

        # Keep the CLI process alive without interactive menu
        logger.info(f"Node running at {host}:{port} (CLI mode)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")

    finally:
        node.stop()


def main():
    """Main entry point for P2P quantum blockchain."""

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

    args = parser.parse_args()

    # Read GPU config from environment and override GPU miners if applicable
    backend = os.getenv("QUIP_GPU_BACKEND", "local").lower()
    devices_csv = os.getenv("QUIP_GPU_DEVICES")
    types_csv = os.getenv("QUIP_GPU_TYPES")

    gpu_devices: List[str] = [d.strip() for d in devices_csv.split(",")] if devices_csv else []
    gpu_types: List[str] = [t.strip() for t in types_csv.split(",")] if types_csv else []

    if backend == "local":
        if gpu_devices:
            args.num_gpu = len(gpu_devices)
    elif backend == "modal":
        if gpu_types:
            args.num_gpu = len(gpu_types)
    else:
        logger.warning(f"Unknown QUIP_GPU_BACKEND '{backend}', defaulting to 'local'")
        backend = "local"

    # Create and start node with gpu metadata
    node = P2PBlockchainNode(
        host=args.host,
        port=args.port,
        competitive=args.competitive,
        num_qpu=args.num_qpu,
        num_sa=args.num_sa,
        num_gpu=args.num_gpu,
        gpu_backend=backend,
        gpu_devices=gpu_devices,
        gpu_types=gpu_types,
    )



    # Attach gpu config onto blockchain for downstream use
    node.blockchain.gpu_backend = backend
    node.blockchain.gpu_devices = gpu_devices
    node.blockchain.gpu_types = gpu_types

    try:
        node.start()

        # Connect to peer if specified
        if args.peer:
            print(f"Connecting to peer: {args.peer}")
            if node.connect_to_peer(args.peer):
                print("✅ Successfully joined network!")
            else:
                print("❌ Failed to join network!")

        # Start mining process
        start_mining_process(node)

        # Run interactive menu
        print(f"\nNode running at {args.host}:{args.port}")
        interactive_menu(node)

    finally:
        node.stop()


if __name__ == "__main__":
    main()