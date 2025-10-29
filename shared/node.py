"""Node class for quantum blockchain network participation."""

import asyncio
import inspect
import json
import logging
import multiprocessing
import os
import socket
from queue import Empty
import time
from blake3 import blake3
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Callable
from multiprocessing.synchronize import Event as EventType
from logging.handlers import QueueListener
import aiohttp

from shared.block_requirements import compute_next_block_requirements, validate_block

if TYPE_CHECKING:
    pass

from shared import block
from shared.block_signer import BlockSigner
from shared.block import Block, MinerInfo
from shared.miner import Miner, MiningResult
from shared.logging_config import init_component_logger
from shared.time_utils import utc_timestamp_float, utc_timestamp
# Global logger for this module (set during Node initialization)
log = None

# Persistent miner handle and worker integration
from shared.miner_worker import MinerHandle, miner_worker_main


class Node:
    """Node that manages multiple miners and handles blockchain network participation."""

    def __init__(self, node_id: str, miners_config: Dict[str, Any], genesis_block: Block,
                 secret: Optional[str] = None,
                 on_block_mined: Optional[Callable] = None,
                 on_mining_started: Optional[Callable] = None,
                 on_mining_stopped: Optional[Callable] = None):
        """
        Initialize a blockchain node with multiple miners.

        Args:
            node_id: Unique identifier for this node
            miners_config: Configuration dict with cpu, gpu, qpu sections
            secret: Secret key for deterministic key generation (random if None)
            on_block_mined: Callback when a block is successfully mined
            on_mining_started: Callback when mining starts
            on_mining_stopped: Callback when mining stops
        """
        self.node_id = node_id
        self.miners_config = miners_config

        self.peers: Dict[str, MinerInfo] = {}

        # Store event callbacks
        self.on_block_mined = on_block_mined
        self.on_mining_started = on_mining_started
        self.on_mining_stopped = on_mining_stopped

        if not secret:
            secret = os.urandom(32).hex()

        seed = blake3(secret.encode()).digest()
        self.crypto = BlockSigner(seed=seed)

        # Set up multiprocessing logging first (before initializing miners)
        self._log_queue = None
        self._log_listener = None
        self._setup_multiprocess_logging()

        # Initialize logger with helper function
        self.logger = init_component_logger('node', node_id)

        # Initialize miners based on config
        self.miners: List[Miner] = []
        self._initialize_miners(cfg=miners_config)

        # Node-level timing and statistics
        self.timing_stats = {
            'total_blocks_attempted': 0,
            'total_blocks_won': 0,
            'total_mining_time': 0,
            'blocks_per_miner': {},
            'wins_per_miner': {}
        }

        # Track timing history for all miners
        self.timing_history = {
            'block_numbers': [],
            'total_mining_times': [],
            'winning_miner_types': [],
            'network_hash_rates': []
        }

        # Difficulty and network management
        self.no_block_timeout = 1800  # 30 minutes in seconds
        self.difficulty_reduction_factor = 0.1

        # Mining state for async coordination
        self._mining_stop_event: Optional[EventType] = None
        self._is_mining = False
        self._current_mining_task: Optional[asyncio.Task] = None

        self.logger.info(f"Node {node_id} initialized with {len(getattr(self, 'miner_handles', []))} miners")
        self.logger.debug(f"  ECDSA Public Key: {self.crypto.ecdsa_public_key_hex[:16]}...")
        self.logger.debug(f"  WOTS+ Public Key: {self.crypto.wots_plus_public_key.hex()[:16]}...")
        for h in getattr(self, 'miner_handles', []):
            self.logger.debug(f"  - {h.miner_id} ({h.miner_type})")

        # Initialize blockchain
        self.chain: List[Block] = []
        self.chain_lock = asyncio.Lock()
        self.chain.append(genesis_block)

    def _setup_multiprocess_logging(self):
        """Set up logging queue and listener for multiprocessing."""
        # Use default multiprocessing context (should be spawn after CLI sets it)
        self._mp_context = multiprocessing

        # Create queue for inter-process logging
        self._log_queue = multiprocessing.Queue()

        # Get the current root logger's handlers to replicate them in the listener
        root_logger = logging.getLogger()
        handlers = []

        # Copy existing handlers from root logger
        for handler in root_logger.handlers:
            handlers.append(handler)

        if handlers:
            # Create listener that will process logs from child processes
            self._log_listener = QueueListener(self._log_queue, *handlers, respect_handler_level=True)
            self._log_listener.start()

    def _initialize_miners(self, cfg: Dict[str, Any]):
        """Initialize persistent miner workers based on configuration (TOML)."""

        self.miner_handles: list[MinerHandle] = []

        # CPU Miners, 1 per cpu
        if cfg.get("cpu") is not None:
            for i in range(cfg["cpu"].get("num_cpus", 1)):
                spec = {
                    "id": f"{self.node_id}-CPU-{i+1}",
                    "kind": "cpu"
                }
                # CPU requires no config at this time.
                self.miner_handles.append(MinerHandle(spec, self._log_queue))

        # GPU Miners, 1 per device or type
        if cfg.get("gpu") is not None:
            gpu_cfg = cfg["gpu"]
            gpu_backend = (gpu_cfg.get("backend") or "local").lower()
            gpu_devices = list(gpu_cfg.get("devices") or [])

            # Extract common GPU config (gpu_utilization, etc.)
            miner_cfg = {}
            if "gpu_utilization" in gpu_cfg:
                miner_cfg["gpu_utilization"] = gpu_cfg["gpu_utilization"]

            if gpu_backend == "local":
                for d in gpu_devices:
                    spec = {
                        "id": f"{self.node_id}-GPU-CUDA-{d}",
                        "kind": "cuda",
                        "cfg": miner_cfg,
                        "args": {"device": str(d)}
                    }
                    self.miner_handles.append(MinerHandle(spec, self._log_queue))
            elif gpu_backend == "modal":
                gpu_types = list(gpu_cfg.get("types") or [])
                for t in gpu_types:
                    spec = {
                        "id": f"{self.node_id}-GPU-MODAL-{t}",
                        "kind": "modal",
                        "cfg": miner_cfg,
                        "args": {"gpu_type": str(t)}
                    }
                    self.miner_handles.append(MinerHandle(spec, self._log_queue))
            elif gpu_backend == "mps":
                # can only have one metal miner
                spec = {
                    "id": f"{self.node_id}-GPU-MPS",
                    "kind": "metal",
                    "cfg": miner_cfg,
                    "args": {"device": "mps"}
                }
                self.miner_handles.append(MinerHandle(spec, self._log_queue))
            else:
                raise ValueError(f"Unknown GPU backend: {gpu_backend}")

        # QPU Miners, 1 per qpu section
        if cfg.get("qpu") is not None:
            spec = {"id": f"{self.node_id}-QPU-1", "kind": "qpu"}
            # QPU requires no config at this time.
            self.miner_handles.append(MinerHandle(spec, self._log_queue))

        # Back-compat summary list for logs (do not assign to typed self.miners)
        self._summary_miners = [(h.miner_id, h.miner_type) for h in self.miner_handles]

    ## Block Reception and Validation Logic ##

    # For now, our chain logic is as follows:
    # - Each node maintains a full copy of the blockchain in memory.
    # - On receipt of a new block, we switch to it if: 
    #   1. It's timestamp is newer than the current block we have at that index. (i.e., a trust me bro model, not secure)
    #   2. We do not have more than 6 blocks after it.
    #   3. It's signature is valid.
    #   4. It meets the previous block quantum proof validation requirements (including difficulty decay)
    # Wrappers around node are responsible for start/stop mining and broadcasting blocks to other nodes.

    def get_block(self, index: int) -> Optional[Block]:
        """Get a block from the blockchain."""
        if len(self.chain) <= index:
            return None
        return self.chain[index]
    
    def get_latest_block(self) -> Block:
        """Get the latest block from the blockchain."""
        return self.chain[-1]

    async def check_block(self, block: Block) -> bool:
        """Check if a block is valid and can be accepted."""
        # 1. Check if we already have this block or a newer one at this index
        cur_block = self.get_block(block.header.index)
        if not block.hash or not block.raw or not block.signature:
            self.logger.error(f"Block {block.header.index} rejected: missing hash, raw, or signature - it's not been finalized/signed.")
            return False
        
        if cur_block is not None:
            if not cur_block.hash:
                raise RuntimeError("Current block is not finalized!")

            # We should not process duplicates...
            if cur_block.hash == block.hash:
                self.logger.warning(f"Block {block.header.index}-{block.hash.hex()[:8]} is a duplicate, ignoring...")
                return False

            # Compare timestamps first - prefer older blocks
            if cur_block.header.timestamp < block.header.timestamp:
                self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: we have an older block at this index ({cur_block.header.timestamp} > {block.header.timestamp}), {cur_block.hash.hex()[:8]}")
                return False
            elif cur_block.header.timestamp == block.header.timestamp:
                if cur_block.hash > block.hash:
                    self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: we have a block with same timestamp and larger hash at this index, {cur_block.hash.hex()[:8]}")
                    return False
            
        # 2. Do we have more than 6 blocks after it?
        head = self.get_latest_block()
        if not head.hash and head.header.index > 0:
            raise RuntimeError("Head block is not finalized!")
    
        if head.header.index > block.header.index + 6:
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: we have more than 6 blocks after it ({head.header.index} > {block.header.index + 6})")
            return False

        prev_block = self.get_block(block.header.index - 1)
        if prev_block is None or prev_block.hash is None:
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: we do not have the previous block ({block.header.index - 1})")
            return False
        
        if prev_block.hash != block.header.previous_hash:
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: previous hash mismatch ({prev_block.hash.hex()[:8]} != {block.header.previous_hash.hex()[:8]})")
            return False

        # 3. Check Signature
        # FIXME: We are not even bothering with checking against known miner info right now.
        block_bytes = block.raw
        signature = block.signature
        if not block_bytes or not signature:
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: missing block bytes or signature")
            return False
        if not self.crypto.verify_combined_signature(
            block.miner_info.ecdsa_public_key,
            block.miner_info.wots_public_key,
            block_bytes,
            signature
        ):
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: invalid signature")
            return False

        # 4. Validate the Quantum Proof and other block artifacts.
        block.quantum_proof.compute_derived_fields()
        if not validate_block(block, prev_block):
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: invalid quantum proof")
            qpjson = block.quantum_proof.to_json()
            qpjson['proof_data'] = qpjson['proof_data'][:10] + "..."
            self.logger.error(f"Quantum Proof: {json.dumps(qpjson)}, rq: {prev_block.next_block_requirements.to_json()}")
            return False

        return True

    async def receive_block(self, block: Block) -> bool:
        """Receive a block from the network."""

        if not await self.check_block(block):
            return False

        head = self.get_latest_block()
        async with self.chain_lock:
            # Reset chain if needed
            if head.header.index >= block.header.index:
                self.logger.warning(f"Resetting chain to accept block {block.header.index} from previous head {head.header.index})")
                self.chain = self.chain[:block.header.index]
            # Accept the block
            self.chain.append(block)

        assert block.hash is not None

        self.logger.info(f"Accepted block {block.header.index}-{block.hash.hex()[:8]} from {block.miner_info.miner_id}")

        # Emit an event so we can stop mining and potentially broadcast to other nodes
        asyncio.create_task(self._emit_block_mined(block))

        return True

    async def _emit_mining_started(self, block: Block) -> None:
        """Emit mining started event with sync callback."""
        if self.on_mining_started:
            try:
                self.on_mining_started(block)
            except Exception as e:
                self.logger.error(f"Error in mining_started callback: {e}")

    async def _emit_mining_stopped(self) -> None:
        """Emit mining stopped event with sync callback."""
        if self.on_mining_stopped:
            try:
                self.on_mining_stopped()
            except Exception as e:
                self.logger.error(f"Error in mining_stopped callback: {e}")

    async def _emit_block_mined(self, block: Block) -> None:
        """Emit block mined event with sync callback."""
        if self.on_block_mined:
            try:
                self.on_block_mined(block)
            except Exception as e:
                self.logger.error(f"Error in block_mined callback: {e}")

    def info(self) -> MinerInfo:
        """Get information about this node."""
        return MinerInfo(
            miner_id=self.node_id,
            miner_type=f"{json.dumps(self.miners_config)}",
            reward_address=self.crypto.ecdsa_public_key_bytes,
            ecdsa_public_key=self.crypto.ecdsa_public_key_bytes,
            wots_public_key=self.crypto.wots_plus_public_key,
            next_wots_public_key=self.crypto.wots_plus_public_key
        )

    async def mine_block(self, previous_block: Block, transactions: List = None) -> Optional[MiningResult]:
        """
        Async method to coordinate mining across all miners of this node for a block.

        Args:
            previous_block: Previous block (contains next block requirements)
            transactions: Optional list of Transaction objects to include in the block

        Returns:
            Block if successful, None if stopped/failed
        """
        if transactions is None:
            transactions = []
        if not self.chain:
            self.logger.info("No existing chain, previous block is genesis")
            self.chain.append(previous_block)

        if (previous_block.header.index) != self.get_latest_block().header.index:
            self.logger.info(f"Node {self.node_id}: Previous block index {previous_block.header.index} does not match latest block index {self.get_latest_block().header.index}, exiting mining task...")
            return None

        if self._is_mining or self._mining_stop_event is not None:
            raise RuntimeError(f"Node {self.node_id}: Already mining")

        # Send the full previous block and its next-block requirements to miners
        requirements = previous_block.next_block_requirements

        handles = self.miner_handles
        if not handles:
            raise ValueError(f"Node {self.node_id}: No miners configured")

        # Set mining state
        self._is_mining = True
        self._mining_stop_event = multiprocessing.Event()

        # Start timing
        start_time = utc_timestamp_float()
        self.timing_stats['total_blocks_attempted'] += 1

        self.logger.info(f"Starting mining with {len(handles)} miners...")

        # Emit mining started event
        asyncio.create_task(self._emit_mining_started(previous_block))

        # Command all workers to start mining with full context
        prev_timestamp = previous_block.header.timestamp
        if previous_block.header.index == 0:
            prev_timestamp = int(start_time)
        for h in handles:
            h.mine(previous_block, self.info(), requirements, prev_timestamp)

        result = None
        try:
            # Wait for first result or timeout
            deadline = utc_timestamp_float() + self.no_block_timeout
            while utc_timestamp_float() < deadline and not self._mining_stop_event.is_set():
                # Async sleep to allow other coroutines to run
                await asyncio.sleep(0.1)

                # Poll each handle's response queue quickly
                for h in handles:
                    # Async sleep to allow other coroutines to run
                    await asyncio.sleep(0.1)
                    try:
                        msg = h.resp.get_nowait()
                    except Empty:
                        continue
                    # MiningResult objects will be put by miners; if dict, may be stats/error
                    if msg is None:
                        continue
                    if hasattr(msg, 'miner_id') and hasattr(msg, 'solutions'):
                        # Sometimes the miners finish a block from the last mine
                        # so we drop those results.
                        if (msg.prev_timestamp != prev_timestamp):
                            self.logger.debug(f"Miner {msg.miner_id} returned result for wrong block timestamp {msg.prev_timestamp} (expected {prev_timestamp})")
                            continue
                        result = msg
                        self._mining_stop_event.set()
                        break
                    if isinstance(msg, dict) and msg.get('op') == 'error':
                        # Log or collect errors as needed
                        pass
                if result is not None:
                    break
        except asyncio.CancelledError:
            self.logger.info("Mining cancelled")
            self._mining_stop_event.set()
        except KeyboardInterrupt:
            self.logger.info("Mining interrupted")
            self._mining_stop_event.set()
        finally:
            # Signal cancellation to all workers if result found
            if result is not None:
                for h in handles:
                    h.cancel()
            # Reset mining state
            self._is_mining = False
            self._mining_stop_event = None

            # Emit mining stopped event
            asyncio.create_task(self._emit_mining_stopped())

        # Record timing and statistics
        total_time = utc_timestamp_float() - start_time
        self.timing_stats['total_mining_time'] += total_time

        if result:
            self.timing_stats['total_blocks_won'] += 1
            winning_miner_id = result.miner_id

            # Update per-miner statistics
            if winning_miner_id not in self.timing_stats['wins_per_miner']:
                self.timing_stats['wins_per_miner'][winning_miner_id] = 0
            self.timing_stats['wins_per_miner'][winning_miner_id] += 1

            self.logger.info(f"{winning_miner_id} won block in {total_time:.2f}s")
        else:
            self.logger.info(f"No solution found in {total_time:.2f}s")

        return result

    async def stop_mining(self) -> None:
        """
        Async method to stop the current mining operation.

        NOTE: You should await this if you want to wait for the node to actually stop mining. 
        """
        if not self._is_mining or self._mining_stop_event is None:
            return

        self.logger.info("Stopping mining...")
        self._mining_stop_event.set()

        # Wait for mining to stop before clearing the mining event.
        while self._is_mining:
            await asyncio.sleep(0.1)

        # Reset state
        self._mining_stop_event = None
    
    def build_block(self, previous_block: Block, mining_result: MiningResult, block_data: bytes, transactions: List = None):
        """Build a block from a mining result.

        Args:
            previous_block: The previous block in the chain
            mining_result: The result from the mining process
            block_data: Arbitrary block data
            transactions: Optional list of Transaction objects to include in the block
        """
        if transactions is None:
            transactions = []

        if previous_block.hash is None:
            raise ValueError("Previous block hash is empty, unsigned/finalized block?")

        header = block.BlockHeader(
            previous_hash=previous_block.hash,
            index=previous_block.header.index + 1,
            timestamp=utc_timestamp(),
            data_hash=blake3(block_data).digest()
        )
        miner_info = self.info()
        quantum_proof = block.QuantumProof(
            nonce=mining_result.nonce,
            salt=mining_result.salt,
            nodes=mining_result.variable_order or mining_result.node_list,
            edges=mining_result.edge_list,
            solutions=mining_result.solutions,
            mining_time=mining_result.mining_time,
            energy=mining_result.energy,
            diversity=mining_result.diversity,
            num_valid_solutions=mining_result.num_valid
        )
        quantum_proof.compute_derived_fields()

        if (quantum_proof.energy is None or quantum_proof.energy != mining_result.energy):
            raise ValueError(f"Miner reported bad energy {mining_result.energy} but we computed {quantum_proof.energy}")
        if (quantum_proof.diversity is None or quantum_proof.diversity != mining_result.diversity):
            raise ValueError(f"Miner reported bad diversity {mining_result.diversity} but we computed {quantum_proof.diversity}")
        if (quantum_proof.num_valid_solutions is None or quantum_proof.num_valid_solutions > mining_result.num_valid):
            raise ValueError(f"Miner reported bad num_valid_solutions {mining_result.num_valid} but we computed {quantum_proof.num_valid_solutions}")

        next_block_requirements = compute_next_block_requirements(previous_block, mining_result, self.logger)
        next_block = block.Block(
            header=header,
            miner_info=miner_info,
            quantum_proof=quantum_proof,
            next_block_requirements=next_block_requirements,
            data=block_data,
            transactions=transactions,
            raw=b"",
            hash=b"",
            signature=b""
        )
        next_block.finalize()
        return next_block

    def sign_block(self, block: Block):
        """Sign a block."""
        block.finalize()
        block_data = block.to_network()
        block.signature = self.crypto.sign_block_data(block_data)
        if not self.crypto.verify_combined_signature(
            block.miner_info.ecdsa_public_key,
            block.miner_info.wots_public_key,
            block_data,
            block.signature
        ):
            raise ValueError("Failed to verify signature")
        return block

    def close(self):
        """Shutdown all persistent miner workers and logging."""
        for h in self.miner_handles:
            try:
                h.close()
            except Exception:
                pass

        # Stop the logging listener
        if self._log_listener:
            try:
                self._log_listener.stop()
            except Exception:
                pass

    ## Peer tracking and statistics ##

    def add_or_update_peer(self, peer_address: str, peer_info: MinerInfo) -> bool:
        """Add or update a peer in the network."""
        was_new = peer_address not in self.peers
        self.peers[peer_address] = peer_info
        return was_new
    
    def remove_peer(self, peer_address: str) -> bool:
        """Remove a peer from the network."""
        if peer_address in self.peers:
            del self.peers[peer_address]
            return True
        return False
    
    def get_peer_info(self, peer_address: str) -> Optional[MinerInfo]:
        """Get information about a peer."""
        return self.peers.get(peer_address)
    
    def get_peers(self) -> Dict[str, MinerInfo]:
        """Get all peers in the network."""
        return self.peers

    def get_stats(self) -> dict:
        """Aggregate comprehensive stats from all miner handles and blockchain state."""
        # Get latest block for blockchain stats
        latest_block = self.get_latest_block()
        
        # Calculate win rate
        win_rate = 0.0
        if self.timing_stats['total_blocks_attempted'] > 0:
            win_rate = self.timing_stats['total_blocks_won'] / self.timing_stats['total_blocks_attempted']
        
        # Calculate average mining time
        avg_mining_time = 0.0
        if self.timing_stats['total_blocks_attempted'] > 0:
            avg_mining_time = self.timing_stats['total_mining_time'] / self.timing_stats['total_blocks_attempted']
        
        stats = {
            # Node identification
            "node_id": self.node_id,
            "timestamp": utc_timestamp(),
            
            # Mining statistics
            "mining": {
                "total_blocks_attempted": self.timing_stats['total_blocks_attempted'],
                "total_blocks_won": self.timing_stats['total_blocks_won'],
                "total_mining_time": self.timing_stats['total_mining_time'],
                "win_rate": win_rate,
                "average_mining_time": avg_mining_time,
                "wins_per_miner": dict(self.timing_stats['wins_per_miner']),
                "is_mining": self._is_mining,
            },
            
            # Blockchain state
            "blockchain": {
                "chain_length": len(self.chain),
                "latest_block_index": latest_block.header.index,
                "latest_block_timestamp": latest_block.header.timestamp,
                "latest_block_hash": latest_block.hash.hex() if latest_block.hash else None,
                "latest_block_miner": latest_block.miner_info.miner_id if latest_block.miner_info else None,
            },
            
            # Individual miner statistics
            "miners": [],
        }
        
        # Add individual miner stats
        for h in getattr(self, 'miner_handles', []):
            try:
                m = h.get_stats()
                stats["miners"].append(m)
            except Exception:
                stats["miners"].append({"miner_id": h.miner_id, "miner_type": h.miner_type})
                
        return stats
    
    def get_mining_summary(self) -> str:
        """Get a summary of mining statistics for this node."""
        lines = [f"\nMining Summary for Node {self.node_id}:"]
        lines.append(f"  Total Blocks Attempted: {self.timing_stats['total_blocks_attempted']}")
        lines.append(f"  Total Blocks Won: {self.timing_stats['total_blocks_won']}")

        if self.timing_stats['total_blocks_attempted'] > 0:
            win_rate = self.timing_stats['total_blocks_won'] / self.timing_stats['total_blocks_attempted'] * 100
            lines.append(f"  Overall Win Rate: {win_rate:.1f}%")

        if self.timing_stats['total_mining_time'] > 0 and self.timing_stats['total_blocks_attempted'] > 0:
            avg_time = self.timing_stats['total_mining_time'] / self.timing_stats['total_blocks_attempted']
            lines.append(f"  Average Mining Time: {avg_time:.2f}s")

        # Per-miner win statistics
        if self.timing_stats['wins_per_miner']:
            lines.append("  Wins by Miner:")
            for miner_id, wins in self.timing_stats['wins_per_miner'].items():
                lines.append(f"    {miner_id}: {wins}")

        # Summary miners from handles
        for mid, mtype in getattr(self, '_summary_miners', []):
            lines.append(f"  - {mid} ({mtype})")

        return "\n".join(lines)

