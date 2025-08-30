"""Node class for quantum blockchain network participation."""

import multiprocessing
import os
import time
from blake3 import blake3
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from multiprocessing.synchronize import Event as EventType

if TYPE_CHECKING:
    pass

from shared.block_signer import BlockSigner
from shared.block import Block
from shared.miner import Miner, MiningResult


# Persistent miner handle and worker integration
from shared.miner_worker import MinerHandle, miner_worker_main


class Node:
    """Node that manages multiple miners and handles blockchain network participation."""

    def __init__(self, node_id: str, miners_config: Dict[str, Any], secret: Optional[str] = None):
        """
        Initialize a blockchain node with multiple miners.

        Args:
            node_id: Unique identifier for this node
            miners_config: Configuration dict with cpu, gpu, qpu sections
            secret: Secret key for deterministic key generation (random if None)
        """
        self.node_id = node_id
        self.miners_config = miners_config

        if not secret:
            secret = os.urandom(32).hex()

        seed = blake3(secret.encode()).digest()
        self.crypto = BlockSigner(seed=seed)

        # Expose public keys for network operations
        self.ecdsa_public_key_hex = self.crypto.ecdsa_public_key_hex
        self.wots_plus_public_key_hex = self.crypto.wots_plus_public_key_hex

        # Keep backward compatibility references
        self.ecdsa_private_key = self.crypto.ecdsa_private_key
        self.ecdsa_public_key = self.crypto.ecdsa_public_key
        self.wots_plus = self.crypto.wots_plus

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
        self.last_block_received_time = time.time()
        self.no_block_timeout = 1800  # 30 minutes in seconds
        self.difficulty_reduction_factor = 0.1

        print(f"Node {node_id} initialized with {len(getattr(self, 'miner_handles', []))} miners:")
        print(f"  ECDSA Public Key: {self.ecdsa_public_key_hex[:16]}...")
        print(f"  WOTS+ Public Key: {self.wots_plus_public_key_hex[:16]}...")
        for h in getattr(self, 'miner_handles', []):
            print(f"  - {h.miner_id} ({h.miner_type})")

    def _initialize_miners(self, cfg: Dict[str, Any]):
        """Initialize persistent miner workers based on configuration (TOML)."""

        self.miner_handles: list[MinerHandle] = []
        ctx = multiprocessing.get_context("spawn")

        # CPU Miners, 1 per cpu
        if cfg.get("cpu") is not None:
            for i in range(cfg["cpu"].get("num_cpus", 1)):
                spec = {
                    "id": f"{self.node_id}-CPU-{i+1}",
                    "kind": "cpu"
                }
                # CPU requires no config at this time.
                self.miner_handles.append(MinerHandle(ctx, spec))

        # GPU Miners, 1 per device or type
        if cfg.get("gpu") is not None:
            gpu_cfg = cfg["gpu"]
            gpu_backend = (gpu_cfg.get("backend") or "local").lower()
            gpu_devices = list(gpu_cfg.get("devices") or [])
            if gpu_backend == "local":
                for d in gpu_devices:
                    spec = {
                        "id": f"{self.node_id}-GPU-CUDA-{d}",
                        "kind": "cuda",
                        "args": {"device": str(d)}
                    }
                    self.miner_handles.append(MinerHandle(ctx, spec))
            elif gpu_backend == "modal":
                gpu_types = list(gpu_cfg.get("types") or [])
                for t in gpu_types:
                    spec = {
                        "id": f"{self.node_id}-GPU-MODAL-{t}",
                        "kind": "modal",
                        "args": {"gpu_type": str(t)}
                    }
                    self.miner_handles.append(MinerHandle(ctx, spec))
            elif gpu_backend == "mps":
                # can only have one metal miner
                spec = {
                    "id": f"{self.node_id}-GPU-MPS", 
                    "kind": "metal",
                    "args": {"device": "mps"}
                }
                self.miner_handles.append(MinerHandle(ctx, spec))
            else:
                raise ValueError(f"Unknown GPU backend: {gpu_backend}")
            
        # QPU Miners, 1 per qpu section
        if cfg.get("qpu") is not None:
            spec = {"id": f"{self.node_id}-QPU-1", "kind": "qpu"}
            # QPU requires no config at this time.
            self.miner_handles.append(MinerHandle(ctx, spec))

        # Back-compat summary list for logs (do not assign to typed self.miners)
        self._summary_miners = [(h.miner_id, h.miner_type) for h in self.miner_handles]


    def sign_block_data(self, block_data: str) -> Tuple[str, str]:
        """Sign block data with WOTS+ and ECDSA."""
        signature_hex, next_wots_key_hex = self.crypto.sign_block_data(block_data)

        # Update local reference to new WOTS+ key
        self.wots_plus_public_key_hex = next_wots_key_hex
        self.wots_plus = self.crypto.wots_plus

        return signature_hex, next_wots_key_hex

    def generate_new_wots_key(self):
        """Generate a new WOTS+ key pair after using the current one."""
        self.crypto.generate_new_wots_key()

        # Update local references
        self.wots_plus = self.crypto.wots_plus
        self.wots_plus_public_key_hex = self.crypto.wots_plus_public_key_hex
        print(f"Node {self.node_id} generated new WOTS+ key: {self.wots_plus_public_key_hex[:16]}...")

    def check_and_adjust_difficulty_for_timeout(self):
        """Check if no block has been received for 30 minutes and adjust difficulty."""
        time_since_last_block = time.time() - self.last_block_received_time

        if time_since_last_block > self.no_block_timeout:
            # Note: Difficulty parameters are now managed at block level, not miner level
            # This method would need to be updated to adjust block-level difficulty parameters
            print(f"Node {self.node_id}: Timeout-based difficulty adjustment needed after {time_since_last_block/60:.1f} minutes")
            print("  Note: Difficulty parameters are now managed at block level")
            return True
        return False

    def adapt_parameters(self, network_stats: dict):
        """Adapt all miners' parameters based on network performance."""
        for miner in self.miners:
            miner.adapt_parameters(network_stats)

    def reset_block_received_time(self):
        """Reset the last block received time when a new block is received."""
        self.last_block_received_time = time.time()
        # For persistent workers, nothing to notify here; miners read network state indirectly
        # via difficulty adjustments before next round.

    def mine_block(self, previous_block: Block, stop_event: EventType) -> Optional[MiningResult]:
        """
        Coordinate mining across all miners of this node for a block.

        Args:
            previous_block: Previous block (contains next block requirements)

        Returns:
            MiningResult if successful, None if stopped/failed
        """
        # Send the full previous block and its next-block requirements to miners
        block = previous_block
        requirements = previous_block.next_block_requirements
        handles = getattr(self, 'miner_handles', [])
        if not handles:
            print(f"Node {self.node_id}: No miners configured")
            return None


        # Start timing
        start_time = time.time()
        self.timing_stats['total_blocks_attempted'] += 1

        # Check for timeout-based difficulty adjustment
        self.check_and_adjust_difficulty_for_timeout()

        print(f"Node {self.node_id} starting mining with {len(handles)} miners...")

        # Command all workers to start mining with full context
        for h in handles:
            h.mine(block, requirements)

        result = None
        try:
            # Wait for first result or timeout
            deadline = time.time() + self.no_block_timeout
            while time.time() < deadline and not stop_event.is_set():
                # Poll each handle's response queue quickly
                for h in handles:
                    try:
                        msg = h.resp.get(timeout=0.1)
                    except Exception:
                        continue
                    # MiningResult objects will be put by miners; if dict, may be stats/error
                    if msg is None:
                        continue
                    if hasattr(msg, 'miner_id') and hasattr(msg, 'solutions'):
                        result = msg
                        stop_event.set()
                        break
                    if isinstance(msg, dict) and msg.get('op') == 'error':
                        # Log or collect errors as needed
                        pass
                if result is not None:
                    break
        except KeyboardInterrupt:
            print(f"Node {self.node_id}: Mining interrupted")
            stop_event.set()
        finally:
            # Signal cancellation to all workers if result found
            if result is not None:
                for h in handles:
                    h.cancel()

        # Record timing and statistics
        total_time = time.time() - start_time
        self.timing_stats['total_mining_time'] += total_time

        if result:
            self.timing_stats['total_blocks_won'] += 1
            winning_miner_id = result.miner_id

            # Update per-miner statistics
            if winning_miner_id not in self.timing_stats['wins_per_miner']:
                self.timing_stats['wins_per_miner'][winning_miner_id] = 0
            self.timing_stats['wins_per_miner'][winning_miner_id] += 1

            print(f"Node {self.node_id}: {winning_miner_id} won block in {total_time:.2f}s")
        else:
            print(f"Node {self.node_id}: No solution found in {total_time:.2f}s")

        return result

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

    def close(self):
        """Shutdown all persistent miner workers."""
        for h in getattr(self, 'miner_handles', []):
            try:
                h.close()
            except Exception:
                pass

    def get_network_identity(self) -> dict:
        """Get network identity information for this node."""
        return {
            "node_id": self.node_id,
            "ecdsa_public_key": self.ecdsa_public_key_hex,
            "wots_plus_public_key": self.wots_plus_public_key_hex,
            "miners": [
                {
                    "miner_id": mid,
                    "miner_type": mtype
                }
                for (mid, mtype) in getattr(self, '_summary_miners', [])
            ]
        }

    def get_stats(self) -> dict:
        """Aggregate stats from all miner handles."""
        stats = {
            "node_id": self.node_id,
            "total_blocks_attempted": self.timing_stats['total_blocks_attempted'],
            "total_blocks_won": self.timing_stats['total_blocks_won'],
            "total_mining_time": self.timing_stats['total_mining_time'],
            "wins_per_miner": dict(self.timing_stats['wins_per_miner']),
            "miners": [],
        }
        for h in getattr(self, 'miner_handles', []):
            try:
                m = h.get_stats()
                stats["miners"].append(m)
            except Exception:
                stats["miners"].append({"miner_id": h.miner_id, "miner_type": h.miner_type})
        return stats

    def verify_signature(self, message: str, signature: str, public_key: str) -> bool:
        """Verify a signature using the crypto manager."""
        return self.crypto.verify_ecdsa_signature(
            public_key,
            message.encode(),
            bytes.fromhex(signature)
        )