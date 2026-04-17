"""Node class for quantum blockchain network participation."""

import asyncio
import inspect
import json
import logging
import multiprocessing
import os
import socket
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from queue import Empty
import time
from blake3 import blake3
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Callable
from multiprocessing.synchronize import Event as EventType
from logging.handlers import QueueListener
import aiohttp

from shared.block_requirements import compute_next_block_requirements, validate_block, compute_current_requirements

_SPAWN_CTX = multiprocessing.get_context('spawn')

# Per-worker singleton: ProcessPoolExecutor reuses worker processes across
# calls, so we instantiate one BlockSigner per worker rather than paying
# its init cost on every verify.
_worker_signer: Optional['BlockSigner'] = None


def _verify_worker_init() -> None:
    """ProcessPoolExecutor initializer — one BlockSigner per worker."""
    global _worker_signer
    from shared.block_signer import BlockSigner as _BS
    _worker_signer = _BS(seed=os.urandom(32))


def _verify_signature_worker(
    ecdsa_pk: bytes,
    wots_pk: bytes,
    message: bytes,
    signature: bytes,
) -> bool:
    """Worker-side signature check. Uses the per-worker BlockSigner."""
    global _worker_signer
    if _worker_signer is None:
        _verify_worker_init()
    assert _worker_signer is not None
    return _worker_signer.verify_combined_signature(
        ecdsa_pk, wots_pk, message, signature
    )


def _validate_quantum_proof_worker(
    block: 'Block',
    prev_block: 'Block',
) -> bool:
    """Worker-side PoW check. Blocks cross the process boundary as dataclasses."""
    block.quantum_proof.compute_derived_fields()
    return validate_block(block, prev_block)

if TYPE_CHECKING:
    pass

from shared import block
from shared.block_signer import BlockSigner
from shared.block import Block, MinerInfo
from shared.miner import Miner, MiningResult
from shared.logging_config import init_component_logger
from shared.system_info import build_descriptor
from shared.time_utils import utc_timestamp_float, utc_timestamp, network_timestamp
from shared.version import PROTOCOL_VERSION
# Global logger for this module (set during Node initialization)
log = None

# Persistent miner handle and worker integration
from shared.miner_worker import MinerHandle, miner_worker_main

# Per-device GPU config keys (inherited from [gpu] defaults)
_GPU_CFG_KEYS = (
    "utilization", "yielding", "enabled", "sms_per_nonce",
)

# Top-level TOML keys that map to GPU device types
_GPU_DEVICE_SECTIONS = {
    "cuda": "cuda",
    "nvidia": "cuda",   # alias
    "metal": "metal",
    "modal": "modal",
}

# Top-level TOML keys that map to QPU device types
_QPU_DEVICE_SECTIONS = (
    "dwave", "ibm", "braket", "pasqal", "ionq", "origin",
)


def _build_gpu_miner_cfg(
    section: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build miner cfg dict from a TOML section.

    Extracts known GPU config keys from *section*, falling
    back to *defaults* for any missing keys.
    """
    base = dict(defaults) if defaults else {}
    for key in _GPU_CFG_KEYS:
        if key in section:
            base[key] = section[key]
    return base



def _normalize_gpu_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a normalized GPU config from the full TOML config.

    Scans for top-level device-type sections (``[cuda.N]``,
    ``[nvidia.N]``, ``[metal]``, ``[modal]``) and merges them
    with ``[gpu]`` global defaults.

    Returns a dict with a ``devices`` list-of-dicts ready for
    ``_initialize_miners``.
    """
    gpu_cfg = dict(cfg.get("gpu") or {})
    devices: List[Dict[str, Any]] = []

    for section_key, dev_type in _GPU_DEVICE_SECTIONS.items():
        section = cfg.get(section_key)
        if section is None:
            continue

        if dev_type in ("cuda", "modal") and isinstance(section, dict):
            # [cuda.0], [cuda.1] — subtables keyed by device id
            if any(isinstance(v, dict) for v in section.values()):
                for dev_id in sorted(section.keys()):
                    sub = section[dev_id]
                    if not isinstance(sub, dict):
                        continue
                    entry: Dict[str, Any] = {"type": dev_type}
                    if dev_type == "cuda":
                        entry["device"] = str(dev_id)
                    entry.update(sub)
                    devices.append(entry)
            else:
                # [cuda] with direct keys (no numbered subtables)
                entry = {"type": dev_type}
                entry.update(section)
                devices.append(entry)
        elif isinstance(section, list):
            # [[cuda]] or [[metal]] — array of tables
            for item in section:
                entry = {"type": dev_type}
                entry.update(item)
                devices.append(entry)
        elif isinstance(section, dict):
            # [metal] or [modal] — single table
            entry = {"type": dev_type}
            entry.update(section)
            devices.append(entry)

    if devices:
        gpu_cfg["devices"] = devices
    return gpu_cfg


def _normalize_qpu_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a normalized QPU config from the full TOML config.

    Scans for top-level device-type sections (``[dwave]``, ``[ibm]``,
    ``[braket]``, ``[pasqal]``, ``[ionq]``, ``[origin]``) and merges
    them with ``[qpu]`` global defaults.

    Returns a dict with a ``devices`` list-of-dicts ready for
    ``_initialize_miners``.
    """
    qpu_cfg = dict(cfg.get("qpu") or {})
    devices: List[Dict[str, Any]] = []

    for section_key in _QPU_DEVICE_SECTIONS:
        section = cfg.get(section_key)
        if section is None:
            continue

        if isinstance(section, list):
            # [[dwave]] — array of tables
            for item in section:
                entry: Dict[str, Any] = {"type": section_key}
                entry.update(item)
                devices.append(entry)
        elif isinstance(section, dict):
            # Check for numbered subtables: [dwave.1], [dwave.2]
            if any(isinstance(v, dict) for v in section.values()):
                for sub_id in sorted(section.keys()):
                    sub = section[sub_id]
                    if not isinstance(sub, dict):
                        continue
                    entry = {"type": section_key}
                    entry.update(sub)
                    devices.append(entry)
            else:
                # [dwave] with direct keys
                entry = {"type": section_key}
                entry.update(section)
                devices.append(entry)

    if devices:
        qpu_cfg["devices"] = devices
    return qpu_cfg


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
        self._descriptor_cache: Optional[Dict[str, Any]] = None

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
        # Secondary index for content-addressed block lookup; kept in
        # lockstep with ``chain`` via ``_index_append`` / ``_index_truncate``.
        self.chain_by_hash: Dict[bytes, Block] = {}
        self.chain_lock = asyncio.Lock()
        self._index_append(genesis_block)

        # CPU-bound verification (SPHINCS+ signature, PoW) is offloaded to
        # this pool so the event loop stays responsive for gossip and other
        # I/O. Lazy-initialized on first check_block to avoid spawn cost
        # during tests that construct a Node but never verify a block.
        self._verify_pool: Optional[ProcessPoolExecutor] = None

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

        # GPU Miners — one per device section ([cuda.N], [metal], etc.)
        has_gpu = (
            cfg.get("gpu") is not None
            or any(cfg.get(k) is not None for k in _GPU_DEVICE_SECTIONS)
        )
        if has_gpu:
            gpu_cfg = _normalize_gpu_config(cfg)
            common_cfg = _build_gpu_miner_cfg(gpu_cfg)

            for dev in gpu_cfg.get("devices", []):
                dev_type = dev["type"].lower()
                if dev.get("enabled") is False:
                    continue
                dev_cfg = _build_gpu_miner_cfg(dev, defaults=common_cfg)

                if dev_type == "cuda":
                    device_id = dev.get("device", "0")
                    spec = {
                        "id": f"{self.node_id}-GPU-CUDA-{device_id}",
                        "kind": "cuda",
                        "cfg": dev_cfg,
                        "args": {"device": str(device_id)},
                    }
                elif dev_type == "metal":
                    spec = {
                        "id": f"{self.node_id}-GPU-MPS",
                        "kind": "metal",
                        "cfg": dev_cfg,
                        "args": {"device": "mps"},
                    }
                elif dev_type == "modal":
                    gpu_type = dev.get("gpu_type", "t4")
                    spec = {
                        "id": f"{self.node_id}-GPU-MODAL-{gpu_type}",
                        "kind": "modal",
                        "cfg": dev_cfg,
                        "args": {"gpu_type": str(gpu_type)},
                    }
                else:
                    raise ValueError(f"Unknown GPU device type: {dev_type}")

                self.miner_handles.append(
                    MinerHandle(spec, self._log_queue),
                )

        # QPU Miners — one per device section ([dwave], [ibm], etc.)
        has_qpu = (
            cfg.get("qpu") is not None
            or any(cfg.get(k) is not None for k in _QPU_DEVICE_SECTIONS)
        )
        if has_qpu:
            qpu_cfg = _normalize_qpu_config(cfg)

            for i, dev in enumerate(qpu_cfg.get("devices", []), start=1):
                dev_type = dev.get("type", "dwave").lower()
                tag = dev_type.upper()

                if dev_type == "dwave":
                    spec = {
                        "id": f"{self.node_id}-QPU-{tag}-{i}",
                        "kind": "qpu",
                        "cfg": {
                            "qpu_type": "dwave",
                            "daily_budget": dev.get("daily_budget"),
                            "qpu_min_blocks_for_estimation": dev.get(
                                "qpu_min_blocks_for_estimation", 5,
                            ),
                            "qpu_ema_alpha": dev.get("qpu_ema_alpha", 0.3),
                        },
                    }
                elif dev_type in (
                    "ibm", "braket", "pasqal", "ionq", "origin",
                ):
                    spec = {
                        "id": f"{self.node_id}-QPU-{tag}-{i}",
                        "kind": "qpu",
                        "cfg": {
                            "qpu_type": dev_type,
                            "token": dev.get("token"),
                            "daily_budget": dev.get("daily_budget"),
                        },
                    }
                else:
                    raise ValueError(f"Unknown QPU device type: {dev_type}")

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

    def get_block_by_hash(self, block_hash: bytes) -> Optional[Block]:
        """Get a block from the canonical chain by its hash.

        Returns None if the hash is not currently on the canonical chain
        (including hashes from blocks that were truncated away during a
        reorg).
        """
        return self.chain_by_hash.get(block_hash)

    def build_locator(self) -> List[bytes]:
        """Build a Bitcoin-style locator of block hashes from tip to genesis.

        The locator is used by the GET_CHAIN_MANIFEST RPC so a peer can
        find the latest common ancestor with our chain in a single round
        trip. Layout: the tip plus the next 10 blocks contiguously, then
        the step size doubles until genesis. Length grows like
        ``~11 + log2(tip)``.

        Returns:
            List of 32-byte block hashes ordered newest-first. Empty when
            the chain itself is empty. Always terminates with the genesis
            block's hash when genesis is finalized.
        """
        locator: List[bytes] = []
        if not self.chain:
            return locator

        index = len(self.chain) - 1
        step = 1
        while index > 0:
            block = self.chain[index]
            if block.hash is not None:
                locator.append(block.hash)
            index = max(0, index - step)
            if len(locator) > 10:
                step *= 2

        genesis = self.chain[0]
        if genesis.hash is not None:
            locator.append(genesis.hash)

        return locator

    def _find_block_by_hash(
        self, target_hash: bytes, full_search: bool = False
    ) -> Optional[Block]:
        """Search chain backward for a block with matching hash.

        Args:
            target_hash: Hash to search for.
            full_search: If True, search the entire chain (used during
                sync to resolve deep forks). If False, search only the
                last 6 blocks (used for normal gossip reorgs).
        """
        stop = 0 if full_search else max(0, len(self.chain) - 7)
        for i in range(len(self.chain) - 1, stop, -1):
            if self.chain[i].hash == target_hash:
                return self.chain[i]
        return None

    def _index_append(self, block: Block) -> None:
        """Append a block to ``chain`` and mirror into ``chain_by_hash``.

        Caller is responsible for holding ``chain_lock`` when the append
        happens outside ``__init__``. Blocks without a finalized ``hash``
        still land in ``chain`` but are omitted from ``chain_by_hash`` —
        hash lookups for unfinalized blocks would be meaningless.
        """
        self.chain.append(block)
        if block.hash is not None:
            self.chain_by_hash[block.hash] = block

    def _index_truncate(self, new_length: int) -> None:
        """Shrink ``chain`` to ``new_length`` blocks and evict dropped hashes.

        Caller is responsible for holding ``chain_lock`` when the
        truncate happens outside ``__init__``. No-op when the chain is
        already that short.
        """
        if new_length >= len(self.chain):
            return
        dropped = self.chain[new_length:]
        self.chain = self.chain[:new_length]
        for block in dropped:
            if block.hash is not None:
                self.chain_by_hash.pop(block.hash, None)

    def _ensure_verify_pool(self) -> ProcessPoolExecutor:
        if self._verify_pool is None:
            self._verify_pool = ProcessPoolExecutor(
                max_workers=max(2, (os.cpu_count() or 2) // 2),
                mp_context=_SPAWN_CTX,
                initializer=_verify_worker_init,
            )
        return self._verify_pool

    async def _run_verify(
        self,
        fn: Callable[..., bool],
        args: tuple,
        inline_fn: Callable[[], bool],
    ) -> bool:
        """Run a CPU-bound verifier off the event loop; inline fallback on failure."""
        pool = self._ensure_verify_pool()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(pool, fn, *args)
        except BrokenExecutor as exc:
            self.logger.warning(
                f"executor-fallback: verify pool broken ({exc}), "
                f"running {fn.__name__} inline — next block will respawn pool"
            )
            self._verify_pool = None
            return inline_fn()
        except (OSError, EOFError) as exc:
            self.logger.warning(
                f"executor-fallback: IPC error ({exc}) in {fn.__name__}, "
                f"running inline"
            )
            return inline_fn()

    async def check_block(self, block: Block, force_reorg: bool = False) -> tuple[bool, str | None]:
        """Check if a block is valid and can be accepted.

        Args:
            block: The block to check.
            force_reorg: If True, skip timestamp comparison to allow chain reorganization
                        during sync (longest chain wins). Default False.

        Returns:
            Tuple of (accepted: bool, rejection_reason: str | None).
            If accepted is True, rejection_reason is None.
            If accepted is False, rejection_reason contains the reason for rejection.
        """
        # 0. Check block protocol version
        if block.header.version != PROTOCOL_VERSION:
            reason = (
                f"incompatible protocol version "
                f"(block: {block.header.version}, local: {PROTOCOL_VERSION})"
            )
            self.logger.error(f"Block {block.header.index} rejected: {reason}")
            return False, reason

        # 1. Check if we already have this block or a newer one at this index
        cur_block = self.get_block(block.header.index)
        if not block.hash or not block.raw or not block.signature:
            reason = "missing hash, raw, or signature - it's not been finalized/signed"
            self.logger.error(f"Block {block.header.index} rejected: {reason}")
            return False, reason

        if cur_block is not None:
            if not cur_block.hash:
                raise RuntimeError("Current block is not finalized!")

            # We should not process duplicates...
            if cur_block.hash == block.hash:
                reason = "duplicate block"
                self.logger.warning(f"Block {block.header.index}-{block.hash.hex()[:8]} is a duplicate, ignoring...")
                return False, reason

            # Skip timestamp comparison during forced reorg (sync mode - longest chain wins)
            if not force_reorg:
                # Compare timestamps first - prefer older blocks
                if cur_block.header.timestamp < block.header.timestamp:
                    reason = f"older block exists (ours: {cur_block.header.timestamp} < incoming: {block.header.timestamp})"
                    self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: we have an older block at this index (ours: {cur_block.header.timestamp} < incoming: {block.header.timestamp}), keeping {cur_block.hash.hex()[:8]}")
                    return False, reason
                elif cur_block.header.timestamp == block.header.timestamp:
                    if cur_block.hash > block.hash:
                        reason = "same timestamp, larger hash exists"
                        self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: we have a block with same timestamp and larger hash at this index, {cur_block.hash.hex()[:8]}")
                        return False, reason
            else:
                self.logger.info(f"Block {block.header.index}-{block.hash.hex()[:8]} accepting via force_reorg (replacing {cur_block.hash.hex()[:8]})")
            
        # 2. Do we have more than 6 blocks after it?
        head = self.get_latest_block()
        if not head.hash and head.header.index > 0:
            raise RuntimeError("Head block is not finalized!")
    
        if not force_reorg and head.header.index > block.header.index + 6:
            reason = f"too old (chain is at {head.header.index}, block is {block.header.index})"
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: we have more than 6 blocks after it ({head.header.index} > {block.header.index + 6})")
            return False, reason

        prev_block = self.get_block(block.header.index - 1)
        if prev_block is None or prev_block.hash is None:
            reason = f"missing previous block ({block.header.index - 1})"
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: we do not have the previous block ({block.header.index - 1})")
            return False, reason

        if prev_block.hash != block.header.previous_hash:
            if force_reorg:
                # During sync, search the entire chain for the ancestor
                # to resolve deep forks (e.g., node mined while disconnected)
                ancestor = self._find_block_by_hash(
                    block.header.previous_hash, full_search=True
                )
                if ancestor is not None:
                    self.logger.info(
                        f"Reorg: common ancestor at block {ancestor.header.index}, "
                        f"truncating from {self.get_latest_block().header.index}"
                    )
                    # Truncate chain to common ancestor (with lock for safety)
                    async with self.chain_lock:
                        self._index_truncate(ancestor.header.index + 1)
                    prev_block = ancestor
                else:
                    reason = f"cannot find ancestor with hash {block.header.previous_hash.hex()[:8]}"
                    self.logger.error(
                        f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: cannot find ancestor "
                        f"with hash {block.header.previous_hash.hex()[:8]} in chain"
                    )
                    return False, reason
            else:
                reason = f"previous hash mismatch ({prev_block.hash.hex()[:8]} != {block.header.previous_hash.hex()[:8]})"
                self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: previous hash mismatch ({prev_block.hash.hex()[:8]} != {block.header.previous_hash.hex()[:8]})")
                return False, reason

        # 3. Check Signature
        # FIXME: We are not even bothering with checking against known miner info right now.
        block_bytes = block.raw
        signature = block.signature
        if not block_bytes or not signature:
            reason = "missing block bytes or signature"
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: {reason}")
            return False, reason
        sig_valid = await self._run_verify(
            _verify_signature_worker,
            (block.miner_info.ecdsa_public_key,
             block.miner_info.wots_public_key,
             block_bytes,
             signature),
            inline_fn=lambda: self.crypto.verify_combined_signature(
                block.miner_info.ecdsa_public_key,
                block.miner_info.wots_public_key,
                block_bytes,
                signature,
            ),
        )
        if not sig_valid:
            reason = "invalid signature"
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: {reason}")
            return False, reason

        # 4. Validate the Quantum Proof and other block artifacts.
        block.quantum_proof.compute_derived_fields()
        qp_valid = await self._run_verify(
            _validate_quantum_proof_worker,
            (block, prev_block),
            inline_fn=lambda: validate_block(block, prev_block),
        )
        if not qp_valid:
            reason = "invalid quantum proof"
            self.logger.error(f"Block {block.header.index}-{block.hash.hex()[:8]} rejected: invalid quantum proof (miner: {block.miner_info.miner_id})")
            qpjson = block.quantum_proof.to_json()
            qpjson['proof_data'] = qpjson['proof_data'][:10] + "..."
            # Compute actual decayed requirements for accurate logging
            original_req = prev_block.next_block_requirements
            actual_req = original_req
            elapsed = block.header.timestamp - prev_block.header.timestamp
            if prev_block.header.index > 0:
                actual_req = compute_current_requirements(original_req, prev_block.header.timestamp, self.logger, block.header.timestamp)
            self.logger.error(
                f"Timestamps: prev_block={prev_block.header.timestamp}, block={block.header.timestamp}, "
                f"elapsed={elapsed}s, mining_time={block.quantum_proof.mining_time}s"
            )
            self.logger.error(
                f"Requirements: original_energy={original_req.difficulty_energy:.2f}, "
                f"decayed_energy={actual_req.difficulty_energy:.2f}"
            )
            self.logger.error(f"Quantum Proof: {json.dumps(qpjson)}, rq: {actual_req.to_json()}")
            return False, reason

        return True, None

    async def receive_block(self, block: Block, force_reorg: bool = False) -> tuple[bool, str | None]:
        """Receive a block from the network.

        Args:
            block: The block to receive.
            force_reorg: If True, skip timestamp comparison to allow chain reorganization
                        during sync (longest chain wins). Default False.

        Returns:
            Tuple of (accepted: bool, rejection_reason: str | None).
            If accepted is True, rejection_reason is None.
            If accepted is False, rejection_reason contains the reason for rejection.
        """
        accepted, reason = await self.check_block(block, force_reorg=force_reorg)
        if not accepted:
            return False, reason

        head = self.get_latest_block()
        async with self.chain_lock:
            # Reset chain if needed
            if head.header.index >= block.header.index:
                self.logger.warning(f"Resetting chain to accept block {block.header.index} from previous head {head.header.index})")
                self._index_truncate(block.header.index)
            # Accept the block
            self._index_append(block)

        assert block.hash is not None

        self.logger.info(f"Accepted block {block.header.index}-{block.hash.hex()[:8]} from {block.miner_info.miner_id}")

        # Emit an event so we can stop mining and potentially broadcast to other nodes
        asyncio.create_task(self._emit_block_mined(block))

        return True, None

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

    def _derive_miner_type_label(self) -> str:
        """Short label for this node's active miners (e.g. 'CPU', 'CPU+GPU').

        Historically this field leaked the full TOML config. The
        documented contract (shared/block.py:MinerInfo.miner_type) is
        a short string consumed by `calculate_adaptive_parameters`.
        """
        kinds = sorted({
            h.miner_type for h in getattr(self, "miner_handles", [])
            if getattr(h, "miner_type", None)
        })
        return "+".join(kinds) if kinds else "UNKNOWN"

    def info(self) -> MinerInfo:
        """Get information about this node."""
        return MinerInfo(
            miner_id=self.node_id,
            miner_type=self._derive_miner_type_label(),
            reward_address=self.crypto.ecdsa_public_key_bytes,
            ecdsa_public_key=self.crypto.ecdsa_public_key_bytes,
            wots_public_key=self.crypto.wots_plus_public_key,
            next_wots_public_key=self.crypto.wots_plus_public_key
        )

    def descriptor(self) -> Dict[str, Any]:
        """Lazily build and cache the NodeDescriptor for this node.

        Returns a JSON-friendly dict. Cached after first call; call
        ``invalidate_descriptor()`` if miners_config changes.
        """
        if self._descriptor_cache is None:
            desc = build_descriptor(self.node_id, self.miners_config)
            self._descriptor_cache = desc.to_dict()
        return self._descriptor_cache

    def invalidate_descriptor(self) -> None:
        """Drop the cached descriptor so it will be rebuilt on next read."""
        self._descriptor_cache = None

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
            self._index_append(previous_block)

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
    
    def _write_validation_debug_report(
        self,
        mining_result: MiningResult,
        quantum_proof,
        previous_block: Block,
        mismatches: List[str],
    ) -> Optional[str]:
        """Write a debug report for block validation mismatches.

        Dumps all data needed to reproduce the energy mismatch:
        nonce, salt, topology dimensions, node ordering, per-solution
        energies (reported vs recomputed), and the full traceback.

        Returns the debug file path, or None on write failure.
        """
        import traceback
        from pathlib import Path
        from shared.quantum_proof_of_work import (
            generate_ising_model_from_nonce,
            energy_of_solution,
        )

        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        block_index = previous_block.header.index + 1

        # Find log directory from root logger's file handlers
        debug_dir = Path("debug")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                debug_dir = Path(handler.baseFilename).parent / "debug"
                break
        debug_dir.mkdir(parents=True, exist_ok=True)

        filename = f"energy_mismatch_block{block_index}_{timestamp}.txt"
        filepath = debug_dir / filename

        proof_nodes = mining_result.variable_order or mining_result.node_list
        proof_edges = mining_result.edge_list

        # Recompute per-solution energies for the report
        try:
            h, J = generate_ising_model_from_nonce(
                mining_result.nonce, proof_nodes, proof_edges,
            )
            per_solution_energies = [
                energy_of_solution(sol, h, J, proof_nodes)
                for sol in mining_result.solutions
            ]
        except Exception as e:
            per_solution_energies = [f"recompute failed: {e}"]

        nodes_sorted = list(sorted(proof_nodes))
        nodes_match_sorted = (proof_nodes == nodes_sorted)

        lines = [
            f"=== Energy Mismatch Debug Report ===",
            f"Timestamp:       {timestamp}",
            f"Block index:     {block_index}",
            f"Miner ID:        {mining_result.miner_id}",
            f"Miner type:      {mining_result.miner_type}",
            f"",
            f"--- Mismatches ---",
            *[f"  {m}" for m in mismatches],
            f"",
            f"--- Reported (from miner) ---",
            f"Energy:          {mining_result.energy}",
            f"Diversity:       {mining_result.diversity}",
            f"Num valid:       {mining_result.num_valid}",
            f"",
            f"--- Recomputed (compute_derived_fields) ---",
            f"Energy:          {quantum_proof.energy}",
            f"Diversity:       {quantum_proof.diversity}",
            f"Num valid:       {quantum_proof.num_valid_solutions}",
            f"",
            f"--- Ising Model Params ---",
            f"Nonce:           {mining_result.nonce}",
            f"Salt:            {mining_result.salt.hex()}",
            f"Num nodes:       {len(proof_nodes)}",
            f"Num edges:       {len(proof_edges)}",
            f"Nodes sorted?    {nodes_match_sorted}",
            f"First 10 nodes:  {proof_nodes[:10]}",
            f"Last 10 nodes:   {proof_nodes[-10:]}",
            f"",
            f"--- Solutions ---",
            f"Num solutions:   {len(mining_result.solutions)}",
        ]
        for i, sol in enumerate(mining_result.solutions):
            recomp = per_solution_energies[i] if i < len(per_solution_energies) else "N/A"
            lines.append(f"  Solution {i}: len={len(sol)}, recomputed_energy={recomp}")
            # Show first/last few spin values for ordering analysis
            if len(sol) > 20:
                lines.append(f"    first 10 spins: {sol[:10]}")
                lines.append(f"    last 10 spins:  {sol[-10:]}")
            else:
                lines.append(f"    spins: {sol}")

        lines.extend([
            f"",
            f"--- Previous Block ---",
            f"Index:           {previous_block.header.index}",
            f"Hash:            {previous_block.hash.hex() if previous_block.hash else 'None'}",
            f"",
            f"--- Traceback ---",
            traceback.format_stack()[-3] if len(traceback.format_stack()) >= 3 else "(unavailable)",
        ])

        try:
            filepath.write_text("\n".join(lines), encoding="utf-8")
            return str(filepath)
        except OSError as e:
            self.logger.warning(f"Failed to write debug report: {e}")
            return None

    def build_block(self, previous_block: Block, mining_result: MiningResult, block_data: bytes, transactions: List = None):
        """Build a block from a mining result.

        Args:
            previous_block: The previous block in the chain
            mining_result: The result from the mining process
            block_data: Arbitrary block data
            transactions: Optional list of Transaction objects to include in the block

        Returns:
            Block if validation passes, None if validation fails
            (debug report written to disk on failure).
        """
        if transactions is None:
            transactions = []

        if previous_block.hash is None:
            raise ValueError("Previous block hash is empty, unsigned/finalized block?")

        header = block.BlockHeader(
            previous_hash=previous_block.hash,
            index=previous_block.header.index + 1,
            timestamp=network_timestamp(),
            data_hash=blake3(block_data).digest()
        )
        miner_info = self.info()
        # Use h_values from the previous block's requirements so the
        # recomputed Ising model matches what the miner was given.
        h_values = getattr(
            previous_block.next_block_requirements, 'h_values', None,
        )
        quantum_proof = block.QuantumProof(
            nonce=mining_result.nonce,
            salt=mining_result.salt,
            nodes=mining_result.variable_order or mining_result.node_list,
            edges=mining_result.edge_list,
            solutions=mining_result.solutions,
            mining_time=mining_result.mining_time,
            h_values=h_values,
            energy=mining_result.energy,
            diversity=mining_result.diversity,
            num_valid_solutions=mining_result.num_valid
        )
        quantum_proof.compute_derived_fields()

        mismatches = []
        if quantum_proof.energy is None or quantum_proof.energy != mining_result.energy:
            mismatches.append(
                f"energy: miner={mining_result.energy}, "
                f"recomputed={quantum_proof.energy}"
            )
        if quantum_proof.diversity is None or quantum_proof.diversity != mining_result.diversity:
            mismatches.append(
                f"diversity: miner={mining_result.diversity}, "
                f"recomputed={quantum_proof.diversity}"
            )
        if quantum_proof.num_valid_solutions is None or quantum_proof.num_valid_solutions > mining_result.num_valid:
            mismatches.append(
                f"num_valid: miner={mining_result.num_valid}, "
                f"recomputed={quantum_proof.num_valid_solutions}"
            )

        if mismatches:
            debug_path = self._write_validation_debug_report(
                mining_result, quantum_proof, previous_block, mismatches,
            )
            self.logger.error(
                "Block validation failed for %s block %d: %s "
                "(debug report: %s)",
                mining_result.miner_id,
                previous_block.header.index + 1,
                "; ".join(mismatches),
                debug_path or "write failed",
            )
            return None

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

        if self._verify_pool is not None:
            try:
                self._verify_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._verify_pool = None

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

