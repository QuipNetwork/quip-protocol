"""
Local telemetry output for QUIP network nodes.

Writes node registry (nodes.json) and per-block statistics
(telemetry/{epoch}/{block}.json) to a configurable directory. The
``epoch`` directory name is the first 16 hex characters of block 1's
hash — content-addressed keying so the dashboard can distinguish
canonical-chain dirs from stale forks left over from reorgs at height
1 or from node restarts mid-cycle. A top-level ``current_epoch.json``
marker names the live epoch so the cache can label each dir as
``live`` or ``stale_fork`` without guessing.

SPDX-License-Identifier: AGPL-3.0-or-later
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.block import Block, MinerInfo


EPOCH_KEY_LEN = 16  # hex chars taken from block 1's hash
_HEX_CHARS = frozenset("0123456789abcdef")

# Statuses whose updates are debounced to avoid write amplification under
# retry storms: rejected-capacity JOINs and failed handshakes both repeat
# at seconds-scale intervals while the record is functionally unchanged.
# ``active`` is deliberately excluded — heartbeat updates are the primary
# liveness signal for the dashboard.
_WRITE_DEBOUNCE_STATUSES = frozenset({"rejected_capacity", "failed"})
_WRITE_DEBOUNCE_SECONDS = 30.0


def _is_epoch_hex(name: str) -> bool:
    """Return True if *name* is a valid hex-encoded epoch dir name."""
    return len(name) == EPOCH_KEY_LEN and all(c in _HEX_CHARS for c in name)


def atomic_write_json(target: Path, payload: dict) -> None:
    """Write *payload* as JSON to *target* via a tempfile + ``os.replace``.

    Module-level counterpart to ``TelemetryManager._atomic_write_json``
    so sibling processes (``telemetry_aggregator``) can reuse the exact
    same on-disk write semantics without pulling in the manager class.
    Raises on failure; the manager catches and logs on behalf of its
    async callers.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        # tempfile.mkstemp creates files with mode 0o600; relax to 0o644
        # so telemetry JSON is readable by the data-dir owner (and group)
        # regardless of who the node process runs as.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, target)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


@dataclass
class NodeRecord:
    """Telemetry record for a known network node.

    ``miner_id`` / ``miner_type`` are intentionally absent: they are
    derivable from ``descriptor.node_name`` and ``descriptor.miners``
    respectively, and the ``MinerInfo``-sourced versions carried stale
    pre-upgrade values that were never refreshed by heartbeat.
    """

    address: str
    ecdsa_public_key_hex: Optional[str] = None
    status: str = "active"
    last_heartbeat: Optional[float] = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    descriptor: Optional[Dict] = None


class TelemetryManager:
    """Manages local JSON telemetry for node registry and block statistics."""

    def __init__(
        self,
        telemetry_dir: str = "telemetry",
        enabled: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self._dir = Path(telemetry_dir)
        self._enabled = enabled
        self._logger = logger or logging.getLogger(__name__)
        self._nodes: Dict[str, NodeRecord] = {}
        # Per-address timestamp of the last ``_write_nodes_json`` call,
        # used by the debounce guard in ``update_node``. Tracks writes
        # (not ``last_seen``) so a steady stream of retries cannot keep
        # sliding the window forward and silently suppress all writes.
        self._last_node_write_at: Dict[str, float] = {}
        # Full block-1 hash; the epoch dir name is its first 16 hex chars.
        self._block_1_hash: Optional[bytes] = None
        self._periodic_task: Optional[asyncio.Task] = None
        # Optional off-thread write pipeline. When ``enable_async_writes``
        # has been called, ``_atomic_write_json`` enqueues onto the
        # queue and a dedicated task drains it via ``asyncio.to_thread``
        # so hot paths (heartbeat, peer discovery, block record) never
        # block the coordinator's event loop on disk I/O.
        self._write_queue: Optional[asyncio.Queue] = None
        self._writer_task: Optional[asyncio.Task] = None

        # Preserve first_seen (and other per-record history) across
        # restarts by loading the prior nodes.json. Without this, every
        # boot reseeds first_seen=now and the "known since" signal is
        # destroyed. Malformed files or individual records are skipped
        # with a warning rather than failing startup.
        if self._enabled:
            self._load_nodes_json()

    # ------------------------------------------------------------------
    # Node telemetry
    # ------------------------------------------------------------------

    def record_initial_peers(self, peers: List[str]) -> None:
        """Seed node records for bootstrap peer addresses."""
        if not self._enabled:
            return
        now = time.time()
        for addr in peers:
            if addr not in self._nodes:
                self._nodes[addr] = NodeRecord(
                    address=addr, status="initial_peer",
                    first_seen=now, last_seen=now,
                )
        self._write_nodes_json()

    def update_node(
        self,
        address: str,
        status: str,
        miner_info: Optional[MinerInfo] = None,
        last_heartbeat: Optional[float] = None,
        descriptor: Optional[Dict] = None,
    ) -> None:
        """Insert or update a node record and rewrite nodes.json."""
        if not self._enabled:
            return
        now = time.time()
        rec = self._nodes.get(address)
        if rec is None:
            rec = NodeRecord(address=address, first_seen=now)
            self._nodes[address] = rec

        prev_status = rec.status
        last_written_at = self._last_node_write_at.get(address)

        rec.status = status
        rec.last_seen = now
        if last_heartbeat is not None:
            rec.last_heartbeat = last_heartbeat
        if miner_info is not None and miner_info.ecdsa_public_key:
            rec.ecdsa_public_key_hex = miner_info.ecdsa_public_key.hex()
        if descriptor is not None:
            rec.descriptor = descriptor

        if (
            status in _WRITE_DEBOUNCE_STATUSES
            and prev_status == status
            and last_written_at is not None
            and (now - last_written_at) < _WRITE_DEBOUNCE_SECONDS
        ):
            return

        self._last_node_write_at[address] = now
        self._write_nodes_json()

    def remove_node(self, address: str) -> None:
        """Mark a node as lost and rewrite nodes.json."""
        if not self._enabled:
            return
        rec = self._nodes.get(address)
        if rec:
            rec.status = "lost"
            rec.last_seen = time.time()
            self._write_nodes_json()

    def _write_nodes_json(self) -> None:
        """Atomically write the node registry to nodes.json."""
        self._dir.mkdir(parents=True, exist_ok=True)
        target = self._dir / "nodes.json"
        active = sum(1 for n in self._nodes.values() if n.status == "active")
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(self._nodes),
            "active_count": active,
            "nodes": {
                addr: _node_record_to_dict(rec)
                for addr, rec in self._nodes.items()
            },
        }
        self._atomic_write_json(target, payload)

    def _load_nodes_json(self) -> None:
        """Populate ``self._nodes`` from an existing nodes.json, if any.

        Inverse of ``_node_record_to_dict``: NodeRecord fields live at
        fixed keys, everything else was flattened from the descriptor.
        Missing/malformed records are skipped with a warning so a
        corrupted file can't block startup.
        """
        target = self._dir / "nodes.json"
        if not target.is_file():
            return
        try:
            payload = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning(
                f"Failed to load {target}: {exc}; starting with empty "
                "node registry"
            )
            return

        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, dict):
            return

        loaded = 0
        for addr, flat in raw_nodes.items():
            if not isinstance(flat, dict):
                continue
            try:
                rec = _node_record_from_dict(addr, flat)
            except (TypeError, ValueError, KeyError) as exc:
                self._logger.warning(
                    f"Skipping malformed node entry {addr}: {exc}"
                )
                continue
            self._nodes[addr] = rec
            loaded += 1

        if loaded:
            self._logger.info(
                f"Loaded {loaded} node record(s) from {target}"
            )

    def log_nodes_table(self) -> None:
        """Log a formatted table of all known nodes."""
        if not self._nodes:
            self._logger.info("=== Known Nodes: none ===")
            return
        active = sum(1 for n in self._nodes.values() if n.status == "active")
        lines = [f"=== Known Nodes ({len(self._nodes)} total, {active} active) ==="]
        lines.append(
            f"{'Address':<30} {'Node':<20} {'Type':<10} "
            f"{'Status':<12} {'Last Heartbeat':<16} {'System'}"
        )
        now = time.time()
        for rec in sorted(self._nodes.values(), key=lambda r: r.last_seen, reverse=True):
            hb = _format_ago(now, rec.last_heartbeat) if rec.last_heartbeat else "never"
            lines.append(
                f"{rec.address:<30} {_descriptor_node_label(rec.descriptor):<20} "
                f"{_descriptor_miner_type(rec.descriptor):<10} {rec.status:<12} "
                f"{hb:<16} {_describe_system(rec.descriptor)}"
            )
        self._logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Block telemetry
    # ------------------------------------------------------------------

    def record_block(self, block: Block) -> None:
        """Write a per-block telemetry JSON file under the live epoch dir."""
        if not self._enabled:
            return
        idx = block.header.index
        if idx == 1 and self._block_1_hash is None and block.hash:
            self._block_1_hash = block.hash

        out_dir = self._dir / self._epoch_dir_name()
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = self._block_to_dict(block)
        self._atomic_write_json(out_dir / f"{idx}.json", payload)
        self._write_current_epoch_marker()

    def reset_epoch(self) -> None:
        """Clear the current block-1 hash so the next block 1 re-keys."""
        self._block_1_hash = None

    def sync_epoch_from_chain(self, chain: list) -> None:
        """Align ``_block_1_hash`` with the chain's canonical block 1.

        Unlike the prior timestamp-keyed behaviour, we update on hash
        change: a reorg at height 1 means a different canonical chain,
        and telemetry should follow it. The stale dir from the old block
        1 stays on disk; the cache labels it ``stale_fork`` using the
        ``current_epoch.json`` marker.
        """
        for block in chain:
            if block.header.index == 1 and block.hash:
                if self._block_1_hash != block.hash:
                    self._block_1_hash = block.hash
                return

    def _epoch_dir_name(self) -> str:
        """Return the live epoch dir name, or 'genesis' if block 1 unknown."""
        if self._block_1_hash is None:
            return "genesis"
        return self._block_1_hash.hex()[:EPOCH_KEY_LEN]

    def _write_current_epoch_marker(self) -> None:
        """Write/update the top-level current_epoch.json marker.

        The marker names the live epoch dir and carries the full block-1
        hash so the cache can distinguish the canonical chain from stale
        fork dirs without scanning every block 1 on disk.
        """
        if self._block_1_hash is None:
            return
        payload = {
            "epoch": self._epoch_dir_name(),
            "block_1_hash": self._block_1_hash.hex(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write_json(self._dir / "current_epoch.json", payload)

    def migrate_legacy_dirs(self) -> Dict[str, List]:
        """Rename pre-hash telemetry dirs to their content-addressed names.

        Reads ``{dir}/1.json`` from every non-genesis subdirectory that
        isn't already a 16-char hex name, takes its ``block_hash`` field,
        and renames the dir to ``block_hash[:16]``. Idempotent; dirs
        without a usable block 1 or with a pre-existing target are left
        alone so migration is safe to retry.

        Returns:
            Summary dict: ``{"migrated": [(old, new), ...], "skipped": [(old, reason), ...]}``.
        """
        result: Dict[str, List] = {"migrated": [], "skipped": []}
        if not self._enabled or not self._dir.is_dir():
            return result

        for entry in self._dir.iterdir():
            if not entry.is_dir() or entry.name == "genesis":
                continue
            if _is_epoch_hex(entry.name):
                continue

            reason = self._migrate_one_dir(entry, result)
            if reason is not None:
                result["skipped"].append((entry.name, reason))

        return result

    def _migrate_one_dir(
        self, entry: Path, result: Dict[str, List],
    ) -> Optional[str]:
        """Rename a single legacy dir; return a skip-reason or None on success."""
        block_1 = entry / "1.json"
        if not block_1.is_file():
            return "missing block 1"
        try:
            data = json.loads(block_1.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return f"read error: {exc}"

        block_hash_hex = data.get("block_hash")
        if (
            not isinstance(block_hash_hex, str)
            or len(block_hash_hex) < EPOCH_KEY_LEN
        ):
            return "invalid block_hash"

        new_name = block_hash_hex[:EPOCH_KEY_LEN]
        target = self._dir / new_name
        if target.exists():
            return f"target {new_name} exists"
        try:
            entry.rename(target)
        except OSError as exc:
            return f"rename failed: {exc}"

        result["migrated"].append((entry.name, new_name))
        self._logger.info(
            "Migrated telemetry dir %s -> %s", entry.name, new_name,
        )
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_periodic_log(self, interval: float = 600.0) -> None:
        """Background task: log the nodes table every *interval* seconds."""
        try:
            while True:
                await asyncio.sleep(interval)
                self.log_nodes_table()
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Cancel the periodic log task and the async writer."""
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            self._periodic_task = None
        if self._writer_task is not None and not self._writer_task.done():
            self._writer_task.cancel()
            self._writer_task = None
        self._write_queue = None

    def enable_async_writes(self, maxsize: int = 1024) -> None:
        """Start the serialized off-thread writer.

        Call once from the event loop; after this, ``update_node`` /
        ``record_block`` enqueue their disk writes instead of running
        them inline. Safe to call more than once (no-op after the first).
        """
        if self._write_queue is not None:
            return
        self._write_queue = asyncio.Queue(maxsize=maxsize)
        self._writer_task = asyncio.create_task(
            self._writer_loop(), name="telemetry-writer",
        )

    async def _writer_loop(self) -> None:
        """Serialized drainer: pop (target, payload) and write off-thread."""
        queue = self._write_queue
        assert queue is not None
        while True:
            item = await queue.get()
            if item is None:  # shutdown sentinel
                return
            target, payload = item
            try:
                await asyncio.to_thread(atomic_write_json, target, payload)
            except Exception as exc:
                self._logger.warning(
                    f"async telemetry write {target} failed: {exc}"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _block_to_dict(block: Block) -> dict:
        """Convert a Block to a telemetry-friendly dict."""
        miner = {}
        if block.miner_info:
            miner = {
                "miner_id": block.miner_info.miner_id,
                "miner_type": block.miner_info.miner_type,
                "ecdsa_public_key": block.miner_info.ecdsa_public_key.hex()
                if block.miner_info.ecdsa_public_key else None,
            }

        proof = {}
        if block.quantum_proof:
            qp = block.quantum_proof
            proof = {
                "energy": qp.energy,
                "diversity": qp.diversity,
                "num_valid_solutions": qp.num_valid_solutions,
                "mining_time": qp.mining_time,
                "nonce": qp.nonce,
                "num_nodes": len(qp.nodes) if qp.nodes else 0,
                "num_edges": len(qp.edges) if qp.edges else 0,
            }

        reqs = {}
        if block.next_block_requirements:
            r = block.next_block_requirements
            reqs = {
                "difficulty_energy": r.difficulty_energy,
                "min_diversity": r.min_diversity,
                "min_solutions": r.min_solutions,
            }

        return {
            "block_index": block.header.index,
            "block_hash": block.hash.hex() if block.hash else None,
            "timestamp": block.header.timestamp,
            "previous_hash": block.header.previous_hash.hex()
            if block.header.previous_hash else None,
            "miner": miner,
            "quantum_proof": proof,
            "requirements": reqs,
        }

    def _atomic_write_json(self, target: Path, payload: dict) -> None:
        """Write JSON to *target* atomically.

        When ``enable_async_writes`` has been called, the write is
        enqueued for off-thread execution; drops on queue-full with
        a warning so a stuck disk can't block the event loop. Without
        async writes the call is inline (still tempfile+rename, just
        synchronous — used during startup and in tests).
        """
        queue = self._write_queue
        if queue is not None:
            try:
                queue.put_nowait((target, payload))
            except asyncio.QueueFull:
                self._logger.warning(
                    f"telemetry write queue full; dropping {target}"
                )
            return
        try:
            atomic_write_json(target, payload)
        except Exception as exc:
            self._logger.warning(f"Failed to write telemetry {target}: {exc}")


def _node_record_to_dict(rec: NodeRecord) -> Dict[str, Any]:
    """Flatten a NodeRecord into a JSON-friendly dict.

    Connection metadata (address, status, heartbeat timestamps, ecdsa
    pubkey) is merged with the descriptor so each node entry is a single
    flat object. The descriptor — built through the
    ``system_info.build_descriptor`` whitelist + ``_scrub()`` — is the
    canonical source of node identity (``node_name``), hardware profile,
    and per-miner detail (``miners``).
    """
    out: Dict[str, Any] = {
        "address": rec.address,
        "status": rec.status,
        "first_seen": rec.first_seen,
        "last_seen": rec.last_seen,
        "last_heartbeat": rec.last_heartbeat,
        "ecdsa_public_key_hex": rec.ecdsa_public_key_hex,
    }
    if rec.descriptor:
        for key, value in rec.descriptor.items():
            # Never let descriptor overwrite connection/status keys.
            if key not in out:
                out[key] = value
    return out


# Connection/status fields owned directly by NodeRecord — everything else
# in the flattened on-disk dict came from the descriptor.
_NODE_RECORD_FIELDS = frozenset({
    "address", "status", "first_seen", "last_seen",
    "last_heartbeat", "ecdsa_public_key_hex",
})


def _node_record_from_dict(address: str, flat: Dict[str, Any]) -> NodeRecord:
    """Inverse of ``_node_record_to_dict`` used when reloading nodes.json.

    Non-NodeRecord keys are reassembled into ``descriptor``. Coerces
    numeric timestamps through ``float`` so older files that recorded
    them as ints still load.
    """
    descriptor = {
        k: v for k, v in flat.items() if k not in _NODE_RECORD_FIELDS
    } or None

    first_seen_raw = flat.get("first_seen")
    last_seen_raw = flat.get("last_seen")
    now = time.time()
    return NodeRecord(
        address=flat.get("address") or address,
        status=flat.get("status") or "active",
        first_seen=float(first_seen_raw) if first_seen_raw is not None else now,
        last_seen=float(last_seen_raw) if last_seen_raw is not None else now,
        last_heartbeat=(
            float(flat["last_heartbeat"])
            if flat.get("last_heartbeat") is not None else None
        ),
        ecdsa_public_key_hex=flat.get("ecdsa_public_key_hex"),
        descriptor=descriptor,
    )


def _descriptor_node_label(descriptor: Optional[Dict]) -> str:
    """Return a short node identifier for log-table display."""
    if not descriptor:
        return "-"
    name = descriptor.get("node_name")
    return name if name else "-"


def _descriptor_miner_type(descriptor: Optional[Dict]) -> str:
    """Derive a miner-type label (e.g. 'CPU', 'CPU+QPU') from a descriptor."""
    if not descriptor:
        return "-"
    miners = descriptor.get("miners") or {}
    kinds = sorted({
        m.get("kind") for m in miners.values()
        if isinstance(m, dict) and m.get("kind")
    })
    return "+".join(kinds) if kinds else "-"


def _format_ago(now: float, ts: float) -> str:
    """Format a timestamp as a human-readable 'X ago' string."""
    delta = int(now - ts)
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    return f"{delta // 3600}h ago"


def _describe_system(descriptor: Optional[Dict]) -> str:
    """Compact single-line system summary extracted from a descriptor dict."""
    if not descriptor:
        return "-"
    sysinfo = descriptor.get("system_info") or {}
    os_info = sysinfo.get("os") or {}
    cpu = sysinfo.get("cpu") or {}
    gpus = sysinfo.get("gpus") or []
    parts = [f"{os_info.get('system', '?')}/{os_info.get('machine', '?')}"]
    cores = cpu.get("logical_cores")
    if cores:
        parts.append(f"{cores}C")
    if gpus:
        counts: Dict[str, int] = {}
        for gpu in gpus:
            name = gpu.get("name") or gpu.get("vendor") or "GPU"
            counts[name] = counts.get(name, 0) + 1
        parts.append(" ".join(
            f"{n}×{name}" if n > 1 else name
            for name, n in counts.items()
        ))
    return " ".join(parts)
