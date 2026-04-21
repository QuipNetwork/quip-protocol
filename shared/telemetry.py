"""
Local telemetry output for QUIP network nodes.

Writes node registry (nodes.json) and per-block statistics
(telemetry/{epoch}/{block}.json) to a configurable directory.

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
        self._epoch_timestamp: Optional[int] = None
        self._periodic_task: Optional[asyncio.Task] = None

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

        rec.status = status
        rec.last_seen = now
        if last_heartbeat is not None:
            rec.last_heartbeat = last_heartbeat
        if miner_info is not None and miner_info.ecdsa_public_key:
            rec.ecdsa_public_key_hex = miner_info.ecdsa_public_key.hex()
        if descriptor is not None:
            rec.descriptor = descriptor

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
        """Write a per-block telemetry JSON file."""
        if not self._enabled:
            return
        idx = block.header.index
        # Auto-detect epoch from block 1
        if idx == 1 and self._epoch_timestamp is None:
            self._epoch_timestamp = block.header.timestamp

        epoch_dir = str(self._epoch_timestamp) if self._epoch_timestamp else "genesis"
        out_dir = self._dir / epoch_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = self._block_to_dict(block)
        self._atomic_write_json(out_dir / f"{idx}.json", payload)

    def set_epoch_timestamp(self, ts: Optional[int]) -> None:
        """Set or reset the current epoch directory timestamp."""
        self._epoch_timestamp = ts

    def sync_epoch_from_chain(self, chain: list) -> None:
        """Extract block-1 timestamp from an existing chain (mid-epoch join)."""
        if self._epoch_timestamp is not None:
            return
        for block in chain:
            if block.header.index == 1:
                self._epoch_timestamp = block.header.timestamp
                break

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
        """Cancel the periodic log task."""
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            self._periodic_task = None

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
        """Write JSON to *target* atomically via temp file + os.replace()."""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target.parent), suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
                # tempfile.mkstemp creates files with mode 0o600; relax to
                # 0o644 so telemetry JSON is readable by the data-dir owner
                # (and group) regardless of who the node process runs as.
                os.chmod(tmp_path, 0o644)
                os.replace(tmp_path, target)
            except BaseException:
                # Clean up temp file on any failure
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
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
