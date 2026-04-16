"""IHAVE/IWANT block inventory for bandwidth-efficient gossip.

Instead of broadcasting full blocks, nodes announce block availability
via IHAVE (just the hash). Peers that don't have the block respond
with IWANT, and only then is the full block transferred.

This reduces gossip bandwidth by ~90% for blocks that most peers
already have (common in well-connected networks).

Usage::

    inv = BlockInventory()

    # When we mine or receive a new block:
    inv.record_have(block_hash)

    # When we receive IHAVE from a peer:
    if inv.should_request(block_hash):
        inv.record_want(block_hash, peer)
        # send IWANT to peer

    # When IWANT times out:
    stale = inv.expire_wants(timeout=5.0)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class _WantEntry:
    """Tracks a pending IWANT request."""
    peer: str
    requested_at: float


class BlockInventory:
    """Tracks block availability across the local node and peers.

    Args:
        max_have: Maximum number of block hashes to remember locally.
        want_timeout: Seconds before a pending IWANT is considered stale.
    """

    def __init__(
        self,
        max_have: int = 5000,
        want_timeout: float = 10.0,
    ):
        self.max_have = max_have
        self.want_timeout = want_timeout

        # Blocks we possess (hash -> timestamp received)
        self._have: Dict[bytes, float] = {}
        # What each peer claims to have (peer -> set of hashes)
        self._peer_has: Dict[str, Set[bytes]] = {}
        # Pending IWANT requests (hash -> WantEntry)
        self._pending_wants: Dict[bytes, _WantEntry] = {}

    def record_have(self, block_hash: bytes) -> None:
        """Record that we possess a block."""
        self._have[block_hash] = time.monotonic()
        # Remove from pending wants if we were waiting for it
        self._pending_wants.pop(block_hash, None)
        # Evict oldest if over capacity
        if len(self._have) > self.max_have:
            self._evict_oldest_have()

    def has_block(self, block_hash: bytes) -> bool:
        """Check if we already have this block."""
        return block_hash in self._have

    def record_ihave(self, peer: str, block_hash: bytes) -> bool:
        """Record that a peer has announced a block via IHAVE.

        Returns True if we should send IWANT (we don't have the block
        and haven't already requested it).
        """
        # Track what peer claims to have
        if peer not in self._peer_has:
            self._peer_has[peer] = set()
        self._peer_has[peer].add(block_hash)

        # Should we request it?
        if block_hash in self._have:
            return False  # Already have it
        if block_hash in self._pending_wants:
            return False  # Already requested from someone
        return True

    def record_want(self, block_hash: bytes, peer: str) -> None:
        """Record that we've sent an IWANT for this block to peer."""
        self._pending_wants[block_hash] = _WantEntry(
            peer=peer,
            requested_at=time.monotonic(),
        )

    def record_block_received(self, block_hash: bytes) -> None:
        """Record that a block was received (via IWANT response or full gossip)."""
        self.record_have(block_hash)

    def get_pending_wants(self) -> List[Tuple[bytes, str]]:
        """Return list of (block_hash, peer) for all pending IWANT requests."""
        return [
            (h, entry.peer) for h, entry in self._pending_wants.items()
        ]

    def expire_wants(self) -> List[Tuple[bytes, str]]:
        """Remove and return IWANT requests that have timed out.

        Returns list of (block_hash, peer) that expired. These blocks
        can be re-requested from a different peer.
        """
        now = time.monotonic()
        expired: List[Tuple[bytes, str]] = []
        stale_hashes: List[bytes] = []

        for block_hash, entry in self._pending_wants.items():
            if now - entry.requested_at > self.want_timeout:
                expired.append((block_hash, entry.peer))
                stale_hashes.append(block_hash)

        for h in stale_hashes:
            del self._pending_wants[h]

        return expired

    def get_peers_with_block(self, block_hash: bytes) -> List[str]:
        """Return peers that have announced they have this block.

        Useful for re-requesting from a different peer after timeout.
        """
        return [
            peer for peer, hashes in self._peer_has.items()
            if block_hash in hashes
        ]

    def remove_peer(self, peer: str) -> None:
        """Remove all tracking state for a disconnected peer."""
        self._peer_has.pop(peer, None)
        # Cancel any pending wants from this peer
        to_remove = [
            h for h, entry in self._pending_wants.items()
            if entry.peer == peer
        ]
        for h in to_remove:
            del self._pending_wants[h]

    def have_count(self) -> int:
        """Number of blocks we have tracked."""
        return len(self._have)

    def pending_want_count(self) -> int:
        """Number of pending IWANT requests."""
        return len(self._pending_wants)

    def _evict_oldest_have(self) -> None:
        """Remove the oldest entries from _have to stay under max_have."""
        if len(self._have) <= self.max_have:
            return
        excess = len(self._have) - self.max_have
        oldest = sorted(self._have.items(), key=lambda x: x[1])[:excess]
        for h, _ in oldest:
            del self._have[h]
