"""SWIM-inspired failure detection for peer health monitoring.

Implements indirect probing: when a direct heartbeat to peer P fails,
K random peers are asked to probe P on our behalf. If all K indirect
probes also fail, P is marked as suspect. After a configurable number
of consecutive suspect rounds, P is declared dead.

This avoids false positives from transient network partitions between
two specific nodes — if other peers can still reach P, it's alive.

State machine per peer::

    ALIVE  --(direct HB fail)--> SUSPECT(probing)
    SUSPECT --(indirect probe OK)--> ALIVE
    SUSPECT --(all probes fail, rounds >= threshold)--> DEAD

Usage::

    detector = SwimDetector(k_probes=3, suspect_rounds=2)
    detector.record_heartbeat_success("peer1")
    detector.record_heartbeat_failure("peer1")

    probes = detector.get_pending_probes()
    # Send PROBE_REQUEST to each probe target via random peers
    # On PROBE_RESPONSE:
    detector.record_probe_result("peer1", "prober2", success=True)

    dead = detector.get_dead_peers()
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


class PeerState(Enum):
    ALIVE = auto()
    SUSPECT = auto()
    DEAD = auto()


@dataclass
class PeerHealth:
    """Health tracking state for a single peer."""
    state: PeerState = PeerState.ALIVE
    suspect_since: float = 0.0
    suspect_rounds: int = 0
    last_heartbeat_ok: float = field(default_factory=time.monotonic)
    # Indirect probing state
    pending_probes: Dict[str, Optional[bool]] = field(default_factory=dict)
    # Membership delta: recent join/leave events to piggyback
    joined_at: float = field(default_factory=time.monotonic)


@dataclass
class ProbeRequest:
    """A request to ask a random peer to probe a suspect target."""
    target: str        # The suspect peer to probe
    prober: str        # The random peer we ask to do the probing
    probe_id: str      # Unique ID for this probe round
    issued_at: float = field(default_factory=time.monotonic)


class SwimDetector:
    """SWIM-style failure detector with indirect probing.

    Args:
        k_probes: Number of random peers to ask for indirect probing.
        suspect_rounds: Consecutive failed probe rounds before declaring dead.
        probe_timeout: Seconds to wait for indirect probe results.
    """

    def __init__(
        self,
        k_probes: int = 3,
        suspect_rounds: int = 2,
        probe_timeout: float = 10.0,
    ):
        self.k_probes = k_probes
        self.suspect_rounds = suspect_rounds
        self.probe_timeout = probe_timeout
        self._peers: Dict[str, PeerHealth] = {}
        self._probe_counter: int = 0
        # Recent membership changes for piggybacking
        self._recent_joins: List[Tuple[str, float]] = []
        self._recent_leaves: List[Tuple[str, float]] = []
        self._membership_ttl: float = 60.0  # keep events for 60s

    def add_peer(self, peer: str) -> None:
        """Register a new peer as alive."""
        if peer not in self._peers:
            self._peers[peer] = PeerHealth()
            self._recent_joins.append((peer, time.monotonic()))

    def remove_peer(self, peer: str) -> None:
        """Remove a peer from tracking."""
        if peer in self._peers:
            del self._peers[peer]
            self._recent_leaves.append((peer, time.monotonic()))

    def get_state(self, peer: str) -> Optional[PeerState]:
        """Get the current state of a peer."""
        health = self._peers.get(peer)
        return health.state if health else None

    def record_heartbeat_success(self, peer: str) -> None:
        """Record a successful direct heartbeat to peer."""
        health = self._peers.get(peer)
        if health is None:
            self.add_peer(peer)
            health = self._peers[peer]
        health.state = PeerState.ALIVE
        health.suspect_rounds = 0
        health.suspect_since = 0.0
        health.last_heartbeat_ok = time.monotonic()
        health.pending_probes.clear()

    def record_heartbeat_failure(self, peer: str) -> None:
        """Record a failed direct heartbeat. Moves ALIVE -> SUSPECT."""
        health = self._peers.get(peer)
        if health is None:
            return
        if health.state == PeerState.ALIVE:
            health.state = PeerState.SUSPECT
            health.suspect_since = time.monotonic()
            health.suspect_rounds = 1
            health.pending_probes.clear()
        elif health.state == PeerState.SUSPECT:
            health.suspect_rounds += 1

    def get_suspects(self) -> List[str]:
        """Return list of peers currently in SUSPECT state."""
        return [
            peer for peer, h in self._peers.items()
            if h.state == PeerState.SUSPECT
        ]

    def create_probe_requests(
        self, all_peers: List[str]
    ) -> List[ProbeRequest]:
        """Create indirect probe requests for all suspect peers.

        Selects up to K random peers (excluding the suspect itself)
        to probe each suspect target.

        Args:
            all_peers: Full list of known peer addresses to choose
                       probers from.

        Returns:
            List of ProbeRequest objects to send.
        """
        requests: List[ProbeRequest] = []
        suspects = self.get_suspects()

        for target in suspects:
            health = self._peers[target]
            # Only create probes if we don't already have pending ones
            if health.pending_probes:
                continue

            # Select K random probers (exclude the target itself)
            candidates = [p for p in all_peers if p != target]
            k = min(self.k_probes, len(candidates))
            if k == 0:
                # No peers to probe through — escalate directly
                health.suspect_rounds += 1
                continue

            probers = random.sample(candidates, k)
            self._probe_counter += 1
            probe_id = f"probe-{self._probe_counter}"

            for prober in probers:
                health.pending_probes[prober] = None  # None = pending
                requests.append(ProbeRequest(
                    target=target,
                    prober=prober,
                    probe_id=probe_id,
                ))

        return requests

    def record_probe_result(
        self, target: str, prober: str, success: bool
    ) -> None:
        """Record the result of an indirect probe.

        If any probe succeeds, the target is moved back to ALIVE.
        If all probes fail, the suspect round is confirmed.
        """
        health = self._peers.get(target)
        if health is None or health.state != PeerState.SUSPECT:
            return

        if prober in health.pending_probes:
            health.pending_probes[prober] = success

        if success:
            # At least one peer can reach the target — it's alive
            health.state = PeerState.ALIVE
            health.suspect_rounds = 0
            health.pending_probes.clear()
            return

        # Check if all probes have completed
        all_done = all(
            v is not None for v in health.pending_probes.values()
        )
        if all_done:
            all_failed = all(
                v is False for v in health.pending_probes.values()
            )
            if all_failed:
                # This round confirmed the suspect
                if health.suspect_rounds >= self.suspect_rounds:
                    health.state = PeerState.DEAD
                # pending_probes cleared for next round
                health.pending_probes.clear()

    def expire_probes(self) -> None:
        """Expire pending probes that have timed out.

        Treats timed-out probes as failures.
        """
        now = time.monotonic()
        for peer, health in self._peers.items():
            if health.state != PeerState.SUSPECT:
                continue
            if not health.pending_probes:
                continue
            # If suspect for longer than probe_timeout with pending
            # probes, treat remaining as failed
            elapsed = now - health.suspect_since
            if elapsed > self.probe_timeout * health.suspect_rounds:
                for prober, result in health.pending_probes.items():
                    if result is None:
                        health.pending_probes[prober] = False
                # Re-evaluate
                all_failed = all(
                    v is False for v in health.pending_probes.values()
                )
                if all_failed:
                    if health.suspect_rounds >= self.suspect_rounds:
                        health.state = PeerState.DEAD
                    health.pending_probes.clear()

    def get_dead_peers(self) -> List[str]:
        """Return list of peers declared dead."""
        return [
            peer for peer, h in self._peers.items()
            if h.state == PeerState.DEAD
        ]

    def get_membership_deltas(self) -> dict:
        """Get recent membership changes for piggybacking on heartbeats.

        Returns a dict with 'joins' and 'leaves' lists, each containing
        (peer_address, timestamp) tuples from the last membership_ttl
        seconds. Prunes old entries.
        """
        now = time.monotonic()
        cutoff = now - self._membership_ttl

        self._recent_joins = [
            (p, t) for p, t in self._recent_joins if t > cutoff
        ]
        self._recent_leaves = [
            (p, t) for p, t in self._recent_leaves if t > cutoff
        ]

        return {
            "joins": [(p, t) for p, t in self._recent_joins],
            "leaves": [(p, t) for p, t in self._recent_leaves],
        }

    def peer_count(self) -> int:
        """Number of peers being tracked."""
        return len(self._peers)

    def alive_count(self) -> int:
        """Number of peers in ALIVE state."""
        return sum(
            1 for h in self._peers.values() if h.state == PeerState.ALIVE
        )
