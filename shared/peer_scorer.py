"""GossipSub-inspired peer scoring for gossip target selection.

Peers earn positive scores for useful behavior (relaying valid blocks,
fast heartbeat responses) and negative scores for bad behavior (invalid
blocks, rate limit violations, protocol errors). Low-scoring peers are
deprioritized for gossip and eventually disconnected.

Usage::

    scorer = PeerScorer(disconnect_threshold=-100.0)
    scorer.record_valid_block("peer1")
    scorer.record_invalid_block("peer1")

    targets = scorer.select_gossip_targets(all_peers, fanout=5)
    # 80% highest-scored, 20% random for diversity

    if scorer.should_disconnect("peer1"):
        disconnect(peer1)
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Scoring weights
_VALID_BLOCK_SCORE = 10.0
_INVALID_BLOCK_SCORE = -50.0
_HEARTBEAT_OK_SCORE = 1.0
_HEARTBEAT_FAIL_SCORE = -2.0
_RATE_LIMIT_VIOLATION_SCORE = -10.0
_PROTOCOL_ERROR_SCORE = -20.0
_DUPLICATE_EXCESS_SCORE = -1.0


@dataclass
class PeerScore:
    """Accumulated score state for a single peer."""
    score: float = 0.0
    valid_blocks: int = 0
    invalid_blocks: int = 0
    heartbeat_ok: int = 0
    heartbeat_fail: int = 0
    rate_limit_violations: int = 0
    protocol_errors: int = 0
    first_seen: float = field(default_factory=time.monotonic)
    last_update: float = field(default_factory=time.monotonic)


class PeerScorer:
    """Tracks peer behavior scores for gossip target selection.

    Args:
        disconnect_threshold: Score below which a peer should be
            disconnected. Default -100.0 allows several bad events
            before triggering.
        decay_rate: Score decays toward 0 by this fraction per minute.
            Allows peers to recover from transient bad behavior.
    """

    def __init__(
        self,
        disconnect_threshold: float = -100.0,
        decay_rate: float = 0.05,
    ):
        self.disconnect_threshold = disconnect_threshold
        self.decay_rate = decay_rate
        self._scores: Dict[str, PeerScore] = {}

    def _get_or_create(self, peer: str) -> PeerScore:
        if peer not in self._scores:
            self._scores[peer] = PeerScore()
        return self._scores[peer]

    def record_valid_block(self, peer: str) -> None:
        """Peer relayed a valid block."""
        ps = self._get_or_create(peer)
        ps.valid_blocks += 1
        ps.score += _VALID_BLOCK_SCORE
        ps.last_update = time.monotonic()

    def record_invalid_block(self, peer: str) -> None:
        """Peer sent an invalid block."""
        ps = self._get_or_create(peer)
        ps.invalid_blocks += 1
        ps.score += _INVALID_BLOCK_SCORE
        ps.last_update = time.monotonic()

    def record_heartbeat_ok(self, peer: str) -> None:
        """Peer responded to heartbeat."""
        ps = self._get_or_create(peer)
        ps.heartbeat_ok += 1
        ps.score += _HEARTBEAT_OK_SCORE
        ps.last_update = time.monotonic()

    def record_heartbeat_fail(self, peer: str) -> None:
        """Peer failed to respond to heartbeat."""
        ps = self._get_or_create(peer)
        ps.heartbeat_fail += 1
        ps.score += _HEARTBEAT_FAIL_SCORE
        ps.last_update = time.monotonic()

    def record_rate_limit_violation(self, peer: str) -> None:
        """Peer was rate-limited."""
        ps = self._get_or_create(peer)
        ps.rate_limit_violations += 1
        ps.score += _RATE_LIMIT_VIOLATION_SCORE
        ps.last_update = time.monotonic()

    def record_protocol_error(self, peer: str) -> None:
        """Peer sent a malformed or unexpected message."""
        ps = self._get_or_create(peer)
        ps.protocol_errors += 1
        ps.score += _PROTOCOL_ERROR_SCORE
        ps.last_update = time.monotonic()

    def record_duplicate_excess(self, peer: str) -> None:
        """Peer sent an excessive duplicate message."""
        ps = self._get_or_create(peer)
        ps.score += _DUPLICATE_EXCESS_SCORE
        ps.last_update = time.monotonic()

    def get_score(self, peer: str) -> float:
        """Get the current score for a peer."""
        ps = self._scores.get(peer)
        return ps.score if ps else 0.0

    def should_disconnect(self, peer: str) -> bool:
        """Whether a peer's score is below the disconnect threshold."""
        return self.get_score(peer) < self.disconnect_threshold

    def remove_peer(self, peer: str) -> None:
        """Remove scoring state for a disconnected peer."""
        self._scores.pop(peer, None)

    def ranked_peers(self) -> List[Tuple[str, float]]:
        """Return all peers sorted by score (highest first)."""
        return sorted(
            ((peer, ps.score) for peer, ps in self._scores.items()),
            key=lambda x: x[1],
            reverse=True,
        )

    def select_gossip_targets(
        self,
        all_peers: List[str],
        fanout: int,
        scored_fraction: float = 0.8,
    ) -> List[str]:
        """Select gossip targets: top-scored + random for diversity.

        Args:
            all_peers: Full list of peer addresses to choose from.
            fanout: Number of targets to select.
            scored_fraction: Fraction of targets chosen by score
                (rest are random for diversity).

        Returns:
            List of selected peer addresses.
        """
        if len(all_peers) <= fanout:
            return list(all_peers)

        n_scored = max(1, int(fanout * scored_fraction))
        n_random = fanout - n_scored

        # Get scored peers sorted by score
        scored = [
            (p, self.get_score(p)) for p in all_peers
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Top N by score
        selected = set()
        for peer, _ in scored[:n_scored]:
            selected.add(peer)

        # Random from remainder for diversity
        remaining = [p for p in all_peers if p not in selected]
        if remaining and n_random > 0:
            randoms = random.sample(
                remaining, min(n_random, len(remaining))
            )
            selected.update(randoms)

        return list(selected)

    def decay_scores(self) -> None:
        """Apply time-based decay to all scores, moving them toward 0.

        Call periodically (e.g., every 60s) to allow peers to recover
        from transient bad behavior.
        """
        for ps in self._scores.values():
            if ps.score > 0:
                ps.score = max(0.0, ps.score * (1 - self.decay_rate))
            elif ps.score < 0:
                ps.score = min(0.0, ps.score * (1 - self.decay_rate))

    def peer_count(self) -> int:
        """Number of peers being tracked."""
        return len(self._scores)

    def get_low_scoring_peers(
        self, threshold: Optional[float] = None
    ) -> List[str]:
        """Return peers below a score threshold.

        Defaults to the disconnect threshold.
        """
        if threshold is None:
            threshold = self.disconnect_threshold
        return [
            peer for peer, ps in self._scores.items()
            if ps.score < threshold
        ]
