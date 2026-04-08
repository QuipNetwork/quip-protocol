"""Tests for PeerScorer gossip target selection."""

import pytest

from shared.peer_scorer import PeerScorer


class TestPeerScorer:
    def test_new_peer_score_zero(self):
        s = PeerScorer()
        assert s.get_score("peer1") == 0.0

    def test_valid_block_increases_score(self):
        s = PeerScorer()
        s.record_valid_block("peer1")
        assert s.get_score("peer1") > 0

    def test_invalid_block_decreases_score(self):
        s = PeerScorer()
        s.record_invalid_block("peer1")
        assert s.get_score("peer1") < 0

    def test_heartbeat_ok_increases_score(self):
        s = PeerScorer()
        s.record_heartbeat_ok("peer1")
        assert s.get_score("peer1") > 0

    def test_heartbeat_fail_decreases_score(self):
        s = PeerScorer()
        s.record_heartbeat_fail("peer1")
        assert s.get_score("peer1") < 0

    def test_should_disconnect_below_threshold(self):
        s = PeerScorer(disconnect_threshold=-50.0)
        # 2 invalid blocks = -100
        s.record_invalid_block("peer1")
        s.record_invalid_block("peer1")
        assert s.should_disconnect("peer1")

    def test_should_not_disconnect_above_threshold(self):
        s = PeerScorer(disconnect_threshold=-100.0)
        s.record_heartbeat_fail("peer1")
        assert not s.should_disconnect("peer1")

    def test_remove_peer(self):
        s = PeerScorer()
        s.record_valid_block("peer1")
        s.remove_peer("peer1")
        assert s.get_score("peer1") == 0.0
        assert s.peer_count() == 0

    def test_ranked_peers(self):
        s = PeerScorer()
        s.record_valid_block("good")
        s.record_valid_block("good")
        s.record_heartbeat_ok("ok")
        s.record_invalid_block("bad")

        ranked = s.ranked_peers()
        assert ranked[0][0] == "good"
        assert ranked[-1][0] == "bad"

    def test_select_gossip_targets_all_when_small(self):
        s = PeerScorer()
        peers = ["a", "b", "c"]
        targets = s.select_gossip_targets(peers, fanout=5)
        assert set(targets) == set(peers)

    def test_select_gossip_targets_respects_fanout(self):
        s = PeerScorer()
        peers = [f"peer{i}" for i in range(20)]
        targets = s.select_gossip_targets(peers, fanout=5)
        assert len(targets) == 5

    def test_select_gossip_targets_prefers_high_scoring(self):
        s = PeerScorer()
        # Give "good" a high score
        for _ in range(10):
            s.record_valid_block("good")
        peers = ["good"] + [f"peer{i}" for i in range(19)]
        # Run multiple times to verify "good" is consistently selected
        appearances = sum(
            1 for _ in range(20)
            if "good" in s.select_gossip_targets(peers, fanout=5)
        )
        assert appearances >= 18, "High-scoring peer should almost always be selected"

    def test_decay_scores(self):
        s = PeerScorer(decay_rate=0.5)
        s.record_valid_block("peer1")
        score_before = s.get_score("peer1")
        s.decay_scores()
        score_after = s.get_score("peer1")
        assert score_after < score_before
        assert score_after > 0  # Positive scores decay toward 0

    def test_decay_negative_scores(self):
        s = PeerScorer(decay_rate=0.5)
        s.record_invalid_block("peer1")
        score_before = s.get_score("peer1")
        s.decay_scores()
        score_after = s.get_score("peer1")
        assert score_after > score_before  # Closer to 0
        assert score_after < 0  # Still negative

    def test_get_low_scoring_peers(self):
        s = PeerScorer(disconnect_threshold=-50.0)
        s.record_invalid_block("bad1")
        s.record_invalid_block("bad1")  # score = -100
        s.record_valid_block("good1")

        low = s.get_low_scoring_peers()
        assert "bad1" in low
        assert "good1" not in low

    def test_rate_limit_violation(self):
        s = PeerScorer()
        s.record_rate_limit_violation("peer1")
        assert s.get_score("peer1") < 0

    def test_protocol_error(self):
        s = PeerScorer()
        s.record_protocol_error("peer1")
        assert s.get_score("peer1") < 0

    def test_peer_count(self):
        s = PeerScorer()
        s.record_valid_block("a")
        s.record_valid_block("b")
        assert s.peer_count() == 2
