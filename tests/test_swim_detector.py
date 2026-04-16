"""Tests for SWIM failure detector."""

import pytest

from shared.swim_detector import SwimDetector, PeerState


class TestSwimDetector:
    def test_new_peer_is_alive(self):
        d = SwimDetector()
        d.add_peer("A")
        assert d.get_state("A") == PeerState.ALIVE
        assert d.peer_count() == 1

    def test_heartbeat_success_keeps_alive(self):
        d = SwimDetector()
        d.add_peer("A")
        d.record_heartbeat_success("A")
        assert d.get_state("A") == PeerState.ALIVE

    def test_heartbeat_failure_moves_to_suspect(self):
        d = SwimDetector()
        d.add_peer("A")
        d.record_heartbeat_failure("A")
        assert d.get_state("A") == PeerState.SUSPECT

    def test_get_suspects(self):
        d = SwimDetector()
        d.add_peer("A")
        d.add_peer("B")
        d.record_heartbeat_failure("A")
        suspects = d.get_suspects()
        assert "A" in suspects
        assert "B" not in suspects

    def test_heartbeat_success_clears_suspect(self):
        d = SwimDetector()
        d.add_peer("A")
        d.record_heartbeat_failure("A")
        assert d.get_state("A") == PeerState.SUSPECT
        d.record_heartbeat_success("A")
        assert d.get_state("A") == PeerState.ALIVE

    def test_create_probe_requests(self):
        d = SwimDetector(k_probes=2)
        d.add_peer("target")
        d.add_peer("prober1")
        d.add_peer("prober2")
        d.record_heartbeat_failure("target")

        probes = d.create_probe_requests(["target", "prober1", "prober2"])
        assert len(probes) == 2
        assert all(p.target == "target" for p in probes)
        # Probers should not include the target
        assert all(p.prober != "target" for p in probes)

    def test_probe_success_revives_peer(self):
        d = SwimDetector(k_probes=2, suspect_rounds=2)
        d.add_peer("A")
        d.add_peer("B")
        d.add_peer("C")
        d.record_heartbeat_failure("A")

        probes = d.create_probe_requests(["A", "B", "C"])
        # One probe succeeds
        d.record_probe_result("A", probes[0].prober, success=True)
        assert d.get_state("A") == PeerState.ALIVE

    def test_all_probes_fail_escalates(self):
        d = SwimDetector(k_probes=2, suspect_rounds=1)
        d.add_peer("A")
        d.add_peer("B")
        d.add_peer("C")
        d.record_heartbeat_failure("A")

        probes = d.create_probe_requests(["A", "B", "C"])
        for p in probes:
            d.record_probe_result("A", p.prober, success=False)

        assert d.get_state("A") == PeerState.DEAD

    def test_dead_after_multiple_suspect_rounds(self):
        d = SwimDetector(k_probes=1, suspect_rounds=2)
        d.add_peer("A")
        d.add_peer("B")

        # Round 1: heartbeat fails, suspect round 1
        d.record_heartbeat_failure("A")
        probes = d.create_probe_requests(["A", "B"])
        for p in probes:
            d.record_probe_result("A", p.prober, success=False)
        # After 1 failed round with suspect_rounds=2, still suspect
        assert d.get_state("A") == PeerState.SUSPECT

        # Round 2: heartbeat fails again, suspect round 2
        d.record_heartbeat_failure("A")
        probes = d.create_probe_requests(["A", "B"])
        for p in probes:
            d.record_probe_result("A", p.prober, success=False)
        assert d.get_state("A") == PeerState.DEAD

    def test_get_dead_peers(self):
        d = SwimDetector(k_probes=1, suspect_rounds=1)
        d.add_peer("A")
        d.add_peer("B")
        d.record_heartbeat_failure("A")
        probes = d.create_probe_requests(["A", "B"])
        for p in probes:
            d.record_probe_result("A", p.prober, success=False)

        dead = d.get_dead_peers()
        assert "A" in dead
        assert "B" not in dead

    def test_remove_peer(self):
        d = SwimDetector()
        d.add_peer("A")
        d.remove_peer("A")
        assert d.get_state("A") is None
        assert d.peer_count() == 0

    def test_membership_deltas(self):
        d = SwimDetector()
        d.add_peer("A")
        d.add_peer("B")
        d.remove_peer("A")

        deltas = d.get_membership_deltas()
        join_peers = [p for p, _ in deltas["joins"]]
        leave_peers = [p for p, _ in deltas["leaves"]]
        assert "A" in join_peers
        assert "B" in join_peers
        assert "A" in leave_peers

    def test_no_probes_without_candidates(self):
        """No probes created when the suspect is the only peer."""
        d = SwimDetector(k_probes=3)
        d.add_peer("A")
        d.record_heartbeat_failure("A")
        probes = d.create_probe_requests(["A"])
        assert len(probes) == 0

    def test_alive_count(self):
        d = SwimDetector()
        d.add_peer("A")
        d.add_peer("B")
        d.add_peer("C")
        d.record_heartbeat_failure("C")
        assert d.alive_count() == 2

    def test_auto_add_on_heartbeat_success(self):
        """Recording heartbeat for unknown peer auto-adds it."""
        d = SwimDetector()
        d.record_heartbeat_success("new_peer")
        assert d.get_state("new_peer") == PeerState.ALIVE
        assert d.peer_count() == 1

    def test_duplicate_probe_requests_not_created(self):
        """Don't create new probes if pending probes already exist."""
        d = SwimDetector(k_probes=2)
        d.add_peer("A")
        d.add_peer("B")
        d.add_peer("C")
        d.record_heartbeat_failure("A")

        probes1 = d.create_probe_requests(["A", "B", "C"])
        assert len(probes1) == 2

        # Second call should not create more probes (pending exist)
        probes2 = d.create_probe_requests(["A", "B", "C"])
        assert len(probes2) == 0
