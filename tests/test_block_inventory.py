"""Tests for BlockInventory IHAVE/IWANT tracking."""

import time
from unittest.mock import patch

import pytest

from _utils import hash_for as _hash
from shared.block_inventory import BlockInventory


class TestBlockInventory:
    def test_record_have(self):
        inv = BlockInventory()
        inv.record_have(_hash(1))
        assert inv.has_block(_hash(1))
        assert not inv.has_block(_hash(2))

    def test_have_count(self):
        inv = BlockInventory()
        inv.record_have(_hash(1))
        inv.record_have(_hash(2))
        assert inv.have_count() == 2

    def test_ihave_triggers_want_for_new_block(self):
        inv = BlockInventory()
        assert inv.record_ihave("peer1", _hash(1)) is True

    def test_ihave_does_not_trigger_want_if_already_have(self):
        inv = BlockInventory()
        inv.record_have(_hash(1))
        assert inv.record_ihave("peer1", _hash(1)) is False

    def test_ihave_does_not_trigger_want_if_already_requested(self):
        inv = BlockInventory()
        inv.record_ihave("peer1", _hash(1))
        inv.record_want(_hash(1), "peer1")
        # Second IHAVE from different peer should not trigger another want
        assert inv.record_ihave("peer2", _hash(1)) is False

    def test_record_want(self):
        inv = BlockInventory()
        inv.record_want(_hash(1), "peer1")
        assert inv.pending_want_count() == 1

    def test_record_block_received_clears_want(self):
        inv = BlockInventory()
        inv.record_want(_hash(1), "peer1")
        inv.record_block_received(_hash(1))
        assert inv.pending_want_count() == 0
        assert inv.has_block(_hash(1))

    def test_get_pending_wants(self):
        inv = BlockInventory()
        inv.record_want(_hash(1), "peer1")
        inv.record_want(_hash(2), "peer2")
        wants = inv.get_pending_wants()
        assert len(wants) == 2
        peers = {peer for _, peer in wants}
        assert peers == {"peer1", "peer2"}

    def test_expire_wants(self):
        inv = BlockInventory(want_timeout=1.0)
        inv.record_want(_hash(1), "peer1")

        # Not expired yet
        expired = inv.expire_wants()
        assert len(expired) == 0

        # Fast-forward time
        with patch("shared.block_inventory.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 2.0
            expired = inv.expire_wants()
            assert len(expired) == 1
            assert expired[0] == (_hash(1), "peer1")
        assert inv.pending_want_count() == 0

    def test_get_peers_with_block(self):
        inv = BlockInventory()
        inv.record_ihave("peer1", _hash(1))
        inv.record_ihave("peer2", _hash(1))
        inv.record_ihave("peer3", _hash(2))

        peers = inv.get_peers_with_block(_hash(1))
        assert set(peers) == {"peer1", "peer2"}

    def test_remove_peer(self):
        inv = BlockInventory()
        inv.record_ihave("peer1", _hash(1))
        inv.record_want(_hash(1), "peer1")

        inv.remove_peer("peer1")
        assert inv.pending_want_count() == 0
        assert inv.get_peers_with_block(_hash(1)) == []

    def test_max_have_eviction(self):
        inv = BlockInventory(max_have=5)
        for i in range(10):
            inv.record_have(_hash(i))
        assert inv.have_count() <= 5

    def test_eviction_keeps_newest(self):
        inv = BlockInventory(max_have=3)
        for i in range(5):
            inv.record_have(_hash(i))
        # Newest 3 should remain
        assert inv.has_block(_hash(4))
        assert inv.has_block(_hash(3))
