"""Tests for LoadMonitor."""

import pytest

from shared.load_monitor import LoadMonitor, NodeLoad


class TestNodeLoad:
    def test_to_dict_round_trip(self):
        load = NodeLoad(
            connection_count=10, max_connections=50,
            cpu_load_avg=1.5, memory_percent=45.3,
            block_queue_depth=100, gossip_queue_depth=50,
        )
        d = load.to_dict()
        restored = NodeLoad.from_dict(d)
        assert restored.connection_count == 10
        assert restored.max_connections == 50
        assert restored.cpu_load_avg == 1.5

    def test_from_dict_defaults(self):
        load = NodeLoad.from_dict({})
        assert load.connection_count == 0
        assert load.max_connections == 50


class TestLoadMonitor:
    def test_not_overloaded_initially(self):
        m = LoadMonitor(max_connections=50)
        m.update(connection_count=10)
        assert not m.is_overloaded()

    def test_overloaded_at_high_watermark(self):
        m = LoadMonitor(max_connections=50, high_watermark=0.8)
        m.update(connection_count=40)
        assert m.is_overloaded()

    def test_hysteresis_stays_overloaded(self):
        m = LoadMonitor(max_connections=50, high_watermark=0.8, low_watermark=0.5)
        m.update(connection_count=42)
        assert m.is_overloaded()

        # Drop below high watermark but above low watermark
        m.update(connection_count=35)
        assert m.is_overloaded(), "Should stay overloaded until below low_watermark"

    def test_hysteresis_clears_at_low_watermark(self):
        m = LoadMonitor(max_connections=50, high_watermark=0.8, low_watermark=0.5)
        m.update(connection_count=42)
        assert m.is_overloaded()

        m.update(connection_count=24)
        assert not m.is_overloaded()

    def test_connections_to_shed(self):
        m = LoadMonitor(max_connections=50, high_watermark=0.8, low_watermark=0.5)
        m.update(connection_count=45)
        assert m.is_overloaded()
        shed = m.connections_to_shed()
        # Target is 50 * 0.5 = 25, excess = 45 - 25 = 20
        assert shed == 20

    def test_connections_to_shed_zero_when_not_overloaded(self):
        m = LoadMonitor(max_connections=50)
        m.update(connection_count=10)
        assert m.connections_to_shed() == 0

    def test_should_accept_join_under_capacity(self):
        m = LoadMonitor(max_connections=50)
        m.update(connection_count=30)
        assert m.should_accept_join()

    def test_should_not_accept_join_at_max(self):
        m = LoadMonitor(max_connections=50)
        m.update(connection_count=50)
        assert not m.should_accept_join()

    def test_should_not_accept_join_while_shedding(self):
        m = LoadMonitor(max_connections=50, high_watermark=0.8, low_watermark=0.5)
        m.update(connection_count=42)
        m.is_overloaded()  # trigger shedding
        m.update(connection_count=35)  # still above low watermark
        assert not m.should_accept_join()

    def test_connection_utilization(self):
        m = LoadMonitor(max_connections=100)
        m.update(connection_count=25)
        assert m.connection_utilization() == 0.25

    def test_snapshot(self):
        m = LoadMonitor(max_connections=50)
        m.update(connection_count=10, block_queue=5, gossip_queue=3)
        snap = m.snapshot()
        assert snap.connection_count == 10
        assert snap.max_connections == 50
        assert snap.block_queue_depth == 5

    def test_queue_overload(self):
        m = LoadMonitor(max_connections=50, queue_threshold=100)
        m.update(connection_count=10, block_queue=150)
        assert m.is_overloaded()
