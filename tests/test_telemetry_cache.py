"""Tests for shared.telemetry_cache – TelemetryCache."""

import asyncio
import json
import os

import pytest

from shared.telemetry_cache import TelemetryCache, EpochInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_json(path, data):
    """Write a JSON file atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _make_block_json(block_index, epoch_ts, energy=-3950.0):
    return {
        "block_index": block_index,
        "block_hash": f"deadbeef{block_index:04d}",
        "timestamp": epoch_ts + block_index * 10,
        "previous_hash": "0" * 64,
        "miner": {
            "miner_id": "test-node",
            "miner_type": "CPU",
            "ecdsa_public_key": "02" * 32,
        },
        "quantum_proof": {
            "energy": energy,
            "diversity": 0.22,
            "num_valid_solutions": 5,
            "mining_time": 5.0,
            "nonce": 12345,
            "num_nodes": 100,
            "num_edges": 200,
        },
        "requirements": {
            "difficulty_energy": -4100.0,
            "min_diversity": 0.15,
            "min_solutions": 5,
        },
    }


def _make_nodes_json(count=2, active=1):
    return {
        "updated_at": "2026-04-02T23:31:53.610902+00:00",
        "node_count": count,
        "active_count": active,
        "nodes": {
            f"127.0.0.{i}:8085": {
                "address": f"127.0.0.{i}:8085",
                "miner_id": f"node-{i}",
                "status": "active" if i < active else "initial_peer",
            }
            for i in range(count)
        },
    }


@pytest.fixture
def telemetry_dir(tmp_path):
    """Create a populated telemetry directory."""
    tdir = tmp_path / "telemetry"
    tdir.mkdir()

    # nodes.json
    _write_json(str(tdir / "nodes.json"), _make_nodes_json())

    # Epoch 1775166921 with 3 blocks
    epoch1 = "1775166921"
    for i in range(1, 4):
        _write_json(
            str(tdir / epoch1 / f"{i}.json"),
            _make_block_json(i, int(epoch1)),
        )

    # Epoch 1775167182 with 5 blocks
    epoch2 = "1775167182"
    for i in range(1, 6):
        _write_json(
            str(tdir / epoch2 / f"{i}.json"),
            _make_block_json(i, int(epoch2)),
        )

    return str(tdir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTelemetryCacheRefresh:
    """Tests for the cache refresh mechanism."""

    @pytest.mark.asyncio
    async def test_refresh_populates_state(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        status = cache.get_status()
        assert status["total_blocks"] == 8  # 3 + 5
        assert len(status["epochs"]) == 2
        assert status["latest_epoch"] == "1775167182"
        assert status["latest_block_index"] == 5
        assert status["node_count"] == 2
        assert status["active_node_count"] == 1

    @pytest.mark.asyncio
    async def test_refresh_detects_new_blocks(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        assert cache._latest_block_index == 5

        # Add a new block
        _write_json(
            os.path.join(telemetry_dir, "1775167182", "6.json"),
            _make_block_json(6, 1775167182),
        )

        callback_data = []
        cache.on_new_block = lambda ep, idx, d: callback_data.append((ep, idx))

        await cache._refresh()

        assert cache._latest_block_index == 6
        assert len(callback_data) == 1
        assert callback_data[0] == ("1775167182", 6)

    @pytest.mark.asyncio
    async def test_refresh_detects_nodes_change(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        # Update nodes.json
        import time
        time.sleep(0.05)  # Ensure mtime changes
        _write_json(
            os.path.join(telemetry_dir, "nodes.json"),
            _make_nodes_json(count=3, active=2),
        )

        callback_data = []
        cache.on_nodes_changed = lambda d: callback_data.append(d)

        await cache._refresh()

        assert cache.get_status()["node_count"] == 3
        assert len(callback_data) == 1

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path):
        tdir = tmp_path / "empty_telemetry"
        tdir.mkdir()
        cache = TelemetryCache(telemetry_dir=str(tdir), refresh_interval=60)
        await cache._refresh()

        status = cache.get_status()
        assert status["total_blocks"] == 0
        assert status["epochs"] == []
        assert status["latest_epoch"] == ""

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self, tmp_path):
        cache = TelemetryCache(
            telemetry_dir=str(tmp_path / "does_not_exist"),
            refresh_interval=60,
        )
        await cache._refresh()
        assert cache.get_status()["total_blocks"] == 0


class TestTelemetryCacheAccessors:
    """Tests for cache accessor methods."""

    @pytest.mark.asyncio
    async def test_get_epochs(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        epochs = cache.get_epochs()
        assert len(epochs) == 2
        assert epochs[0]["epoch"] == "1775166921"
        assert epochs[0]["block_count"] == 3
        assert epochs[1]["epoch"] == "1775167182"
        assert epochs[1]["block_count"] == 5

    @pytest.mark.asyncio
    async def test_get_nodes(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        nodes = cache.get_nodes()
        assert nodes is not None
        assert nodes["node_count"] == 2
        assert "127.0.0.0:8085" in nodes["nodes"]

    @pytest.mark.asyncio
    async def test_get_block_cache_hit(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        block = cache.get_block("1775166921", 1)
        assert block is not None
        assert block["block_index"] == 1

        # Second call should hit cache
        block2 = cache.get_block("1775166921", 1)
        assert block2 is block  # same object

    @pytest.mark.asyncio
    async def test_get_block_not_found(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        assert cache.get_block("1775166921", 99) is None
        assert cache.get_block("9999999999", 1) is None

    @pytest.mark.asyncio
    async def test_get_latest(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        latest = cache.get_latest()
        assert latest is not None
        assert latest["epoch"] == "1775167182"
        assert latest["block_index"] == 5
        assert latest["block"]["block_index"] == 5


class TestTelemetryCacheETags:
    """Tests for ETag generation."""

    @pytest.mark.asyncio
    async def test_status_etag(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        etag = cache.status_etag()
        assert "1775167182" in etag
        assert "5" in etag

    @pytest.mark.asyncio
    async def test_nodes_etag(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        etag = cache.nodes_etag()
        assert "2026-04-02" in etag

    @pytest.mark.asyncio
    async def test_block_etag(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        block = cache.get_block("1775166921", 1)
        etag = cache.block_etag(block)
        assert etag == "deadbeef0001"


class TestTelemetryCacheLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=0.1)
        await cache.start()

        assert cache._refresh_task is not None
        assert cache.get_status()["total_blocks"] == 8

        await cache.stop()
        assert cache._refresh_task is None

    @pytest.mark.asyncio
    async def test_block_cache_eviction(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        cache._block_cache_max = 3
        await cache._refresh()

        # Access several blocks to fill cache beyond max
        cache.get_block("1775166921", 1)
        cache.get_block("1775166921", 2)
        cache.get_block("1775166921", 3)
        cache.get_block("1775167182", 1)

        assert len(cache._block_cache) <= 3
