"""Tests for shared.telemetry_cache – TelemetryCache."""

import asyncio
import json
import os

import pytest

from shared.telemetry_cache import TelemetryCache, EpochInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

from _utils import write_json as _write_json  # noqa: E402


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
    """Build a nodes.json payload matching the current TelemetryManager shape.

    Each entry is a flat merge of connection metadata and descriptor
    fields; ``miner_id`` / ``miner_type`` are no longer at the top level
    (derive them from ``node_name`` / ``miners`` when needed).
    """
    return {
        "updated_at": "2026-04-02T23:31:53.610902+00:00",
        "node_count": count,
        "active_count": active,
        "nodes": {
            f"127.0.0.{i}:8085": {
                "address": f"127.0.0.{i}:8085",
                "status": "active" if i < active else "initial_peer",
                "node_name": f"node-{i}",
                "miners": {"cpu": {"kind": "CPU", "num_cpus": 1}},
            }
            for i in range(count)
        },
    }


# Hex-named epoch dirs (16 chars) replace legacy timestamp-named dirs.
EPOCH_A = "aaaaaaaaaaaaaaaa"  # oldest in fixture (3 blocks)
EPOCH_B = "bbbbbbbbbbbbbbbb"  # newest in fixture, marked live (5 blocks)


def _write_marker(tdir, epoch_hex, block_1_hash_full="bb" * 32):
    """Write a top-level current_epoch.json marker."""
    _write_json(os.path.join(tdir, "current_epoch.json"), {
        "epoch": epoch_hex,
        "block_1_hash": block_1_hash_full,
        "updated_at": "2026-04-02T23:31:53.610902+00:00",
    })


@pytest.fixture
def telemetry_dir(tmp_path):
    """Create a populated telemetry directory (hex-keyed, with marker)."""
    tdir = tmp_path / "telemetry"
    tdir.mkdir()

    _write_json(str(tdir / "nodes.json"), _make_nodes_json())

    # EPOCH_A: 3 blocks (stale fork — no marker pointing here)
    for i in range(1, 4):
        _write_json(
            str(tdir / EPOCH_A / f"{i}.json"),
            _make_block_json(i, 1775166921),
        )

    # EPOCH_B: 5 blocks (live — marker points here)
    for i in range(1, 6):
        _write_json(
            str(tdir / EPOCH_B / f"{i}.json"),
            _make_block_json(i, 1775167182),
        )

    _write_marker(str(tdir), EPOCH_B)
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
        # latest_epoch is driven by the marker, not by max(dir_names).
        assert status["latest_epoch"] == EPOCH_B
        assert status["latest_block_index"] == 5
        assert status["node_count"] == 2
        assert status["active_node_count"] == 1

    @pytest.mark.asyncio
    async def test_refresh_detects_new_blocks(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        assert cache._latest_block_index == 5

        _write_json(
            os.path.join(telemetry_dir, EPOCH_B, "6.json"),
            _make_block_json(6, 1775167182),
        )

        callback_data = []
        cache.on_new_block = lambda ep, idx, d: callback_data.append((ep, idx))

        await cache._refresh()

        assert cache._latest_block_index == 6
        assert len(callback_data) == 1
        assert callback_data[0] == (EPOCH_B, 6)

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
        by_name = {e["epoch"]: e for e in epochs}
        assert by_name[EPOCH_A]["block_count"] == 3
        assert by_name[EPOCH_B]["block_count"] == 5

    @pytest.mark.asyncio
    async def test_get_epochs_marks_live_vs_stale_fork(self, telemetry_dir):
        """Marker file determines which epoch is live; the rest are stale forks.

        The dashboard uses the ``status`` field to separate the canonical
        chain from fork directories left over from reorgs at height 1 or
        from node restarts mid-cycle.
        """
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        by_name = {e["epoch"]: e for e in cache.get_epochs()}
        assert by_name[EPOCH_B]["status"] == "live"
        assert by_name[EPOCH_A]["status"] == "stale_fork"

    @pytest.mark.asyncio
    async def test_no_marker_all_epochs_stale(self, tmp_path):
        """Without a marker we can't identify a live epoch — everything is stale."""
        tdir = tmp_path / "telemetry"
        tdir.mkdir()
        _write_json(
            str(tdir / EPOCH_A / "1.json"),
            _make_block_json(1, 1775166921),
        )

        cache = TelemetryCache(telemetry_dir=str(tdir), refresh_interval=60)
        await cache._refresh()

        epochs = cache.get_epochs()
        assert len(epochs) == 1
        assert epochs[0]["status"] == "stale_fork"
        # latest_epoch also empty since no marker defines "live".
        assert cache.get_status()["latest_epoch"] == ""

    @pytest.mark.asyncio
    async def test_ignores_legacy_timestamp_dirs(self, tmp_path):
        """Dirs that aren't exactly 16 hex chars are filtered out.

        Legacy timestamp-named dirs stay on disk until migrated but the
        cache never surfaces them, preventing pre-migration confusion.
        """
        tdir = tmp_path / "telemetry"
        tdir.mkdir()
        # Legacy dir: 10-char numeric timestamp, happens to be valid hex.
        _write_json(
            str(tdir / "1775167182" / "1.json"),
            _make_block_json(1, 1775167182),
        )

        cache = TelemetryCache(telemetry_dir=str(tdir), refresh_interval=60)
        await cache._refresh()

        assert cache.get_epochs() == []
        assert cache.get_status()["total_blocks"] == 0

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

        block = cache.get_block(EPOCH_A, 1)
        assert block is not None
        assert block["block_index"] == 1

        # Second call should hit cache
        block2 = cache.get_block(EPOCH_A, 1)
        assert block2 is block  # same object

    @pytest.mark.asyncio
    async def test_get_block_not_found(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        assert cache.get_block(EPOCH_A, 99) is None
        assert cache.get_block("9" * 16, 1) is None

    @pytest.mark.asyncio
    async def test_get_latest(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        latest = cache.get_latest()
        assert latest is not None
        assert latest["epoch"] == EPOCH_B
        assert latest["block_index"] == 5
        assert latest["block"]["block_index"] == 5


class TestTelemetryCacheETags:
    """Tests for ETag generation."""

    @pytest.mark.asyncio
    async def test_status_etag(self, telemetry_dir):
        cache = TelemetryCache(telemetry_dir=telemetry_dir, refresh_interval=60)
        await cache._refresh()

        etag = cache.status_etag()
        assert EPOCH_B in etag
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

        block = cache.get_block(EPOCH_A, 1)
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

        cache.get_block(EPOCH_A, 1)
        cache.get_block(EPOCH_A, 2)
        cache.get_block(EPOCH_A, 3)
        cache.get_block(EPOCH_B, 1)

        assert len(cache._block_cache) <= 3
