"""Tests for shared.telemetry – TelemetryManager."""

import json
import logging
import time

import pytest

from shared.block import Block, BlockHeader, BlockRequirements, MinerInfo, QuantumProof
from shared.telemetry import TelemetryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_miner_info(miner_id="node1-CPU-0", miner_type="CPU"):
    return MinerInfo(
        miner_id=miner_id,
        miner_type=miner_type,
        reward_address=b"\x01" * 32,
        ecdsa_public_key=b"\x02" * 32,
        wots_public_key=b"\x03" * 32,
        next_wots_public_key=b"\x04" * 32,
    )


def _make_block(index=1, timestamp=1700000001):
    header = BlockHeader(
        version=1,
        previous_hash=b"\x00" * 32,
        index=index,
        timestamp=timestamp,
        data_hash=b"\xaa" * 32,
    )
    proof = QuantumProof(
        nonce=42,
        salt=b"\xbb" * 8,
        nodes=list(range(10)),
        edges=[(i, i + 1) for i in range(9)],
        solutions=[[1, -1] * 5 for _ in range(5)],
        mining_time=12.5,
    )
    proof.energy = -3950.0
    proof.diversity = 0.22
    proof.num_valid_solutions = 12
    reqs = BlockRequirements(
        difficulty_energy=-4100.0,
        min_diversity=0.15,
        min_solutions=5,
        timeout_to_difficulty_adjustment_decay=600,
    )
    block = Block(
        header=header,
        miner_info=_make_miner_info(),
        quantum_proof=proof,
        next_block_requirements=reqs,
        data=b"test",
    )
    block.hash = b"\xff" * 32
    return block


# ---------------------------------------------------------------------------
# Tests – disabled mode
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_disabled_skips_writes(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=False)
        tm.update_node("1.2.3.4:20049", "active", _make_miner_info())
        tm.record_block(_make_block())
        assert not (tmp_path / "tel").exists()


# ---------------------------------------------------------------------------
# Tests – node telemetry
# ---------------------------------------------------------------------------

class TestNodeTelemetry:
    def test_update_node_writes_json(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.update_node("1.2.3.4:20049", "active", _make_miner_info())

        data = json.loads((tmp_path / "tel" / "nodes.json").read_text())
        assert data["node_count"] == 1
        assert data["active_count"] == 1
        node = data["nodes"]["1.2.3.4:20049"]
        assert node["address"] == "1.2.3.4:20049"
        assert node["status"] == "active"
        assert node["ecdsa_public_key_hex"] == (b"\x02" * 32).hex()
        # miner_id / miner_type are no longer at top level — they are
        # derived from the descriptor when one has been attached.
        assert "miner_id" not in node
        assert "miner_type" not in node

    def test_update_node_flattens_descriptor(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        descriptor = {
            "descriptor_version": 1,
            "node_name": "test-node",
            "public_host": "1.2.3.4",
            "public_port": 20049,
            "miners": {"cpu": {"kind": "CPU", "num_cpus": 4}},
            "system_info": {"os": {"system": "Linux"}},
        }
        tm.update_node(
            "1.2.3.4:20049", "active", _make_miner_info(),
            descriptor=descriptor,
        )

        data = json.loads((tmp_path / "tel" / "nodes.json").read_text())
        node = data["nodes"]["1.2.3.4:20049"]
        # Descriptor keys are flattened onto the entry.
        assert node["node_name"] == "test-node"
        assert node["public_host"] == "1.2.3.4"
        assert node["miners"]["cpu"]["kind"] == "CPU"
        assert node["system_info"]["os"]["system"] == "Linux"
        # Connection metadata sits alongside.
        assert node["address"] == "1.2.3.4:20049"
        assert node["status"] == "active"
        # No nested "descriptor" wrapper and no legacy top-level fields.
        assert "descriptor" not in node
        assert "miner_id" not in node
        assert "miner_type" not in node

    def test_connection_metadata_survives_hostile_descriptor(self, tmp_path):
        """A peer-supplied descriptor cannot overwrite our connection fields."""
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        hostile = {
            "node_name": "peer",
            "address": "0.0.0.0:0",           # attempt to hijack
            "status": "lost",                 # attempt to flip status
            "last_heartbeat": 0,
        }
        tm.update_node("1.2.3.4:20049", "active", descriptor=hostile)

        data = json.loads((tmp_path / "tel" / "nodes.json").read_text())
        node = data["nodes"]["1.2.3.4:20049"]
        assert node["address"] == "1.2.3.4:20049"
        assert node["status"] == "active"

    def test_update_node_preserves_first_seen(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.update_node("1.2.3.4:20049", "active", _make_miner_info())
        first = tm._nodes["1.2.3.4:20049"].first_seen
        time.sleep(0.01)
        tm.update_node("1.2.3.4:20049", "active", _make_miner_info(), last_heartbeat=time.time())
        assert tm._nodes["1.2.3.4:20049"].first_seen == first

    def test_remove_node_marks_lost(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.update_node("1.2.3.4:20049", "active")
        tm.remove_node("1.2.3.4:20049")

        data = json.loads((tmp_path / "tel" / "nodes.json").read_text())
        assert data["nodes"]["1.2.3.4:20049"]["status"] == "lost"

    def test_remove_unknown_node_is_noop(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.remove_node("unknown:20049")  # should not raise

    def test_record_initial_peers(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.record_initial_peers(["a.b.c:20049", "d.e.f:20049"])

        data = json.loads((tmp_path / "tel" / "nodes.json").read_text())
        assert data["node_count"] == 2
        assert data["nodes"]["a.b.c:20049"]["status"] == "initial_peer"
        assert data["nodes"]["d.e.f:20049"]["status"] == "initial_peer"

    def test_initial_peers_not_overwritten(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.update_node("a.b.c:20049", "active", _make_miner_info())
        tm.record_initial_peers(["a.b.c:20049"])
        # Should not overwrite active status
        assert tm._nodes["a.b.c:20049"].status == "active"

    def test_multiple_nodes_tracking(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.update_node("a:20049", "active", _make_miner_info("n1", "CPU"))
        tm.update_node("b:20049", "active", _make_miner_info("n2", "GPU"))
        tm.update_node("c:20049", "failed")
        tm.remove_node("a:20049")

        data = json.loads((tmp_path / "tel" / "nodes.json").read_text())
        assert data["node_count"] == 3
        assert data["active_count"] == 1
        assert data["nodes"]["a:20049"]["status"] == "lost"
        assert data["nodes"]["b:20049"]["status"] == "active"
        assert data["nodes"]["c:20049"]["status"] == "failed"


class TestWriteDebounce:
    """Regression tests for the retry-storm debounce on ``update_node``."""

    def test_same_status_within_window_debounces(self, tmp_path, monkeypatch):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        writes = []
        orig_write = tm._write_nodes_json

        def spy():
            writes.append(1)
            orig_write()

        monkeypatch.setattr(tm, "_write_nodes_json", spy)

        # First call: always writes (new record).
        tm.update_node("1.2.3.4:20049", "rejected_capacity",
                       descriptor={"runtime": {"quip_version": "0.1.4"}})
        assert len(writes) == 1

        # Retry storm: several updates within the debounce window.
        for _ in range(5):
            tm.update_node("1.2.3.4:20049", "rejected_capacity",
                           descriptor={"runtime": {"quip_version": "0.1.4"}})
        assert len(writes) == 1, "debounced same-status updates must not rewrite"

        # last_seen still advances in memory so a late consumer isn't misled.
        assert tm._nodes["1.2.3.4:20049"].last_seen >= tm._nodes["1.2.3.4:20049"].first_seen

    def test_debounce_expires_after_window(self, tmp_path, monkeypatch):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        writes = []
        orig_write = tm._write_nodes_json

        def spy():
            writes.append(1)
            orig_write()

        monkeypatch.setattr(tm, "_write_nodes_json", spy)

        base = time.time()
        monkeypatch.setattr("shared.telemetry.time.time", lambda: base)
        tm.update_node("peer:20049", "rejected_capacity",
                       descriptor={"runtime": {"quip_version": "0.1.4"}})
        assert len(writes) == 1

        # Still inside the window — no new write.
        monkeypatch.setattr("shared.telemetry.time.time", lambda: base + 10.0)
        tm.update_node("peer:20049", "rejected_capacity",
                       descriptor={"runtime": {"quip_version": "0.1.4"}})
        assert len(writes) == 1

        # Past the window — write fires again.
        monkeypatch.setattr("shared.telemetry.time.time", lambda: base + 31.0)
        tm.update_node("peer:20049", "rejected_capacity",
                       descriptor={"runtime": {"quip_version": "0.1.4"}})
        assert len(writes) == 2

    def test_status_change_bypasses_debounce(self, tmp_path, monkeypatch):
        """A transition rejected_capacity -> active must rewrite immediately."""
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        writes = []
        orig_write = tm._write_nodes_json

        def spy():
            writes.append(1)
            orig_write()

        monkeypatch.setattr(tm, "_write_nodes_json", spy)

        tm.update_node("peer:20049", "rejected_capacity",
                       descriptor={"runtime": {"quip_version": "0.1.4"}})
        tm.update_node("peer:20049", "active", _make_miner_info())
        assert len(writes) == 2

    def test_active_status_never_debounced(self, tmp_path, monkeypatch):
        """Heartbeat/liveness updates must not be throttled."""
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        writes = []
        orig_write = tm._write_nodes_json

        def spy():
            writes.append(1)
            orig_write()

        monkeypatch.setattr(tm, "_write_nodes_json", spy)

        for _ in range(4):
            tm.update_node("peer:20049", "active", _make_miner_info(),
                           last_heartbeat=time.time())
        assert len(writes) == 4


# ---------------------------------------------------------------------------
# Tests – block telemetry
# ---------------------------------------------------------------------------

def _hash(byte: int) -> bytes:
    """Build a 32-byte hash filled with *byte* — stable/predictable in tests."""
    return bytes([byte]) * 32


def _make_block_with_hash(index, timestamp, block_hash):
    block = _make_block(index=index, timestamp=timestamp)
    block.hash = block_hash
    return block


class TestBlockTelemetry:
    def test_record_block_uses_block_hash_for_dir_name(self, tmp_path):
        """Epoch dir name = first 16 hex chars of block 1's hash.

        Content-addressed keying so the dashboard can match the dir to a
        specific chain instance, not a node-local observation (timestamp).
        """
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        h = _hash(0xAB)  # hex() -> "abab..."
        tm.record_block(_make_block_with_hash(1, 1700000001, h))

        expected_dir = h.hex()[:16]  # "abababababababab"
        assert (tmp_path / "tel" / expected_dir / "1.json").exists()

    def test_record_block_writes_json_fields(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        h = _hash(0xAB)
        tm.record_block(_make_block_with_hash(1, 1700000001, h))

        epoch_dir = h.hex()[:16]
        data = json.loads((tmp_path / "tel" / epoch_dir / "1.json").read_text())
        assert data["block_index"] == 1
        assert data["block_hash"] == h.hex()
        assert data["miner"]["miner_id"] == "node1-CPU-0"
        assert data["quantum_proof"]["energy"] == -3950.0
        assert data["quantum_proof"]["mining_time"] == 12.5
        assert data["quantum_proof"]["num_nodes"] == 10
        assert data["requirements"]["difficulty_energy"] == -4100.0

    def test_genesis_dir_before_block_1(self, tmp_path):
        """Blocks recorded before block 1 is seen land in telemetry/genesis/."""
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.record_block(_make_block(index=0, timestamp=1700000000))
        assert (tmp_path / "tel" / "genesis" / "0.json").exists()

    def test_epoch_set_on_block_1(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        h = _hash(0xCD)
        tm.record_block(_make_block_with_hash(1, 1700000001, h))
        assert tm._block_1_hash == h
        # Subsequent blocks use same epoch dir (derived from block 1's hash)
        tm.record_block(_make_block_with_hash(2, 1700000050, _hash(0x99)))
        assert (tmp_path / "tel" / h.hex()[:16] / "2.json").exists()

    def test_reset_epoch_clears_block_1_hash(self, tmp_path):
        """reset_epoch() clears internal state so the next block 1 re-keys."""
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        h_a = _hash(0x01)
        h_b = _hash(0x02)
        tm.record_block(_make_block_with_hash(1, 1700000001, h_a))
        tm.reset_epoch()
        tm.record_block(_make_block_with_hash(1, 1700099999, h_b))

        assert (tmp_path / "tel" / h_a.hex()[:16] / "1.json").exists()
        assert (tmp_path / "tel" / h_b.hex()[:16] / "1.json").exists()

    def test_record_block_writes_current_epoch_marker(self, tmp_path):
        """Atomic marker file names the live epoch dir for the cache."""
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        h = _hash(0xEF)
        tm.record_block(_make_block_with_hash(1, 1700000001, h))

        marker = tmp_path / "tel" / "current_epoch.json"
        assert marker.is_file()
        data = json.loads(marker.read_text())
        assert data["epoch"] == h.hex()[:16]
        assert data["block_1_hash"] == h.hex()
        assert "updated_at" in data

    def test_current_epoch_marker_updates_on_rekey(self, tmp_path):
        """Reorg at block 1 re-keys: marker must follow the new hash.

        The cache labels dirs live/stale against this marker, so it must
        track the current canonical chain's block 1 — not the first one
        ever seen by the node.
        """
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        h_a = _hash(0xAA)
        h_b = _hash(0xBB)

        tm.record_block(_make_block_with_hash(1, 1700000001, h_a))
        tm.reset_epoch()
        tm.record_block(_make_block_with_hash(1, 1700000001, h_b))

        data = json.loads((tmp_path / "tel" / "current_epoch.json").read_text())
        assert data["epoch"] == h_b.hex()[:16]
        assert data["block_1_hash"] == h_b.hex()

    def test_sync_epoch_from_chain(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        h = _hash(0x11)
        chain = [
            _make_block_with_hash(0, 1700000000, _hash(0x00)),
            _make_block_with_hash(1, 1700000001, h),
        ]
        tm.sync_epoch_from_chain(chain)
        assert tm._block_1_hash == h

    def test_sync_epoch_rekeys_on_block_1_hash_change(self, tmp_path):
        """If block 1's hash changes (reorg-at-height-1), re-key.

        Follows the canonical chain. Stale dir from the previous block 1
        remains on disk; the cache marks it stale_fork using the marker.
        """
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        h_a = _hash(0x11)
        h_b = _hash(0x22)

        tm.sync_epoch_from_chain([_make_block_with_hash(1, 1, h_a)])
        assert tm._block_1_hash == h_a

        tm.sync_epoch_from_chain([_make_block_with_hash(1, 1, h_b)])
        assert tm._block_1_hash == h_b


# ---------------------------------------------------------------------------
# Tests – legacy dir migration
# ---------------------------------------------------------------------------


def _write_legacy_block(path, index, block_hash_hex):
    """Write a legacy-format telemetry block JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "block_index": index,
        "block_hash": block_hash_hex,
        "timestamp": 1_700_000_000 + index,
        "miner": {"miner_id": "legacy"},
    }))


class TestMigrateLegacyDirs:
    """Tests for TelemetryManager.migrate_legacy_dirs."""

    def test_renames_timestamp_dir_to_hash_prefix(self, tmp_path):
        """A timestamp-named dir with a valid block 1 is renamed in place."""
        tel = tmp_path / "tel"
        block_1_hash = (b"\xAB" * 32).hex()
        _write_legacy_block(tel / "1700000001" / "1.json", 1, block_1_hash)
        _write_legacy_block(tel / "1700000001" / "2.json", 2, "ff" * 32)

        tm = TelemetryManager(str(tel), enabled=True)
        result = tm.migrate_legacy_dirs()

        new_name = block_1_hash[:16]
        assert (tel / new_name / "1.json").is_file()
        assert (tel / new_name / "2.json").is_file()
        assert not (tel / "1700000001").exists()
        assert (("1700000001", new_name)) in result["migrated"]

    def test_idempotent_already_migrated(self, tmp_path):
        """A dir whose name is already 16-hex is left alone."""
        tel = tmp_path / "tel"
        hex_name = "abababababababab"
        _write_legacy_block(tel / hex_name / "1.json", 1, "ab" * 32)

        tm = TelemetryManager(str(tel), enabled=True)
        result = tm.migrate_legacy_dirs()

        assert (tel / hex_name / "1.json").is_file()
        assert result["migrated"] == []

    def test_skips_genesis_dir(self, tmp_path):
        """The genesis dir (pre-block-1 staging) is never migrated."""
        tel = tmp_path / "tel"
        _write_legacy_block(tel / "genesis" / "0.json", 0, "00" * 32)

        tm = TelemetryManager(str(tel), enabled=True)
        result = tm.migrate_legacy_dirs()

        assert (tel / "genesis" / "0.json").is_file()
        assert result["migrated"] == []

    def test_skips_dir_without_block_1(self, tmp_path):
        """If 1.json is missing we can't compute the new name — skip."""
        tel = tmp_path / "tel"
        _write_legacy_block(tel / "1700000001" / "2.json", 2, "ff" * 32)

        tm = TelemetryManager(str(tel), enabled=True)
        result = tm.migrate_legacy_dirs()

        assert (tel / "1700000001" / "2.json").is_file()
        assert result["migrated"] == []
        assert any("1700000001" in s[0] for s in result["skipped"])

    def test_skips_when_target_already_exists(self, tmp_path):
        """If the hash-named target already exists, keep both; don't clobber."""
        tel = tmp_path / "tel"
        block_1_hash = (b"\xAB" * 32).hex()
        new_name = block_1_hash[:16]

        _write_legacy_block(tel / "1700000001" / "1.json", 1, block_1_hash)
        _write_legacy_block(tel / new_name / "1.json", 1, block_1_hash)

        tm = TelemetryManager(str(tel), enabled=True)
        result = tm.migrate_legacy_dirs()

        assert (tel / "1700000001" / "1.json").is_file()
        assert (tel / new_name / "1.json").is_file()
        assert result["migrated"] == []

    def test_disabled_is_noop(self, tmp_path):
        """With telemetry disabled, migration does nothing."""
        tel = tmp_path / "tel"
        _write_legacy_block(tel / "1700000001" / "1.json", 1, "ab" * 32)

        tm = TelemetryManager(str(tel), enabled=False)
        result = tm.migrate_legacy_dirs()

        assert (tel / "1700000001").exists()
        assert result == {"migrated": [], "skipped": []}


# ---------------------------------------------------------------------------
# Tests – log output
# ---------------------------------------------------------------------------

class TestLogTable:
    def test_log_nodes_table(self, tmp_path, caplog):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.update_node("1.2.3.4:20049", "active", _make_miner_info())
        tm.update_node("5.6.7.8:20049", "failed")

        with caplog.at_level(logging.INFO):
            tm.log_nodes_table()

        assert "Known Nodes (2 total, 1 active)" in caplog.text
        assert "1.2.3.4:20049" in caplog.text
        assert "5.6.7.8:20049" in caplog.text

    def test_log_empty_nodes(self, tmp_path, caplog):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        with caplog.at_level(logging.INFO):
            tm.log_nodes_table()
        assert "none" in caplog.text
