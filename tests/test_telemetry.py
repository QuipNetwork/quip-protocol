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
        assert node["miner_id"] == "node1-CPU-0"
        assert node["status"] == "active"
        assert node["ecdsa_public_key_hex"] == (b"\x02" * 32).hex()

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


# ---------------------------------------------------------------------------
# Tests – block telemetry
# ---------------------------------------------------------------------------

class TestBlockTelemetry:
    def test_record_block_creates_epoch_dir(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        block = _make_block(index=1, timestamp=1700000001)
        tm.record_block(block)
        assert (tmp_path / "tel" / "1700000001" / "1.json").exists()

    def test_record_block_writes_json_fields(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        block = _make_block(index=1, timestamp=1700000001)
        tm.record_block(block)

        data = json.loads((tmp_path / "tel" / "1700000001" / "1.json").read_text())
        assert data["block_index"] == 1
        assert data["block_hash"] == (b"\xff" * 32).hex()
        assert data["miner"]["miner_id"] == "node1-CPU-0"
        assert data["quantum_proof"]["energy"] == -3950.0
        assert data["quantum_proof"]["mining_time"] == 12.5
        assert data["quantum_proof"]["num_nodes"] == 10
        assert data["requirements"]["difficulty_energy"] == -4100.0

    def test_genesis_dir_before_block_1(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        block = _make_block(index=0, timestamp=1700000000)
        tm.record_block(block)
        assert (tmp_path / "tel" / "genesis" / "0.json").exists()

    def test_epoch_set_on_block_1(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.record_block(_make_block(index=1, timestamp=1700000001))
        assert tm._epoch_timestamp == 1700000001
        # Subsequent blocks use same epoch dir
        tm.record_block(_make_block(index=2, timestamp=1700000050))
        assert (tmp_path / "tel" / "1700000001" / "2.json").exists()

    def test_epoch_reset_creates_new_dir(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.record_block(_make_block(index=1, timestamp=1700000001))
        tm.set_epoch_timestamp(None)
        tm.record_block(_make_block(index=1, timestamp=1700099999))

        assert (tmp_path / "tel" / "1700000001" / "1.json").exists()
        assert (tmp_path / "tel" / "1700099999" / "1.json").exists()

    def test_sync_epoch_from_chain(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        chain = [_make_block(index=0, timestamp=1700000000),
                 _make_block(index=1, timestamp=1700000001)]
        tm.sync_epoch_from_chain(chain)
        assert tm._epoch_timestamp == 1700000001

    def test_sync_epoch_noop_if_already_set(self, tmp_path):
        tm = TelemetryManager(str(tmp_path / "tel"), enabled=True)
        tm.set_epoch_timestamp(9999)
        tm.sync_epoch_from_chain([_make_block(index=1, timestamp=1111)])
        assert tm._epoch_timestamp == 9999


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
