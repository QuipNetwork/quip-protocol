"""Tests for `BaseMiner.mine_work_item` (Phase 3 substrate-mode entry point).

Exercises the protocol-neutral mining loop against a real CPU SA miner with
a very-relaxed difficulty so the loop terminates in a few seconds. Verifies:

  - the loop accepts a `SubstrateMiningContext` and returns a `MiningResult`
  - the result is shaped correctly for `encode_quantum_proof`
  - the bridge objects (`_BridgePrevBlock`, `_BridgeNodeInfo`) carry the
    substrate snapshot fields under the legacy names the hooks expect
  - `stop_event` is observed and the loop exits cleanly with `None`
  - the worker process dispatches `op="mine_work_item"` correctly through
    `MinerHandle.mine_work_item(context)`
"""
from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from shared.base_miner import (
    _BridgeNodeInfo,
    _BridgePrevBlock,
    _block_requirements_from_difficulty,
)
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.substrate_submitter import encode_quantum_proof
from shared.substrate_types import (
    CANONICAL_H_VALUES,
    SubstrateDifficulty,
    SubstrateMiningContext,
)


@pytest.fixture(scope="module")
def cpu_miner():
    """Module-scoped CPU SA miner. Creating the SA sampler is expensive
    (loads the default Zephyr topology) so we keep one instance across all
    tests in this module."""
    # Local imports — keep the test module import-cheap. The SA miner pulls in
    # dimod / dwave-neal which take several hundred ms to import.
    from CPU.sa_miner import SimulatedAnnealingMiner
    miner = SimulatedAnnealingMiner(miner_id="test")
    yield miner


@pytest.fixture
def relaxed_context(cpu_miner) -> SubstrateMiningContext:
    """Build a SubstrateMiningContext over the miner's actual topology with
    a very-relaxed difficulty so mine_work_item terminates in a handful of
    iterations.

    The thresholds mirror what `Phase 2 bootstrap` seeds onto the dev chain
    via `Sudo.set_difficulty`, just scaled down so we don't wait minutes.
    """
    return SubstrateMiningContext(
        block_number=1,
        parent_hash=b"\xab" * 32,
        topology_hash=b"\xcd" * 32,
        nodes=list(cpu_miner.sampler.nodes),
        edges=[(int(u), int(v)) for u, v in cpu_miner.sampler.edges],
        difficulty=SubstrateDifficulty(
            min_solutions=1,
            max_energy_milli=0,         # any non-positive energy passes
            min_diversity_milli=0,
            min_quality_milli=0,
        ),
        miner_account_bytes=b"\x42" * 32,
        h_values=CANONICAL_H_VALUES,
    )


def test_bridge_prev_block_carries_substrate_fields(relaxed_context):
    bridge = _BridgePrevBlock.from_context(relaxed_context)
    # cur_index = prev_block.header.index + 1 must equal context.block_number
    assert bridge.header.index + 1 == relaxed_context.block_number
    assert bridge.hash == relaxed_context.parent_hash


def test_bridge_node_info_exposes_hex_account(relaxed_context):
    bridge = _BridgeNodeInfo.from_context(relaxed_context)
    assert bridge.miner_id == "0x" + relaxed_context.miner_account_bytes.hex()
    assert bridge.miner_account_bytes == relaxed_context.miner_account_bytes


def test_block_requirements_converts_milli_to_float():
    difficulty = SubstrateDifficulty(
        min_solutions=5,
        max_energy_milli=-2_500_000,
        min_diversity_milli=200,
        min_quality_milli=900,
    )
    req = _block_requirements_from_difficulty(difficulty)
    assert req.difficulty_energy == -2500.0
    assert req.min_diversity == 0.2
    assert req.min_solutions == 5
    # Decay is disabled in substrate mode — sentinel is large enough that
    # compute_current_requirements would no-op even if accidentally called.
    assert req.timeout_to_difficulty_adjustment_decay >= 2**30


def test_mine_work_item_returns_result(cpu_miner, relaxed_context):
    stop = mp.Event()
    result = cpu_miner.mine_work_item(relaxed_context, stop)
    assert isinstance(result, MiningResult)
    assert result.num_valid >= relaxed_context.difficulty.min_solutions
    assert result.energy <= relaxed_context.difficulty.max_energy
    assert len(result.salt) == 32
    # node_list and edge_list should match the topology the miner ran with
    assert sorted(result.node_list) == sorted(relaxed_context.nodes)


def test_mine_work_item_result_encodes_to_quantum_proof(cpu_miner, relaxed_context):
    stop = mp.Event()
    result = cpu_miner.mine_work_item(relaxed_context, stop)
    proof = encode_quantum_proof(result, relaxed_context)
    # Shape matches pallet QuantumProof: hex hashes, int nonce/salt, vec
    # nodes/edges/solutions, milli-precision h_values.
    assert proof["topology_hash"] == "0x" + relaxed_context.topology_hash.hex()
    assert proof["nonce"] == result.nonce
    assert proof["salt"] == "0x" + result.salt.hex()
    assert set(proof["nodes"]) == set(relaxed_context.nodes)
    assert len(proof["edges"]) == len(relaxed_context.edges)
    assert len(proof["solutions"]) >= relaxed_context.difficulty.min_solutions
    # Every spin in every solution is ±1 (chain rejects 0).
    for sol in proof["solutions"]:
        assert all(s in (-1, 1) for s in sol)
    assert proof["h_values"] == [-1000, 0, 1000]


def test_mine_work_item_observes_stop_event(cpu_miner, relaxed_context):
    """Set stop_event BEFORE calling mine_work_item and assert it returns
    None within one polling cycle. Mirrors how the controller cancels work
    on a new chain head."""
    impossibly_hard = SubstrateMiningContext(
        block_number=relaxed_context.block_number,
        parent_hash=relaxed_context.parent_hash,
        topology_hash=relaxed_context.topology_hash,
        nodes=relaxed_context.nodes,
        edges=relaxed_context.edges,
        # A solution must have energy <= -10^15 which is unreachable for a
        # 1368-node graph with ±1 couplings (true GSE is ~-4100).
        difficulty=SubstrateDifficulty(
            min_solutions=5,
            max_energy_milli=-10**18,
            min_diversity_milli=1000,
            min_quality_milli=1000,
        ),
        miner_account_bytes=relaxed_context.miner_account_bytes,
    )
    stop = mp.Event()
    stop.set()
    start = time.time()
    result = cpu_miner.mine_work_item(impossibly_hard, stop)
    elapsed = time.time() - start
    assert result is None
    # The cpu miner does one round of preprocessing/sampling before the
    # stop is observed, so allow a few seconds. The point is it isn't
    # mining for minutes.
    assert elapsed < 30, f"mine_work_item ignored stop_event for {elapsed:.1f}s"


@pytest.mark.timeout(120)
def test_miner_handle_dispatches_mine_work_item(relaxed_context):
    """End-to-end through the 2-process worker.

    Spawns a CPU SA miner worker, sends a `mine_work_item` op, and verifies
    a `MiningResult` comes back via the response queue. Same scaffolding the
    Phase 4 controller will use.
    """
    spec = {"id": "test-cpu-0", "kind": "cpu", "args": {}}
    handle = MinerHandle(spec=spec)
    try:
        handle.mine_work_item(relaxed_context)
        # Drain response with a generous timeout — CPU SA on Z(9,2) at
        # min_solutions=1 / max_energy=0 typically lands in <10s.
        deadline = time.time() + 100
        while time.time() < deadline:
            try:
                msg = handle.resp.get(timeout=5)
            except Exception:
                continue
            if isinstance(msg, MiningResult):
                assert msg.num_valid >= 1
                assert msg.energy <= 0
                return
            if isinstance(msg, dict) and msg.get("op") == "error":
                pytest.fail(f"worker errored: {msg.get('message')}")
        pytest.fail("worker did not produce a MiningResult before deadline")
    finally:
        handle.cancel()
        handle.req.put({"op": "shutdown"})
        handle.proc.join(timeout=5)
        if handle.proc.is_alive():
            handle.proc.terminate()
            handle.proc.join(timeout=2)
