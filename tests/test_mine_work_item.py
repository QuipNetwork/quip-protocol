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

import logging
import multiprocessing as mp
import time

import pytest

from shared.base_miner import (
    _BridgeNodeInfo,
    _BridgePrevBlock,
)
from shared.work_context import requirements_from_context, resolve_ising
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.quantum_proof_of_work import derive_nonce
from substrate.submitter import encode_quantum_proof
from shared.allowed_value_spec import AllowedValueSet
from substrate.types import (
    SubstrateDifficulty,
    SubstrateMiningContext,
)


_BIN_SPEC = AllowedValueSet((-1000, 1000))
_TER_SPEC = AllowedValueSet((-1000, 0, 1000))


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
        last_proof_block_hash=b"\xab" * 32,
        topology_hash=b"\xcd" * 32,
        nodes=list(cpu_miner.sampler.nodes),
        edges=[(int(u), int(v)) for u, v in cpu_miner.sampler.edges],
        difficulty=SubstrateDifficulty(
            min_solutions=1,
            max_energy_milli=0,         # any non-positive energy passes
            min_diversity_milli=0,
        ),
        miner_account_bytes=b"\x42" * 32,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
        block_hash=b"\x55" * 32,
        block_number=1,
    )


def test_bridge_prev_block_carries_substrate_fields(relaxed_context):
    bridge = _BridgePrevBlock.from_work_context(relaxed_context)
    # PoW: header.index is a placeholder (0) since the nonce no longer
    # depends on a block-number input; hash carries the last proof block hash.
    assert bridge.header.index == 0
    assert bridge.hash == relaxed_context.last_proof_block_hash


def test_bridge_node_info_exposes_hex_account(relaxed_context):
    bridge = _BridgeNodeInfo.from_work_context(relaxed_context)
    assert bridge.miner_id == "0x" + relaxed_context.miner_account_bytes.hex()
    assert bridge.miner_account_bytes == relaxed_context.miner_account_bytes


def test_bridge_prev_block_handles_mempool_context():
    """Phase 8b: mempool work uses order_id as the prev-block index and a
    zero-hash placeholder. Subclass hooks that log the index get a
    meaningful value; the hash is never fed back into the chain."""
    from shared.mempool_types import MempoolJobContext
    ctx = MempoolJobContext(
        order_id=42,
        nodes=(0, 1, 2),
        edges=((0, 1), (1, 2)),
        h_values=(0, 0, 0),
        j_values=(0, 0),
    )
    bridge = _BridgePrevBlock.from_work_context(ctx)
    assert bridge.header.index == 42
    assert bridge.hash == b"\x00" * 32


def test_bridge_node_info_handles_mempool_context():
    from shared.mempool_types import MempoolJobContext
    ctx = MempoolJobContext(
        order_id=7,
        nodes=(0, 1),
        edges=((0, 1),),
        h_values=(0, 0),
        j_values=(0,),
    )
    bridge = _BridgeNodeInfo.from_work_context(ctx)
    assert bridge.miner_id == "mempool-order-7"
    assert bridge.miner_account_bytes == b"\x00" * 32


def test_requirements_from_context_pow_path():
    """`requirements_from_context` maps PoW milli fields to float requirements."""
    ctx = SubstrateMiningContext(
        last_proof_block_hash=b"\xab" * 32,
        topology_hash=b"\xcd" * 32,
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        difficulty=SubstrateDifficulty(
            min_solutions=5,
            max_energy_milli=-2_500_000,
            min_diversity_milli=200,
        ),
        miner_account_bytes=b"\x42" * 32,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
        block_hash=b"\x66" * 32,
        block_number=2,
    )
    req = requirements_from_context(ctx)
    assert req.difficulty_energy == -2500.0
    assert req.min_diversity == 0.2
    assert req.min_solutions == 5
    # Decay is disabled in substrate mode.
    assert req.timeout_to_difficulty_adjustment_decay >= 2**30
    # AllowedValueSpec instances are carried through for the PoW path.
    assert req.allowed_h_values is not None
    assert req.allowed_j_values is not None


def test_requirements_from_context_mempool_unset_floors():
    """Mempool quality floors are Option<T>; unset maps to:
      - difficulty_energy = +inf (no upper bound)
      - min_diversity    = 0.0
      - min_solutions    = 1 (chain requires ≥1 valid submission)"""
    from shared.mempool_types import MempoolJobContext
    ctx = MempoolJobContext(
        order_id=1,
        nodes=(0, 1),
        edges=((0, 1),),
        h_values=(0, 0),
        j_values=(0,),
    )
    req = requirements_from_context(ctx)
    assert req.difficulty_energy == float("inf")
    assert req.min_diversity == 0.0
    assert req.min_solutions == 1


def test_requirements_from_context_mempool_with_floors():
    from shared.mempool_types import MempoolJobContext
    ctx = MempoolJobContext(
        order_id=2,
        nodes=(0, 1),
        edges=((0, 1),),
        h_values=(0, 0),
        j_values=(0,),
        min_energy_milli=-1500,
        min_diversity_milli=300,
        min_solutions=4,
    )
    req = requirements_from_context(ctx)
    assert req.difficulty_energy == -1.5
    assert req.min_diversity == 0.3
    assert req.min_solutions == 4


def test_resolve_ising_mempool_returns_stored_h_j():
    """Mempool's resolve_ising surfaces the stored (h, J) dicts directly;
    nonce is 0 (telemetry placeholder). nodes/edges params are unused on
    this path since the chain doesn't re-derive the model."""
    from shared.mempool_types import MempoolJobContext
    ctx = MempoolJobContext(
        order_id=99,
        nodes=(10, 20, 30),
        edges=((10, 20), (20, 30)),
        h_values=(1000, -500, 250),     # millivalues
        j_values=(-100, 750),
    )
    h, J, nonce = resolve_ising(
        ctx, salt=b"\x00" * 32, nodes=ctx.nodes, edges=ctx.edges,
    )
    assert h == {10: 1.0, 20: -0.5, 30: 0.25}
    assert J == {(10, 20): -0.1, (20, 30): 0.75}
    assert nonce == 0


def test_make_feeder_pow_returns_random_feeder(relaxed_context):
    """``SubstrateMiningContext.make_feeder`` builds a RandomIsingFeeder
    seeded with the snapshot's round identity. Each pop derives a fresh
    salt + nonce — the loop relies on this in lieu of the old inline
    ``fresh_salt()`` / ``resolve_ising()`` calls."""
    from shared.ising_feeder import RandomIsingFeeder
    from shared.ising_model import IsingModel

    feeder = relaxed_context.make_feeder(
        relaxed_context.nodes, relaxed_context.edges, buffer_size=2,
    )
    try:
        assert isinstance(feeder, RandomIsingFeeder)
        model = feeder.pop_blocking()
        assert isinstance(model, IsingModel)
        # PoW path: each iteration must derive a fresh nonce from a
        # fresh salt; the next pop must produce a different one.
        model_b = feeder.pop_blocking()
        assert model.salt != model_b.salt
        assert model.nonce != model_b.nonce
    finally:
        feeder.stop()


def test_make_feeder_mempool_returns_fixed_feeder():
    """``MempoolJobContext.make_feeder`` builds a FixedIsingFeeder that
    cycles the order's stored (h, J) — placeholder nonce / salt are
    zero bytes (chain doesn't re-derive)."""
    from shared.ising_feeder import FixedIsingFeeder
    from shared.mempool_types import MempoolJobContext

    ctx = MempoolJobContext(
        order_id=99,
        nodes=(10, 20, 30),
        edges=((10, 20), (20, 30)),
        h_values=(1000, -500, 250),
        j_values=(-100, 750),
    )
    feeder = ctx.make_feeder(ctx.nodes, ctx.edges)
    try:
        assert isinstance(feeder, FixedIsingFeeder)
        m = feeder.pop_blocking()
        # h, J decoded from milli; chain-side floats.
        assert m.h == {10: 1.0, 20: -0.5, 30: 0.25}
        assert m.J == {(10, 20): -0.1, (20, 30): 0.75}
        # Same model on repeat — cycle of length 1.
        m2 = feeder.pop_blocking()
        assert m2 is m
        assert m.nonce == b"\x00" * 32
        assert m.salt == b"\x00" * 32
    finally:
        feeder.stop()


def test_resolve_ising_pow_uses_derive_nonce(relaxed_context, cpu_miner):
    """PoW's resolve_ising must produce the same (h, J, nonce) as the legacy
    direct invocation of derive_nonce + generate_ising_model_from_nonce —
    when called with the sampler's node/edge iteration order."""
    from shared.quantum_proof_of_work import (
        derive_nonce,
        generate_ising_model_from_nonce,
    )
    salt = b"\x42" * 32
    sampler_nodes = list(cpu_miner.sampler.nodes)
    sampler_edges = list(cpu_miner.sampler.edges)
    legacy_nonce = derive_nonce(
        relaxed_context.last_proof_block_hash,
        relaxed_context.miner_account_bytes,
        salt,
    )
    legacy_h, legacy_J = generate_ising_model_from_nonce(
        legacy_nonce,
        sampler_nodes,
        sampler_edges,
        allowed_h=relaxed_context.allowed_h_values,
        allowed_j=relaxed_context.allowed_j_values,
    )
    h, J, nonce = resolve_ising(
        relaxed_context, salt=salt, nodes=sampler_nodes, edges=sampler_edges,
    )
    assert nonce == legacy_nonce
    assert h == legacy_h
    assert J == legacy_J


def test_mine_work_item_returns_result(cpu_miner, relaxed_context):
    stop = mp.Event()
    result = cpu_miner.mine_work_item(relaxed_context, stop)
    assert isinstance(result, MiningResult)
    assert result.num_valid >= relaxed_context.difficulty.min_solutions
    assert result.energy <= relaxed_context.difficulty.max_energy
    assert len(result.salt) == 32
    # node_list and edge_list should match the topology the miner ran with
    assert sorted(result.node_list) == sorted(relaxed_context.nodes)


def test_mine_work_item_drives_streaming_batch_when_available(
    cpu_miner, relaxed_context, monkeypatch,
):
    """When a backend implements ``_sample_batch`` (the streaming pipeline
    used by QPU async dispatch and GPU multi-problem dispatch), the mine
    loop MUST drive it and must NOT fall back to the single-shot
    ``_sample``. Regression for the production bypass that quietly ran
    every QPU/GPU dispatch through the slow synchronous path.
    """
    import types

    calls = {"batch": 0, "sample": 0}
    real_sample = cpu_miner._sample

    def fake_batch(self, prev_hash, miner_id, cur_index, nodes, edges,
                   *, num_reads, num_sweeps, **kw):
        # Mirror the QPU/GPU pattern: pull from the loop-owned feeder
        # internally and return (nonce, salt, sampleset). The loop must
        # NOT also pop the feeder itself.
        calls["batch"] += 1
        model = self._feeder.pop_blocking()
        ss = real_sample(
            model.h, model.J, num_reads=num_reads, num_sweeps=num_sweeps,
            nonce_seed=model.nonce, **kw,
        )
        return [(model.nonce, model.salt, ss)]

    def forbidden_sample(self, *a, **k):
        calls["sample"] += 1
        raise AssertionError(
            "_sample called although _sample_batch is available — "
            "the streaming bypass regressed"
        )

    monkeypatch.setattr(cpu_miner, "_sample_batch",
                        types.MethodType(fake_batch, cpu_miner))
    monkeypatch.setattr(cpu_miner, "_sample",
                        types.MethodType(forbidden_sample, cpu_miner))

    result = cpu_miner.mine_work_item(relaxed_context, mp.Event())
    assert isinstance(result, MiningResult)
    assert calls["batch"] >= 1, "streaming _sample_batch was never called"
    assert calls["sample"] == 0, "loop used single-shot _sample, not streaming"


def test_mine_work_item_handles_mempool_context(cpu_miner):
    """Phase 8b: mine_work_item must accept a MempoolJobContext and mine
    against the directly-provided (h, J) rather than deriving via nonce.

    Uses an all-zero Ising over the CPU miner's actual topology — every
    spin assignment has energy 0, so any sample with `min_solutions=1`
    and no other floors passes the requirements check.
    """
    from shared.mempool_types import MempoolJobContext

    nodes = tuple(int(n) for n in cpu_miner.sampler.nodes)
    edges = tuple((int(u), int(v)) for u, v in cpu_miner.sampler.edges)
    ctx = MempoolJobContext(
        order_id=12345,
        nodes=nodes,
        edges=edges,
        h_values=tuple(0 for _ in nodes),     # all-zero field
        j_values=tuple(0 for _ in edges),     # all-zero coupling
        min_energy_milli=None,
        min_diversity_milli=None,
        min_solutions=1,
    )
    stop = mp.Event()
    result = cpu_miner.mine_work_item(ctx, stop)
    assert isinstance(result, MiningResult)
    # mempool's resolve_ising returns 0 as the placeholder nonce, which
    # evaluate_sampleset encodes as the 32-byte zero buffer.
    assert result.nonce == b"\x00" * 32
    # all-zero ising → any sample has energy 0
    assert result.energy <= 0.0
    assert result.num_valid >= 1
    assert sorted(result.node_list) == sorted(nodes)


def test_mine_work_item_result_encodes_to_quantum_proof(cpu_miner, relaxed_context):
    stop = mp.Event()
    result = cpu_miner.mine_work_item(relaxed_context, stop)
    proof = encode_quantum_proof(result, relaxed_context)
    # Shape matches the post-MR-!20 pallet QuantumProof. Nonce is a U256
    # int; salt is a `[u8; 32]` array; solutions are BoundedVec<BoundedVec<u8>>
    # of bit-packed spins.
    assert proof["topology_hash"] == "0x" + relaxed_context.topology_hash.hex()
    assert proof["nonce"] == int.from_bytes(result.nonce, "big")
    assert bytes(proof["salt"]) == result.salt

    # nodes / edges / h_values are NOT in the proof anymore — the chain
    # looks them up from the registered topology.
    assert "nodes" not in proof
    assert "edges" not in proof
    assert "h_values" not in proof

    solutions, = proof["solutions"]
    assert len(solutions) >= relaxed_context.difficulty.min_solutions
    expected_byte_len = (len(relaxed_context.nodes) + 7) // 8  # 1 bit per spin
    for wrapped_sol in solutions:
        inner_sol, = wrapped_sol
        assert len(inner_sol) == expected_byte_len
        assert all(0 <= b <= 0xFF for b in inner_sol)


def test_mine_work_item_observes_stop_event(cpu_miner, relaxed_context):
    """Set stop_event BEFORE calling mine_work_item and assert it returns
    None within one polling cycle. Mirrors how the controller cancels work
    on a new chain head."""
    impossibly_hard = SubstrateMiningContext(
        last_proof_block_hash=relaxed_context.last_proof_block_hash,
        topology_hash=relaxed_context.topology_hash,
        nodes=relaxed_context.nodes,
        edges=relaxed_context.edges,
        # A solution must have energy <= -10^15 which is unreachable for a
        # 1368-node graph with ±1 couplings (true GSE is ~-4100).
        difficulty=SubstrateDifficulty(
            min_solutions=5,
            max_energy_milli=-10**18,
            min_diversity_milli=1000,
        ),
        miner_account_bytes=relaxed_context.miner_account_bytes,
        allowed_h_values=relaxed_context.allowed_h_values,
        allowed_j_values=relaxed_context.allowed_j_values,
        allowed_spin_values=relaxed_context.allowed_spin_values,
        block_hash=relaxed_context.block_hash,
        block_number=relaxed_context.block_number,
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


def test_mine_work_item_post_evaluation_stop_check(
    cpu_miner, relaxed_context, monkeypatch,
):
    """Cancel-race regression (SUBMISSIONSTORM.md phase 4): if
    evaluate_sampleset returns a valid result but stop_event was set
    during evaluation, mine_work_item must return None — NOT the
    now-stale result. Without this check, a result produced after
    cancel still surfaces as fresh against the next dispatch.

    Simulated by monkeypatching `evaluate_sampleset` to also set
    `stop_event` before returning its result. Without the post-eval
    check, mine_work_item would return that result; with it, returns
    None.
    """
    stop = mp.Event()

    fake_result = MiningResult(
        miner_id="test",
        miner_type="CPU",
        nonce=(7).to_bytes(32, "big"),
        salt=b"\x33" * 32,
        timestamp=0,
        prev_timestamp=0,
        solutions=[[1 for _ in relaxed_context.nodes]],
        energy=-1.0,
        diversity=0.5,
        num_valid=1,
        mining_time=0,
        node_list=list(relaxed_context.nodes),
        edge_list=list(relaxed_context.edges),
        variable_order=list(relaxed_context.nodes),
    )

    def fake_scoring(*args, **kwargs):
        stop.set()  # Simulate cancel landing during scoring.
        return fake_result

    monkeypatch.setattr(cpu_miner, "evaluate_sampleset", fake_scoring)
    result = cpu_miner.mine_work_item(relaxed_context, stop)

    # Without the post-evaluation stop check this would return
    # `fake_result`; with the fix it returns None and the controller's
    # next dispatch is free to start fresh.
    assert result is None


def test_requirements_from_context_mempool_zero_energy_floor():
    """min_energy_milli=0 is a valid (very relaxed) floor — must NOT map to
    +inf the way None does. Zero divides to 0.0 which allows any non-positive
    energy sample."""
    from shared.mempool_types import MempoolJobContext
    ctx = MempoolJobContext(
        order_id=3,
        nodes=(0, 1),
        edges=((0, 1),),
        h_values=(0, 0),
        j_values=(0,),
        min_energy_milli=0,
    )
    req = requirements_from_context(ctx)
    assert req.difficulty_energy == 0.0
    assert req.difficulty_energy != float("inf")


def test_mempool_job_context_from_job_order():
    """from_job_order must copy all IsingParams fields into the context."""
    from shared.mempool_types import (
        IsingParams, JobMode, JobOrder, MempoolJobContext,
        OrderStatus, OrderTiming, ResultDelivery, RewardResolution,
    )
    ising = IsingParams(
        nodes=(10, 20),
        edges=((10, 20),),
        h_values=(500, -500),
        j_values=(250,),
        min_energy_milli=-1000,
        min_diversity_milli=100,
        min_solutions=2,
    )
    order = JobOrder(
        spec_id=b"\x01" * 32,
        proposer=b"\x02" * 32,
        ising_params=ising,
        reward=1000,
        mode=JobMode.open(),
        resolution=RewardResolution.single_best(),
        timing=OrderTiming(deadline_blocks=10, block_wait=1),
        delivery=ResultDelivery.on_chain_only(),
        status=OrderStatus.OPENED,
        created_at=100,
        first_solution_at=None,
        solution_count=0,
    )
    ctx = MempoolJobContext.from_job_order(order_id=5, order=order)
    assert ctx.order_id == 5
    assert ctx.nodes == (10, 20)
    assert ctx.edges == ((10, 20),)
    assert ctx.h_values == (500, -500)
    assert ctx.j_values == (250,)
    assert ctx.min_energy_milli == -1000
    assert ctx.min_diversity_milli == 100
    assert ctx.min_solutions == 2


def test_mempool_job_context_rejects_mismatched_h_values():
    """MempoolJobContext.__post_init__ must raise if h_values length != nodes."""
    from shared.mempool_types import MempoolJobContext
    with pytest.raises(ValueError, match="h_values length"):
        MempoolJobContext(
            order_id=1,
            nodes=(0, 1, 2),
            edges=(),
            h_values=(0, 0),       # 2 != 3 nodes
            j_values=(),
        )


def test_mempool_job_context_rejects_mismatched_j_values():
    """MempoolJobContext.__post_init__ must raise if j_values length != edges."""
    from shared.mempool_types import MempoolJobContext
    with pytest.raises(ValueError, match="j_values length"):
        MempoolJobContext(
            order_id=1,
            nodes=(0, 1),
            edges=((0, 1),),
            h_values=(0, 0),
            j_values=(0, 0),       # 2 != 1 edge
        )


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
        dispatch_id = handle.mine_work_item(relaxed_context)
        assert dispatch_id == 1
        # Drain response with a generous timeout — CPU SA on Z(9,2) at
        # min_solutions=1 / max_energy=0 typically lands in <10s.
        deadline = time.time() + 100
        while time.time() < deadline:
            try:
                msg = handle.resp.get(timeout=5)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("op") == "mine_result":
                result = msg["result"]
                assert isinstance(result, MiningResult)
                assert result.num_valid >= 1
                assert result.energy <= 0
                assert msg.get("dispatch_id") == dispatch_id
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


def test_mine_work_item_nonce_matches_chain_derivation(cpu_miner, relaxed_context):
    """The produced nonce must equal `derive_nonce(last_proof_block_hash,
    miner_account_bytes, salt)` — byte-exact against the Rust pallet's
    validation derivation. A regression that accidentally uses the legacy
    `ising_nonce_from_block` (string identity) would silently produce
    proofs the chain rejects."""
    stop = mp.Event()
    result = cpu_miner.mine_work_item(relaxed_context, stop)
    expected = derive_nonce(
        relaxed_context.last_proof_block_hash,
        relaxed_context.miner_account_bytes,
        result.salt,
    )
    assert result.nonce == expected


@pytest.mark.timeout(120)
def test_miner_handle_emits_work_item_done_sentinel_on_cancel(relaxed_context):
    """When `mine_work_item` returns None (cancelled or no valid result),
    the worker must put a `{"op": "work_item_done", "id": ...}` sentinel on
    resp_q so the controller's cancel→clear→dispatch cycle has an
    observable acknowledgment. Without this the controller would hang on
    `resp_q.get()` after every cancel."""
    spec = {"id": "test-cpu-cancel", "kind": "cpu", "args": {}}
    handle = MinerHandle(spec=spec)
    try:
        impossibly_hard = SubstrateMiningContext(
            last_proof_block_hash=relaxed_context.last_proof_block_hash,
            topology_hash=relaxed_context.topology_hash,
            nodes=relaxed_context.nodes,
            edges=relaxed_context.edges,
            difficulty=SubstrateDifficulty(
                min_solutions=5,
                max_energy_milli=-10**18,
                min_diversity_milli=1000,
            ),
            miner_account_bytes=relaxed_context.miner_account_bytes,
            allowed_h_values=relaxed_context.allowed_h_values,
            allowed_j_values=relaxed_context.allowed_j_values,
            allowed_spin_values=relaxed_context.allowed_spin_values,
            block_hash=relaxed_context.block_hash,
            block_number=relaxed_context.block_number,
        )
        handle.mine_work_item(impossibly_hard)
        # Cancel after a brief moment so the worker enters the loop and
        # then observes the stop event.
        time.sleep(0.5)
        handle.cancel()
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                msg = handle.resp.get(timeout=5)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("op") == "work_item_done":
                assert msg.get("id") == "test-cpu-cancel"
                assert "dispatch_id" in msg
                return
            if isinstance(msg, dict) and msg.get("op") == "error":
                pytest.fail(f"worker errored: {msg.get('message')}")
            if isinstance(msg, dict) and msg.get("op") == "mine_result":
                pytest.fail("expected work_item_done sentinel, got a result")
        pytest.fail("no work_item_done sentinel before deadline")
    finally:
        handle.req.put({"op": "shutdown"})
        handle.proc.join(timeout=5)
        if handle.proc.is_alive():
            handle.proc.terminate()
            handle.proc.join(timeout=2)


def test_miner_handle_error_sentinel_on_missing_context():
    """`op=mine_work_item` with no `context` key must produce an
    `{"op": "error", ...}` sentinel keyed by miner id. Without this the
    controller cannot distinguish a stuck worker from a malformed request."""
    spec = {"id": "test-cpu-bad", "kind": "cpu", "args": {}}
    handle = MinerHandle(spec=spec)
    try:
        handle.stop_event.clear()
        handle.req.put({"op": "mine_work_item"})  # no "context"
        msg = handle.resp.get(timeout=20)
        assert isinstance(msg, dict)
        assert msg["op"] == "error"
        assert "context" in msg["message"].lower()
        assert msg["id"] == "test-cpu-bad"
    finally:
        handle.req.put({"op": "shutdown"})
        handle.proc.join(timeout=5)
        if handle.proc.is_alive():
            handle.proc.terminate()
            handle.proc.join(timeout=2)


# ----------------------------------------------------------------------
# Anticipatory-submission preview channel (Task 6a)
# ----------------------------------------------------------------------


def test_mine_work_item_emits_preview_on_floor_improvement(
    cpu_miner, relaxed_context, monkeypatch,
):
    """A ``preview_cb`` must fire when a strictly-better-floor candidate is
    stashed, with a payload carrying the fields the controller needs to
    encode+submit later. It must NOT fire again without a strict floor
    improvement (throttle).

    Drives ``mine_work_item`` with a stubbed ``evaluate_sampleset`` that
    returns successively-improving then non-improving floors so the
    throttle behaviour is deterministic:
      - iter 1: floor -1.0 → first stash → preview fires.
      - iter 2: floor -2.0 (better) → preview fires again.
      - iter 3: floor -1.5 (worse than best -2.0) → still stored in the
        bounded heap but best-by-floor unchanged → preview does NOT fire.
    Stops after iter 3.
    """
    previews: list = []

    def capture(payload):
        previews.append(payload)

    floors = [-1.0, -2.0, -1.5]
    energies = [-1.0, -2.0, -1.5]
    iter_idx = [0]

    def fake_evaluate(sampleset, requirements, nodes, edges,
                      nonce, salt, prev_timestamp, start_time,
                      strict_energy=True, live_threshold_energy=None):
        i = iter_idx[0]
        iter_idx[0] += 1
        if i >= len(floors):
            stop.set()
            return None
        n = len(nodes)
        return MiningResult(
            miner_id="test",
            miner_type="CPU",
            nonce=(i + 1).to_bytes(32, "big"),
            salt=bytes([i + 1]) * 32,
            timestamp=0,
            prev_timestamp=0,
            solutions=[[-1] * n],
            energy=energies[i],
            diversity=1.0,
            num_valid=3,
            mining_time=0,
            node_list=list(nodes),
            edge_list=list(edges),
            submit_floor_energy=floors[i],
        )

    # Live threshold strict enough that the submit gate never passes for
    # any of our fake floors (-1.0/-2.0/-1.5 → milli -1000/-2000/-1500):
    # submit needs floor_milli < live, and -2000 < -3000 is False. This
    # keeps the loop running through all three iters so we can observe the
    # throttle. The precheck ``near_live`` gate must NOT short-circuit the
    # lenient evaluate, so we widen the precheck margin to admit every iter
    # regardless of the real sampler's energy.
    live_var = mp.Value("q", -3000)
    monkeypatch.setattr(cpu_miner, "RATCHET_PRECHECK_MARGIN_MILLI", 10**18)
    stop = mp.Event()
    cpu_miner._live_max_energy_milli = live_var
    monkeypatch.setattr(cpu_miner, "evaluate_sampleset", fake_evaluate)
    try:
        cpu_miner.mine_work_item(relaxed_context, stop, preview_cb=capture)
    finally:
        del cpu_miner._live_max_energy_milli

    assert len(previews) == 2, (
        f"expected exactly 2 previews (iter1 -1.0, iter2 -2.0; iter3 -1.5 "
        f"is no improvement), got {len(previews)}: "
        f"{[p['submit_floor_energy'] for p in previews]}"
    )
    # First preview: the iter-1 candidate (floor -1.0).
    p0 = previews[0]
    for field in (
        "dispatch_id", "miner_type", "nonce", "salt", "solutions",
        "submit_floor_energy", "energy", "num_valid", "diversity",
    ):
        assert field in p0, f"preview payload missing required field {field!r}"
    assert p0["submit_floor_energy"] == -1.0
    assert p0["nonce"] == (1).to_bytes(32, "big")
    # The preview carries the real source backend so the controller's
    # anticipatory path records accurate per-backend attribution.
    assert p0["miner_type"] == "CPU"
    # Second preview: the improved iter-2 candidate (floor -2.0).
    assert previews[1]["submit_floor_energy"] == -2.0
    assert previews[1]["nonce"] == (2).to_bytes(32, "big")


def test_mine_work_item_preview_cb_default_none_is_noop(
    cpu_miner, relaxed_context,
):
    """Backward-compat: callers passing no ``preview_cb`` (the default)
    mine exactly as before — no error, a normal result comes back."""
    stop = mp.Event()
    result = cpu_miner.mine_work_item(relaxed_context, stop)
    assert isinstance(result, MiningResult)


def test_mine_work_item_preview_cb_failure_does_not_break_mining(
    cpu_miner, relaxed_context,
):
    """A throwing ``preview_cb`` must never break the mining loop — the
    dispatch still completes and returns a valid result."""
    def boom(_payload):
        raise RuntimeError("preview channel exploded")

    stop = mp.Event()
    result = cpu_miner.mine_work_item(relaxed_context, stop, preview_cb=boom)
    assert isinstance(result, MiningResult)


# ----------------------------------------------------------------------
# submit_floor_energy parity — chain rejects on the WORST selected
# solution, not the best. evaluate_sampleset must surface the worst-case
# so the substrate submit gate doesn't ship a proof the chain will reject.
# ----------------------------------------------------------------------


def test_evaluate_sampleset_submit_floor_is_worst_recomputed_energy():
    """``submit_floor_energy`` must be the MAX (least negative) of the
    independently-recomputed Ising energies for the diverse-selected
    solutions — not the sampler-reported value and not the BEST.

    Motivation: the chain re-derives each submitted solution's energy
    via ``energy_of_solution`` (i64 milli) and filters with strict
    ``< max_energy_milli`` before checking
    ``valid_solution_count >= min_solutions``. Sampler-reported energies
    drift from chain-computed ones at the milli boundary, so submitting
    on the headline best can ship a proof where mid-pack solutions fail
    the chain's per-solution check and trigger
    ``Error::InsufficientSolutions`` — observed in production at block
    126 of the local deployment when the ratchet's threshold sat 3 milli
    above the actual sampler ceiling.
    """
    import dimod
    from shared.quantum_proof_of_work import evaluate_sampleset
    from shared.miner_types import BlockRequirements

    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3)]
    # 5 distinct samples — all-ones plus four single-flips.
    samples = [
        [1, 1, 1, 1],
        [1, 1, 1, -1],
        [1, 1, -1, 1],
        [1, -1, 1, 1],
        [-1, 1, 1, 1],
    ]
    # Sampler reports a misleading flat -100 for every sample. If the
    # floor were sourced from the sampler it would be -100 and the test
    # would fail — pinning the recompute-is-authoritative invariant.
    sampleset = dimod.SampleSet.from_samples(
        samples, vartype=dimod.SPIN, energy=[-100.0] * 5,
    )

    h = {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}
    J = {(0, 1): -1.0, (1, 2): -1.0, (2, 3): -1.0}
    # True Ising energies (manually verified):
    #   [1,1,1,1]  -> -7  (all ferromagnetic, all couplers satisfied)
    #   [1,1,1,-1] -> -3
    #   [1,1,-1,1] -> -1
    #   [1,-1,1,1] -> -1
    #   [-1,1,1,1] -> -3
    # worst = -1; best = -7.

    requirements = BlockRequirements(
        difficulty_energy=-2.0,
        min_diversity=0.0,
        min_solutions=5,
        timeout_to_difficulty_adjustment_decay=10_000,
    )

    result = evaluate_sampleset(
        sampleset, requirements, nodes, edges,
        nonce=b"\x00" * 32, salt=b"\x00" * 32,
        prev_timestamp=0, start_time=time.time(),
        miner_id="test", miner_type="CPU",
        h=h, J=J,
        # Lenient — accept all 5, mirroring the substrate ratchet path
        # where evaluate_sampleset is called with strict_energy=False.
        strict_energy=False,
    )

    assert result is not None
    assert result.submit_floor_energy == pytest.approx(-1.0), (
        "submit_floor_energy must be the WORST recomputed energy "
        "across the selected 5 — that's what the chain effectively "
        f"gates against. got {result.submit_floor_energy}"
    )


def test_attempt_log_records_num_valid_and_diversity_on_stored_iteration(
    cpu_miner, relaxed_context,
):
    """An iteration that post-processes (``post_processed=true``) but does
    NOT submit must still record ``num_valid`` and ``diversity_milli`` on
    the per-iteration attempt log — diagnostic loss otherwise (we can't
    see how close stored / rejected attempts came to the chain floor).

    Drives ``mine_work_item`` through one substrate-ratchet iteration with
    a live decay threshold so strict the submit gate cannot pass, then
    asserts the captured ``AttemptLogger.record(...)`` kwargs carry
    integer ``num_valid`` and ``diversity_milli`` for the stored case
    (``result_kind="stored"``). Regression for the case where the submit
    gate rebinds the local ``result`` to ``None`` before log kwargs are
    assembled.
    """
    from unittest.mock import MagicMock

    # Capture every record() call made by the loop.
    captured = []
    recording_logger = MagicMock()
    recording_logger.record.side_effect = lambda **kw: (
        captured.append(kw), stop.set(),
    )

    # Use a live threshold that matches the snapshot (max_energy_milli=0
    # in relaxed_context). With min_solutions=1 the ratchet and submit
    # gates compare the same energy against the same threshold, so an
    # iter that post-processes will also submit — and `num_valid` +
    # `diversity_milli` must still be populated on the submitted record.
    # (The previous setup forced submit to fail by setting live well
    # below any reachable energy, but that exploited an old bug where
    # the ratchet gate read the dispatch snapshot while the submit gate
    # read live; both gates now read live, so that contradiction is
    # no longer expressible with min_solutions=1.)
    live_var = mp.Value('q', 0)

    # Install both onto the miner. ``mine_work_item`` reads
    # ``_attempt_logger`` and ``_live_max_energy_milli`` directly off
    # ``self``; module-scope fixture means we must clean up after.
    cpu_miner._attempt_logger = recording_logger
    cpu_miner._live_max_energy_milli = live_var

    stop = mp.Event()
    try:
        cpu_miner.mine_work_item(relaxed_context, stop)
    finally:
        del cpu_miner._attempt_logger
        del cpu_miner._live_max_energy_milli

    assert captured, "expected at least one AttemptLogger.record call"

    # Find an iteration that post-processed; assert num_valid and
    # diversity_milli are populated regardless of whether the result
    # was stored or submitted. The regression we're guarding against
    # is the submit gate rebinding the local ``result`` to None
    # before log kwargs are assembled — the values are captured into
    # ``post_num_valid`` / ``post_diversity_milli`` earlier in the
    # iteration so the rebind can't strip them.
    post_processed = [k for k in captured if k.get("post_processed")]
    assert post_processed, (
        "expected at least one post_processed=True record; got "
        f"{[k.get('result_kind') for k in captured]}"
    )
    rec = post_processed[0]
    assert rec["result_kind"] in ("stored", "submitted"), (
        f"expected result_kind in (stored, submitted), got {rec['result_kind']}"
    )
    assert isinstance(rec["num_valid"], int) and rec["num_valid"] >= 1, (
        f"num_valid must be a populated int on post-processed "
        f"iteration, got {rec['num_valid']!r}"
    )
    assert isinstance(rec["diversity_milli"], int), (
        f"diversity_milli must be a populated int on post-processed "
        f"iteration, got {rec['diversity_milli']!r}"
    )


def test_precheck_skips_evaluate_for_no_hope_iter(
    cpu_miner, relaxed_context, monkeypatch,
):
    """Pre-check must skip the expensive lenient ``evaluate_sampleset`` when
    the iter's best energy is FAR above (worse than) the live threshold plus
    ``RATCHET_PRECHECK_MARGIN_MILLI``.

    Arranges a fake sampler that returns a sampleset whose best energy is
    +100.0 (100_000 milli) — far above the live threshold of 0 milli plus the
    2000-milli margin. The spy on ``evaluate_sampleset`` must NOT be called,
    and the attempt log must record ``result_kind="rejected"`` with
    ``post_processed=False``.
    """
    import dimod
    from unittest.mock import MagicMock

    # Sampleset with a single sample at energy +100 — no-hope (far above live).
    BAD_ENERGY = 100.0
    bad_ss = dimod.SampleSet.from_samples(
        [{n: 1 for n in relaxed_context.nodes}],
        vartype=dimod.SPIN,
        energy=[BAD_ENERGY],
    )

    # Spy on evaluate_sampleset — must NOT be called for a no-hope iter.
    spy_calls = []

    def spy_evaluate(sampleset, *args, **kwargs):
        spy_calls.append(sampleset)
        return None

    # Capture the first attempt-log record, then stop.
    captured = []
    recording_logger = MagicMock()

    def _capture(**kw):
        captured.append(kw)
        stop.set()

    recording_logger.record.side_effect = _capture

    def fake_sample(*args, **kwargs):
        # Return the bad-energy sampleset regardless of (h, J).
        return bad_ss

    stop = mp.Event()
    # Live threshold = 0 milli. With RATCHET_PRECHECK_MARGIN_MILLI=2000,
    # iter_best_milli = 100_000 >> 0 + 2000 so near_live is False.
    live_var = mp.Value('q', 0)

    monkeypatch.setattr(cpu_miner, "_sample", fake_sample)
    monkeypatch.setattr(cpu_miner, "evaluate_sampleset", spy_evaluate)
    cpu_miner._attempt_logger = recording_logger
    cpu_miner._live_max_energy_milli = live_var
    try:
        cpu_miner.mine_work_item(relaxed_context, stop)
    finally:
        del cpu_miner._attempt_logger
        del cpu_miner._live_max_energy_milli

    assert captured, "expected at least one AttemptLogger.record call"
    rec = captured[0]
    assert not spy_calls, (
        "evaluate_sampleset must NOT be called for a no-hope iter "
        f"(best_energy={BAD_ENERGY}, live_threshold_milli=0, margin=2000)"
    )
    assert rec["result_kind"] == "rejected", (
        f"no-hope iter must be logged as 'rejected', got {rec['result_kind']!r}"
    )
    assert rec["post_processed"] is False, (
        f"no-hope iter must have post_processed=False, got {rec['post_processed']!r}"
    )


def test_precheck_evaluates_iter_near_live_threshold(
    cpu_miner, relaxed_context, monkeypatch,
):
    """Pre-check must call ``evaluate_sampleset`` when the iter's best energy
    is at or below the live threshold — i.e. clearly within the
    ``RATCHET_PRECHECK_MARGIN_MILLI`` window.

    Arranges a fake sampler that returns a sampleset whose best energy is
    -1.0 (-1000 milli) — below the live threshold of 0 milli. The spy on
    ``evaluate_sampleset`` MUST be called for that iter.
    """
    import dimod
    from unittest.mock import MagicMock

    # Sampleset with best energy -1.0 — clearly within margin of threshold 0.
    GOOD_ENERGY = -1.0
    good_ss = dimod.SampleSet.from_samples(
        [{n: -1 for n in relaxed_context.nodes}],
        vartype=dimod.SPIN,
        energy=[GOOD_ENERGY],
    )

    spy_calls = []

    def spy_evaluate(sampleset, *args, **kwargs):
        spy_calls.append(sampleset)
        return None  # Return None so the loop continues; stop via logger.

    captured = []
    recording_logger = MagicMock()

    def _capture(**kw):
        captured.append(kw)
        stop.set()

    recording_logger.record.side_effect = _capture

    def fake_sample(*args, **kwargs):
        return good_ss

    stop = mp.Event()
    # Live threshold = 0 milli; GOOD_ENERGY milli = -1000 << 0 + 2000.
    live_var = mp.Value('q', 0)

    monkeypatch.setattr(cpu_miner, "_sample", fake_sample)
    monkeypatch.setattr(cpu_miner, "evaluate_sampleset", spy_evaluate)
    cpu_miner._attempt_logger = recording_logger
    cpu_miner._live_max_energy_milli = live_var
    try:
        cpu_miner.mine_work_item(relaxed_context, stop)
    finally:
        del cpu_miner._attempt_logger
        del cpu_miner._live_max_energy_milli

    assert spy_calls, (
        "evaluate_sampleset MUST be called when iter best energy "
        f"({GOOD_ENERGY}) is within margin of live threshold (0 milli)"
    )
    assert captured, "expected at least one AttemptLogger.record call"
    rec = captured[0]
    assert rec.get("post_processed") is True, (
        f"near-live iter must have post_processed=True, got {rec.get('post_processed')!r}"
    )


def test_attempt_log_records_qpu_access_time_us_when_sample_records_qpu_timing(
    cpu_miner, relaxed_context,
):
    """The mining loop must forward D-Wave's per-iteration QPU access
    time into the attempt log so the dashboard can graph real QPU time.

    Simulates a QPU iteration by wrapping ``_sample`` so it appends a
    fake microsecond value to ``timing_stats['qpu_access_time']`` —
    the same list the production ``_record_qpu_timing`` writes to.
    The base mining loop snapshots that list before and after each
    sample and pulls out the new entry; the captured record must carry
    it as ``qpu_access_time_us``.
    """
    from unittest.mock import MagicMock

    INJECTED_QPU_TIME_US = 12_345

    captured = []
    recording_logger = MagicMock()
    recording_logger.record.side_effect = lambda **kw: (
        captured.append(kw), stop.set(),
    )

    original_sample = cpu_miner._sample

    def _sample_with_qpu_timing(*args, **kwargs):
        sampleset = original_sample(*args, **kwargs)
        cpu_miner.timing_stats['qpu_access_time'].append(INJECTED_QPU_TIME_US)
        return sampleset

    cpu_miner._sample = _sample_with_qpu_timing
    cpu_miner._attempt_logger = recording_logger
    stop = mp.Event()
    try:
        cpu_miner.mine_work_item(relaxed_context, stop)
    finally:
        cpu_miner._sample = original_sample
        del cpu_miner._attempt_logger
        # Don't leak the injected timing values into the module-scoped
        # miner fixture — other tests assert on timing_stats contents.
        cpu_miner.timing_stats['qpu_access_time'].clear()

    assert captured, "expected at least one AttemptLogger.record call"
    rec = captured[0]
    assert rec["qpu_access_time_us"] == INJECTED_QPU_TIME_US, (
        "mining loop must forward the per-iteration qpu_access_time "
        "appended inside _sample() to the attempt-log record"
    )


def test_attempt_log_qpu_access_time_us_is_none_for_non_qpu_backends(
    cpu_miner, relaxed_context,
):
    """CPU/CUDA/etc. backends do not write to
    ``timing_stats['qpu_access_time']``. The attempt record must still
    carry the key — value ``None`` — so downstream parsers can rely on
    a uniform schema rather than presence checks."""
    from unittest.mock import MagicMock

    captured = []
    recording_logger = MagicMock()
    recording_logger.record.side_effect = lambda **kw: (
        captured.append(kw), stop.set(),
    )

    cpu_miner._attempt_logger = recording_logger
    stop = mp.Event()
    try:
        cpu_miner.mine_work_item(relaxed_context, stop)
    finally:
        del cpu_miner._attempt_logger

    assert captured, "expected at least one AttemptLogger.record call"
    rec = captured[0]
    assert "qpu_access_time_us" in rec
    assert rec["qpu_access_time_us"] is None


def test_stored_solution_iter_matches_attempt_iter(
    cpu_miner, relaxed_context, tmp_path,
):
    """The ``iter`` field in stored-solution files must equal the ``iter``
    field in the attempt-log row for the same iteration.

    Before the fix, attempt-log rows used ``progress + 1`` (1-based) while
    ``SolutionStore.record`` received ``progress`` (0-based), so the two
    records for the SAME iteration carried different ``iter`` values and
    cross-referencing via ``query_by_dispatch`` / ``query_stored_solutions``
    was broken.

    Drives ``mine_work_item`` through one substrate-ratchet iteration that
    produces a stored or submitted candidate, then reads back both artefacts
    and asserts their ``iter`` values are equal.
    """
    from shared.mining_attempt_log import (
        AttemptLogger,
        SolutionStore,
        query_by_dispatch,
        query_stored_solutions,
    )

    DISPATCH_ID = 9900

    real_logger = AttemptLogger(
        cpu_miner.miner_id, log_dir=tmp_path, miner_type=cpu_miner.miner_type,
    )
    real_store = SolutionStore(cpu_miner.miner_id, log_dir=tmp_path)

    # Stop after the first stored/submitted record so we don't run forever.
    stop = mp.Event()
    original_record = real_logger.record

    def _record_and_maybe_stop(**kw):
        original_record(**kw)
        if kw.get("result_kind") in ("stored", "submitted"):
            stop.set()

    real_logger.record = _record_and_maybe_stop  # type: ignore[method-assign]

    live_var = mp.Value('q', 0)
    cpu_miner._attempt_logger = real_logger
    cpu_miner._solution_store = real_store
    cpu_miner._current_dispatch_id = DISPATCH_ID
    cpu_miner._live_max_energy_milli = live_var
    try:
        cpu_miner.mine_work_item(relaxed_context, stop)
    finally:
        real_logger.record = original_record  # type: ignore[method-assign]
        del cpu_miner._attempt_logger
        del cpu_miner._solution_store
        del cpu_miner._current_dispatch_id
        del cpu_miner._live_max_energy_milli

    # Read back attempt rows and find the stored/submitted one.
    attempt_rows = query_by_dispatch(
        cpu_miner.miner_id, DISPATCH_ID, log_dir=tmp_path,
    )
    stored_rows = [
        r for r in attempt_rows
        if r.get("result_kind") in ("stored", "submitted")
    ]
    assert stored_rows, (
        "expected at least one stored/submitted attempt row; "
        f"got result_kinds={[r.get('result_kind') for r in attempt_rows]}"
    )
    attempt_iter = stored_rows[0]["iter"]
    assert attempt_iter >= 1, f"expected 1-based iter, got {attempt_iter}"

    # Read back the stored solution record.
    sol_records = query_stored_solutions(
        DISPATCH_ID, log_dir=tmp_path, miner_id=cpu_miner.miner_id,
    )
    assert sol_records, (
        "expected at least one stored-solution file; "
        "SolutionStore.record was not called for a stored/submitted iter"
    )
    solution_iter = sol_records[0]["iter"]

    assert attempt_iter == solution_iter, (
        f"attempt-log iter ({attempt_iter}) != stored-solution iter "
        f"({solution_iter}); cross-reference is broken"
    )


def test_mined_log_nonce_comes_from_result_not_loop_variable(
    cpu_miner, relaxed_context, caplog,
):
    """The ``mined!`` INFO log line must display the *submitted* result's
    nonce (``result.nonce``), not the loop's last-iterated nonce variable.

    In the substrate-ratchet path the submit gate walks ``top_k`` and
    returns the best-energy stashed candidate, which may have been produced
    by an *earlier* iteration with a different nonce than the current one.
    Before the fix the log used the loop variable ``nonce``/``salt``, so
    operators saw nonce X in the ``mined!`` line while the submitter used
    nonce Y from the returned result.

    Setup:
    - Iter 1: ``evaluate_sampleset`` returns ``result_A`` (nonce_A, energy -2.0,
      floor -1.5).  Live threshold is -3000 milli so the submit gate fails
      (int(-1500) < -3000 is False) but ``result_A`` is stored in ``top_k``.
    - After iter 1's ``AttemptLogger.record`` fires, the live threshold is
      relaxed to 0 milli.
    - Iter 2: a fresh nonce (``nonce_B``) is drawn from the feeder.
      ``evaluate_sampleset`` returns ``result_B`` (nonce_B, energy -1.8,
      floor -1.5).  Submit gate walks top_k = [result_A, result_B] sorted
      by energy; result_A is better (-2.0 < -1.8), its floor (-1500 milli)
      < 0 milli → submit gate selects ``result_A``.  The loop variable
      ``nonce`` at this point is ``nonce_B``.

    Assertion: the ``mined!`` log line contains ``result_A.nonce.hex()[:16]``
    and does NOT contain ``nonce_B.hex()[:16]`` (which are different).
    """
    from unittest.mock import MagicMock

    # Predetermined nonce for the stashed result from iter 1 (32 bytes).
    nonce_A = bytes(range(32))
    salt_A = b"\xaa" * 32

    # Tracking state shared across the mock closures.
    iter_count = [0]
    captured_iter2_loop_nonce: list = []

    def fake_evaluate(sampleset, requirements, nodes, edges,
                      nonce, salt, prev_timestamp, start_time,
                      strict_energy=True, live_threshold_energy=None):
        iter_count[0] += 1
        n = len(nodes)
        fake_solutions = [[-1] * n]
        if iter_count[0] == 1:
            # Iter 1: return result_A with predetermined nonce_A.
            return MiningResult(
                miner_id="test",
                miner_type="CPU",
                nonce=nonce_A,
                salt=salt_A,
                timestamp=0,
                prev_timestamp=0,
                solutions=fake_solutions,
                energy=-2.0,
                diversity=1.0,
                num_valid=1,
                mining_time=0,
                node_list=list(nodes),
                edge_list=list(edges),
                submit_floor_energy=-1.5,  # int(-1500) < -3000 → False → no submit
            )
        else:
            # Iter 2+: record the loop's nonce (iter 2's nonce_B) for the
            # divergence assertion, then return result_B (also with nonce=nonce
            # from this iter — but submit gate will prefer result_A from top_k).
            if iter_count[0] == 2:
                captured_iter2_loop_nonce.append(nonce)
            return MiningResult(
                miner_id="test",
                miner_type="CPU",
                nonce=nonce,   # result_B.nonce == loop nonce for this iter
                salt=salt,
                timestamp=0,
                prev_timestamp=0,
                solutions=fake_solutions,
                energy=-1.8,
                diversity=1.0,
                num_valid=1,
                mining_time=0,
                node_list=list(nodes),
                edge_list=list(edges),
                submit_floor_energy=-1.5,  # int(-1500) < 0 → True → submittable
            )

    # Live threshold starts very strict (-3000 milli) so iter 1 stores but
    # does not submit.  After iter 1's record() fires we relax to 0 so iter 2's
    # submit gate can pass for the stashed result_A.
    live_var = mp.Value('q', -3000)

    recording_logger = MagicMock()

    def _on_record(**kw):
        # After the first record (iter 1), relax the live threshold so that
        # the submit gate passes on iter 2.
        if kw.get("result_kind") == "stored":
            with live_var.get_lock():
                live_var.value = 0

    recording_logger.record.side_effect = _on_record

    stop = mp.Event()
    cpu_miner._attempt_logger = recording_logger
    cpu_miner._live_max_energy_milli = live_var
    monkeypatched_evaluate = cpu_miner.evaluate_sampleset
    cpu_miner.evaluate_sampleset = fake_evaluate  # type: ignore[method-assign]

    returned_result = None
    logger_name = cpu_miner.logger.name
    try:
        with caplog.at_level(logging.INFO, logger=logger_name):
            returned_result = cpu_miner.mine_work_item(relaxed_context, stop)
    finally:
        cpu_miner.evaluate_sampleset = monkeypatched_evaluate
        del cpu_miner._attempt_logger
        del cpu_miner._live_max_energy_milli

    assert returned_result is not None, (
        "mine_work_item must return a result when a candidate clears the "
        "live threshold; returned None instead"
    )
    # The submit gate chose result_A (better energy) from top_k — even though
    # the loop's last nonce was nonce_B (iter 2).  The returned result must
    # carry nonce_A.
    assert returned_result.nonce == nonce_A, (
        f"expected submit gate to return result_A (nonce_A), "
        f"got nonce={returned_result.nonce.hex()}"
    )

    # Verify iter 2 ran so we actually tested the divergence scenario.
    assert captured_iter2_loop_nonce, (
        "evaluate_sampleset was never called for iter 2; "
        "the divergence scenario was never exercised"
    )
    nonce_B = captured_iter2_loop_nonce[0]
    assert nonce_B != nonce_A, (
        "iter-2 loop nonce equals nonce_A — divergence scenario collapsed; "
        "adjust the test setup so the two iters produce different loop nonces"
    )

    # Core assertion: the ``mined!`` line must log result.nonce (nonce_A),
    # not the loop variable nonce (nonce_B).
    mined_lines = [r.message for r in caplog.records if "mined!" in r.message]
    assert mined_lines, (
        "no 'mined!' INFO log line found; check logger name or caplog setup"
    )
    mined_line = mined_lines[0]

    expected_nonce_prefix = nonce_A.hex()[:16]
    assert expected_nonce_prefix in mined_line, (
        f"mined! log must show submitted result's nonce "
        f"(0x{expected_nonce_prefix}...) but got: {mined_line!r}"
    )

    # Regression check: the loop's last-iterated nonce (nonce_B) must NOT
    # appear in the log line.
    wrong_nonce_prefix = nonce_B.hex()[:16]
    assert wrong_nonce_prefix not in mined_line, (
        f"mined! log shows loop variable's nonce (0x{wrong_nonce_prefix}...) "
        f"instead of the submitted result's nonce — "
        f"this is the pre-fix bug; log line: {mined_line!r}"
    )
