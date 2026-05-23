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
)
from shared.work_context import requirements_from_context, resolve_ising
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.quantum_proof_of_work import derive_nonce
from shared.substrate_submitter import encode_quantum_proof
from shared.allowed_value_spec import AllowedValueSet
from shared.substrate_types import (
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

    # Force the submit gate to fail: set the live threshold so far below
    # any reachable Ising energy that no floor can clear it. The post-
    # processing branch still fires because ratchet_threshold (= the
    # context's relaxed difficulty_energy) is easily beaten.
    live_var = mp.Value('q', -10**18)

    # Install both onto the miner. ``mine_work_item`` reads
    # ``_attempt_logger`` and ``_live_max_energy_milli`` directly off
    # ``self``; module-scope fixture means we must clean up after.
    cpu_miner._attempt_logger = recording_logger
    cpu_miner._live_max_energy_milli = live_var

    stop = mp.Event()
    try:
        result = cpu_miner.mine_work_item(relaxed_context, stop)
    finally:
        del cpu_miner._attempt_logger
        del cpu_miner._live_max_energy_milli

    # Submit gate forced to fail → loop returns None (stop was set by
    # the mock as soon as one record was emitted).
    assert result is None
    assert captured, "expected at least one AttemptLogger.record call"

    # Find an iteration that post-processed; the test setup guarantees
    # the first such record will be ``stored`` (no prior stored_best to
    # beat, so any valid post-processed result becomes stored) — submit
    # blocked by the impossibly-low live threshold.
    post_processed = [k for k in captured if k.get("post_processed")]
    assert post_processed, (
        "expected at least one post_processed=True record; got "
        f"{[k.get('result_kind') for k in captured]}"
    )
    rec = post_processed[0]
    assert rec["result_kind"] == "stored", (
        f"expected result_kind='stored' (submit gate forced to fail), "
        f"got {rec['result_kind']}"
    )
    assert isinstance(rec["num_valid"], int) and rec["num_valid"] >= 1, (
        f"num_valid must be a populated int on post-processed stored "
        f"iteration, got {rec['num_valid']!r}"
    )
    assert isinstance(rec["diversity_milli"], int), (
        f"diversity_milli must be a populated int on post-processed "
        f"stored iteration, got {rec['diversity_milli']!r}"
    )
