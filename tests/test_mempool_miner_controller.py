"""Unit + integration tests for `shared.mempool_miner_controller`.

Unit tests cover:
  - `_should_accept_job` mode + topology filtering
  - submission/claim error classifiers

The integration test brings up a real MinerCore + MempoolMinerController
against the docker chain, has //Alice propose a Z(2,2) Ising job, and
asserts the solver picks it up, mines a solution, submits it, and the
chain emits SolutionAccepted.

Auto-skipped when the docker chain at `ws://localhost:9944` isn't reachable.
"""
from __future__ import annotations

import asyncio
import os
import socket
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.mempool_miner_controller import (
    CLAIM_STALE_ERRORS,
    MempoolMinerController,
    SOLUTION_FATAL_ERRORS,
    SOLUTION_STALE_ERRORS,
    _classify_claim,
    _classify_solution,
)
from shared.mempool_types import (
    IsingParams,
    JobMode,
    JobOrder,
    MinerType,
    OrderStatus,
    OrderTiming,
    ResultDelivery,
    RewardResolution,
)
from shared.quantum_proof_of_work import (
    DEFAULT_ALLOWED_H,
    DEFAULT_ALLOWED_J,
    DEFAULT_ALLOWED_SPIN,
)
from shared.topology_hash import topology_hash


def _topology_hash(nodes, edges) -> bytes:
    """Helper: canonical topology hash with the default allowed-value specs."""
    return topology_hash(
        nodes, edges, DEFAULT_ALLOWED_H, DEFAULT_ALLOWED_J, DEFAULT_ALLOWED_SPIN
    )


DEFAULT_URL = os.environ.get("QUIP_SUBSTRATE_URL", "ws://localhost:9944")


def _chain_reachable(url: str) -> bool:
    bare = url.split("://", 1)[1]
    host, _, port_str = bare.partition(":")
    port = int(port_str) if port_str else 9944
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, socket.timeout):
        return False


# ----------------------------------------------------------------------
# Topology hash parity with the CLI helper
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Error classifiers
# ----------------------------------------------------------------------


def test_classify_solution_success():
    assert _classify_solution(None) == "ok"


@pytest.mark.parametrize("name", SOLUTION_STALE_ERRORS)
def test_classify_solution_stale_names(name):
    assert _classify_solution(f"Module(error={name}, index=42)") == "stale"


@pytest.mark.parametrize("name", SOLUTION_FATAL_ERRORS)
def test_classify_solution_fatal_names(name):
    assert _classify_solution(f"Module(error={name}, index=42)") == "fatal"


def test_classify_solution_unknown_is_fatal():
    assert _classify_solution("Module(error=NeverSeenError)") == "fatal"


def test_classify_claim_success():
    assert _classify_claim(None) == "ok"


@pytest.mark.parametrize("name", CLAIM_STALE_ERRORS)
def test_classify_claim_stale_names(name):
    assert _classify_claim(f"Module(error={name})") == "stale"


# ----------------------------------------------------------------------
# Eligibility filter (mode + topology)
# ----------------------------------------------------------------------


def _bare_controller(account: bytes, solver_type: MinerType, sampler_hash: bytes):
    """Construct a MempoolMinerController bypassing __init__ for filter-only tests."""
    c = MempoolMinerController.__new__(MempoolMinerController)
    c.signer = MagicMock()
    c.signer.account_id_bytes.return_value = account
    c.solver_type = solver_type
    c.sampler_topology_hash = sampler_hash
    c.allowed_h_values = DEFAULT_ALLOWED_H
    c.allowed_j_values = DEFAULT_ALLOWED_J
    c.allowed_spin_values = DEFAULT_ALLOWED_SPIN
    return c


def _open_order_with_topology(
    topology_hash_inputs: tuple,
    mode: JobMode,
) -> JobOrder:
    nodes, edges = topology_hash_inputs
    return JobOrder(
        spec_id=b"\x42" * 32,
        proposer=b"\x11" * 32,
        ising_params=IsingParams(
            nodes=tuple(nodes),
            edges=tuple(edges),
            h_values=tuple(0 for _ in nodes),
            j_values=tuple(0 for _ in edges),
        ),
        reward=1_000_000_000_000,
        mode=mode,
        resolution=RewardResolution.single_best(),
        timing=OrderTiming(deadline_blocks=100, block_wait=10),
        delivery=ResultDelivery.on_chain_only(),
        status=OrderStatus.OPENED,
        created_at=1,
        first_solution_at=None,
        solution_count=0,
    )


def test_should_accept_topology_mismatch_rejected():
    nodes = (0, 1, 2)
    edges = ((0, 1), (1, 2))
    sampler_hash = _topology_hash(nodes, edges)
    other_hash = _topology_hash((0, 1, 2, 3), ((0, 1),))
    c = _bare_controller(b"\xAA" * 32, MinerType.CPU, sampler_hash)
    # Job is over a different topology — must be rejected.
    order = _open_order_with_topology(((0, 1, 2, 3), ((0, 1),)), JobMode.open())
    assert c._should_accept_job(order) is False
    assert other_hash != sampler_hash


def test_should_accept_open_mode_passes():
    nodes = (0, 1, 2)
    edges = ((0, 1), (1, 2))
    sampler_hash = _topology_hash(nodes, edges)
    c = _bare_controller(b"\xAA" * 32, MinerType.CPU, sampler_hash)
    order = _open_order_with_topology((nodes, edges), JobMode.open())
    assert c._should_accept_job(order) is True


def test_should_accept_bid_with_matching_account():
    nodes = (0, 1, 2)
    edges = ((0, 1), (1, 2))
    sampler_hash = _topology_hash(nodes, edges)
    account = b"\xAA" * 32
    c = _bare_controller(account, MinerType.CPU, sampler_hash)
    order = _open_order_with_topology(
        (nodes, edges),
        JobMode.bid(miners=(account, b"\xBB" * 32)),
    )
    assert c._should_accept_job(order) is True


def test_should_accept_bid_with_matching_miner_type():
    nodes = (0, 1, 2)
    edges = ((0, 1), (1, 2))
    sampler_hash = _topology_hash(nodes, edges)
    c = _bare_controller(b"\xAA" * 32, MinerType.GPU, sampler_hash)
    order = _open_order_with_topology(
        (nodes, edges),
        JobMode.bid(miner_types=(MinerType.GPU, MinerType.QPU_DWAVE)),
    )
    assert c._should_accept_job(order) is True


def test_should_reject_bid_with_no_matching_criteria():
    nodes = (0, 1, 2)
    edges = ((0, 1), (1, 2))
    sampler_hash = _topology_hash(nodes, edges)
    c = _bare_controller(b"\xAA" * 32, MinerType.CPU, sampler_hash)
    order = _open_order_with_topology(
        (nodes, edges),
        JobMode.bid(
            miners=(b"\xBB" * 32,),
            miner_types=(MinerType.GPU,),
        ),
    )
    assert c._should_accept_job(order) is False


# ----------------------------------------------------------------------
# Active-order dispatch-gate tracking
# ----------------------------------------------------------------------


def _bare_active_order_controller(num_handles: int = 3):
    """Construct a MempoolMinerController with the minimum state to exercise
    `_record_handle_terminal_for_active` and `_clear_active_order`."""
    from shared.mempool_miner_controller import MempoolControllerStats

    c = MempoolMinerController.__new__(MempoolMinerController)
    c._active_order = None
    c._active_order_done_handles = set()
    c._dispatch_contexts = {}
    c.miner_handles = [MagicMock(miner_id=f"h{i}") for i in range(num_handles)]
    c.stats = MempoolControllerStats()
    return c


def test_record_handle_terminal_one_error_leaves_active_order_set():
    """A single handle erroring on the active order MUST NOT clear
    `_active_order` — siblings may still produce a valid result that
    `_handle_result` would otherwise drop as stale."""
    c = _bare_active_order_controller(num_handles=3)
    ctx = MagicMock(order_id=42)
    c._dispatch_contexts[("h0", 1)] = ctx
    c._dispatch_contexts[("h1", 2)] = ctx
    c._dispatch_contexts[("h2", 3)] = ctx
    c._active_order = 42

    c._record_handle_terminal_for_active("h0", 1)

    assert c._active_order == 42, "premature clear loses sibling work"
    assert c._active_order_done_handles == {"h0"}


def test_record_handle_terminal_all_handles_done_clears():
    """When every handle has reported terminal for the active order
    without a winning submission, the gate must release so the next
    JobProposed is accepted."""
    c = _bare_active_order_controller(num_handles=3)
    ctx = MagicMock(order_id=42)
    c._dispatch_contexts[("h0", 1)] = ctx
    c._dispatch_contexts[("h1", 2)] = ctx
    c._dispatch_contexts[("h2", 3)] = ctx
    c._active_order = 42

    c._record_handle_terminal_for_active("h0", 1)
    c._record_handle_terminal_for_active("h1", 2)
    assert c._active_order == 42  # still waiting on h2
    c._record_handle_terminal_for_active("h2", 3)

    assert c._active_order is None
    assert c._active_order_done_handles == set()


def test_record_handle_terminal_ignores_stale_dispatch_ids():
    """Terminal events for dispatches that don't match the active order
    (e.g., a late cancel-completion from a previous order) must not
    advance the done-handles counter."""
    c = _bare_active_order_controller(num_handles=2)
    old_ctx = MagicMock(order_id=41)
    cur_ctx = MagicMock(order_id=42)
    c._dispatch_contexts[("h0", 1)] = old_ctx  # previous order
    c._dispatch_contexts[("h0", 5)] = cur_ctx  # current order
    c._dispatch_contexts[("h1", 6)] = cur_ctx
    c._active_order = 42

    c._record_handle_terminal_for_active("h0", 1)  # stale: from order 41
    assert c._active_order_done_handles == set()
    assert c._active_order == 42

    c._record_handle_terminal_for_active("h0", 5)
    c._record_handle_terminal_for_active("h1", 6)
    assert c._active_order is None


def test_record_handle_terminal_no_op_when_no_active_order():
    """Terminal events after the active order has already cleared (e.g.,
    a winning submission already completed and `_handle_result` cleared
    the gate) must be no-ops."""
    c = _bare_active_order_controller(num_handles=2)
    ctx = MagicMock(order_id=42)
    c._dispatch_contexts[("h0", 1)] = ctx
    # _active_order already cleared (e.g., by _handle_result success path)
    c._active_order = None

    c._record_handle_terminal_for_active("h0", 1)

    assert c._active_order is None
    assert c._active_order_done_handles == set()


# ----------------------------------------------------------------------
# on_new_block — event-manager entry point (post-migration)
# ----------------------------------------------------------------------


def _bare_event_controller(num_handles: int = 1):
    """Bare controller wired for ``on_new_block`` / ``_handle_result`` tests.

    Bypasses ``__init__`` (no pool/network). Sets only the attributes the
    code paths under test read; everything else stays unset so a regression
    that introduces a new dependency fails loudly rather than silently.
    """
    from shared.mempool_miner_controller import MempoolControllerStats

    c = MempoolMinerController.__new__(MempoolMinerController)
    c._active_order = None
    c._active_order_done_handles = set()
    c._dispatch_contexts = {}
    c._pending = deque()
    c._pending_seen = set()
    c._submitted_orders = set()
    c._claimable = set()
    c.miner_handles = [MagicMock(miner_id=f"h{i}") for i in range(num_handles)]
    c.stats = MempoolControllerStats()
    # After the pool.get("rpc") removal: parent owns build_client (compose+sign
    # only) and pool_client (swap-aware reads + submit). The MagicMock surface
    # lets each test wire just the methods it cares about.
    c.build_client = MagicMock()
    c.pool_client = MagicMock()
    c.signer = MagicMock()
    c.signer.account_id_bytes.return_value = b"\xAA" * 32
    c.solver_type = MinerType.CPU
    c.sampler_topology_hash = b"\xCD" * 32
    c.allowed_h_values = DEFAULT_ALLOWED_H
    c.allowed_j_values = DEFAULT_ALLOWED_J
    c.allowed_spin_values = DEFAULT_ALLOWED_SPIN
    c.on_solution_submitted = None
    c.on_reward_claimed = None
    c.core = None
    return c


def _make_ctx(*, block_number: int = 7, block_hash_byte: int = 0x11):
    """Build a ``SubstrateMiningContext``-shaped object via SimpleNamespace.

    Mempool only reads ``ctx.block_hash`` and ``ctx.block_number``; the
    real dataclass would force us to fill PoW-only fields we don't care
    about here. Mirrors ``test_miner_controller_on_new_head.py``'s pattern.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        block_hash=bytes([block_hash_byte]) * 32,
        block_number=block_number,
    )


@pytest.mark.asyncio
async def test_on_new_block_none_ctx_is_noop():
    """``ctx is None`` (no topology registered yet) must not process events."""
    c = _bare_event_controller()
    process_calls = []

    async def fake_process_head(block_hash, block_number):
        process_calls.append((block_hash, block_number))

    c._process_head = fake_process_head  # type: ignore[assignment]
    await c.on_new_block(None)
    assert process_calls == []


@pytest.mark.asyncio
async def test_on_new_block_routes_to_process_head():
    """A real ctx routes ``block_hash`` + ``block_number`` into ``_process_head``."""
    c = _bare_event_controller()
    process_calls = []

    async def fake_process_head(block_hash, block_number):
        process_calls.append((block_hash, block_number))

    c._process_head = fake_process_head  # type: ignore[assignment]
    ctx = _make_ctx(block_number=42, block_hash_byte=0x33)
    await c.on_new_block(ctx)
    assert process_calls == [(bytes([0x33]) * 32, 42)]


# ----------------------------------------------------------------------
# Per-block event routing — JobProposed / OrderExpired through _process_head
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_head_routes_job_proposed_through_consider():
    """A ``JobProposed`` event in the block's events list is routed."""
    c = _bare_event_controller()
    block_hash = b"\x10" * 32

    async def fake_get_events_at(bh):
        assert bh == block_hash
        return [
            {
                "module_id": "QuantumComputeMempool",
                "event_id": "JobProposed",
                "attributes": {"order_id": 5},
            }
        ]

    c.pool_client.get_events_at = fake_get_events_at

    considered: list[int] = []

    async def fake_consider(order_id):
        considered.append(order_id)

    c._consider_order = fake_consider  # type: ignore[assignment]
    await c._process_head(block_hash, 100)
    assert considered == [5]
    assert c.stats.events_seen == 1
    assert c.stats.heads_observed == 1


@pytest.mark.asyncio
async def test_process_head_routes_order_expired_to_claimable():
    """``OrderExpired`` for a previously-submitted order joins ``_claimable``."""
    c = _bare_event_controller()
    c._submitted_orders.add(99)

    async def fake_get_events_at(bh):
        return [
            {
                "module_id": "QuantumComputeMempool",
                "event_id": "OrderExpired",
                "attributes": {"order_id": 99},
            }
        ]

    c.pool_client.get_events_at = fake_get_events_at
    await c._process_head(b"\x10" * 32, 100)
    assert 99 in c._claimable


@pytest.mark.asyncio
async def test_process_head_ignores_non_mempool_events():
    """Events from other pallets are not counted as mempool events."""
    c = _bare_event_controller()

    async def fake_get_events_at(bh):
        return [
            {
                "module_id": "System",
                "event_id": "ExtrinsicSuccess",
                "attributes": {},
            }
        ]

    c.pool_client.get_events_at = fake_get_events_at
    await c._process_head(b"\x10" * 32, 100)
    assert c.stats.events_seen == 0
    assert c.stats.heads_observed == 1


# ----------------------------------------------------------------------
# _handle_result — submission path coverage
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_result_stale_order_dropped():
    """Late results from a previously-cancelled order are dropped, not submitted."""
    from shared.mempool_miner_controller import _MempoolResultEnvelope

    c = _bare_event_controller()
    c._active_order = 100

    ctx = MagicMock(order_id=99)  # different from active
    envelope = _MempoolResultEnvelope(
        result=MagicMock(solutions=[[1, -1]]),
        context=ctx,
        handle_id="h0",
    )

    submit_calls = []

    async def fake_submit(*args, **kwargs):
        submit_calls.append((args, kwargs))
        return MagicMock(error=None)

    c.build_client.build_signed_extrinsic = AsyncMock(return_value="0xdeadbeef")
    c.pool_client.submit_signed_extrinsic = fake_submit
    await c._handle_result(envelope)
    assert submit_calls == []
    assert c.stats.results_received == 1
    assert c.stats.solutions_submitted == 0


@pytest.mark.asyncio
async def test_handle_result_ok_marks_submitted_and_clears_active():
    """A successful submission records the order in ``_submitted_orders`` and
    clears ``_active_order`` so the dispatch gate can release."""
    from shared.mempool_miner_controller import _MempoolResultEnvelope
    from shared.miner_types import MiningResult

    c = _bare_event_controller(num_handles=2)
    c._active_order = 42

    result = MiningResult(
        miner_id="h0",
        miner_type="CPU",
        nonce=b"\x00" * 32,
        salt=b"\x00" * 32,
        timestamp=0,
        prev_timestamp=0,
        solutions=[[1, -1]],
        energy=-1.0,
        diversity=0.5,
        num_valid=1,
        mining_time=10,
        node_list=[],
        edge_list=[],
    )
    ctx = MagicMock(order_id=42)
    envelope = _MempoolResultEnvelope(
        result=result, context=ctx, handle_id="h0",
    )

    async def fake_submit(*args, **kwargs):
        return MagicMock(error=None, extrinsic_hash="0xdeadbeef")

    c.build_client.build_signed_extrinsic = AsyncMock(return_value="0xdeadbeef")
    c.pool_client.submit_signed_extrinsic = fake_submit
    await c._handle_result(envelope)
    assert c._active_order is None
    assert 42 in c._submitted_orders
    assert c.stats.solutions_submitted == 1


# ----------------------------------------------------------------------
# Claim loop — direct exercise of _claim_expired_orders
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_expired_orders_success_removes_from_claimable():
    """A claimable order with an OK receipt is removed from ``_claimable``
    and the rewards counter advances."""
    c = _bare_event_controller()
    c._claimable.add(7)

    async def fake_submit(*args, **kwargs):
        return MagicMock(error=None, extrinsic_hash="0xfeed")

    c.build_client.build_signed_extrinsic = AsyncMock(return_value="0xdeadbeef")
    c.pool_client.submit_signed_extrinsic = fake_submit
    await c._claim_expired_orders()
    assert 7 not in c._claimable
    assert c.stats.rewards_claimed == 1


@pytest.mark.asyncio
async def test_claim_expired_orders_not_expired_retries_next_tick():
    """``OrderNotExpired`` keeps the order in ``_claimable`` for retry."""
    c = _bare_event_controller()
    c._claimable.add(7)

    async def fake_submit(*args, **kwargs):
        return MagicMock(
            error="Module(error=OrderNotExpired)", extrinsic_hash="0xfeed",
        )

    c.build_client.build_signed_extrinsic = AsyncMock(return_value="0xdeadbeef")
    c.pool_client.submit_signed_extrinsic = fake_submit
    await c._claim_expired_orders()
    assert 7 in c._claimable
    assert c.stats.rewards_claimed == 0


@pytest.mark.asyncio
async def test_claim_expired_orders_not_winner_gives_up():
    """``NotWinner`` is stale-class-but-terminal — drop, don't retry."""
    c = _bare_event_controller()
    c._claimable.add(7)

    async def fake_submit(*args, **kwargs):
        return MagicMock(
            error="Module(error=NotWinner)", extrinsic_hash="0xfeed",
        )

    c.build_client.build_signed_extrinsic = AsyncMock(return_value="0xdeadbeef")
    c.pool_client.submit_signed_extrinsic = fake_submit
    await c._claim_expired_orders()
    assert 7 not in c._claimable


# ----------------------------------------------------------------------
# Live-chain integration test
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not _chain_reachable(DEFAULT_URL),
    reason=f"substrate chain not reachable at {DEFAULT_URL}",
)
@pytest.mark.timeout(180)
async def test_controller_submits_solution_end_to_end(tmp_path):
    """Bring up a MempoolMinerController, have //Alice propose a Z(2,2) job,
    and verify the controller mines + submits a solution that the chain
    accepts.

    Self-contained: registers a fresh hybrid solver, funds it, registers
    a job spec, proposes a job. Then runs the controller until the chain
    emits SolutionAccepted for our solver.
    """
    from dwave_topologies.topologies.zephyr import zephyr

    from CPU.sa_miner import SimulatedAnnealingMiner  # noqa: F401 — ensures sampler module loads
    from shared.keystore_hybrid import generate
    from shared.mempool_miner_controller import MempoolMinerController
    from shared.mempool_types import MinerType
    from shared.miner_bootstrap import _resolve_dev_signer
    from shared.miner_worker import MinerHandle
    from substrate.client import SubstrateClient

    # Fresh hybrid keystore for this test run — keeps state independent
    # from any leftover registration on the chain.
    keystore = generate(tmp_path / "hybrid_signing.json")
    solver_type = MinerType.CPU

    # Tiny topology so the chain validation is fast. `zephyr(m, t)` is the
    # wrapper that gives the CPU SA sampler its expected `.properties`
    # attribute; the underlying graph is the same as `dnx.zephyr_graph(m, t)`.
    topology = zephyr(2, 2)
    sampler_nodes = tuple(int(n) for n in topology.nodes)
    sampler_edges = tuple((int(u), int(v)) for u, v in topology.edges)
    sampler_topology_hash = _topology_hash(
        sampler_nodes, sampler_edges
    )

    setup_client = SubstrateClient(url=DEFAULT_URL)
    await setup_client.connect()
    try:
        alice = _resolve_dev_signer("//Alice")

        # Fund the solver from //Alice.
        balance = await setup_client.query_balance(
            keystore.signer.account_id_bytes()
        )
        if balance < 2_000_000_000_000:
            receipt = await setup_client.submit_extrinsic(
                "Balances",
                "transfer_keep_alive",
                {
                    "dest": {
                        "Id": "0x" + keystore.signer.account_id_bytes().hex()
                    },
                    "value": 10_000_000_000_000,
                },
                alice,
                wait_for="inblock",
            )
            if receipt.error:
                pytest.fail(f"funding: {receipt.error}")

        # Register as a CPU solver.
        existing = await setup_client.query_solver(
            keystore.signer.account_id_bytes()
        )
        if existing is None:
            r = await setup_client.submit_extrinsic(
                "QuantumComputeMempool",
                "register_solver",
                {"solver_type": solver_type.to_scale_variant()},
                keystore.signer,
                wait_for="inblock",
            )
            if r.error:
                pytest.fail(f"register_solver: {r.error}")

        # Register a JobSpec under //Alice and pull the spec_id from the
        # inclusion-block events.
        name = b"phase8c-test-" + os.urandom(8).hex().encode()
        register_receipt = await setup_client.submit_extrinsic(
            "QuantumComputeMempool",
            "register_job_spec",
            {
                "name": (list(name),),
                "formulation": "Ising",
                "validation_program": None,
                "transform_program": None,
            },
            alice,
            wait_for="inblock",
        )
        if register_receipt.error:
            pytest.fail(f"register_job_spec: {register_receipt.error}")
        register_block = bytes.fromhex(register_receipt.block_hash[2:])
        events = await setup_client.get_events_at(register_block)
        spec_id = None
        for ev in events:
            if (
                ev["module_id"] == "QuantumComputeMempool"
                and ev["event_id"] == "JobSpecRegistered"
            ):
                raw = ev["attributes"].get("spec_id")
                spec_id = (
                    bytes.fromhex(raw[2:] if raw.startswith("0x") else raw)
                    if isinstance(raw, str)
                    else bytes(raw)
                )
                break
        if spec_id is None:
            pytest.fail(f"JobSpecRegistered event missing: {events}")

        # Propose an all-zero Ising — every spin assignment has energy 0
        # so the SA sampler trivially solves it. Open mode, no quality
        # floors. Reward is 2 UNIT (> MinReward). Build the params via
        # IsingParams.to_scale_dict() so we go through the same encoding
        # path the test_mempool_client.py::test_propose_job_and_query_back
        # test exercises.
        from shared.mempool_types import IsingParams
        ising = IsingParams(
            nodes=sampler_nodes,
            edges=sampler_edges,
            h_values=tuple(0 for _ in sampler_nodes),
            j_values=tuple(0 for _ in sampler_edges),
            min_energy_milli=None,
            min_diversity_milli=None,
            min_solutions=None,
        )
        propose_receipt = await setup_client.submit_extrinsic(
            "QuantumComputeMempool",
            "propose_job",
            {
                "spec_id": "0x" + spec_id.hex(),
                "ising_params": ising.to_scale_dict(),
                "reward": 2_000_000_000_000,
                "mode": {"Open": None},
                "resolution": {"SingleBest": None},
                "deadline_blocks": 100,
                "block_wait": 5,
                "delivery": {"OnChainOnly": None},
            },
            alice,
            wait_for="inblock",
        )
        if propose_receipt.error:
            pytest.fail(f"propose_job: {propose_receipt.error}")
    finally:
        await setup_client.close()

    # Bring up a CPU miner bound to the same Z(2,2) topology.
    spec = {
        "id": "test-mempool-cpu",
        "kind": "cpu",
        "args": {"topology": topology},
    }
    handle = MinerHandle(spec=spec)

    from substrate.pool import ValidatorPool
    pool = ValidatorPool(urls=[DEFAULT_URL])
    controller = MempoolMinerController(
        pool=pool,
        signer=keystore.signer,
        miner_handles=[handle],
        sampler_topology_hash=sampler_topology_hash,
        allowed_h_values=DEFAULT_ALLOWED_H,
        allowed_j_values=DEFAULT_ALLOWED_J,
        allowed_spin_values=DEFAULT_ALLOWED_SPIN,
        solver_type=solver_type,
    )
    solution_submitted = asyncio.Event()

    async def on_solution(order_id, result):
        solution_submitted.set()

    controller.on_solution_submitted = on_solution

    run_task = asyncio.create_task(controller.run())
    try:
        try:
            await asyncio.wait_for(solution_submitted.wait(), timeout=120)
        except asyncio.TimeoutError:
            pytest.fail(
                f"controller did not submit a solution in 120s. "
                f"stats={controller.stats}"
            )
        assert controller.stats.solutions_submitted >= 1
        assert controller.stats.solution_errors == 0

        # Chain-side counter: the solver's `solutions_submitted` should have
        # incremented at least once.
        client2 = SubstrateClient(url=DEFAULT_URL)
        await client2.connect()
        try:
            info = await client2.query_solver(
                keystore.signer.account_id_bytes()
            )
            assert info is not None
            assert info.solutions_submitted >= 1
        finally:
            await client2.close()
    finally:
        controller.shutdown()
        try:
            await asyncio.wait_for(run_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        await pool.close()
        handle.req.put({"op": "shutdown"})
        handle.proc.join(timeout=5)
        if handle.proc.is_alive():
            handle.proc.terminate()
            handle.proc.join(timeout=2)
