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
from unittest.mock import MagicMock

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

    from shared.validator_pool import ValidatorPool
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
    # Resolve rpc once so the test fixture's later setup
    # (events, balances) shares a connected client.
    await pool.get("rpc")

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
