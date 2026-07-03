"""Live-chain integration tests for mempool priority mining (T9).

Drives the PRODUCTION scheduler stack (``quip_cli._build_scheduler_stack``)
end-to-end against a throwaway dev chain: pow as idle filler, a mempool job
preempting a busy handle (cancel -> mandatory done-sentinel -> dispatch),
failure isolation (a mempool-fatal receipt parks the stack while pow keeps
mining), and the one-account/one-solver-type constraint the supervisor's
owner election protects.

Starting the dev chain (from a ``quip-protocol-rs`` checkout)::

    cargo build --release -p quip-network-node
    ./target/release/quip-network-node --dev --tmp --rpc-port 9945
    QUIP_SUBSTRATE_URL=ws://127.0.0.1:9945 pytest tests/test_mempool_priority_integration.py

The suite auto-skips when the chain is unreachable, when ``system_chain`` is
not a dev chain (the tests sudo-reseed difficulty, so they must never point at
a real network), or when the runtime lacks the MinerRegistry pallet, the
per-topology ``QuantumPow.Difficulties`` map, or the QuantumComputeMempool
pallet.

Manual QPU smoke procedure (NOT automated — per project policy the operator
runs every QPU/D-Wave job, never test automation):

    1. Run a qpu config with an explicit vendor-section opt-in
       (``[dwave] mempool = true``)::

           quip-miner qpu --config <qpu toml>

    2. Verify in the logs that mempool job dispatch to the QPU handle is
       idle-only and the QPU handle is never preempted: every
       "dispatched mempool job" line naming the QPU handle must occur while
       that handle was idle, and there must be no cancel/preempt of an
       in-flight QPU dispatch (the WorkScheduler never preempts QPU handles —
       a preemption would strand already-paid QPU work).

Manual multi-backend supervisor boot (NOT automated on dev boxes — a
``[metal]`` / ``[cuda.N]`` section spawns a child that starts REAL GPU
mining, so the full supervisor path is a manual check)::

    quip-miner --config <cpu+gpu toml>

    Look for: the mempool owner-election echo ("supervisor: mempool owner is
    cpu; disabling mempool on ..."), exactly ONE solver registration for the
    shared signer account (Guard D+ runs only on the owner child), and no
    ``SolverAlreadyRegistered`` crash on any non-owner child.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import pytest

import quip_cli
from shared.keystore_hybrid import generate
from shared.miner_config import SubmissionConfig
from shared.topology_hash import topology_hash
from substrate.client import SubstrateClient
from substrate.mempool_types import (
    IsingParams,
    JobMode,
    MempoolJobContext,
    MinerType,
    ResultDelivery,
    RewardResolution,
)
from substrate.miner_bootstrap import (
    DEFAULT_SEED_DIFFICULTY,
    DEV_CHAIN_PREFIXES,
    BootstrapConfig,
    _maybe_seed_chain,
    _resolve_dev_signer,
    _sudo_call,
)
from substrate.pool import ValidatorPool
from substrate.solver_registration import SolverGuardOutcome, ensure_solver_registered
from substrate.types import SubstrateMiningContext


logger = logging.getLogger(__name__)

DEFAULT_URL = os.environ.get("QUIP_SUBSTRATE_URL", "ws://localhost:9944")

# The genesis-shipped job spec. QUERIED from chain state, never hardcoded by
# hash — the spec_id differs per chain build.
_GENESIS_SPEC_NAME = "plain-ising-v1"

_CPU_MINER_CONFIG = {"cpu": {"num_cpus": 1}}

# Unwinnable difficulty: sudo-reseeding this makes the single CPU handle mine
# a pow item it can never win, so a proposed job deterministically finds the
# handle BUSY (the preemption path). Strictness is ENERGY-ONLY on purpose:
# the controller live-pushes only the energy threshold to busy workers
# (`set_live_threshold_milli`); diversity / min_solutions are baked into the
# work item at dispatch. An unreachable diversity floor would therefore stay
# frozen inside the strict-round item across the later relaxed reseed and
# wedge the recovery phase, while the energy floor relaxes live. CPU SA on
# the seeded Z(9,2) reaches ~-3500 energy; -10000.0 is out of reach.
_STRICT_DIFFICULTY = {
    "min_solutions": DEFAULT_SEED_DIFFICULTY.min_solutions,
    "max_energy_milli": -10_000_000,
    "min_diversity_milli": DEFAULT_SEED_DIFFICULTY.min_diversity_milli,
}
# The relaxed dev seed (`DEFAULT_SEED_DIFFICULTY`) — pow proofs land in
# seconds-to-minutes on CPU. Restored in every teardown.
_RELAXED_DIFFICULTY = {
    "min_solutions": DEFAULT_SEED_DIFFICULTY.min_solutions,
    "max_energy_milli": DEFAULT_SEED_DIFFICULTY.max_energy_milli,
    "min_diversity_milli": DEFAULT_SEED_DIFFICULTY.min_diversity_milli,
}


# ----------------------------------------------------------------------
# Collection-time skip gating
# ----------------------------------------------------------------------


def _chain_reachable(url: str) -> bool:
    bare = url.split("://", 1)[1]
    host, _, port_str = bare.partition(":")
    port = int(port_str) if port_str else 9944
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, socket.timeout):
        return False


def _gate_reason(url: str) -> Optional[str]:
    """Skip reason for this module, or None when the chain is usable.

    Mirrors the probe pattern of the sibling live tests
    (tests/test_substrate_miner_controller.py): reachability first, then the
    dev-chain name guard, then metadata probes for the pallets/storage this
    suite depends on — so CI-without-chain and testnet-pointed runs both
    skip cleanly instead of failing mid-bootstrap.
    """
    if not _chain_reachable(url):
        return f"substrate chain not reachable at {url}"
    try:
        from substrateinterface import SubstrateInterface

        si = SubstrateInterface(url=url)
        chain_name = str(si.chain)
        if not any(chain_name.startswith(p) for p in DEV_CHAIN_PREFIXES):
            return (
                f"chain {chain_name!r} is not a dev chain (allowed prefixes: "
                f"{', '.join(DEV_CHAIN_PREFIXES)}); these tests sudo-reseed "
                "difficulty and must never run against a real network"
            )
        md = si.get_metadata()

        def _pallet(name: str):
            try:
                return md.get_metadata_pallet(name)
            except Exception:  # noqa: BLE001 — treat any probe failure as absent
                return None

        if _pallet("MinerRegistry") is None:
            return "chain runtime lacks the MinerRegistry pallet"
        quantum_pow = _pallet("QuantumPow")
        if (
            quantum_pow is None
            or quantum_pow.get_storage_function("Difficulties") is None
        ):
            return "chain runtime lacks per-topology QuantumPow.Difficulties"
        if _pallet("QuantumComputeMempool") is None:
            return "chain runtime lacks the QuantumComputeMempool pallet"
    except Exception as exc:  # noqa: BLE001 — probe failure means skip, not error
        return f"chain probe at {url} failed: {exc}"
    return None


_GATE_REASON = _gate_reason(DEFAULT_URL)

pytestmark = pytest.mark.skipif(
    _GATE_REASON is not None, reason=_GATE_REASON or "live dev chain available"
)


# ----------------------------------------------------------------------
# Polling / chain helpers (no sleeps-as-synchronization)
# ----------------------------------------------------------------------


async def _wait_for(
    predicate,
    *,
    timeout: float,
    desc: str,
    tasks: tuple = (),
    interval: float = 0.5,
    fail: bool = True,
):
    """Poll ``predicate`` (sync or async) until truthy, with a deadline.

    ``tasks`` are run-tasks that must stay alive for the wait to make sense
    (controller.run() / mempool_stack.run()); if one exits, fail immediately
    with its exception instead of burning the whole deadline. With
    ``fail=False`` a timeout returns None so the caller can retry a
    different way (e.g. re-propose a job the head poll may have skipped).
    """
    deadline = time.monotonic() + timeout
    while True:
        for task in tasks:
            if task.done():
                exc = None if task.cancelled() else task.exception()
                pytest.fail(
                    f"task {task.get_name()} exited while waiting for {desc}: {exc!r}"
                )
        value = predicate()
        if inspect.isawaitable(value):
            value = await value
        if value:
            return value
        if time.monotonic() >= deadline:
            if fail:
                pytest.fail(f"timed out after {timeout:.0f}s waiting for {desc}")
            return None
        await asyncio.sleep(interval)


def _receipt_block_bytes(receipt) -> bytes:
    block_hash = receipt.block_hash
    assert block_hash and block_hash.startswith("0x"), (
        f"receipt carries no block hash: {receipt}"
    )
    return bytes.fromhex(block_hash[2:])


async def _set_difficulty(
    client: SubstrateClient, topology_hash_bytes: bytes, difficulty: dict
) -> None:
    """Sudo-reseed the per-topology difficulty (dev chain only).

    Hard 120s cap: the in-block watch subscription can die silently
    (observed live — a hang here once wedged the whole suite past its
    pytest-timeout, in teardown). A loud TimeoutError beats a silent hang.
    """
    await asyncio.wait_for(
        _sudo_call(
            client,
            _resolve_dev_signer("//Alice"),
            "QuantumPow",
            "set_difficulty",
            {
                "topology_hash": "0x" + topology_hash_bytes.hex(),
                "difficulty": dict(difficulty),
            },
        ),
        timeout=120,
    )


async def _fund_fresh_keystore(client: SubstrateClient, keystore_path):
    """Mint a fresh hybrid keystore and fund it from //Alice."""
    keystore = generate(keystore_path)
    alice = _resolve_dev_signer("//Alice")
    balance = await client.query_balance(keystore.signer.account_id_bytes())
    if balance < 2_000_000_000_000:
        receipt = await client.submit_extrinsic(
            "Balances",
            "transfer_keep_alive",
            {
                "dest": {"Id": "0x" + keystore.signer.account_id_bytes().hex()},
                "value": 10_000_000_000_000,
            },
            alice,
            wait_for="inblock",
        )
        assert receipt.error is None, f"funding transfer failed: {receipt.error}"
    return keystore


async def _bootstrap_mining_identity(client: SubstrateClient, keystore_path):
    """Seed the chain (idempotent, difficulty force-reset to the relaxed
    default) and fund + pow-register a fresh keystore."""
    keystore = await _fund_fresh_keystore(client, keystore_path)
    await _maybe_seed_chain(
        client,
        BootstrapConfig(
            validators=(DEFAULT_URL,),
            signer_key_path=keystore_path,
            seed_chain=True,
            seed_topology_mt=(9, 2),
            force_reseed_difficulty=True,
        ),
    )
    if await client.query_miner(keystore.signer.account_id_bytes()) is None:
        receipt = await client.submit_extrinsic(
            "QuantumPow", "register_miner", {}, keystore.signer, wait_for="inblock"
        )
        assert receipt.error is None, f"register_miner failed: {receipt.error}"
    return keystore


async def _genesis_spec_id(client: SubstrateClient) -> bytes:
    """Query the genesis job spec's id from JobSpecs (never hardcoded)."""
    iface = client._iface  # noqa: SLF001 — same internal pattern as _sudo_call
    entries = await client._run(  # noqa: SLF001
        lambda: list(iface.query_map("QuantumComputeMempool", "JobSpecs"))
    )
    for key, value in entries:
        record = getattr(value, "value", value) or {}
        if record.get("name") == _GENESIS_SPEC_NAME:
            raw = getattr(key, "value", key)
            if isinstance(raw, str):
                return bytes.fromhex(raw[2:] if raw.startswith("0x") else raw)
            return bytes(raw)
    pytest.fail(f"genesis job spec {_GENESIS_SPEC_NAME!r} not found in JobSpecs")


async def _propose_matching_job(
    client: SubstrateClient,
    snapshot,
    *,
    block_wait: int = 5,
    deadline_blocks: int = 50,
) -> int:
    """Propose (as //Alice) a job the running sampler is eligible for.

    The producer's eligibility filter hashes the job's nodes+edges under the
    chain's allowed-value specs and requires exact equality with the
    sampler's topology hash — so the job is built from the CHAIN's registered
    topology (the same snapshot the stack bound to), with h/j values legal
    under the seeded ternary-h / binary-j specs.
    """
    alice = _resolve_dev_signer("//Alice")
    spec_id = await _genesis_spec_id(client)
    nodes = tuple(int(n) for n in snapshot.nodes)
    edges = tuple((int(u), int(v)) for u, v in snapshot.edges)
    h_pattern = (-1000, 0, 1000)
    j_pattern = (-1000, 1000)
    ising = IsingParams(
        nodes=nodes,
        edges=edges,
        h_values=tuple(h_pattern[i % 3] for i in range(len(nodes))),
        j_values=tuple(j_pattern[i % 2] for i in range(len(edges))),
        min_energy_milli=None,
        min_diversity_milli=None,
        min_solutions=1,
    )
    receipt = await client.submit_extrinsic(
        "QuantumComputeMempool",
        "propose_job",
        {
            "spec_id": "0x" + spec_id.hex(),
            "ising_params": ising.to_scale_dict(),
            "reward": 2_000_000_000_000,
            "mode": JobMode.open().to_scale_dict(),
            "resolution": RewardResolution.single_best().to_scale_dict(),
            "deadline_blocks": deadline_blocks,
            "block_wait": block_wait,
            "delivery": ResultDelivery.on_chain_only().to_scale_dict(),
        },
        alice,
        wait_for="inblock",
    )
    assert receipt.error is None, f"propose_job failed: {receipt.error}"
    events = await client.get_events_at(_receipt_block_bytes(receipt))
    order_ids = [
        int(ev["attributes"]["order_id"])
        for ev in events
        if ev["module_id"] == "QuantumComputeMempool"
        and ev["event_id"] == "JobProposed"
        and isinstance(ev["attributes"], dict)
        and "order_id" in ev["attributes"]
    ]
    assert order_ids, f"JobProposed event not found in propose block: {events}"
    return max(order_ids)


async def _propose_job_seen_by_producer(stack_fixture, tasks) -> list:
    """Propose a matching job and wait for the producer to accept it.

    The producer discovers jobs by polling System.Events at each observed
    head; the head poll can (rarely) skip a block, silently losing the
    JobProposed event. Propose once, and if the producer's accepted counter
    doesn't move within the first budget, propose a second fresh order —
    the tests then assert against the union of proposed order ids.
    """
    accepted_before = stack_fixture.stack.producer.stats.jobs_accepted
    order_ids: list = []
    for budget in (75.0, 150.0):
        order_ids.append(
            await _propose_matching_job(stack_fixture.client, stack_fixture.snapshot)
        )
        seen = await _wait_for(
            lambda: stack_fixture.stack.producer.stats.jobs_accepted > accepted_before,
            timeout=budget,
            desc=f"producer accepts a proposed job (orders={order_ids})",
            tasks=tasks,
            fail=False,
        )
        if seen:
            return order_ids
    pytest.fail(f"producer never accepted any proposed job (orders={order_ids})")


# ----------------------------------------------------------------------
# The live production stack
# ----------------------------------------------------------------------


class _ListHandler(logging.Handler):
    """Capture LogRecords on a logger-level handler.

    pytest's ``caplog`` cannot be used here: ``MinerCore`` re-routes the ROOT
    logger's handlers into its log-writer child process (removing caplog's
    handler with them). A handler attached directly to the
    ``substrate.work_scheduler`` logger survives that reset.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@dataclass
class _LiveStack:
    """Handles to the running production stack yielded to a test."""

    client: SubstrateClient
    keystore: Any
    snapshot: Any
    pool: Any
    core: Any
    handle: Any
    controller: Any
    scheduler: Any
    stack: Any
    controller_task: asyncio.Task
    stack_task: asyncio.Task
    routed: list  # every WorkResult delivered to the mempool consumer
    scheduler_records: list  # LogRecords from the substrate.work_scheduler logger


@asynccontextmanager
async def _live_scheduler_stack(tmp_path, name: str) -> AsyncIterator[_LiveStack]:
    """Bring up the production scheduler stack against the dev chain.

    Mirrors `quip_cli._run_concurrent_miner`'s startup: bootstrap identity
    (seed + fund + register_miner), Guard D+ solver registration, chain-bound
    topology, `_prepare_core`, `_build_scheduler_stack`, then the same task
    lifecycle `_orchestrate_controllers` drives (scheduler.start() +
    controller.run() + mempool_stack.run()) — minus the signal handlers.

    Teardown is fully guarded (each step individually) and always restores
    the relaxed difficulty, so a hung component or a mid-test failure can
    neither wedge the suite nor leave the shared chain strict.
    """
    env_overrides = {
        # No in-process telemetry sibling (would bind a REST port per test).
        "QUIP_TELEMETRY_EXTERNAL": "1",
        # Isolate stats snapshots + mining-attempt logs from any real miner.
        "QUIP_RUNTIME_DIR": str(tmp_path / "runtime"),
    }
    saved_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)

    client = SubstrateClient(url=DEFAULT_URL)
    await client.connect()
    pool = core = None
    controller = scheduler = stack = None
    controller_task = stack_task = None
    snapshot = None
    scheduler_logger = logging.getLogger("substrate.work_scheduler")
    log_handler = _ListHandler()
    saved_level = scheduler_logger.level
    scheduler_logger.addHandler(log_handler)
    scheduler_logger.setLevel(logging.INFO)
    # Full INFO log of the run into tmp_path — the only way to diagnose a
    # live-chain failure after the fact. Attached to the ROOT logger before
    # MinerCore construction so its log-writer child adopts the same file.
    root_logger = logging.getLogger()
    saved_root_level = root_logger.level
    debug_log_path = tmp_path / "t9-live-stack.log"
    file_handler = logging.FileHandler(debug_log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    logger.info("T9 live stack log: %s", debug_log_path)
    try:
        keystore = await _bootstrap_mining_identity(
            client, tmp_path / f"{name}.json"
        )
        outcome = await ensure_solver_registered(client, keystore.signer, "cpu")
        assert outcome in (
            SolverGuardOutcome.REGISTERED,
            SolverGuardOutcome.ALREADY_REGISTERED,
        ), f"solver guard failed: {outcome}"

        topology, snapshot = await quip_cli._topology_from_chain(
            client,
            account_bytes=keystore.signer.account_id_bytes(),
            miner_config=_CPU_MINER_CONFIG,
        )
        core = quip_cli._prepare_core("cpu", _CPU_MINER_CONFIG, topology)
        pool = ValidatorPool(urls=[DEFAULT_URL])
        controller, scheduler, stack = await quip_cli._build_scheduler_stack(
            client=client,
            pool=pool,
            core=core,
            keystore=keystore,
            topology=topology,
            miner_kind="cpu",
            telemetry_port=18086,
            submission_config=SubmissionConfig(),
            mempool_enabled=True,
            mempool_min_reward=0,
        )
        assert stack is not None, "mempool stack missing with mempool_enabled=True"
        # 30s claim cadence is production-appropriate but slow for a test.
        stack.submitter.claim_poll_interval = 3.0

        # Record every job WorkResult the scheduler delivers so tests can
        # assert (handle_id, dispatch_id, order_id) of the winning result.
        routed: list = []
        inner_on_job_result = scheduler._on_job_result
        assert inner_on_job_result is not None

        async def _record_job_result(work):
            routed.append(work)
            await inner_on_job_result(work)

        scheduler._on_job_result = _record_job_result

        scheduler.start()
        controller_task = asyncio.create_task(controller.run(), name="pow-controller")
        stack_task = asyncio.create_task(stack.run(), name="mempool-stack")
        await asyncio.sleep(0)
        yield _LiveStack(
            client=client,
            keystore=keystore,
            snapshot=snapshot,
            pool=pool,
            core=core,
            handle=core.miner_handles[0],
            controller=controller,
            scheduler=scheduler,
            stack=stack,
            controller_task=controller_task,
            stack_task=stack_task,
            routed=routed,
            scheduler_records=log_handler.records,
        )
    finally:
        scheduler_logger.removeHandler(log_handler)
        scheduler_logger.setLevel(saved_level)
        root_logger.removeHandler(file_handler)
        file_handler.close()
        root_logger.setLevel(saved_root_level)
        for component in (controller, stack, scheduler):
            if component is None:
                continue
            try:
                component.shutdown()
            except Exception:  # noqa: BLE001 — teardown must reach every step
                logger.exception("shutdown() raised during teardown")
        for task in (controller_task, stack_task):
            if task is None:
                continue
            try:
                await asyncio.wait_for(task, timeout=20)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except Exception:  # noqa: BLE001 — surface, don't mask test result
                logger.exception(
                    "%s raised during teardown", task.get_name()
                )
        if scheduler is not None:
            try:
                await scheduler.stop()
            except Exception:  # noqa: BLE001
                logger.exception("scheduler.stop() raised during teardown")
        if snapshot is not None:
            # Always leave the shared chain mineable for the next test/run.
            # A FRESH client matters: the long-lived one can hold a dead
            # websocket after a multi-minute test (observed BrokenPipeError).
            try:
                restore_client = SubstrateClient(url=DEFAULT_URL)
                await restore_client.connect()
                try:
                    await _set_difficulty(
                        restore_client, snapshot.topology_hash, _RELAXED_DIFFICULTY
                    )
                finally:
                    await restore_client.close()
            except Exception:  # noqa: BLE001
                logger.exception("relaxed-difficulty restore failed in teardown")
        if pool is not None:
            try:
                # Capped: a client whose websocket died mid-test can wedge
                # its serialized executor; close must not hang the suite.
                await asyncio.wait_for(pool.shutdown(), timeout=20)
            except Exception:  # noqa: BLE001
                logger.exception("pool.shutdown() raised during teardown")
        if core is not None:
            try:
                core.close()
            except Exception:  # noqa: BLE001
                logger.exception("core.close() raised during teardown")
        try:
            await asyncio.wait_for(client.close(), timeout=20)
        except Exception:  # noqa: BLE001
            logger.exception("client.close() raised during teardown")
        # Last resort, DIRECT socket closes (bypassing each client's
        # executor): a watch whose websocket died wedges the executor
        # thread in a blocking recv; the capped async closes above queue
        # BEHIND the wedge and get abandoned, and asyncio's loop shutdown
        # (shutdown_default_executor) then waits on that thread forever —
        # hanging pytest after the test finished. Closing the socket makes
        # the blocked recv raise so the thread can exit.
        for owner in (client, getattr(stack, "build_client", None),
                      getattr(controller, "build_client", None)):
            iface = getattr(owner, "_iface", None)
            if iface is None:
                continue
            try:
                iface.close()
            except Exception:  # noqa: BLE001 — dead-socket close is routine
                pass
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _stable_busy_pow_dispatch(stack_fixture) -> int:
    """The handle's pow dispatch id if busy on the SAME pow item across two
    observations 0.5s apart, else 0 (caller retries).

    Two spaced reads rule out catching the instant of a dispatch handoff;
    the context type check rules out mistaking a mempool dispatch for pow.
    """
    handle = stack_fixture.handle
    first = handle._active_dispatch_id
    if first == 0:
        return 0
    context = stack_fixture.scheduler.dispatch_context(handle.miner_id, first)
    if not isinstance(context, SubstrateMiningContext):
        return 0
    await asyncio.sleep(0.5)
    return first if handle._active_dispatch_id == first else 0


# ----------------------------------------------------------------------
# Tests (serial; each brings its own funded keystore and fresh orders,
# and every teardown restores the relaxed difficulty)
# ----------------------------------------------------------------------


@pytest.mark.timeout(900)
async def test_job_preempts_busy_pow_and_pow_resumes_live(tmp_path):
    """THE headline: on a 1-handle node a mempool job preempts busy pow
    mining, the solution + claim land on chain, and pow fully recovers."""
    async with _live_scheduler_stack(tmp_path, "preempt") as s:
        tasks = (s.controller_task, s.stack_task)

        # Sanity: a job built from the chain snapshot hashes to the exact
        # topology the producer's eligibility filter demands.
        assert (
            topology_hash(
                s.snapshot.nodes,
                s.snapshot.edges,
                s.snapshot.allowed_h_values,
                s.snapshot.allowed_j_values,
                s.snapshot.allowed_spin_values,
            )
            == s.stack.producer.sampler_topology_hash
        )

        # (b) Relaxed difficulty: pow lands at least one proof end-to-end.
        await _wait_for(
            lambda: s.controller.stats.proofs_submitted >= 1,
            timeout=300,
            desc="first pow proof under relaxed difficulty",
            tasks=tasks,
        )

        # (c) STRICT reseed: the handle now mines a pow item it can never
        # win (the controller pushes the strict live threshold to the worker
        # on the next heads), so it stays BUSY until something preempts it.
        heads_before = s.controller.stats.heads_observed
        await _set_difficulty(s.client, s.snapshot.topology_hash, _STRICT_DIFFICULTY)
        await _wait_for(
            lambda: s.controller.stats.heads_observed >= heads_before + 2,
            timeout=60,
            desc="two heads after the strict reseed (live-threshold push)",
            tasks=tasks,
        )
        busy_id = await _wait_for(
            lambda: _stable_busy_pow_dispatch(s),
            timeout=90,
            desc="handle busy on an unwinnable pow work item",
            tasks=tasks,
            interval=0.25,
        )

        # (d) Propose a matching job. With the single handle pinned busy on
        # pow, the ONLY way the scheduler can run the job is its preemption
        # protocol — `_dispatch_to_handle` raises on a busy handle, so a job
        # result on this handle proves cancel -> done-sentinel -> dispatch
        # happened in order. `win.dispatch_id > busy_id` on the same handle
        # additionally proves the observed pow item was the victim (pow's
        # work key was frozen by the strict difficulty, so no other
        # dispatch could have replaced busy_id in between).
        contexts_before = s.scheduler.stats.contexts_dispatched
        order_ids = await _propose_job_seen_by_producer(s, tasks)

        def _job_routed():
            return any(
                isinstance(w.context, MempoolJobContext)
                and w.context.order_id in order_ids
                for w in s.routed
            )

        await _wait_for(
            _job_routed,
            timeout=180,
            desc=f"mempool job result routed (orders={order_ids})",
            tasks=tasks,
        )
        assert s.scheduler.stats.results_routed_job >= 1
        assert any(
            "dispatched mempool job" in record.getMessage()
            for record in s.scheduler_records
        ), "expected the scheduler's 'dispatched mempool job' INFO log"
        win = next(
            w
            for w in s.routed
            if isinstance(w.context, MempoolJobContext)
            and w.context.order_id in order_ids
        )
        assert win.handle_id == s.handle.miner_id
        assert win.dispatch_id > busy_id, (
            f"job dispatch id {win.dispatch_id} should postdate the busy pow "
            f"dispatch {busy_id} it preempted"
        )

        # The winning solution must land in-block (submitter bookkeeping
        # only records accepted submissions).
        await _wait_for(
            lambda: any(o in s.stack.submitter.submitted_orders for o in order_ids),
            timeout=120,
            desc="submit_solution accepted in-block",
            tasks=tasks,
        )
        assert s.stack.submitter.stats.solutions_submitted >= 1

        # (e) Pow resumes: the idle refill hands the handle a pow context
        # again after the job completes...
        def _pow_backfilled():
            active = s.handle._active_dispatch_id
            if active == 0 or active <= win.dispatch_id:
                return False
            return isinstance(
                s.scheduler.dispatch_context(s.handle.miner_id, active),
                SubstrateMiningContext,
            )

        await _wait_for(
            _pow_backfilled,
            timeout=90,
            desc="pow context re-dispatched after the job completed",
            tasks=tasks,
        )
        # At least two dispatches beyond the pre-job baseline: the job's own
        # fan-out plus the pow idle refill that followed it.
        assert s.scheduler.stats.contexts_dispatched >= contexts_before + 2

        # ...and after restoring the relaxed difficulty, proofs land again
        # (full recovery).
        proofs_before = s.controller.stats.proofs_submitted
        await _set_difficulty(s.client, s.snapshot.topology_hash, _RELAXED_DIFFICULTY)
        await _wait_for(
            lambda: s.controller.stats.proofs_submitted > proofs_before,
            timeout=300,
            desc="pow proof after the relaxed reseed (full recovery)",
            tasks=tasks,
        )

        # (f) Claim: block_wait=5 expires the order ~5 blocks after the
        # first solution; the producer routes OrderExpired to the submitter,
        # whose claim loop then wins the reward.
        await _wait_for(
            lambda: s.stack.submitter.stats.rewards_claimed >= 1,
            timeout=180,
            desc="claim_reward accepted after order expiry",
            tasks=tasks,
        )


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    os.environ.get("QUIP_T9_PARK") != "1",
    reason="opt-in (QUIP_T9_PARK=1): the usurpation storm this test "
    "deliberately generates (tipped mempool submits replacing pooled pow "
    "extrinsics on the shared signer) can kill an in-process client's "
    "watch websocket; the executor thread wedged in its blocking recv "
    "then hangs the pytest PROCESS at loop shutdown even though the "
    "test's own assertions pass. Root cause is the known in-process "
    "shared-websocket architecture (see the per-connection-process "
    "roadmap); re-enable by default once build clients are isolated.",
)
async def test_mempool_disable_parks_but_pow_continues_live(tmp_path):
    """Failure isolation: a mempool-fatal submit receipt (SolverNotRegistered
    after an out-of-band deregister) parks the mempool stack while the run
    tasks stay alive and pow keeps mining.

    VERIFIED LIVE (2026-07-03, dev chain, all fixes in): the park fires
    ~18s after the job lands — submit_solution retries the txpool nonce
    race, outranks pooled pow traffic via its tip, gets the decoded
    ``Module(error=SolverNotRegistered)`` receipt, and
    ``MEMPOOL PARKED ... pow mining continues`` follows with both run
    tasks alive and fresh pow dispatches on the handle. The opt-in guard
    exists for the process-level aftermath, not the assertions.
    """
    async with _live_scheduler_stack(tmp_path, "park") as s:
        tasks = (s.controller_task, s.stack_task)

        # Pull the registration out from under the running stack — the
        # production trigger for the SolverNotRegistered fatal class.
        receipt = await s.client.submit_extrinsic(
            "QuantumComputeMempool",
            "deregister_solver",
            {},
            s.keystore.signer,
            wait_for="inblock",
        )
        assert receipt.error is None, f"deregister_solver failed: {receipt.error}"
        assert (
            await s.client.query_solver(s.keystore.signer.account_id_bytes()) is None
        )

        # Difficulty stays RELAXED: pow keeps winning while the job path
        # runs into the fatal receipt.
        await _propose_job_seen_by_producer(s, tasks)
        await _wait_for(
            lambda: s.stack.parked,
            timeout=240,
            desc="mempool stack parked on the SolverNotRegistered receipt",
            tasks=tasks,
        )

        # The park must be contained: both run tasks alive, no process exit.
        assert not s.controller_task.done(), "pow controller task must survive"
        assert not s.stack_task.done(), "mempool stack task must survive a park"
        assert s.stack.submitter.stats.solution_errors >= 1

        # Pow is unaffected: the scheduler keeps DISPATCHING pow work and
        # the worker keeps mining it after the park. This is the T7
        # failure-isolation claim (a mempool-fatal receipt must not stop
        # pow); proof-LANDING recovery is covered end-to-end by the
        # headline test. A landed-proof assertion here would additionally
        # depend on the shared websocket surviving the tipped-usurpation
        # submit traffic this test generates — a known, separately-tracked
        # substrate-interface watch-reliability weakness, not the
        # containment property under test.
        dispatched_before = s.scheduler.stats.contexts_dispatched

        def _pow_still_mining():
            active = s.handle._active_dispatch_id
            if active == 0:
                return False
            context = s.scheduler.dispatch_context(s.handle.miner_id, active)
            return isinstance(context, SubstrateMiningContext) and (
                s.scheduler.stats.contexts_dispatched >= dispatched_before
            )

        await _wait_for(
            _pow_still_mining,
            timeout=120,
            desc="pow work item active on the handle after the mempool park",
            tasks=tasks,
        )
        # The producer is parked too: new heads stop polling mempool events.
        heads_before = s.stack.producer.stats.heads_observed
        await asyncio.sleep(15)  # ~2-3 dev blocks
        assert s.stack.producer.stats.heads_observed == heads_before
        assert not s.controller_task.done()
        assert not s.stack_task.done()


@pytest.mark.timeout(120)
async def test_one_account_one_solver_type_live(tmp_path):
    """One account, one solver type: a second child electing a different
    kind degrades to TYPE_MISMATCH (returned, not raised) and the original
    registration stays intact — the live proof that a mis-elected child
    can't crash the node or clobber the owner's registration."""
    client = SubstrateClient(url=DEFAULT_URL)
    await client.connect()
    try:
        keystore = await _fund_fresh_keystore(client, tmp_path / "solver.json")

        first = await ensure_solver_registered(client, keystore.signer, "cpu")
        assert first is SolverGuardOutcome.REGISTERED

        second = await ensure_solver_registered(client, keystore.signer, "gpu")
        assert second is SolverGuardOutcome.TYPE_MISMATCH

        info = await client.query_solver(keystore.signer.account_id_bytes())
        assert info is not None, "original registration must survive the mismatch"
        assert info.solver_type == MinerType.CPU
    finally:
        await client.close()
