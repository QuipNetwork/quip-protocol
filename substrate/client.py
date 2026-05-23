"""Async wrapper around `substrate-interface` for the quip-miner control loop.

`substrate-interface` is sync only; we wrap each blocking call in
`loop.run_in_executor` so the controller's event loop never stalls. The
methods used here are all idempotent and fast (single RPC calls), so the
default executor is fine.

What this module provides for Phase 1:
  - connect / close
  - chain head + finalized-head queries
  - `state_call` wrapper for the `QuantumPowApi_mining_snapshot` runtime API
  - storage queries for `QuantumPow.Miners`, `QuantumPow.Difficulty`,
    `System.Account` balance
  - signed-extrinsic submission with stage selection (`sent` / `inblock` /
    `finalized`)
  - new-best-head subscription (controller-facing in Phase 4)

What this module deliberately does NOT do yet:
  - SCALE-encode the `QuantumProof` payload (that's `substrate_submitter.py`)
  - own work-item dispatching (that's `substrate_miner_controller.py`)
  - cache anything across calls — each method is stateless beyond the open
    websocket

SHALLOW IFACE POLICY
--------------------
substrate-interface SCALE-decodes block headers, blocks, and events on
the receive side. When a runtime upgrade introduces a `DigestItem` (or
other) variant the bundled metadata doesn't recognise, that decode
raises `NotImplementedError` and the call (or worse, the subscription
reader thread) dies. Three iface methods accept `ignore_decoding_errors`
to suppress per-field failures and return whatever decoded successfully:

  - `subscribe_block_headers` — pump only needs the wake signal
  - `get_block_header` — we only consume `header.number`
  - `get_block` — `_fetch_extrinsic_dispatch_error` only reads
    `extrinsics[i].extrinsic_hash`

All three are called with `ignore_decoding_errors=True` in this module.
Adding new shallow-read call sites should follow the same pattern.

Storage queries (`iface.query`) and `iface.get_events` do NOT accept the
flag and MUST decode their typed value. Decode failures on those paths
indicate either a metadata-mismatch bug we want to surface immediately
or a missing pallet definition — silently swallowing them would corrupt
our model of chain state. Consumers of `get_events_at` and
`_fetch_extrinsic_dispatch_error` already match event keys defensively
(`module_id` / `pallet_name`, etc.) so a future renaming on the
chain-side surfaces as "no match" rather than a crash.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence

from scalecodec.base import ScaleBytes
from scalecodec.types import GenericExtrinsic
from scalecodec.utils.ss58 import ss58_decode
from substrateinterface import SubstrateInterface
from websocket import WebSocketException

from shared.hybrid_signer import HybridSigner
from shared.logging_config import get_logger
from shared.mempool_types import (
    IsingParams,
    JobMode,
    JobOrder,
    MempoolSolverInfo,
    MinerType,
    OrderStatus,
    OrderTiming,
    ResultDelivery,
    RewardResolution,
)
from shared.signer import Signer
from substrate.types import (
    ExtrinsicReceipt,
    MinerInfo,
    PowConstants,
    SubstrateDifficulty,
    SubstrateMiningContext,
    WinningSolution,
    WinningSolutionWithNonce,
)


logger = get_logger("substrate_client")


WaitFor = Literal["sent", "inblock", "finalized"]


@dataclass(frozen=True)
class ValidatorAttempt:
    """One entry in the connect-attempt log surfaced by NoValidatorReachable."""

    url: str
    exc_type: str
    message: str


class NoValidatorReachable(RuntimeError):
    """Raised when every URL in a SubstrateClient's validator list refuses.

    Carries the structured attempt log so callers (CLI guards, tests) can
    inspect each failure programmatically; `str(exc)` renders the operator-
    facing message.
    """

    def __init__(self, attempts: Sequence[ValidatorAttempt]) -> None:
        self.attempts = tuple(attempts)
        super().__init__(self._render(self.attempts))

    @staticmethod
    def _render(attempts: Sequence[ValidatorAttempt]) -> str:
        lines = ["no validators reachable. Attempts:"]
        for a in attempts:
            lines.append(f"  {a.url}  -> {a.exc_type}: {a.message}")
        return "\n".join(lines)


class SubstrateClient:
    """Async-friendly wrapper over a substrate websocket with failover.

    Usage:

        # Single URL (back-compat):
        client = SubstrateClient(url="ws://localhost:9944")

        # Validator list — tried in order at connect() time:
        client = SubstrateClient(urls=["ws://primary:9944", "ws://standby:9944"])

        await client.connect()
        head = await client.get_head()
        ...
        await client.close()

    The class is intentionally a thin adapter — no caching, no state machine.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        urls: Optional[Sequence[str]] = None,
        pool: Optional[Any] = None,
    ) -> None:
        # `pool` is typed as `Any` to avoid a circular import with
        # \`substrate.pool\`. Duck-typed: pool just needs an
        # `async advance_rotation(from_url: str) -> str` method.
        if url is not None and urls is not None:
            raise ValueError(
                "SubstrateClient: pass exactly one of `url=` or `urls=`, not both"
            )
        if url is None and urls is None:
            raise ValueError(
                "SubstrateClient: pass `url=<str>` or `urls=<list of str>`"
            )
        if urls is not None:
            url_list = list(urls)
            if not url_list:
                raise ValueError(
                    "SubstrateClient: `urls=` must contain at least one validator URL"
                )
        else:
            url_list = [url]  # type: ignore[list-item]
        self._urls: tuple[str, ...] = tuple(url_list)
        # `url` retained for back-compat reads — points at whichever URL is
        # currently connected (or the first one before connect()).
        self.url: str = self._urls[0]
        self.current_url: str = self._urls[0]
        # Public read-only view of the validator rotation. Controllers
        # propagate this to their separate subscription clients so both
        # clients see the same failover surface.
        self.urls: tuple[str, ...] = self._urls
        self._current_index: int = 0
        # Guard against `_run` triggering a reconnect that, while running,
        # encounters another websocket exception. Without this we could
        # recurse failover->failover->… One failover attempt per `_run`,
        # period.
        self._reentrant_failover: bool = False
        # Optional ValidatorPool. When set, `_run`'s failover handler
        # delegates the URL choice to the pool instead of running the
        # client's own circular rotation — so multiple clients sharing
        # one pool advance in lockstep on a validator death.
        self._pool: Optional[Any] = pool
        self._iface: Optional[SubstrateInterface] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # `SubstrateInterface` keeps a single websocket and isn't safe
        # against concurrent calls from different threads (its internal
        # state machine corrupts mid-decode, producing spurious
        # `No decoding class found` errors and ScaleType objects that
        # leak into the public return value). Serialize every blocking
        # call behind this lock — same fix the standalone faucet uses
        # for the same problem.
        #
        # Lazy-instantiated in `connect()` so the `asyncio.Lock` binds
        # to the running event loop (Lock captures the loop at __init__
        # time). Don't move this to `SubstrateClient.__init__` — clients
        # are typically constructed before the loop is set up.
        self._call_lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._iface is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._call_lock = asyncio.Lock()
        await self._connect_from_index(0)

    async def reconnect(self, *, target_url: Optional[str] = None) -> None:
        """Drop the current iface and rotate to the next healthy validator.

        Default walk (no `target_url`): start at `(current_index + 1)`
        and walk circularly, so a live validator gets picked even if it
        sits *behind* the dead one in the list, and a transient blip on
        the primary can still recover by wrap-around.

        Pool-driven walk (`target_url` set): start at the URL the pool
        chose. If that URL is also dead, the walk continues forward
        from there — the pool may simply have been one step behind
        reality.

        Raises `NoValidatorReachable` if every URL refuses.
        """
        await self._close_iface()
        if target_url is not None:
            try:
                start = self._urls.index(target_url)
            except ValueError:
                # Pool gave us a URL we don't know about — shouldn't
                # happen since pool was built from our URL list, but
                # be defensive: fall back to the default walk.
                start = (self._current_index + 1) % len(self._urls)
        else:
            start = (self._current_index + 1) % len(self._urls)
        await self._connect_from_index(start)

    async def _connect_from_index(self, start: int) -> None:
        attempts: list[ValidatorAttempt] = []
        n = len(self._urls)
        for offset in range(n):
            idx = (start + offset) % n
            candidate = self._urls[idx]
            try:
                # SubstrateInterface() opens the websocket eagerly in its
                # constructor. Use `_raw_run` here so a connect failure does
                # NOT trigger the failover-from-_run path; we're already
                # iterating validators ourselves.
                self._iface = await self._raw_run(
                    lambda u=candidate: SubstrateInterface(url=u)
                )
            except Exception as exc:  # noqa: BLE001 — logged + collected below
                attempts.append(
                    ValidatorAttempt(
                        url=candidate,
                        exc_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                logger.warning(
                    "substrate client could not reach %s: %s: %s",
                    candidate,
                    type(exc).__name__,
                    exc,
                )
                continue
            self.url = candidate
            self.current_url = candidate
            self._current_index = idx
            logger.info("substrate client connected: url=%s", candidate)
            return
        raise NoValidatorReachable(attempts)

    async def close(self) -> None:
        await self._close_iface()

    async def _close_iface(self) -> None:
        if self._iface is None:
            return
        iface = self._iface
        self._iface = None
        try:
            await self._raw_run(iface.close)
        except Exception as exc:  # noqa: BLE001 — dead-socket close races
            # A failing close() on a websocket that's already dropped is
            # routine during failover; don't let it mask the underlying
            # reconnect work.
            logger.debug("ignoring substrate iface close() error: %s", exc)

    # ------------------------------------------------------------------
    # Chain head queries
    # ------------------------------------------------------------------

    async def get_head(self) -> bytes:
        """Best block hash as raw bytes."""
        return bytes.fromhex(
            _strip_0x(await self._run(lambda: self._iface.get_chain_head()))
        )

    async def has_call(self, module: str, function: str) -> bool:
        """Return True iff the runtime metadata exposes ``module.function``.

        Used by the ``identify`` flow to prefer ``System.remark_with_event``
        over plain ``System.remark`` when the runtime supports it.
        substrate-interface raises ``ValueError`` from
        ``get_metadata_call_function`` when the call is absent — we treat
        any exception as "not present" so an unreachable metadata cache
        falls back rather than crashing the caller.
        """
        def _probe() -> bool:
            try:
                self._iface.get_metadata_call_function(module, function)
                return True
            except Exception:
                return False
        return await self._run(_probe)

    async def get_finalized_head(self) -> bytes:
        return bytes.fromhex(
            _strip_0x(await self._run(lambda: self._iface.get_chain_finalised_head()))
        )

    async def get_block_number(self, at: Optional[bytes] = None) -> int:
        block_hash = _hex(at) if at is not None else None
        # Shallow header read — see SHALLOW IFACE POLICY at top of module.
        # We only need `number`; opaque DigestItem variants from runtime
        # upgrades must not block us from learning the block height.
        header = await self._run(
            lambda: self._iface.get_block_header(
                block_hash=block_hash, ignore_decoding_errors=True
            )
        )
        # Some substrate-interface builds surface `number` as a hex string,
        # others as an int — accept either.
        return _coerce_block_number(header["header"]["number"])

    # ------------------------------------------------------------------
    # Runtime API: mining snapshot
    # ------------------------------------------------------------------

    async def get_mining_snapshot(
        self,
        *,
        miner_account_bytes: bytes,
        at: Optional[bytes] = None,
        topology_hash: Optional[bytes] = None,
    ) -> Optional[SubstrateMiningContext]:
        """Call `QuantumPowApi_mining_snapshot` at the given block hash.

        Returns `None` if the chain has no registered topology (e.g. fresh
        dev chain that hasn't been seeded yet — Phase 2 bootstrap handles
        that case via `Sudo.sudo(QuantumPow.register_topology(...))`).

        `miner_account_bytes` is the 32-byte AccountId of the signer; the
        snapshot itself doesn't include this but `SubstrateMiningContext`
        does. Required so the returned context is self-sufficient and so a
        missing signer fails fast rather than silently mining under a
        fabricated zero account.

        The snapshot carries a `last_proof_block_hash` (the last proof block hash —
        `block_hash(LastProofBlock)`) which is the only "time" input the
        miner needs. It's stable across an entire round, so no `at`-vs-head
        offset is required.
        """
        if len(miner_account_bytes) != 32:
            raise ValueError(
                "miner_account_bytes must be the 32-byte SCALE AccountId32, got "
                f"{len(miner_account_bytes)}"
            )
        block_hash = _hex(at) if at is not None else None
        # Encode the parameter: Option<H256>.
        if topology_hash is None:
            scale_param = "0x00"
        else:
            if len(topology_hash) != 32:
                raise ValueError(
                    f"topology_hash must be 32 bytes, got {len(topology_hash)}"
                )
            scale_param = "0x01" + topology_hash.hex()

        raw = await self._run(
            lambda: self._iface.rpc_request(
                "state_call",
                ["QuantumPowApi_mining_snapshot", scale_param, block_hash],
            )
        )
        # Distinguish RPC-level errors (transport, bad method, bad params)
        # from `Option::None` ("no topology registered yet"). Both used to
        # look like `result is None`, which silently swallowed real failures.
        if "error" in raw:
            raise RuntimeError(
                f"state_call mining_snapshot rpc error: {raw['error']}"
            )
        encoded = raw.get("result")
        if encoded is None:
            return None
        decoded = _decode_mining_snapshot(encoded)
        if decoded is None:
            return None
        return SubstrateMiningContext(
            last_proof_block_hash=decoded["last_proof_block_hash"],
            topology_hash=decoded["topology_hash"],
            nodes=decoded["nodes"],
            edges=decoded["edges"],
            difficulty=decoded["difficulty"],
            miner_account_bytes=miner_account_bytes,
            allowed_h_values=decoded["allowed_h_values"],
            allowed_j_values=decoded["allowed_j_values"],
            allowed_spin_values=decoded["allowed_spin_values"],
        )

    # ------------------------------------------------------------------
    # Storage queries
    # ------------------------------------------------------------------

    async def query_miner(self, account: bytes) -> Optional[MinerInfo]:
        """Return `QuantumPow.Miners[account]`, or `None` if not registered."""
        if len(account) != 32:
            raise ValueError(f"account must be 32 bytes, got {len(account)}")
        result = await self._run(
            lambda: self._iface.query("QuantumPow", "Miners", [account])
        )
        if result is None or result.value is None:
            return None
        v = result.value
        return MinerInfo(
            registered_at=int(v["registered_at"]),
            deposit=int(v["deposit"]),
            proofs_submitted=int(v["proofs_submitted"]),
            proofs_won=int(v["proofs_won"]),
            rewards_earned=int(v["rewards_earned"]),
        )

    async def query_difficulty(self) -> Optional[SubstrateDifficulty]:
        """Return the raw `QuantumPow.Difficulty` storage value.

        This is the *post-adjust baseline* — the value set after the most
        recent winning proof, **without** decay applied. Use
        :meth:`query_current_difficulty` for the live threshold a fresh
        proof would have to clear right now.

        Suitable for "is the chain seeded?" checks and historical
        comparisons; not suitable for deciding what difficulty a miner
        currently faces.
        """
        result = await self._run(lambda: self._iface.query("QuantumPow", "Difficulty"))
        if result is None or not _result_was_found(result) or result.value is None:
            return None
        v = result.value
        return SubstrateDifficulty(
            min_solutions=int(v["min_solutions"]),
            max_energy_milli=int(v["max_energy_milli"]),
            min_diversity_milli=int(v["min_diversity_milli"]),
        )

    async def query_current_difficulty(
        self, *, at: Optional[bytes] = None
    ) -> SubstrateDifficulty:
        """Call `QuantumPowApi_current_difficulty` for the live threshold.

        Returns the difficulty a proof submitted *right now* would have to
        clear — i.e. the post-adjust baseline (`QuantumPow.Difficulty`)
        with decay applied for elapsed blocks since the last winning proof.
        This matches what `submit_proof` validation will actually require.

        Storage queries via :meth:`query_difficulty` return the undecayed
        baseline and will under-state difficulty whenever the chain has
        been mining slowly. Use this method whenever miner-facing code
        needs to reason about the active threshold.
        """
        block_hash = _hex(at) if at is not None else None
        raw = await self._run(
            lambda: self._iface.rpc_request(
                "state_call",
                ["QuantumPowApi_current_difficulty", "0x", block_hash],
            )
        )
        if "error" in raw:
            raise RuntimeError(
                f"state_call current_difficulty rpc error: {raw['error']}"
            )
        encoded = raw.get("result")
        if encoded is None:
            raise RuntimeError(
                "state_call current_difficulty returned no result"
            )
        data = ScaleBytes(encoded)
        difficulty = _decode_difficulty_config(data)
        if data.get_remaining_length() != 0:
            raise ValueError(
                "trailing bytes after current_difficulty decode: "
                f"{data.get_remaining_length()} bytes"
            )
        return difficulty

    async def query_last_proof_block_number(self) -> int:
        """Return the ``QuantumPow.LastProofBlock`` storage value.

        The block number — not the hash — of the most recent winning
        proof. Genesis sentinel is ``0`` (no proof has ever won).

        The number is the second of the two "time" inputs to local
        decay computation (the other being ``epoch_length``). Unlike
        ``Difficulty``, this updates on every winning proof, so the
        controller re-reads it whenever ``last_proof_block_hash``
        changes in a fresh snapshot.
        """
        result = await self._run(
            lambda: self._iface.query("QuantumPow", "LastProofBlock")
        )
        if result is None or result.value is None:
            return 0
        return int(result.value)

    async def query_pow_constants(self) -> PowConstants:
        """Read the four ``pallet_quantum_pow`` constants needed for decay.

        These come from chain metadata, not storage — they're baked at
        runtime build time and only change with a runtime upgrade. The
        controller calls this once per session and caches the result;
        re-reading per head would waste an RPC.

        Returns
        -------
        PowConstants
            ``epoch_length``, ``curve_c_easy_milli``, ``curve_c_knee_milli``,
            ``curve_c_hard_milli``.
        """

        def _read() -> PowConstants:
            epoch_length = self._iface.get_constant("QuantumPow", "EpochLength")
            c_easy = self._iface.get_constant("QuantumPow", "CurveCEasyMilli")
            c_knee = self._iface.get_constant("QuantumPow", "CurveCKneeMilli")
            c_hard = self._iface.get_constant("QuantumPow", "CurveCHardMilli")
            return PowConstants(
                epoch_length=int(epoch_length.value),
                curve_c_easy_milli=int(c_easy.value),
                curve_c_knee_milli=int(c_knee.value),
                curve_c_hard_milli=int(c_hard.value),
            )

        return await self._run(_read)

    async def query_winning_solution(
        self, block_number: int, *, at: Optional[bytes] = None
    ) -> Optional[WinningSolutionWithNonce]:
        """Call `QuantumPowApi_winning_solution(block_number)`.

        Returns the persisted winning solution for ``block_number`` with
        the BLAKE3-derived nonce computed server-side. Returns ``None`` for
        blocks with no recorded winner (e.g. genesis, or pre-state-pruning
        depth on archival nodes).

        Use this in preference to decoding `QuantumPow.WinningSolutions`
        storage directly — the runtime API also packages the nonce, saving
        clients from running BLAKE3 over `(parent_hash, miner, block, salt)`.
        """
        if not 0 <= block_number < 2**32:
            raise ValueError(
                f"block_number must fit in u32, got {block_number}"
            )
        block_hash = _hex(at) if at is not None else None
        scale_param = "0x" + block_number.to_bytes(4, "little").hex()
        raw = await self._run(
            lambda: self._iface.rpc_request(
                "state_call",
                ["QuantumPowApi_winning_solution", scale_param, block_hash],
            )
        )
        if "error" in raw:
            raise RuntimeError(
                f"state_call winning_solution rpc error: {raw['error']}"
            )
        encoded = raw.get("result")
        if encoded is None:
            return None
        return _decode_winning_solution_with_nonce(encoded)

    async def query_last_proof_block_number(self) -> Optional[int]:
        """Return ``QuantumPow.LastProofBlock`` or ``None`` if unset.

        Used after a successful submit_proof to verify the runtime actually
        recorded *our* proof — paired with ``query_winning_solution`` to
        compare miner/nonce against the receipt. Storage value is a u32
        block number.
        """
        result = await self._run(
            lambda: self._iface.query("QuantumPow", "LastProofBlock")
        )
        if result is None or not _result_was_found(result) or result.value is None:
            return None
        return int(result.value)

    async def query_balance(self, account: bytes) -> int:
        if len(account) != 32:
            raise ValueError(f"account must be 32 bytes, got {len(account)}")
        result = await self._run(
            lambda: self._iface.query("System", "Account", [account])
        )
        if result is None or result.value is None:
            return 0
        return int(result.value["data"]["free"])

    # ------------------------------------------------------------------
    # QuantumComputeMempool storage queries
    # ------------------------------------------------------------------

    async def query_solver(self, account: bytes) -> Optional[MempoolSolverInfo]:
        """Return `QuantumComputeMempool.Solvers[account]`, or None."""
        if len(account) != 32:
            raise ValueError(f"account must be 32 bytes, got {len(account)}")
        result = await self._run(
            lambda: self._iface.query(
                "QuantumComputeMempool", "Solvers", [account]
            )
        )
        if result is None or result.value is None:
            return None
        v = result.value
        return MempoolSolverInfo(
            account=_decode_account_id(v["account"]),
            solver_type=MinerType.from_scale_variant(str(v["solver_type"])),
            registered_at=int(v["registered_at"]),
            solutions_submitted=int(v["solutions_submitted"]),
            rewards_earned=int(v["rewards_earned"]),
        )

    async def query_job_order(self, order_id: int) -> Optional[JobOrder]:
        """Return `QuantumComputeMempool.JobOrders[order_id]`, or None.

        The decoded `JobOrder` carries the full ising problem definition
        (nodes, edges, h_values, j_values) plus quality floors, reward,
        mode, and lifecycle status — enough to drive `mine_work_item`
        without a follow-on query.
        """
        result = await self._run(
            lambda: self._iface.query(
                "QuantumComputeMempool", "JobOrders", [order_id]
            )
        )
        if result is None or result.value is None:
            return None
        v = result.value

        # The decoded value tree mixes raw bytes (e.g. spec_id), hex strings,
        # ints, dicts (for IsingParams), and strings (for OrderStatus + the
        # JobMode/ResultDelivery/RewardResolution tagged enums).
        ising = IsingParams.from_scale_value(v["ising_params"])
        mode = _decode_job_mode(v["mode"])
        resolution = RewardResolution.from_scale_value(v["resolution"])
        delivery = _decode_result_delivery(v["delivery"])
        status = OrderStatus.from_scale_variant(str(v["status"]))
        timing = OrderTiming(
            deadline_blocks=int(v["timing"]["deadline_blocks"]),
            block_wait=int(v["timing"]["block_wait"]),
        )

        return JobOrder(
            spec_id=_decode_hash(v["spec_id"]),
            proposer=_decode_account_id(v["proposer"]),
            ising_params=ising,
            reward=int(v["reward"]),
            mode=mode,
            resolution=resolution,
            timing=timing,
            delivery=delivery,
            status=status,
            created_at=int(v["created_at"]),
            first_solution_at=(
                int(v["first_solution_at"])
                if v.get("first_solution_at") is not None
                else None
            ),
            solution_count=int(v["solution_count"]),
        )

    # ------------------------------------------------------------------
    # System events
    # ------------------------------------------------------------------

    async def get_events_at(self, block_hash: bytes) -> list[dict]:
        """Decode `System.Events` storage at the given block.

        Returns a list of event dicts shaped as
        `{"module_id": str, "event_id": str, "attributes": dict | list}`.
        Mempool-side callers filter on `module_id == "QuantumComputeMempool"`
        and dispatch by `event_id`.

        substrate-interface's `get_events` does the SCALE decode for us;
        we coerce the iterable to plain Python dicts so the caller doesn't
        have to deal with ScaleType wrappers.
        """
        if len(block_hash) != 32:
            raise ValueError(f"block_hash must be 32 bytes, got {len(block_hash)}")
        raw = await self._run(
            lambda: self._iface.get_events(block_hash="0x" + block_hash.hex())
        )
        out: list[dict] = []
        for er in raw or []:
            value = getattr(er, "value", er)
            event_field = value.get("event", value) if isinstance(value, dict) else value
            if not isinstance(event_field, dict):
                logger.debug(
                    "get_events_at: skipping non-dict event record: %s",
                    type(event_field).__name__,
                )
                continue
            module_id = event_field.get("module_id") or event_field.get("event_module")
            event_id = event_field.get("event_id") or event_field.get("event_name")
            # substrate-interface may use "attributes" (metadata v14+) or
            # "params" (older EventRecord path) depending on chain metadata
            # version. Try both before the nested fallback.
            attrs = event_field.get("attributes") or event_field.get("params")
            if attrs is None:
                inner_event = event_field.get("event")
                if isinstance(inner_event, dict):
                    attrs = inner_event.get("attributes")
            if attrs is None and module_id:
                logger.debug(
                    "get_events_at: no attributes/params for %s.%s",
                    module_id, event_id,
                )
            out.append({
                "module_id": str(module_id) if module_id is not None else "",
                "event_id": str(event_id) if event_id is not None else "",
                "attributes": attrs,
            })
        return out

    # ------------------------------------------------------------------
    # Extrinsic submission
    # ------------------------------------------------------------------

    async def submit_extrinsic(
        self,
        call_module: str,
        call_function: str,
        call_params: dict,
        signer: Signer,
        wait_for: WaitFor = "inblock",
    ) -> ExtrinsicReceipt:
        """Compose, sign, and submit an extrinsic.

        Branches on `signer.signature_kind()`:
          - Sr25519: delegates to substrate-interface's
            `create_signed_extrinsic` (works on `MultiSignature` chains).
          - Hybrid:  bypasses substrate-interface entirely because the
            chain's `Signature` type is now `HybridTxSignature`
            (composite struct), not `MultiSignature` (enum). We build the
            wire bytes manually via `_build_hybrid_signed_extrinsic`.
        """
        kind = signer.signature_kind()
        if kind == "Sr25519":
            return await self._submit_sr25519_extrinsic(
                call_module, call_function, call_params, signer, wait_for,
            )
        if kind == "Hybrid":
            return await self._submit_hybrid_extrinsic(
                call_module, call_function, call_params, signer, wait_for,
            )
        raise NotImplementedError(
            f"submit_extrinsic does not support signature_kind={kind}"
        )

    async def _submit_sr25519_extrinsic(
        self,
        call_module: str,
        call_function: str,
        call_params: dict,
        signer: Signer,
        wait_for: WaitFor,
    ) -> ExtrinsicReceipt:
        keypair = signer.keypair  # type: ignore[attr-defined]

        def _build_and_submit() -> Any:
            call = self._iface.compose_call(
                call_module=call_module,
                call_function=call_function,
                call_params=call_params,
            )
            extrinsic: GenericExtrinsic = self._iface.create_signed_extrinsic(
                call=call, keypair=keypair
            )
            return self._iface.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for == "inblock",
                wait_for_finalization=wait_for == "finalized",
            )

        receipt = await self._run(_build_and_submit)
        return _coerce_receipt(receipt)

    async def _submit_hybrid_extrinsic(
        self,
        call_module: str,
        call_function: str,
        call_params: dict,
        signer: Signer,
        wait_for: WaitFor,
    ) -> ExtrinsicReceipt:
        """Build, sign with HybridSigner, and submit a v4 extrinsic.

        substrate-interface 1.8.1 doesn't know about the new chain's
        `HybridTxSignature` envelope, `AuthorizeCall` / `WeightReclaim`
        signed extensions, or the metadata-hash mode field. We construct
        the SCALE-encoded wire bytes ourselves; submission and
        inclusion-wait both go through substrate-interface's RPC layer.
        """
        if not isinstance(signer, HybridSigner):
            raise TypeError(
                f"hybrid signing requires a HybridSigner, got {type(signer).__name__}"
            )

        def _do() -> dict:
            ext_bytes, ext_hash = _build_hybrid_signed_extrinsic(
                iface=self._iface,
                signer=signer,
                call_module=call_module,
                call_function=call_function,
                call_params=call_params,
            )
            ext_hex = "0x" + ext_bytes.hex()

            if wait_for == "sent":
                resp = self._iface.rpc_request("author_submitExtrinsic", [ext_hex])
                if "error" in resp:
                    raise RuntimeError(
                        f"author_submitExtrinsic rejected: {resp['error']}"
                    )
                return {
                    "extrinsic_hash": ext_hash,
                    "block_hash": None,
                    "finalized": False,
                    "result": resp.get("result"),
                }

            # Inclusion/finalization: use the subscription RPC with a
            # result_handler that pulls the next status update. Mirrors
            # `substrate_interface.SubstrateInterface.submit_extrinsic` so the
            # error-handling path is well-trodden.
            def _result_handler(message, update_nr, subscription_id):  # noqa: ARG001
                params = message.get("params") or {}
                result = params.get("result")
                if isinstance(result, dict):
                    res = {k.lower(): v for k, v in result.items()}
                    if "finalized" in res and wait_for == "finalized":
                        self._iface.rpc_request("author_unwatchExtrinsic", [subscription_id])
                        return {
                            "extrinsic_hash": ext_hash,
                            "block_hash": res["finalized"],
                            "finalized": True,
                        }
                    if "inblock" in res and wait_for == "inblock":
                        self._iface.rpc_request("author_unwatchExtrinsic", [subscription_id])
                        return {
                            "extrinsic_hash": ext_hash,
                            "block_hash": res["inblock"],
                            "finalized": False,
                        }
                elif isinstance(result, str) and result in _HYBRID_TERMINAL_FAILURES:
                    # `dropped` / `invalid` / `usurped` / `retracted` /
                    # `finalityTimeout` are all terminal — without raising
                    # here the subscription would silently keep waiting and
                    # callers would hang forever.
                    self._iface.rpc_request("author_unwatchExtrinsic", [subscription_id])
                    raise ValueError(f"hybrid extrinsic rejected: {result}")
                return None  # non-terminal status — keep waiting

            response = self._iface.rpc_request(
                "author_submitAndWatchExtrinsic",
                [ext_hex],
                result_handler=_result_handler,
            )
            return response

        result = await self._run(_do)
        block_hash = result.get("block_hash")
        error_msg = None
        if block_hash:
            # author_submitAndWatchExtrinsic only reports that the extrinsic
            # made it into a block — a `System.ExtrinsicFailed` event still
            # means the chain rejected the dispatch. Without this fetch
            # callers see `is_success=True` for a chain-rejected proof and
            # silently treat it as accepted. Fetch the block's events and
            # surface any ExtrinsicFailed matching our hash.
            try:
                error_msg = await self._run(
                    lambda: _fetch_extrinsic_dispatch_error(
                        self._iface,
                        block_hash=block_hash,
                        ext_hash=result["extrinsic_hash"],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # Event fetch is best-effort — surface the inability to
                # classify as a non-fatal error rather than letting the
                # extrinsic look fully successful.
                error_msg = f"event fetch failed: {exc}"
                logger.warning("hybrid extrinsic event fetch failed: %s", exc)
        return ExtrinsicReceipt(
            extrinsic_hash=str(result.get("extrinsic_hash") or result.get("result") or ""),
            block_hash=str(block_hash or "") or None,
            is_finalized=bool(result.get("finalized", False)),
            error=error_msg,
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def subscribe_new_heads(
        self,
        callback: Callable[[bytes, int], Awaitable[None]],
        finalized_only: bool = False,
    ) -> None:
        """Block until cancelled, invoking `callback(block_hash, block_number)`
        from a single pump task tied to the *current* best head.

        Pump model (vs. the old per-header dispatch): the websocket reader
        thread only signals — it never schedules per-number lookups. A
        single asyncio task waits on the signal, queries ``get_chain_head``
        (or ``get_chain_finalised_head``) once per wake, fetches the block
        number, and invokes the callback. If more headers arrived during
        the work, the signal is set again and the next iteration starts
        immediately. That coalesces a burst of N headers into one (or two)
        callback invocations driven by *current* chain state.

        Why this matters: the old model spawned one
        ``run_coroutine_threadsafe(_handle_head(number))`` per header and
        each coroutine did its own ``get_block_hash(block_id=number)``
        through the serialised ``_run`` lock. Under RPC load, those
        per-number lookups completed out of order and overwrote the
        controller's "latest head" slot with *historical* hashes —
        ``mining_snapshot(at=stale_hash)`` then correctly returns a stale
        ``last_proof_block_hash`` and the miner spins on an already-won
        round. See incident 2026-05-21 in the v0.2 branch history.

        ``substrate-interface`` invokes the subscribe dispatch on its
        websocket reader thread. We do **not** make synchronous RPC calls
        from that thread (that would deadlock the reader waiting on its
        own response); the reader's only job here is
        ``loop.call_soon_threadsafe(pump_event.set)``.

        The subscribe call itself bypasses ``_call_lock``: substrate-interface
        keeps the websocket in receive mode for the subscription's lifetime,
        which would otherwise hold the lock forever and deadlock every other
        ``_run`` caller. Callers that need to share a client across submissions
        and subscriptions should pass a dedicated ``subscription_client`` to
        the controller (the controller already enforces this).
        """
        loop = asyncio.get_running_loop()
        pump_event = asyncio.Event()

        def _dispatch(obj, update_nr: int, subscription_id: str):  # noqa: ARG001
            # Reader-thread side: signal only. No RPC, no per-header
            # bookkeeping. We don't even look at the header payload — the
            # pump fetches the current best head on its own, so any header
            # at all is enough to know "something changed".
            try:
                loop.call_soon_threadsafe(pump_event.set)
            except RuntimeError:
                # Loop is closed (controller is tearing down). Swallow —
                # the pump task will be cancelled on the same path.
                pass

        async def _pump() -> None:
            head_fn_name = (
                "get_chain_finalised_head" if finalized_only else "get_chain_head"
            )
            while True:
                await pump_event.wait()
                pump_event.clear()

                # If close() or reconnect() cleared `_iface` between the
                # signal arriving and us waking up, swallow this wake.
                # The next reconnect will install a new iface and the
                # subscription will resume.
                iface = self._iface
                if iface is None:
                    logger.debug(
                        "subscribe_new_heads pump: iface is None — skipping"
                    )
                    continue

                # Catch broadly: anything that escapes here would silently
                # kill the pump task while the executor-side subscribe
                # keeps running and setting `pump_event`. That deadlocks
                # the controller (head_signal never fires) without any
                # log or stats counter. Sources of non-connection
                # exceptions include: header SCALE shape drift after a
                # runtime upgrade (KeyError on `header["header"]["number"]`),
                # `_iface` cleared mid-call (AttributeError), and
                # exceptions raised by `callback`. `_run` already handles
                # WebSocketException/ConnectionError via failover; any
                # exception that propagates here means failover gave up
                # (NoValidatorReachable) or the failure is non-transport.
                # Log and continue — the next subscribe callback will
                # re-arm pump_event and we'll retry against the latest
                # chain state.
                try:
                    head_hex = await self._run(
                        lambda: getattr(self._iface, head_fn_name)()
                    )
                    head_bytes = bytes.fromhex(_strip_0x(head_hex))
                    # Same DigestItem-resilience as the subscribe call: a
                    # header we can't fully SCALE-decode is still useful —
                    # we only need the `number` field.
                    header = await self._run(
                        lambda: self._iface.get_block_header(
                            block_hash=head_hex, ignore_decoding_errors=True
                        )
                    )
                    number = _coerce_block_number(header["header"]["number"])
                    await callback(head_bytes, number)
                except asyncio.CancelledError:
                    raise
                except NoValidatorReachable:
                    # Failover gave up — keep retrying on pump_event is
                    # pointless because the subscription is also dead.
                    # Propagate so the outer `_subscribe_heads` (or the
                    # controller wrapping it) sees the exhausted-failover
                    # state and escalates instead of silently going inert.
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "subscribe_new_heads pump iteration failed "
                        "(%s: %s); will retry on next chain head",
                        type(exc).__name__,
                        exc,
                    )

        # Prime the pump so the first head is delivered without waiting
        # for the next chain notification. Important on startup against a
        # quiet chain.
        pump_event.set()
        pump_task = asyncio.create_task(_pump(), name="subscribe-new-heads-pump")

        try:
            # Bypass `_call_lock` for the long-lived subscribe — see docstring.
            # `ignore_decoding_errors=True` keeps a header that
            # substrate-interface can't fully decode (e.g. an unknown
            # `DigestItem` variant after a runtime upgrade) from killing
            # the whole subscription. The pump doesn't read the header
            # payload anyway — it only needs the wake signal — so we
            # don't lose information by tolerating these decode failures.
            await loop.run_in_executor(
                None,
                lambda: self._iface.subscribe_block_headers(
                    _dispatch,
                    finalized_only=finalized_only,
                    ignore_decoding_errors=True,
                ),
            )
        finally:
            pump_task.cancel()
            pump_exc: Optional[BaseException] = None
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
            except NoValidatorReachable as exc:
                # Failover exhausted inside the pump. Capture and
                # re-raise after the finally completes so the outer
                # caller (controller's `_subscribe_heads`) sees the
                # exhausted-failover signal and shuts down instead of
                # waiting on a dead subscription. Without this, the
                # `except Exception` branch below would log + swallow.
                pump_exc = exc
            except Exception:  # noqa: BLE001
                logger.exception("subscribe_new_heads pump task crashed")
            if pump_exc is not None:
                raise pump_exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run(self, fn):
        """Run a blocking substrate-interface call with failover-on-loss.

        Almost all callers go through this method, so the `_call_lock`
        taken inside `_raw_run` guarantees serial access to the underlying
        `SubstrateInterface`. The exception is `subscribe_new_heads`,
        which intentionally bypasses the lock — see its docstring.

        Failover semantics: a `WebSocketException` or `ConnectionError`
        from the call body triggers one `reconnect()` attempt, then the
        *original* exception is re-raised so the caller's retry logic
        runs. The bound-method lambdas in this file capture
        `self._iface` lazily, so a follow-up call after failover hits
        the fresh iface. If `reconnect()` itself can't find a healthy
        validator, `NoValidatorReachable` propagates instead.
        """
        try:
            return await self._raw_run(fn)
        except (WebSocketException, ConnectionError) as exc:
            if self._reentrant_failover:
                # We're already inside a failover attempt — bubble up the
                # raw error so the outer failover handler sees it.
                raise
            logger.warning(
                "substrate _run failed on %s (%s: %s); attempting failover",
                self.current_url,
                type(exc).__name__,
                exc,
            )
            self._reentrant_failover = True
            try:
                # May raise NoValidatorReachable — that's the intended
                # signal upstream. The original exception is shadowed in
                # that case; operators get the structured attempt log.
                if self._pool is not None:
                    # Pool-driven rotation: ask the pool which URL to
                    # move to. The pool coordinates across all slots so
                    # exactly one rotation event fires per validator
                    # death even if multiple slots noticed.
                    target = await self._pool.advance_rotation(
                        from_url=self.current_url
                    )
                    await self.reconnect(target_url=target)
                else:
                    await self.reconnect()
            finally:
                self._reentrant_failover = False
            # Reconnect succeeded. The current call is still lost — the
            # underlying lambda was bound to the dead iface, and we can't
            # safely retry it in general (e.g. a partially-submitted
            # extrinsic must not be resubmitted blindly). Surface the
            # original exception so the caller's retry path decides.
            raise

    async def _raw_run(self, fn):
        """Lock-guarded executor dispatch without failover plumbing.

        Used internally by `_run` (which wraps this with failover) and by
        `connect()`/`reconnect()` (which are themselves the failover loop
        and must not recurse into it).

        When `_call_lock` is `None` we're before `connect()` finishes,
        in which case `connect()` itself is the only caller (single-
        threaded setup) and the lock would be redundant.
        """
        loop = self._loop or asyncio.get_running_loop()
        if self._call_lock is None:
            return await loop.run_in_executor(None, fn)
        async with self._call_lock:
            return await loop.run_in_executor(None, fn)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


# Terminal transaction-pool states returned by `author_submitAndWatchExtrinsic`
# that indicate the extrinsic will never finalize. Without enumerating these
# the subscription handler silently falls through and the caller hangs.
# Source: substrate-interface `pool::watch::TransactionStatus` variants
# (`dropped`, `invalid`, `usurped`, `retracted`, `finalityTimeout`).
_HYBRID_TERMINAL_FAILURES: frozenset = frozenset(
    {"dropped", "invalid", "usurped", "retracted", "finalitytimeout"}
)


def _canonical_hex(value: Any) -> Optional[str]:
    """Lower-case hex (no 0x) for `bytes`/`bytearray`/`str`, or None."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, str):
        return _strip_0x(value).lower()
    return None


def _phase_extrinsic_idx(phase: Any) -> Optional[int]:
    """Extract the extrinsic index from a SCALE-decoded event phase.

    substrate-interface surfaces ``phase`` in several shapes depending on
    decoder version / event source:

    - ``{"ApplyExtrinsic": 1}`` — common
    - ``{"ApplyExtrinsic": {"extrinsic_idx": 1}}`` — newer typed variant
    - bare int / numeric str — some flattened decoders
    """
    if isinstance(phase, dict):
        applied = phase.get("ApplyExtrinsic")
        if applied is None:
            return None
        if isinstance(applied, dict):
            # Use explicit None checks — `0` is a legal extrinsic_idx
            # and `or` would fall through to the second lookup.
            idx = applied.get("extrinsic_idx")
            if idx is None:
                idx = applied.get("index")
            try:
                return int(idx) if idx is not None else None
            except (TypeError, ValueError):
                return None
        try:
            return int(applied)
        except (TypeError, ValueError):
            return None
    if isinstance(phase, (int, str)):
        try:
            return int(phase)
        except (TypeError, ValueError):
            return None
    return None


def _fetch_extrinsic_dispatch_error(
    iface: SubstrateInterface, *, block_hash: str, ext_hash: str
) -> Optional[str]:
    """Return a `Module(error=...)` string if the chain rejected dispatch.

    `author_submitAndWatchExtrinsic` only reports "this hit a block" — but
    the runtime may still emit `System.ExtrinsicFailed` for that extrinsic
    index. The sr25519 path inspects this via substrate-interface's typed
    `ExtrinsicReceipt`; the hybrid path bypasses that, so we query the
    block's events directly and stringify any `ExtrinsicFailed` matching
    our extrinsic.

    Returns:
      - ``None`` on confirmed dispatch success (either ExtrinsicSuccess
        seen, or no failure event found *and* the extrinsic was located).
      - A non-empty string when dispatch failed OR when we could not
        locate the extrinsic in the block at all. The "not found" path
        used to silently return None, masking acceptance failures —
        callers treat any truthy string as "do not mark this as won".
    """
    # Shallow block read — see SHALLOW IFACE POLICY at top of module.
    # We only need `extrinsics[i].extrinsic_hash`; if a digest item or
    # author field doesn't decode (e.g. runtime upgrade), the extrinsics
    # vec is still intact and that's all we look at.
    try:
        block = iface.get_block(
            block_hash=block_hash,
            include_author=False,
            ignore_decoding_errors=True,
        )
    except NotImplementedError as exc:
        # Belt-and-suspenders: even with `ignore_decoding_errors=True`,
        # some substrate-interface versions raise on unknown types
        # during outer-frame decoding (vs inner-field decoding). Surface
        # as a non-OK receipt rather than crashing into the controller.
        return f"unclassified: get_block decode failure ({exc})"
    if block is None:
        return f"unclassified: get_block returned None for {block_hash}"
    target_hash = _canonical_hex(ext_hash)
    if target_hash is None:
        return (
            "unclassified: target extrinsic_hash has unsupported shape: "
            f"{type(ext_hash).__name__}"
        )
    extrinsics = block.get("extrinsics") or []
    ext_idx: Optional[int] = None
    for idx, ext in enumerate(extrinsics):
        eh = getattr(ext, "extrinsic_hash", None)
        if eh is None and isinstance(ext, dict):
            eh = ext.get("extrinsic_hash")
        canon = _canonical_hex(eh)
        if canon is not None and canon == target_hash:
            ext_idx = idx
            break
    if ext_idx is None:
        # Loud non-success — see docstring. The proof receipt looked OK
        # but we cannot prove the runtime applied *our* extrinsic in
        # this block, so callers must NOT close the work key.
        return (
            f"unclassified: extrinsic {target_hash[:16]}… not found in "
            f"block {_strip_0x(block_hash)[:16]}… "
            f"({len(extrinsics)} extrinsics)"
        )
    events = iface.get_events(block_hash=block_hash) or []
    for ev in events:
        v = ev.value if hasattr(ev, "value") else (ev if isinstance(ev, dict) else None)
        if not isinstance(v, dict):
            continue
        applied_idx = _phase_extrinsic_idx(v.get("phase") or {})
        if applied_idx != ext_idx:
            continue
        event = v.get("event") or v
        module_id = (
            event.get("module_id")
            or event.get("pallet")
            or event.get("pallet_name")
        )
        event_id = (
            event.get("event_id")
            or event.get("variant")
            or event.get("name")
            or event.get("event_name")
        )
        if module_id == "System" and event_id == "ExtrinsicFailed":
            attrs = event.get("attributes") or event.get("fields") or {}
            return f"Module(ExtrinsicFailed, attrs={attrs!r})"
        if module_id == "System" and event_id == "ExtrinsicSuccess":
            # Authoritative success for this extrinsic — short-circuit.
            return None
    return None


def _strip_0x(s: str) -> str:
    return s[2:] if s.startswith("0x") else s


def _decode_hash(value) -> bytes:
    """Decode a substrate Hash storage field to 32 raw bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return bytes.fromhex(_strip_0x(value))
    raise ValueError(f"unrecognized Hash shape: {value!r}")


def _decode_account_id(value) -> bytes:
    """Decode an `AccountId32` storage field to 32 raw bytes.

    substrate-interface may surface accounts as raw bytes, hex strings, or
    SS58 strings depending on the field's typedef. We canonicalize to raw
    bytes so callers compare with `signer.account_id_bytes()` cleanly.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        if value.startswith("0x"):
            return bytes.fromhex(value[2:])
        # SS58 — decode via scalecodec's helper.
        return bytes.fromhex(ss58_decode(value))
    raise ValueError(f"unrecognized AccountId shape: {value!r}")


def _decode_job_mode(value) -> JobMode:
    """Decode a `JobMode` tagged-enum storage value."""
    if isinstance(value, str):
        # Bare-string SCALE encoding for the no-field variant.
        if value == "Open":
            return JobMode.open()
        raise ValueError(f"unrecognized JobMode variant: {value!r}")
    if isinstance(value, dict) and len(value) == 1:
        (tag, inner), = value.items()
        if tag == "Open":
            return JobMode.open()
        if tag == "Bid":
            miners_raw = inner.get("miners") if isinstance(inner, dict) else None
            types_raw = inner.get("miner_types") if isinstance(inner, dict) else None
            miners = (
                tuple(_decode_account_id(a) for a in miners_raw)
                if miners_raw is not None
                else None
            )
            mtypes = (
                tuple(MinerType.from_scale_variant(str(mt)) for mt in types_raw)
                if types_raw is not None
                else None
            )
            return JobMode.bid(miners=miners, miner_types=mtypes)
    raise ValueError(f"unrecognized JobMode shape: {value!r}")


def _decode_result_delivery(value) -> ResultDelivery:
    """Decode a `ResultDelivery` tagged-enum storage value."""
    if isinstance(value, str):
        if value == "OnChainOnly":
            return ResultDelivery.on_chain_only()
        raise ValueError(f"unrecognized ResultDelivery variant: {value!r}")
    if isinstance(value, dict) and len(value) == 1:
        (tag, inner), = value.items()
        if tag == "OnChainOnly":
            return ResultDelivery.on_chain_only()
        endpoint = inner.get("endpoint") if isinstance(inner, dict) else inner
        if isinstance(endpoint, str):
            endpoint = bytes.fromhex(_strip_0x(endpoint))
        elif isinstance(endpoint, list):
            endpoint = bytes(endpoint)
        if endpoint is None:
            raise ValueError(
                f"_decode_result_delivery: {tag!r} requires a non-None endpoint"
            )
        if tag == "Callback":
            return ResultDelivery.callback(endpoint)
        if tag == "CallbackWithPoll":
            return ResultDelivery.callback_with_poll(endpoint)
    raise ValueError(f"unrecognized ResultDelivery shape: {value!r}")


def _encode_compact_u32(n: int) -> bytes:
    """SCALE compact encoding for u32. Mirrors substrate's `Compact<u32>`."""
    if n < 0:
        raise ValueError(f"compact u32 must be non-negative, got {n}")
    if n < 0x40:
        return bytes([n << 2])
    if n < 0x4000:
        return ((n << 2) | 0b01).to_bytes(2, "little")
    if n < 0x4000_0000:
        return ((n << 2) | 0b10).to_bytes(4, "little")
    raise NotImplementedError("compact u32 big-int mode not needed here")


def _encode_compact_u128(n: int) -> bytes:
    """SCALE compact encoding for u128 — the tip field is `Compact<Balance>`
    and balances are u128. For values up to 2^30 the layout matches u32."""
    if n < 0:
        raise ValueError(f"compact must be non-negative, got {n}")
    if n < 0x40:
        return bytes([n << 2])
    if n < 0x4000:
        return ((n << 2) | 0b01).to_bytes(2, "little")
    if n < 0x4000_0000:
        return ((n << 2) | 0b10).to_bytes(4, "little")
    # Big-int mode: top 6 bits of first byte encode `(n_bytes - 4)`, low 2
    # bits are 0b11. Then little-endian bytes of the value.
    raw = n.to_bytes((n.bit_length() + 7) // 8, "little")
    # SCALE compact big-int mode caps at 67 bytes: the top 6 bits of the
    # mode byte encode `n_bytes - 4`, so max n_bytes = (0xff >> 2) + 4 = 67.
    if len(raw) > 67:
        raise OverflowError(
            f"compact value needs {len(raw)} bytes, exceeds 67-byte SCALE limit"
        )
    return bytes([((len(raw) - 4) << 2) | 0b11]) + raw


def _build_hybrid_signed_extrinsic(
    *,
    iface,
    signer,
    call_module: str,
    call_function: str,
    call_params: dict,
) -> tuple[bytes, str]:
    """Construct a hybrid-signed extrinsic byte-by-byte.

    Returns (wire_bytes, extrinsic_hash_hex). Bypasses substrate-interface's
    `create_signed_extrinsic` because that path assumes the chain uses
    `MultiSignature` — the hybrid chain uses `HybridTxSignature`, which is
    a composite struct of (public, signature) rather than a tagged enum.

    Layout follows Substrate's signed v4 extrinsic format:

        compact_len(body) || body

        body = (version_byte | 0x80) ||
               MultiAddress::Id || AccountId32(32 bytes) ||
               HybridTxSignature(3828 bytes: public[1344] || signature[2484]) ||
               extra(signed-extension extras in metadata order) ||
               call(SCALE-encoded call bytes)

    The signing payload is `call || extra || additional`, blake2_256-hashed
    when > 256 bytes. The signer applies the hybrid domain prefix on top of
    that — see `HybridSigner.sign` / `prepare_message`.
    """
    # 1. Compose the call bytes via substrate-interface (call shape doesn't
    #    depend on the signer; we just want the SCALE-encoded body).
    call = iface.compose_call(
        call_module=call_module,
        call_function=call_function,
        call_params=call_params,
    )
    raw_call = call.data.data if hasattr(call.data, "data") else call.data
    if hasattr(raw_call, "tobytes"):
        call_bytes = bytes(raw_call)
    elif isinstance(raw_call, str):
        call_bytes = bytes.fromhex(_strip_0x(raw_call))
    else:
        call_bytes = bytes(raw_call)

    # 2. Fetch chain state needed for the signed extensions.
    account = signer.account_id_bytes()
    nonce = iface.get_account_nonce(account_address="0x" + account.hex())
    genesis_hex = iface.get_block_hash(block_id=0)
    rv = iface.rpc_request("state_getRuntimeVersion", [])["result"]
    spec_version = int(rv["specVersion"])
    tx_version = int(rv["transactionVersion"])
    genesis_bytes = bytes.fromhex(_strip_0x(genesis_hex))

    # 3. Signed-extension extras, in metadata order. Empty composites
    #    (AuthorizeCall / CheckNonZeroSender / CheckSpecVersion / CheckTxVersion /
    #    CheckGenesis / CheckWeight / WeightReclaim) encode to 0 bytes.
    extra = (
        b""                                  # AuthorizeCall
        + b""                                # CheckNonZeroSender
        + b""                                # CheckSpecVersion
        + b""                                # CheckTxVersion
        + b""                                # CheckGenesis
        + b"\x00"                            # CheckMortality: Era::immortal
        + _encode_compact_u32(int(nonce))    # CheckNonce
        + b""                                # CheckWeight
        + _encode_compact_u128(0)            # ChargeTransactionPayment tip=0
        + b"\x00"                            # CheckMetadataHash: Mode::Disabled
        + b""                                # WeightReclaim
    )

    # 4. Signed-extension additional_signed, in metadata order. CheckMortality
    #    with an immortal era uses the genesis hash here.
    additional = (
        b""                                  # AuthorizeCall
        + b""                                # CheckNonZeroSender
        + spec_version.to_bytes(4, "little") # CheckSpecVersion
        + tx_version.to_bytes(4, "little")   # CheckTxVersion
        + genesis_bytes                      # CheckGenesis
        + genesis_bytes                      # CheckMortality (immortal -> genesis)
        + b""                                # CheckNonce
        + b""                                # CheckWeight
        + b""                                # ChargeTransactionPayment
        + b"\x00"                            # CheckMetadataHash: Option::None
        + b""                                # WeightReclaim
    )

    # 5. Sign payload = call || extra || additional. Blake2_256 if > 256 bytes
    #    per Substrate's SignaturePayload::using_encoded convention.
    payload = call_bytes + extra + additional
    payload_to_sign = (
        hashlib.blake2b(payload, digest_size=32).digest()
        if len(payload) > 256
        else payload
    )
    signature_bytes = signer.sign(payload_to_sign)

    # 6. SCALE-encode HybridTxSignature = public(1344) || signature(2484).
    hybrid_sig_scale = signer.public_bytes() + signature_bytes

    # 7. Assemble the wire body and length-prefix the whole extrinsic.
    body = (
        bytes([0x84])                        # v4 | 0x80 signed flag
        + b"\x00"                            # MultiAddress::Id discriminator
        + account                            # AccountId32 (32 bytes)
        + hybrid_sig_scale
        + extra
        + call_bytes
    )
    full_extrinsic = _encode_compact_u32(len(body)) + body
    ext_hash = "0x" + hashlib.blake2b(full_extrinsic, digest_size=32).digest().hex()
    return full_extrinsic, ext_hash


def _result_was_found(result) -> bool:
    """Whether a substrate-interface storage query actually hit data.

    Substrate's `StorageValue<_, T>` returns `T::default()` instead of `None`
    when storage is empty (an OptionQuery quirk in substrate-interface's
    decoder). The `meta_info` dict carries an explicit `result_found` flag
    which is the only reliable way to distinguish empty-with-default from
    actually-stored. Returns True if the field is missing — that path is
    only taken for query types where the result is unambiguously decoded.
    """
    meta = getattr(result, "meta_info", None) or {}
    return meta.get("result_found", True)


def _hex(b: bytes) -> str:
    return "0x" + b.hex()


def _read_exact(data: ScaleBytes, n: int) -> bytes:
    """Read `n` bytes from the SCALE buffer, raising on short reads.

    ``ScaleBytes.get_next_bytes`` silently returns a partial slice when the
    underlying buffer is exhausted (and bumps the offset past the end), so
    every subsequent read sees an empty slice. The downstream effect is
    that decode errors surface several fields *after* the actual
    truncation. Surfacing the short read at the field where it happened
    makes error messages diagnostic instead of misleading.
    """
    chunk = data.get_next_bytes(n)
    if len(chunk) != n:
        raise ValueError(
            f"short read: wanted {n} bytes, got {len(chunk)}"
        )
    return bytes(chunk)


def _decode_u32(data: ScaleBytes) -> int:
    return int.from_bytes(_read_exact(data, 4), "little")


def _decode_i32(data: ScaleBytes) -> int:
    return int.from_bytes(_read_exact(data, 4), "little", signed=True)


def _decode_i64(data: ScaleBytes) -> int:
    return int.from_bytes(_read_exact(data, 8), "little", signed=True)


def _decode_u128(data: ScaleBytes) -> int:
    return int.from_bytes(_read_exact(data, 16), "little")


def _decode_u256_le(data: ScaleBytes) -> bytes:
    """Read a 32-byte little-endian ``U256`` and return the raw bytes.

    The Rust pallet returns `nonce: U256` from BLAKE3, which `parity-scale-codec`
    serialises little-endian. The Python side treats nonces as opaque 32-byte
    blobs (see ``MiningResult.nonce`` and ``derive_nonce``) — the *byte order*
    used by miners/validators is the BLAKE3 digest order (big-endian by the
    `blake3` library convention), so we re-reverse on the wire to recover it.
    """
    raw_le = _read_exact(data, 32)
    return bytes(reversed(raw_le))


def _decode_difficulty_config(data: ScaleBytes) -> "SubstrateDifficulty":
    min_solutions = _decode_field("min_solutions", data, _decode_u32)
    max_energy_milli = _decode_field("max_energy_milli", data, _decode_i64)
    min_diversity_milli = _decode_field("min_diversity_milli", data, _decode_u32)
    return SubstrateDifficulty(
        min_solutions=min_solutions,
        max_energy_milli=max_energy_milli,
        min_diversity_milli=min_diversity_milli,
    )


def _decode_allowed_value_spec(data: ScaleBytes):
    """Decode a SCALE-encoded ``AllowedValueSpec<BoundedVec<i32>>``.

    Variant tag byte (0 = Set, 1 = IntegerRange, 2 = ContinuousRange)
    followed by the variant-specific payload. The pallet's bounded vec
    encodes as a plain ``Vec<i32>`` on the wire (the bound is enforced
    by the encoder, not represented in the bytes).
    """
    from shared.allowed_value_spec import (
        AllowedValueContinuousRange,
        AllowedValueIntegerRange,
        AllowedValueSet,
    )

    tag = _read_exact(data, 1)[0]
    if tag == 0:
        length = _decode_compact_u32(data)
        values = tuple(_decode_i32(data) for _ in range(length))
        return AllowedValueSet(values)
    if tag == 1:
        return AllowedValueIntegerRange(min=_decode_i32(data), max=_decode_i32(data))
    if tag == 2:
        return AllowedValueContinuousRange(min=_decode_i32(data), max=_decode_i32(data))
    raise ValueError(f"unknown AllowedValueSpec variant tag: {tag}")


def _decode_compact_u32(data: ScaleBytes) -> int:
    """Decode a SCALE compact-encoded u32 length prefix.

    Mode is encoded in the low 2 bits of the first byte:
      0b00 → single-byte (value in upper 6 bits)
      0b01 → two-byte
      0b10 → four-byte
      0b11 → big-integer mode (rejected here: u32 length prefixes never need
             more than 4 bytes, and a malformed/malicious payload claiming
             mode 0b11 could otherwise drive an OOM allocation downstream).
    """
    first = _read_exact(data, 1)[0]
    mode = first & 0b11
    if mode == 0:
        return first >> 2
    if mode == 1:
        return ((first >> 2) | (_read_exact(data, 1)[0] << 6))
    if mode == 2:
        rest = _read_exact(data, 3)
        return (first >> 2) | (rest[0] << 6) | (rest[1] << 14) | (rest[2] << 22)
    raise ValueError("compact big-integer mode not valid for u32 length prefix")


def _decode_mining_snapshot(encoded_hex: str) -> Optional[dict]:
    """Decode SCALE ``Option<MiningSnapshot<...>>`` from the runtime API.

    Layout:
      - 1 byte option tag (0x00 = None, 0x01 = Some)
      - last_proof_block_hash: H256 (32 bytes) — `block_hash(LastProofBlock)`
      - difficulty: DifficultyConfig {min_solutions: u32,
            max_energy_milli: i64, min_diversity_milli: u32}
      - topology_hash: H256 (32 bytes)
      - nodes: Vec<u32>
      - edges: Vec<(u32, u32)>
      - allowed_h_values: AllowedValueSpec<BoundedVec<i32>>
      - allowed_j_values: AllowedValueSpec<BoundedVec<i32>>
      - allowed_spin_values: AllowedValueSpec<BoundedVec<i32>>

    Decode failures are re-raised with the failing field name so a runtime
    API shape change (or truncated transport) lands a usable error. Trailing
    bytes raise — a future runtime upgrade that appends a field would
    otherwise silently drop it.
    """
    data = ScaleBytes(encoded_hex)
    try:
        tag = _read_exact(data, 1)
        if tag[0] == 0:
            if data.get_remaining_length() != 0:
                raise ValueError(
                    f"trailing bytes after Option::None tag: "
                    f"{data.get_remaining_length()} bytes"
                )
            return None
        last_proof_block_hash = _decode_field(
            "last_proof_block_hash", data, lambda d: _read_exact(d, 32)
        )
        difficulty = _decode_difficulty_config(data)
        topology_hash = _decode_field("topology_hash", data, lambda d: _read_exact(d, 32))
        nodes_len = _decode_field("nodes_len", data, _decode_compact_u32)
        nodes = [_decode_field("nodes[%d]" % i, data, _decode_u32)
                 for i in range(nodes_len)]
        edges_len = _decode_field("edges_len", data, _decode_compact_u32)
        edges = [
            (
                _decode_field("edges[%d].0" % i, data, _decode_u32),
                _decode_field("edges[%d].1" % i, data, _decode_u32),
            )
            for i in range(edges_len)
        ]
        allowed_h = _decode_field("allowed_h_values", data, _decode_allowed_value_spec)
        allowed_j = _decode_field("allowed_j_values", data, _decode_allowed_value_spec)
        allowed_spin = _decode_field(
            "allowed_spin_values", data, _decode_allowed_value_spec
        )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 — rewrap with context
        raise ValueError(f"failed to decode mining snapshot: {exc}") from exc

    if data.get_remaining_length() != 0:
        raise ValueError(
            f"trailing bytes after mining snapshot decode: "
            f"{data.get_remaining_length()} bytes; runtime API shape likely "
            "changed"
        )
    return {
        "last_proof_block_hash": last_proof_block_hash,
        "difficulty": difficulty,
        "topology_hash": topology_hash,
        "nodes": nodes,
        "edges": edges,
        "allowed_h_values": allowed_h,
        "allowed_j_values": allowed_j,
        "allowed_spin_values": allowed_spin,
    }


def _decode_winning_solution_with_nonce(
    encoded_hex: str,
) -> Optional["WinningSolutionWithNonce"]:
    """Decode SCALE ``Option<WinningSolutionWithNonce<AccountId, Balance, BlockNumber>>``.

    Layout (matches `pallet_quantum_pow::types::WinningSolutionWithNonce`):
      - 1 byte option tag (0x00 = None, 0x01 = Some)
      - solution.miner: AccountId32 = [u8; 32]
      - solution.salt: [u8; 32]
      - solution.energy_milli: i64
      - solution.reward: u128 (Balance)
      - solution.submitted_at: BlockNumber (u32)
      - solution.difficulty: DifficultyConfig
      - solution.last_proof_block_hash: H256 (last proof block hash the proof used)
      - nonce: U256 (little-endian on the wire; reversed to recover the
        BLAKE3 digest order Python miners use)
    """
    data = ScaleBytes(encoded_hex)
    tag = _read_exact(data, 1)
    if tag[0] == 0:
        if data.get_remaining_length() != 0:
            raise ValueError(
                "trailing bytes after winning_solution Option::None tag: "
                f"{data.get_remaining_length()} bytes"
            )
        return None
    try:
        miner = _decode_field("miner", data, lambda d: _read_exact(d, 32))
        salt = _decode_field("salt", data, lambda d: _read_exact(d, 32))
        energy_milli = _decode_field("energy_milli", data, _decode_i64)
        reward = _decode_field("reward", data, _decode_u128)
        submitted_at = _decode_field("submitted_at", data, _decode_u32)
        difficulty = _decode_difficulty_config(data)
        last_proof_block_hash = _decode_field(
            "last_proof_block_hash", data, lambda d: _read_exact(d, 32)
        )
        nonce = _decode_field("nonce", data, _decode_u256_le)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 — rewrap with context
        raise ValueError(f"failed to decode winning solution: {exc}") from exc
    if data.get_remaining_length() != 0:
        raise ValueError(
            "trailing bytes after winning_solution decode: "
            f"{data.get_remaining_length()} bytes; runtime API shape likely changed"
        )
    return WinningSolutionWithNonce(
        solution=WinningSolution(
            miner=miner,
            salt=salt,
            energy_milli=energy_milli,
            reward=reward,
            submitted_at=submitted_at,
            difficulty=difficulty,
            last_proof_block_hash=last_proof_block_hash,
        ),
        nonce=nonce,
    )


def _decode_field(name: str, data: ScaleBytes, fn: Callable[[ScaleBytes], Any]) -> Any:
    """Run a decoder, re-raising with the failing field name on error."""
    try:
        return fn(data)
    except Exception as exc:  # noqa: BLE001 — rewrap with context
        raise ValueError(f"failed to decode field {name!r}: {exc}") from exc


def _coerce_block_number(raw: Any) -> int:
    """Accept the `header.number` field as int or hex string."""
    if isinstance(raw, str):
        return int(raw, 16) if raw.startswith("0x") else int(raw)
    if raw is None:
        raise ValueError("header missing 'number' field")
    return int(raw)


def _coerce_receipt(receipt: Any) -> ExtrinsicReceipt:
    """Convert a substrate-interface ExtrinsicReceipt to our typed wrapper.

    Fails loud if the receipt does not advertise success/failure: a missing
    ``is_success`` attribute used to fall through to "success" which masked
    submission failures from callers that classify by ``receipt.error``.
    """
    is_success = getattr(receipt, "is_success", None)
    if is_success is None:
        raise RuntimeError(
            "substrate-interface receipt is missing `is_success`; cannot "
            "classify outcome (receipt_type=%s)" % type(receipt).__name__
        )
    error_msg: Optional[str] = None
    if is_success is False:
        try:
            error_msg = str(receipt.error_message)
        except AttributeError:
            # Receipt advertised failure but provided no error_message —
            # surface a grep-able sentinel so callers don't conflate it with
            # legitimate error strings.
            error_msg = "substrate-interface receipt missing error_message"
            logger.error(
                "extrinsic failed but receipt has no error_message: %s",
                type(receipt).__name__,
            )
    events: list = []
    raw_events = getattr(receipt, "triggered_events", None) or []
    try:
        for ev in raw_events:
            events.append(ev.value if hasattr(ev, "value") else dict(ev))
    except (AttributeError, TypeError) as exc:
        # Don't silently swallow — empty events combined with no error would
        # look like a successful submission that produced no side effects.
        logger.warning("failed to decode triggered_events: %s", exc)
    return ExtrinsicReceipt(
        extrinsic_hash=str(getattr(receipt, "extrinsic_hash", "")),
        block_hash=str(getattr(receipt, "block_hash", "") or "") or None,
        is_finalized=bool(getattr(receipt, "finalized", False)),
        error=error_msg,
        events=events,
    )


__all__ = [
    "SubstrateClient",
    "WaitFor",
]
