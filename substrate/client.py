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
import json
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from scalecodec.base import ScaleBytes
from scalecodec.types import GenericExtrinsic
from substrateinterface import SubstrateInterface
from websocket import WebSocketException

from shared.hybrid_signer import HybridSigner
from shared.logging_config import get_logger

# Aliased: `topology_hash` is also a local param/field name throughout this
# module, so import the canonical-hash function under a distinct name.
from shared.topology_hash import topology_hash as compute_topology_hash
from substrate.mempool_types import (
    IsingParams,
    JobOrder,
    MempoolSolverInfo,
    MinerType,
    OrderStatus,
    OrderTiming,
    RewardResolution,
)
from shared.signer import Signer, strip_0x
from substrate.types import (
    ExtrinsicReceipt,
    MinerInfo,
    PowConstants,
    SubstrateDifficulty,
    SubstrateMiningContext,
    WinningSolutionWithNonce,
)


from substrate.scale_codec import (
    _build_hybrid_signed_extrinsic,
    _decode_account_id,
    _decode_difficulty_config,
    _decode_h256_vec,
    _decode_hash,
    _decode_job_mode,
    _decode_mining_snapshot,
    _decode_result_delivery,
    _decode_winning_solution_with_nonce,
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


class NoRegisteredTopology(RuntimeError):
    """Raised when the chain has no registered topology to mine against.

    A fresh dev chain that hasn't been seeded has no ``DefaultTopology`` /
    ``mining_snapshot``; the operator must seed it (e.g.
    ``quip-miner bootstrap --seed-chain``) before any miner can derive work.
    """


@dataclass(frozen=True)
class TopologyBinding:
    """The local↔chain topology hashes a miner needs before it starts.

    ``expected_hash`` is the canonical hash of the *local* sampler topology
    under the chain's allowed-value specs — the same recipe the runtime's
    ``hash_topology`` uses. ``chain_hash`` is what the chain has registered
    (``snapshot.topology_hash``). PoW requires them to match (the pallet
    rejects a mismatched proof via ``InvalidTopology``); mempool binds its
    sampler to ``expected_hash`` regardless. ``snapshot`` is the full mining
    snapshot they were derived from, carried so callers don't re-query.
    """

    expected_hash: bytes
    chain_hash: bytes
    snapshot: SubstrateMiningContext

    @property
    def matches(self) -> bool:
        """True when the local topology hashes to the chain's registered one."""
        return self.expected_hash == self.chain_hash


# Page size for `state_getKeysPaged` storage-key enumeration (e.g. counting
# `WinningSolutions`). 1000 is the conventional substrate RPC page cap.
_KEYS_PAGE_SIZE = 1000


def _count_map_keys(
    iface,
    storage_module: str,
    storage_function: str,
    *,
    page_size: int = _KEYS_PAGE_SIZE,
) -> int:
    """Count the keys of a storage map via ``state_getKeysPaged``.

    Enumerates *keys only* under the map's storage prefix (cheap — no value
    decode), paging until a short/empty page. Synchronous; callers wrap it
    in ``_run``. Returns ``0`` for an empty map.
    """
    prefix = iface.generate_storage_hash(
        storage_module=storage_module,
        storage_function=storage_function,
    )
    total = 0
    start = None
    while True:
        page = iface.rpc_request("state_getKeysPaged", [prefix, page_size, start])
        keys = page.get("result") or []
        if not keys:
            break
        total += len(keys)
        if len(keys) < page_size:
            break
        start = keys[-1]
    return total


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
    ) -> None:
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
        # Points at whichever URL is currently connected (or the first one
        # before connect()).
        self.current_url: str = self._urls[0]
        # Public read-only view of the validator rotation. Controllers
        # propagate this to their separate subscription clients so both
        # clients see the same failover surface.
        self.urls: tuple[str, ...] = self._urls
        self._current_index: int = 0
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

    async def reconnect(self) -> None:
        """Drop the current iface and rotate to the next healthy validator.

        Starts at ``(current_index + 1)`` and walks circularly, so a
        live validator gets picked even if it sits behind the dead one
        in the list, and a transient blip on the primary can still
        recover by wrap-around.

        Raises ``NoValidatorReachable`` if every URL refuses.
        """
        await self._close_iface()
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
            self.current_url = candidate
            self._current_index = idx
            logger.info("substrate client connected: url=%s", candidate)
            return
        raise NoValidatorReachable(attempts)

    async def ensure_connected(self) -> bool:
        """Cheap pre-submit liveness probe; reconnect if the socket is dead.

        Long mining gaps let an idle validator websocket drop without the
        client noticing — the failure then surfaces as a ``BrokenPipeError``
        at submit time, where it eats the failover round-trip on the
        critical proof path. Calling this immediately before a submit moves
        that cost off the hot path: a one-shot ``system_health`` RPC confirms
        the socket, and any connection-class failure triggers the existing
        :meth:`reconnect` (validator rotation) before we sign + ship.

        Returns ``True`` if the existing connection answered the probe,
        ``False`` if a reconnect was performed. Propagates
        ``NoValidatorReachable`` when reconnect finds no live validator —
        the caller cannot submit anyway, so failing loud here is correct.
        """
        if self._iface is None:
            await self.connect()
            return False
        try:
            await self._raw_run(lambda: self._iface.rpc_request("system_health", []))
            return True
        except (
            WebSocketException,
            ConnectionError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning(
                "ensure_connected: health probe failed on %s (%s: %s); "
                "reconnecting before submit",
                self.current_url,
                type(exc).__name__,
                exc,
            )
            await self.reconnect()
            return False

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
            strip_0x(await self._run(lambda: self._iface.get_chain_head()))
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

    async def has_storage(self, module: str, function: str) -> bool:
        """Return True iff the runtime metadata exposes storage ``module.function``.

        Mirrors :meth:`has_call` for storage items: substrate-interface returns
        ``None`` (or raises) for an absent storage function, so we treat both as
        "not present". Used to pick the right winning-solution counter across the
        ``WinningSolutions``→``QBlockCount`` rename (quip-protocol-rs MR !35).
        """

        def _probe() -> bool:
            try:
                return (
                    self._iface.get_metadata_storage_function(module, function)
                    is not None
                )
            except Exception:
                return False

        return await self._run(_probe)

    async def get_finalized_head(self) -> bytes:
        return bytes.fromhex(
            strip_0x(await self._run(lambda: self._iface.get_chain_finalised_head()))
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
        # Resolve the target block. When `at` is None we want the current
        # best head — the mempool controller relies on `block_hash` /
        # `block_number` advancing every block, so we cannot collapse this
        # to "let the RPC pick"; we have to read the head ourselves.
        if at is None:
            resolved_block_hash = bytes.fromhex(
                strip_0x(await self._run(lambda: self._iface.get_chain_head()))
            )
        else:
            if len(at) != 32:
                raise ValueError(f"at must be 32 bytes, got {len(at)}")
            resolved_block_hash = at
        block_hash = _hex(resolved_block_hash)
        # Encode the parameter: Option<H256>.
        if topology_hash is None:
            scale_param = "0x00"
        else:
            if len(topology_hash) != 32:
                raise ValueError(
                    f"topology_hash must be 32 bytes, got {len(topology_hash)}"
                )
            scale_param = "0x01" + topology_hash.hex()

        # Distinguish RPC-level errors (transport, bad method, bad params)
        # from `Option::None` ("no topology registered yet"). Both used to
        # look like `result is None`, which silently swallowed real failures.
        encoded = await self._state_call(
            "QuantumPowApi_mining_snapshot", scale_param, block_hash
        )
        if encoded is None:
            return None
        decoded = _decode_mining_snapshot(encoded)
        if decoded is None:
            return None
        # Pull the block number for `resolved_block_hash`. The mempool
        # controller routes events per-block, so it needs the number; the
        # PoW controller carries it as ride-along data.
        header = await self._run(
            lambda: self._iface.get_block_header(
                block_hash=_hex(resolved_block_hash),
                ignore_decoding_errors=True,
            )
        )
        block_number = _coerce_block_number(header["header"]["number"])
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
            block_hash=resolved_block_hash,
            block_number=block_number,
        )

    async def resolve_topology_binding(
        self, topology: Any, *, miner_account_bytes: bytes
    ) -> TopologyBinding:
        """Resolve the chain mining snapshot + local↔chain topology hashes.

        Fetches the current head and mining snapshot, then computes the
        canonical hash of ``topology`` (the local sampler graph: any object
        exposing ``.nodes`` and ``.edges``) under the snapshot's allowed-value
        specs — the same recipe the runtime's ``hash_topology`` uses. Owning
        this here keeps the chain-consensus hash recipe out of callers: they
        compare :attr:`TopologyBinding.matches` and read the carried snapshot
        rather than re-deriving the hash themselves.

        Args:
            topology: Local sampler topology with ``.nodes`` / ``.edges``.
            miner_account_bytes: 32-byte AccountId for the snapshot identity.

        Returns:
            A :class:`TopologyBinding` with both hashes and the snapshot.

        Raises:
            NoRegisteredTopology: the chain has no registered topology yet.
        """
        head = await self.get_head()
        snapshot = await self.get_mining_snapshot(
            at=head,
            miner_account_bytes=miner_account_bytes,
        )
        if snapshot is None:
            raise NoRegisteredTopology("chain has no registered topology")
        expected_hash = compute_topology_hash(
            topology.nodes,
            topology.edges,
            snapshot.allowed_h_values,
            snapshot.allowed_j_values,
            snapshot.allowed_spin_values,
        )
        return TopologyBinding(
            expected_hash=expected_hash,
            chain_hash=snapshot.topology_hash,
            snapshot=snapshot,
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
        v = _storage_value(result)
        if v is None:
            return None
        return MinerInfo(
            registered_at=int(v["registered_at"]),
            deposit=int(v["deposit"]),
            proofs_submitted=int(v["proofs_submitted"]),
            proofs_won=int(v["proofs_won"]),
            rewards_earned=int(v["rewards_earned"]),
        )

    async def query_proofs_submitted(self, account: bytes) -> Optional[int]:
        """Return ``QuantumPow.Miners[account].proofs_submitted``, or ``None``.

        Returns None when the miner is not registered or the RPC fails. Safe
        to call on the hot path — callers should treat None as "unavailable"
        and not raise. Use ``query_miner`` when the full ``MinerInfo`` is needed.
        """
        info = await self.query_miner(account)
        if info is None:
            return None
        return info.proofs_submitted

    async def query_difficulty(
        self, topology_hash: Optional[bytes] = None
    ) -> Optional[SubstrateDifficulty]:
        """Return the raw ``QuantumPow.Difficulties[topology_hash]`` map entry.

        This is the *post-adjust baseline* — the value recorded after the most
        recent winning proof for the given topology, **without** decay applied.
        Use :meth:`query_current_difficulty` for the live threshold a fresh
        proof would have to clear right now, or :meth:`query_difficulty_for`
        for the decayed value from the runtime API.

        When ``topology_hash`` is ``None``, the chain's ``DefaultTopology``
        storage value is read first; if that is also absent the method returns
        ``None`` without querying the map.

        Suitable for "is the chain seeded?" checks and historical comparisons;
        not suitable for deciding what difficulty a miner currently faces.

        Args:
            topology_hash: 32-byte topology hash to look up, or ``None`` to
                resolve from ``QuantumPow.DefaultTopology``.

        Returns:
            The stored difficulty config, or ``None`` if the map entry is
            absent or no default topology has been set.
        """
        if topology_hash is None:
            default_result = await self._run(
                lambda: self._iface.query("QuantumPow", "DefaultTopology")
            )
            resolved = _storage_value(default_result, check_found=True)
            if resolved is None:
                return None
            topology_hash = _decode_hash(resolved)
        result = await self._run(
            lambda: self._iface.query(
                "QuantumPow", "Difficulties", ["0x" + topology_hash.hex()]
            )
        )
        v = _storage_value(result, check_found=True)
        if v is None:
            return None
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
        encoded = await self._state_call(
            "QuantumPowApi_current_difficulty", "0x", block_hash
        )
        if encoded is None:
            raise RuntimeError("state_call current_difficulty returned no result")
        data = ScaleBytes(encoded)
        difficulty = _decode_difficulty_config(data)
        if data.get_remaining_length() != 0:
            raise ValueError(
                "trailing bytes after current_difficulty decode: "
                f"{data.get_remaining_length()} bytes"
            )
        return difficulty

    async def query_mineable_topologies(
        self, *, at: Optional[bytes] = None
    ) -> list[bytes]:
        """Call ``QuantumPowApi_mineable_topologies`` for the set of mineable hashes.

        Returns the full list of topology hashes that are currently whitelisted
        for mining. On a fresh chain with no topologies registered this is an
        empty list (not an error).

        Args:
            at: Optional 32-byte block hash to query at. ``None`` uses the
                current best head.

        Returns:
            List of 32-byte topology hashes, in the order the runtime returns them.
        """
        block_hash = _hex(at) if at is not None else None
        encoded = await self._state_call(
            "QuantumPowApi_mineable_topologies", "0x", block_hash
        )
        if encoded is None:
            raise RuntimeError(
                "state_call QuantumPowApi_mineable_topologies returned no result"
            )
        data = ScaleBytes(encoded)
        hashes = _decode_h256_vec(data)
        if data.get_remaining_length() != 0:
            raise ValueError(
                "trailing bytes after mineable_topologies decode: "
                f"{data.get_remaining_length()} bytes"
            )
        return hashes

    async def query_difficulty_for(
        self, topology_hash: bytes, *, at: Optional[bytes] = None
    ) -> Optional[SubstrateDifficulty]:
        """Call ``QuantumPowApi_difficulty_for(topology_hash)`` for the decayed value.

        Unlike :meth:`query_difficulty`, which reads the raw map entry, this
        method calls the runtime API so decay is applied for elapsed blocks
        since the last winning proof. Returns ``None`` when the topology is
        not registered.

        The SCALE parameter is the raw 32-byte H256 topology hash (no compact
        prefix — the runtime API receives it as a fixed-size value).

        Args:
            topology_hash: 32-byte topology hash to query.
            at: Optional 32-byte block hash to query at. ``None`` uses the
                current best head.

        Returns:
            Decayed difficulty config, or ``None`` if the topology is absent.
        """
        if len(topology_hash) != 32:
            raise ValueError(
                f"topology_hash must be 32 bytes, got {len(topology_hash)}"
            )
        block_hash = _hex(at) if at is not None else None
        encoded = await self._state_call(
            "QuantumPowApi_difficulty_for",
            "0x" + topology_hash.hex(),
            block_hash,
        )
        if encoded is None:
            raise RuntimeError(
                "state_call QuantumPowApi_difficulty_for returned no result"
            )
        data = ScaleBytes(encoded)
        tag = data.get_next_bytes(1)
        if not tag:
            raise ValueError("empty response from QuantumPowApi_difficulty_for")
        if tag[0] == 0x00:
            if data.get_remaining_length() != 0:
                raise ValueError(
                    "trailing bytes after difficulty_for Option::None: "
                    f"{data.get_remaining_length()} bytes"
                )
            return None
        if tag[0] != 0x01:
            raise ValueError(
                f"unknown Option variant tag for difficulty_for: 0x{tag[0]:02x}"
            )
        difficulty = _decode_difficulty_config(data)
        if data.get_remaining_length() != 0:
            raise ValueError(
                "trailing bytes after difficulty_for decode: "
                f"{data.get_remaining_length()} bytes"
            )
        return difficulty

    async def add_mineable_topology(
        self,
        topology_hash: bytes,
        signer: "Signer",
        *,
        wait_for: "WaitFor" = "inblock",
    ) -> "ExtrinsicReceipt":
        """Submit ``QuantumPow.add_mineable_topology(topology_hash)`` via sudo.

        Adds ``topology_hash`` to the on-chain whitelist of mineable topologies.
        Requires a root/sudo key.

        Args:
            topology_hash: 32-byte hash of the topology to whitelist.
            signer: Root/sudo signer for the extrinsic.
            wait_for: Submission stage — ``"sent"``, ``"inblock"``, or
                ``"finalized"``.

        Returns:
            Extrinsic receipt at the requested stage.
        """
        if len(topology_hash) != 32:
            raise ValueError(
                f"topology_hash must be 32 bytes, got {len(topology_hash)}"
            )
        return await self.submit_extrinsic(
            call_module="QuantumPow",
            call_function="add_mineable_topology",
            call_params={"topology_hash": "0x" + topology_hash.hex()},
            signer=signer,
            wait_for=wait_for,
        )

    async def remove_mineable_topology(
        self,
        topology_hash: bytes,
        signer: "Signer",
        *,
        wait_for: "WaitFor" = "inblock",
    ) -> "ExtrinsicReceipt":
        """Submit ``QuantumPow.remove_mineable_topology(topology_hash)`` via sudo.

        Removes ``topology_hash`` from the on-chain whitelist of mineable
        topologies. Requires a root/sudo key.

        Args:
            topology_hash: 32-byte hash of the topology to remove.
            signer: Root/sudo signer for the extrinsic.
            wait_for: Submission stage — ``"sent"``, ``"inblock"``, or
                ``"finalized"``.

        Returns:
            Extrinsic receipt at the requested stage.
        """
        if len(topology_hash) != 32:
            raise ValueError(
                f"topology_hash must be 32 bytes, got {len(topology_hash)}"
            )
        return await self.submit_extrinsic(
            call_module="QuantumPow",
            call_function="remove_mineable_topology",
            call_params={"topology_hash": "0x" + topology_hash.hex()},
            signer=signer,
            wait_for=wait_for,
        )

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
            raise ValueError(f"block_number must fit in u32, got {block_number}")
        block_hash = _hex(at) if at is not None else None
        scale_param = "0x" + block_number.to_bytes(4, "little").hex()
        encoded = await self._state_call(
            "QuantumPowApi_winning_solution", scale_param, block_hash
        )
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
        v = _storage_value(result, check_found=True)
        if v is None:
            return None
        return int(v)

    async def query_block_timestamp_ms(self, block_hash: bytes) -> Optional[int]:
        """Return the ``Timestamp.Now`` storage value at ``block_hash``.

        Reads the on-chain millisecond timestamp for the given block.
        pallet-timestamp stores the block production time as a ``u64``
        (milliseconds since UNIX epoch), which the chain updates every block.

        Args:
            block_hash: The 32-byte block hash to query at.

        Returns:
            The timestamp in milliseconds, or ``None`` if the read fails
            (best-effort — a missing timestamp must never crash the caller).
        """
        try:
            result = await self._run(
                lambda: self._iface.query(
                    "Timestamp", "Now", block_hash=_hex(block_hash)
                )
            )
            return int(result.value)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "query_block_timestamp_ms failed for %s: %s: %s",
                block_hash.hex(),
                type(exc).__name__,
                exc,
            )
            return None

    async def query_winning_solution_count(self) -> int:
        """Return the number of recorded winning solutions (qblocks).

        The ordinal *solution number* is this count; the solution currently
        being mined is ``count + 1`` (computed by the controller, cached per
        round). Returns ``0`` on a fresh chain with no winners yet.

        Two runtime layouts are supported (quip-protocol-rs MR !35 renamed the
        storage ``WinningSolutions``→``QBlocks`` and added a direct counter):

        - New runtime: ``QBlockCount`` is a ``u64`` ``StorageValue`` — one O(1)
          read.
        - Old runtime: solutions are keyed in the ``WinningSolutions`` map by
          the block number they won at; count its keys via ``state_getKeysPaged``
          (*keys only* — one cheap RPC per ``KEYS_PAGE_SIZE`` keys, no value
          scan).
        """
        if await self.has_storage("QuantumPow", "QBlockCount"):
            result = await self._run(
                lambda: self._iface.query("QuantumPow", "QBlockCount")
            )
            value = _storage_value(result)
            return int(value) if value is not None else 0
        return await self._run(
            lambda: _count_map_keys(
                self._iface,
                "QuantumPow",
                "WinningSolutions",
            )
        )

    async def query_balance(self, account: bytes) -> int:
        if len(account) != 32:
            raise ValueError(f"account must be 32 bytes, got {len(account)}")
        result = await self._run(
            lambda: self._iface.query("System", "Account", [account])
        )
        v = _storage_value(result)
        if v is None:
            return 0
        return int(v["data"]["free"])

    # ------------------------------------------------------------------
    # QuantumComputeMempool storage queries
    # ------------------------------------------------------------------

    async def query_solver(self, account: bytes) -> Optional[MempoolSolverInfo]:
        """Return `QuantumComputeMempool.Solvers[account]`, or None."""
        if len(account) != 32:
            raise ValueError(f"account must be 32 bytes, got {len(account)}")
        result = await self._run(
            lambda: self._iface.query("QuantumComputeMempool", "Solvers", [account])
        )
        v = _storage_value(result)
        if v is None:
            return None
        return MempoolSolverInfo(
            account=_decode_account_id(v["account"]),
            solver_type=MinerType.from_scale_variant(str(v["solver_type"])),
            registered_at=int(v["registered_at"]),
            solutions_submitted=int(v["solutions_submitted"]),
            rewards_earned=int(v["rewards_earned"]),
        )

    async def register_solver(
        self, signer: Signer, solver_type: MinerType
    ) -> ExtrinsicReceipt:
        """Register ``signer``'s account as a ``QuantumComputeMempool`` solver.

        Wraps the ``register_solver`` extrinsic so callers don't hand-encode the
        pallet name or the ``solver_type`` SCALE variant. Idempotency (skip when
        already registered) is the caller's concern via :meth:`query_solver`.
        """
        return await self.submit_extrinsic(
            "QuantumComputeMempool",
            "register_solver",
            {"solver_type": solver_type.to_scale_variant()},
            signer,
            wait_for="inblock",
        )

    async def deregister_solver(self, signer: Signer) -> ExtrinsicReceipt:
        """Deregister ``signer``'s ``QuantumComputeMempool`` solver registration."""
        return await self.submit_extrinsic(
            "QuantumComputeMempool",
            "deregister_solver",
            {},
            signer,
            wait_for="inblock",
        )

    async def query_job_order(self, order_id: int) -> Optional[JobOrder]:
        """Return `QuantumComputeMempool.JobOrders[order_id]`, or None.

        The decoded `JobOrder` carries the full ising problem definition
        (nodes, edges, h_values, j_values) plus quality floors, reward,
        mode, and lifecycle status — enough to drive `mine_work_item`
        without a follow-on query.
        """
        result = await self._run(
            lambda: self._iface.query("QuantumComputeMempool", "JobOrders", [order_id])
        )
        v = _storage_value(result)
        if v is None:
            return None

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
            event_field = (
                value.get("event", value) if isinstance(value, dict) else value
            )
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
                    module_id,
                    event_id,
                )
            out.append(
                {
                    "module_id": str(module_id) if module_id is not None else "",
                    "event_id": str(event_id) if event_id is not None else "",
                    "attributes": attrs,
                }
            )
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
        """Compose, sign, and submit an extrinsic on this connection.

        Convenience wrapper that pairs :meth:`build_signed_extrinsic`
        with :meth:`submit_signed_extrinsic`. Callers that want
        swap-aware submission should call the two halves separately and
        ship the hex through ``ValidatorPool`` — see ``PoolClient``.
        """
        ext_hex = await self.build_signed_extrinsic(
            call_module,
            call_function,
            call_params,
            signer,
        )
        return await self.submit_signed_extrinsic(ext_hex, wait_for=wait_for)

    async def build_signed_extrinsic(
        self,
        call_module: str,
        call_function: str,
        call_params: dict,
        signer: Signer,
        *,
        tip: int = 0,
    ) -> str:
        """Compose + sign an extrinsic locally; return hex-encoded wire bytes.

        Branches on ``signer.signature_kind()``:
          - Sr25519: substrate-interface's ``create_signed_extrinsic``
            (works on ``MultiSignature`` chains); we then extract the
            SCALE bytes via ``extrinsic.data.to_hex()``.
          - Hybrid:  manual SCALE assembly via
            ``_build_hybrid_signed_extrinsic`` — substrate-interface
            doesn't know about the chain's ``HybridTxSignature`` envelope.

        ``tip`` is the ``ChargeTransactionPayment`` tip in the chain's base
        unit (plancks). A higher tip raises the extrinsic's transaction-pool
        priority, improving the odds of inclusion in the next block — the
        proof race is won by being in the right block, not by being first to
        broadcast. ``tip=0`` reproduces the exact pre-tip wire bytes.

        Pair with :meth:`submit_signed_extrinsic` so the signer's key
        material can stay in the calling process even when the submit
        side crosses an IPC boundary (e.g., ``ValidatorPool.send``).
        """
        if tip < 0:
            raise ValueError(f"tip must be non-negative, got {tip}")
        kind = signer.signature_kind()
        if kind == "Sr25519":
            return await self._build_sr25519_extrinsic_hex(
                call_module,
                call_function,
                call_params,
                signer,
                tip=tip,
            )
        if kind == "Hybrid":
            return await self._build_hybrid_extrinsic_hex(
                call_module,
                call_function,
                call_params,
                signer,
                tip=tip,
            )
        raise NotImplementedError(
            f"build_signed_extrinsic does not support signature_kind={kind}"
        )

    async def _build_sr25519_extrinsic_hex(
        self,
        call_module: str,
        call_function: str,
        call_params: dict,
        signer: Signer,
        *,
        tip: int = 0,
    ) -> str:
        keypair = signer.keypair  # type: ignore[attr-defined]

        def _build() -> str:
            call = self._iface.compose_call(
                call_module=call_module,
                call_function=call_function,
                call_params=call_params,
            )
            extrinsic: GenericExtrinsic = self._iface.create_signed_extrinsic(
                call=call,
                keypair=keypair,
                tip=tip,
            )
            return extrinsic.data.to_hex()

        # Build is pure compose+sign: it reads chain metadata + nonce
        # but does NOT broadcast anything, so retrying after a WS
        # reconnect is safe. Submission is the non-idempotent step,
        # handled separately by ``submit_signed_extrinsic``.
        return await self._run(_build, idempotent=True)

    async def _build_hybrid_extrinsic_hex(
        self,
        call_module: str,
        call_function: str,
        call_params: dict,
        signer: Signer,
        *,
        tip: int = 0,
    ) -> str:
        if not isinstance(signer, HybridSigner):
            raise TypeError(
                f"hybrid signing requires a HybridSigner, got {type(signer).__name__}"
            )

        def _build() -> str:
            ext_bytes, _ext_hash = _build_hybrid_signed_extrinsic(
                iface=self._iface,
                signer=signer,
                call_module=call_module,
                call_function=call_function,
                call_params=call_params,
                tip=tip,
            )
            return "0x" + ext_bytes.hex()

        # Same idempotent-retry rationale as the sr25519 path: build is
        # pure compose+sign with no broadcast side effect.
        return await self._run(_build, idempotent=True)

    async def submit_signed_extrinsic(
        self,
        extrinsic_hex: str,
        wait_for: WaitFor = "inblock",
    ) -> ExtrinsicReceipt:
        """Submit a pre-signed extrinsic; return the typed receipt.

        ``extrinsic_hex`` is the SCALE-encoded signed extrinsic with or
        without a ``0x`` prefix — typically the output of
        :meth:`build_signed_extrinsic`. The extrinsic hash is computed
        locally from the bytes (``blake2_256``), so callers don't need
        to ship it through IPC alongside the hex.

        ``wait_for``:
          - ``"sent"``: return as soon as the node accepts the extrinsic.
          - ``"inblock"``: wait until inclusion in a block (default).
          - ``"finalized"``: wait until the including block is finalized.

        For inclusion/finalization, fetches block events to surface
        ``System.ExtrinsicFailed`` as ``receipt.error`` — without this
        callers would treat chain-rejected dispatches as successful.
        """
        if not extrinsic_hex.startswith("0x"):
            extrinsic_hex = "0x" + extrinsic_hex
        raw = bytes.fromhex(extrinsic_hex[2:])
        ext_hash = "0x" + hashlib.blake2b(raw, digest_size=32).hexdigest()

        def _do() -> dict:
            if wait_for == "sent":
                resp = self._iface.rpc_request(
                    "author_submitExtrinsic", [extrinsic_hex]
                )
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
            # `substrate_interface.SubstrateInterface.submit_extrinsic` so
            # the error-handling path is well-trodden.
            def _result_handler(message, update_nr, subscription_id):  # noqa: ARG001
                params = message.get("params") or {}
                result = params.get("result")
                if isinstance(result, dict):
                    res = {k.lower(): v for k, v in result.items()}
                    if "finalized" in res and wait_for == "finalized":
                        self._iface.rpc_request(
                            "author_unwatchExtrinsic", [subscription_id]
                        )
                        return {
                            "extrinsic_hash": ext_hash,
                            "block_hash": res["finalized"],
                            "finalized": True,
                        }
                    if "inblock" in res and wait_for == "inblock":
                        self._iface.rpc_request(
                            "author_unwatchExtrinsic", [subscription_id]
                        )
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
                    self._iface.rpc_request(
                        "author_unwatchExtrinsic", [subscription_id]
                    )
                    raise ValueError(f"extrinsic rejected: {result}")
                return None  # non-terminal status — keep waiting

            return self._iface.rpc_request(
                "author_submitAndWatchExtrinsic",
                [extrinsic_hex],
                result_handler=_result_handler,
            )

        result = await self._run(_do)
        block_hash = result.get("block_hash")
        error_msg = None
        if block_hash:
            # author_submitAndWatchExtrinsic only reports inclusion — a
            # `System.ExtrinsicFailed` event still means the chain rejected
            # the dispatch. Without this fetch callers see is_success=True
            # for a chain-rejected proof.
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
                logger.warning("extrinsic event fetch failed: %s", exc)
        return ExtrinsicReceipt(
            extrinsic_hash=str(
                result.get("extrinsic_hash") or result.get("result") or ""
            ),
            block_hash=str(block_hash or "") or None,
            is_finalized=bool(result.get("finalized", False)),
            error=error_msg,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _state_call(
        self, method: str, scale_param: str, block_hash: Optional[str]
    ) -> Optional[str]:
        """Dispatch a ``state_call`` RPC and return the hex-encoded result.

        Raises ``RuntimeError`` on an RPC-level error (transport, bad method,
        bad params).  Returns ``None`` when the runtime returns
        ``Option::None`` — callers apply their own None semantics on top.

        Args:
            method: Runtime API method name, e.g. ``QuantumPowApi_mining_snapshot``.
            scale_param: SCALE-encoded hex parameter string (e.g. ``"0x00"``).
            block_hash: Block hash hex string or ``None`` for the current best head.

        Returns:
            Hex-encoded result string, or ``None`` if the runtime returned no value.
        """
        raw = await self._run(
            lambda: self._iface.rpc_request(
                "state_call",
                [method, scale_param, block_hash],
            )
        )
        if "error" in raw:
            raise RuntimeError(f"state_call {method} rpc error: {raw['error']}")
        return raw.get("result")

    async def _run(self, fn, *, idempotent: bool = False):
        """Run a blocking substrate-interface call with failover-on-loss.

        All callers go through this method, so the `_call_lock` taken
        inside `_raw_run` guarantees serial access to the underlying
        `SubstrateInterface`.

        Failover semantics: a connection-class error (``WebSocketException``,
        ``ConnectionError``, or a ``json.JSONDecodeError`` from an empty
        response frame — substrate-interface doesn't realise the WS is
        dead and tries to parse ``""`` as JSON) triggers one
        ``reconnect()`` attempt.

        After the reconnect, behaviour depends on ``idempotent``:
          - ``idempotent=False`` (default, the safe choice for arbitrary
            extrinsic-submitting callers): re-raise the original exception
            so the caller decides whether to retry. A partially-submitted
            extrinsic must NOT be resubmitted blindly.
          - ``idempotent=True``: the caller is asserting this call has no
            chain side effects (pure read, or compose+sign with the
            extrinsic not yet broadcast). We retry the call once on the
            fresh iface. If the retry also fails, the second exception
            propagates.

        ``reconnect()`` itself can raise ``NoValidatorReachable`` if no
        URL in the failover list comes back up; that propagates instead.
        """
        try:
            return await self._raw_run(fn)
        except (WebSocketException, ConnectionError, json.JSONDecodeError) as exc:
            logger.warning(
                "substrate _run failed on %s (%s: %s); attempting failover",
                self.current_url,
                type(exc).__name__,
                exc,
            )
            # May raise NoValidatorReachable — that's the intended
            # signal upstream. `reconnect()` does not call `_run`, so
            # no recursion risk.
            await self.reconnect()
            if not idempotent:
                # Surface the original exception so the caller's retry
                # path decides. Default for any path that could have
                # broadcast a non-idempotent extrinsic.
                raise
            # Idempotent path: retry once on the fresh iface. The
            # lambdas in this file capture ``self._iface`` lazily, so
            # the retry hits the post-reconnect iface naturally.
            logger.info(
                "retrying idempotent call on %s after reconnect",
                self.current_url,
            )
            return await self._raw_run(fn)

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
        return strip_0x(value).lower()
    return None


def _as_int_or_none(x: Any) -> Optional[int]:
    """Return ``int(x)`` or ``None`` if *x* is ``None`` or non-coercible."""
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
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
            return _as_int_or_none(idx)
        return _as_int_or_none(applied)
    if isinstance(phase, (int, str)):
        return _as_int_or_none(phase)
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
            f"block {strip_0x(block_hash)[:16]}… "
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
            event.get("module_id") or event.get("pallet") or event.get("pallet_name")
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


def _storage_value(result, *, check_found: bool = False):
    """Extract the decoded value from a storage query result, or return None.

    Centralises the two recurring empty-result guards:
    - ``result is None`` — the executor raised or the iface returned None.
    - ``result.value is None`` — the entry was present but decoded to None.
    - ``not _result_was_found(result)`` — OptionQuery default masking a
      missing entry; only checked when ``check_found=True``.

    Returns ``result.value`` when all guards pass, ``None`` otherwise.
    Callers that need a different sentinel (e.g. ``0``) compare the return
    value to ``None`` themselves.
    """
    if result is None:
        return None
    if check_found and not _result_was_found(result):
        return None
    if result.value is None:
        return None
    return result.value


def _hex(b: bytes) -> str:
    return "0x" + b.hex()


def _coerce_block_number(raw: Any) -> int:
    """Accept the `header.number` field as int or hex string."""
    if isinstance(raw, str):
        return int(raw, 16) if raw.startswith("0x") else int(raw)
    if raw is None:
        raise ValueError("header missing 'number' field")
    return int(raw)


__all__ = [
    "NoRegisteredTopology",
    "SubstrateClient",
    "TopologyBinding",
    "WaitFor",
]
