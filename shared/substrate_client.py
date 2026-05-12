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
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Literal, Optional

from scalecodec.base import ScaleBytes
from scalecodec.types import GenericExtrinsic
from substrateinterface import SubstrateInterface

from shared.logging_config import get_logger
from shared.signer import Signer, Sr25519Signer
from shared.substrate_types import (
    ExtrinsicReceipt,
    MinerInfo,
    SubstrateDifficulty,
    SubstrateMiningContext,
)


logger = get_logger("substrate_client")


WaitFor = Literal["sent", "inblock", "finalized"]


class SubstrateClient:
    """Async-friendly wrapper over a single substrate websocket connection.

    Usage:

        client = SubstrateClient(url="ws://localhost:9944")
        await client.connect()
        head = await client.get_head()
        snapshot = await client.get_mining_snapshot(at=head)
        ...
        await client.close()

    The class is intentionally a thin adapter — no caching, no state machine.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._iface: Optional[SubstrateInterface] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._iface is not None:
            return
        self._loop = asyncio.get_running_loop()
        # SubstrateInterface() opens the websocket eagerly in its constructor.
        self._iface = await self._run(lambda: SubstrateInterface(url=self.url))
        logger.info("substrate client connected: url=%s", self.url)

    async def close(self) -> None:
        if self._iface is None:
            return
        iface = self._iface
        self._iface = None

        def _close() -> None:
            try:
                iface.close()
            except AttributeError:
                # Older substrate-interface versions don't expose close(); the
                # underlying websocket closes when the object is gc'd.
                pass

        await self._run(_close)

    # ------------------------------------------------------------------
    # Chain head queries
    # ------------------------------------------------------------------

    async def get_head(self) -> bytes:
        """Best block hash as raw bytes."""
        return bytes.fromhex(_strip_0x(await self._run(self._iface.get_chain_head)))

    async def get_finalized_head(self) -> bytes:
        return bytes.fromhex(_strip_0x(await self._run(self._iface.get_chain_finalised_head)))

    async def get_block_number(self, at: Optional[bytes] = None) -> int:
        block_hash = _hex(at) if at is not None else None
        header = await self._run(lambda: self._iface.get_block_header(block_hash=block_hash))
        return int(header["header"]["number"])

    # ------------------------------------------------------------------
    # Runtime API: mining snapshot
    # ------------------------------------------------------------------

    async def get_mining_snapshot(
        self,
        at: Optional[bytes] = None,
        topology_hash: Optional[bytes] = None,
        miner_account_bytes: Optional[bytes] = None,
    ) -> Optional[SubstrateMiningContext]:
        """Call `QuantumPowApi_mining_snapshot` at the given block hash.

        Returns `None` if the chain has no registered topology (e.g. fresh
        dev chain that hasn't been seeded yet — Phase 2 bootstrap handles
        that case via `Sudo.sudo(QuantumPow.register_topology(...))`).

        `miner_account_bytes` is the 32-byte AccountId of the signer; the
        snapshot itself doesn't include this but `SubstrateMiningContext`
        does. Callers pass it through so the resulting context is
        self-sufficient.
        """
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
        encoded = raw.get("result")
        if encoded is None:
            return None
        decoded = self._decode_mining_snapshot(encoded)
        if decoded is None:
            return None
        if miner_account_bytes is None:
            # Caller must provide this for the context to be useful — but we
            # accept None and return a context with an empty account to make
            # debugging easier. The controller will reject this in practice.
            miner_account_bytes = b"\x00" * 32
        return SubstrateMiningContext(
            block_number=decoded["block_number"],
            parent_hash=decoded["parent_hash"],
            topology_hash=decoded["topology_hash"],
            nodes=decoded["nodes"],
            edges=decoded["edges"],
            difficulty=decoded["difficulty"],
            miner_account_bytes=miner_account_bytes,
        )

    def _decode_mining_snapshot(self, encoded_hex: str) -> Optional[dict]:
        """Decode SCALE `Option<MiningSnapshot<BlockNumber, Hash, Nodes, Edges>>`.

        Layout:
          - 1 byte option tag (0x00 = None, 0x01 = Some)
          - block_number: u32 (BlockNumber on this chain)
          - parent_hash: H256 (32 bytes)
          - difficulty: DifficultyConfig {min_solutions: u32,
                max_energy_milli: i64, min_diversity_milli: u32,
                min_quality_milli: u32}
          - topology_hash: H256 (32 bytes)
          - nodes: Vec<u32>
          - edges: Vec<(u32, u32)>
        """
        data = ScaleBytes(encoded_hex)
        tag = data.get_next_bytes(1)
        if tag == b"\x00":
            return None
        block_number = _decode_u32(data)
        parent_hash = bytes(data.get_next_bytes(32))
        min_solutions = _decode_u32(data)
        max_energy_milli = _decode_i64(data)
        min_diversity_milli = _decode_u32(data)
        min_quality_milli = _decode_u32(data)
        topology_hash = bytes(data.get_next_bytes(32))
        nodes_len = _decode_compact_u32(data)
        nodes = [_decode_u32(data) for _ in range(nodes_len)]
        edges_len = _decode_compact_u32(data)
        edges = [(_decode_u32(data), _decode_u32(data)) for _ in range(edges_len)]
        return {
            "block_number": block_number,
            "parent_hash": parent_hash,
            "difficulty": SubstrateDifficulty(
                min_solutions=min_solutions,
                max_energy_milli=max_energy_milli,
                min_diversity_milli=min_diversity_milli,
                min_quality_milli=min_quality_milli,
            ),
            "topology_hash": topology_hash,
            "nodes": nodes,
            "edges": edges,
        }

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
        result = await self._run(lambda: self._iface.query("QuantumPow", "Difficulty"))
        if result is None or result.value is None:
            return None
        v = result.value
        return SubstrateDifficulty(
            min_solutions=int(v["min_solutions"]),
            max_energy_milli=int(v["max_energy_milli"]),
            min_diversity_milli=int(v["min_diversity_milli"]),
            min_quality_milli=int(v["min_quality_milli"]),
        )

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

        The signing path delegates to `substrate-interface` for the sr25519
        case (via `Sr25519Signer.keypair`). When `HybridSigner` lands in
        Phase 7, this method will branch on `signer.signature_kind()`.
        """
        if signer.signature_kind() == "Sr25519":
            keypair = signer.keypair  # type: ignore[attr-defined]
        else:
            raise NotImplementedError(
                f"submit_extrinsic for signature_kind={signer.signature_kind()} "
                "lands in Phase 7 (hybrid sig support)"
            )

        def _build_and_submit() -> dict:
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

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def subscribe_new_heads(
        self,
        callback: Callable[[bytes, int], Awaitable[None]],
        finalized_only: bool = False,
    ) -> None:
        """Block until cancelled, invoking `callback(block_hash, block_number)`
        on every new head.

        Each invocation of `callback` is scheduled on the event loop via
        `loop.call_soon_threadsafe` to keep the user-supplied coroutine off
        the substrate-interface worker thread.
        """
        loop = asyncio.get_running_loop()

        def _dispatch(header: dict, update_nr: int, subscription_id: str):  # noqa: ARG001
            raw_number = header.get("number")
            if isinstance(raw_number, str):
                number = int(raw_number, 16) if raw_number.startswith("0x") else int(raw_number)
            else:
                number = int(raw_number)
            # substrate-interface's block-header subscription delivers parent
            # + state roots but not the new block's hash. Resolve it via a
            # follow-up RPC; the work is dominated by the subsequent snapshot
            # fetch, so the extra round trip is not a concern.
            block_hash_hex = self._iface.get_block_hash(block_id=number)
            block_hash_bytes = bytes.fromhex(_strip_0x(block_hash_hex))
            asyncio.run_coroutine_threadsafe(
                callback(block_hash_bytes, number), loop
            )

        await self._run(
            lambda: self._iface.subscribe_block_headers(
                _dispatch, finalized_only=finalized_only
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run(self, fn):
        """Run a blocking substrate-interface call on the default executor."""
        loop = self._loop or asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _strip_0x(s: str) -> str:
    return s[2:] if s.startswith("0x") else s


def _hex(b: bytes) -> str:
    return "0x" + b.hex()


def _decode_u32(data: ScaleBytes) -> int:
    return int.from_bytes(data.get_next_bytes(4), "little")


def _decode_i64(data: ScaleBytes) -> int:
    return int.from_bytes(data.get_next_bytes(8), "little", signed=True)


def _decode_compact_u32(data: ScaleBytes) -> int:
    """Decode a SCALE compact-encoded u32 length prefix.

    Mode is encoded in the low 2 bits of the first byte:
      0b00 → single-byte (value in upper 6 bits)
      0b01 → two-byte
      0b10 → four-byte
      0b11 → big-integer (followed by `((first >> 2) + 4)` little-endian bytes)
    """
    first = data.get_next_bytes(1)[0]
    mode = first & 0b11
    if mode == 0:
        return first >> 2
    if mode == 1:
        return ((first >> 2) | (data.get_next_bytes(1)[0] << 6))
    if mode == 2:
        rest = data.get_next_bytes(3)
        return (first >> 2) | (rest[0] << 6) | (rest[1] << 14) | (rest[2] << 22)
    n_bytes = (first >> 2) + 4
    return int.from_bytes(data.get_next_bytes(n_bytes), "little")


def _coerce_receipt(receipt: Any) -> ExtrinsicReceipt:
    """Convert a substrate-interface ExtrinsicReceipt to our typed wrapper."""
    is_error = getattr(receipt, "is_success", None)
    error_msg: Optional[str] = None
    if is_error is False:
        try:
            error_msg = str(receipt.error_message)
        except AttributeError:
            error_msg = "unknown error"
    events: list = []
    try:
        for ev in receipt.triggered_events or []:
            events.append(ev.value if hasattr(ev, "value") else dict(ev))
    except (AttributeError, TypeError):
        pass
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
    "Sr25519Signer",
]
