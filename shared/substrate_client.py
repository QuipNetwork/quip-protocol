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
from shared.signer import Signer
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
        await self._run(iface.close)

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

        **Off-by-one note**: The runtime API reads `System::block_number()`
        and `System::parent_hash()` at the queried block's state. At chain
        head N that gives `block_number=N`, `parent_hash=hash(N-1)`. The
        miner's nonce, though, must match what `submit_proof` sees at
        validation time — when the extrinsic lands in block N+1, where
        `System::block_number()=N+1` and `System::parent_hash()=hash(N)`.

        We bake that +1 offset into the returned context so
        `mine_work_item` doesn't need to know about chain timing:
          - `context.block_number = decoded.block_number + 1`
          - `context.parent_hash = at` (the head hash, which IS hash(N))

        This only works when `at` is explicit. Callers that pass `at=None`
        get the raw snapshot values — useful for state probes (Phase 2
        bootstrap idempotency check) but not for actual mining.
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
        # Apply the +1 offset for mining (see method docstring) when the
        # caller pinned a specific block hash. Otherwise return raw
        # snapshot values for state-probe callers.
        if at is not None:
            mining_block_number = decoded["block_number"] + 1
            mining_parent_hash = at
        else:
            mining_block_number = decoded["block_number"]
            mining_parent_hash = decoded["parent_hash"]
        return SubstrateMiningContext(
            block_number=mining_block_number,
            parent_hash=mining_parent_hash,
            topology_hash=decoded["topology_hash"],
            nodes=decoded["nodes"],
            edges=decoded["edges"],
            difficulty=decoded["difficulty"],
            miner_account_bytes=miner_account_bytes,
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
        result = await self._run(lambda: self._iface.query("QuantumPow", "Difficulty"))
        if result is None or not _result_was_found(result) or result.value is None:
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
        from shared.hybrid_signer import HybridSigner

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
                elif result in ("dropped", "invalid"):
                    raise ValueError(f"hybrid extrinsic rejected: {result}")
                # otherwise keep waiting

            response = self._iface.rpc_request(
                "author_submitAndWatchExtrinsic",
                [ext_hex],
                result_handler=_result_handler,
            )
            return response

        result = await self._run(_do)
        return ExtrinsicReceipt(
            extrinsic_hash=str(result.get("extrinsic_hash") or result.get("result") or ""),
            block_hash=str(result.get("block_hash") or "") or None,
            is_finalized=bool(result.get("finalized", False)),
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
        on every new head.

        ``substrate-interface`` invokes the subscription callback on its
        websocket reader thread. We do **not** make synchronous RPC calls
        from that thread (that would deadlock the reader waiting on its own
        response). Instead, the block-hash lookup and the user coroutine are
        both scheduled on the asyncio loop via ``run_coroutine_threadsafe``,
        where ``_run`` puts the blocking RPC on the executor.
        """
        loop = asyncio.get_running_loop()

        async def _handle_head(number: int) -> None:
            # The dispatch callback can fire after `close()` has cleared
            # `self._iface` — the reader thread keeps draining for a beat
            # after the websocket is told to close. Bail quietly in that
            # window rather than raising AttributeError into the future's
            # done-callback.
            iface = self._iface
            if iface is None:
                return
            block_hash_hex = await self._run(
                lambda: iface.get_block_hash(block_id=number)
            )
            block_hash_bytes = bytes.fromhex(_strip_0x(block_hash_hex))
            await callback(block_hash_bytes, number)

        def _dispatch(obj: dict, update_nr: int, subscription_id: str):  # noqa: ARG001
            # subscribe_block_headers delivers `{"header": {...}, "author": ...}`
            # on the main path; some code paths surface the bare header
            # directly. Accept both. Exceptions on the reader thread are
            # otherwise silently swallowed.
            try:
                if isinstance(obj, dict) and isinstance(obj.get("header"), dict):
                    header = obj["header"]
                elif isinstance(obj, dict):
                    header = obj
                else:
                    logger.warning(
                        "subscribe_new_heads: unexpected payload shape: %s",
                        type(obj).__name__,
                    )
                    return
                number = _coerce_block_number(header.get("number"))
                future = asyncio.run_coroutine_threadsafe(
                    _handle_head(number), loop
                )
                future.add_done_callback(_log_dispatch_exception)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "subscribe_new_heads dispatch failed at update_nr=%s: %s",
                    update_nr, exc,
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
    if len(raw) > 67:
        raise OverflowError("compact value exceeds 64-byte limit")
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
    import hashlib

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


def _decode_i64(data: ScaleBytes) -> int:
    return int.from_bytes(_read_exact(data, 8), "little", signed=True)


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
        block_number = _decode_field("block_number", data, _decode_u32)
        parent_hash = _decode_field("parent_hash", data, lambda d: _read_exact(d, 32))
        min_solutions = _decode_field("min_solutions", data, _decode_u32)
        max_energy_milli = _decode_field("max_energy_milli", data, _decode_i64)
        min_diversity_milli = _decode_field("min_diversity_milli", data, _decode_u32)
        min_quality_milli = _decode_field("min_quality_milli", data, _decode_u32)
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


def _log_dispatch_exception(future: Any) -> None:
    """Log exceptions raised by `subscribe_new_heads` dispatch coroutines.

    Without this hook, exceptions become unretrieved-future warnings at
    interpreter shutdown — the subscription appears to silently stop.
    """
    try:
        future.result()
    except Exception as exc:  # noqa: BLE001
        logger.exception("subscribe_new_heads callback raised: %s", exc)


__all__ = [
    "SubstrateClient",
    "WaitFor",
]
