# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Stateless SCALE encode/decode helpers for the QuantumPow + mempool pallets.

Extracted from ``substrate/client.py``: none of these functions depend on the
``SubstrateClient`` connection object. They turn raw SCALE bytes / runtime-API
hex into typed values, and build the hybrid-signed extrinsic wire bytes. Keeping
them out of ``client.py`` lets that module stay focused on RPC/session concerns.
``SubstrateClient`` and the decode-path tests import these helpers directly.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Optional

from scalecodec.base import ScaleBytes
from scalecodec.utils.ss58 import ss58_decode

from substrate.mempool_types import JobMode, MinerType, ResultDelivery
from shared.signer import strip_0x
from substrate.types import (
    SubstrateDifficulty,
    WinningSolution,
    WinningSolutionWithNonce,
)


def _decode_hash(value) -> bytes:
    """Decode a substrate Hash storage field to 32 raw bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return bytes.fromhex(strip_0x(value))
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
            endpoint = bytes.fromhex(strip_0x(endpoint))
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
    """SCALE compact encoding for u32. Mirrors substrate's `Compact<u32>`.

    For values below 2^30 this is identical to ``_encode_compact_u128``;
    the only divergence is the narrower u32 ceiling, which is kept as an
    explicit guard before delegating to the shared encoder.
    """
    if n < 0:
        raise ValueError(f"compact u32 must be non-negative, got {n}")
    if n >= 0x4000_0000:
        raise NotImplementedError("compact u32 big-int mode not needed here")
    return _encode_compact_u128(n)


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
    tip: int = 0,
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
    if isinstance(raw_call, str):
        call_bytes = bytes.fromhex(strip_0x(raw_call))
    else:
        call_bytes = bytes(raw_call)

    # 2. Fetch chain state needed for the signed extensions.
    account = signer.account_id_bytes()
    nonce = iface.get_account_nonce(account_address="0x" + account.hex())
    genesis_hex = iface.get_block_hash(block_id=0)
    rv = iface.rpc_request("state_getRuntimeVersion", [])["result"]
    spec_version = int(rv["specVersion"])
    tx_version = int(rv["transactionVersion"])
    genesis_bytes = bytes.fromhex(strip_0x(genesis_hex))

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
        + _encode_compact_u128(tip)          # ChargeTransactionPayment tip
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
