"""Unit tests for the SCALE decoders behind the new runtime-API calls.

Covers:
  - `_decode_winning_solution_with_nonce` (round-trip + `None` + trailing-bytes)
  - The shape of `QuantumPowApi_current_difficulty` decoding via
    `_decode_difficulty_config`

These don't need a live chain — they hand-build the SCALE bytes and check the
decoder produces the expected typed view. Matches the style of
`test_mining_snapshot_decode.py`.
"""
from __future__ import annotations

import pytest

from substrate.scale_codec import (
    _decode_difficulty_config,
    _decode_winning_solution_with_nonce,
)
from substrate.types import (
    SubstrateDifficulty,
    WinningSolution,
    WinningSolutionWithNonce,
)
from scalecodec.base import ScaleBytes


def _encode_difficulty(d: SubstrateDifficulty) -> bytes:
    return (
        d.min_solutions.to_bytes(4, "little")
        + d.max_energy_milli.to_bytes(8, "little", signed=True)
        + d.min_diversity_milli.to_bytes(4, "little")
    )


def _build_winning_solution_hex(
    *,
    miner: bytes,
    salt: bytes,
    energy_milli: int,
    reward: int,
    submitted_at: int,
    difficulty: SubstrateDifficulty,
    last_proof_block_hash: bytes,
    nonce: bytes,
    is_some: bool = True,
    topology_hash: bytes | None = None,
    device_access_time_us: int | None = None,
) -> str:
    if not is_some:
        return "0x00"
    parts: list[bytes] = [b"\x01"]
    parts.append(miner)
    parts.append(salt)
    parts.append(energy_milli.to_bytes(8, "little", signed=True))
    parts.append(reward.to_bytes(16, "little"))
    parts.append(submitted_at.to_bytes(4, "little"))
    parts.append(_encode_difficulty(difficulty))
    parts.append(last_proof_block_hash)
    # Spec-111 QBlock appends topology_hash + device_access_time_us before
    # the runtime-API nonce. Pass both to build the 111 shape; omit both
    # for the legacy 110 shape. (Passing only one is a caller bug.)
    if topology_hash is not None or device_access_time_us is not None:
        assert topology_hash is not None and device_access_time_us is not None
        parts.append(topology_hash)
        parts.append(device_access_time_us.to_bytes(8, "little"))
    # U256 on the wire is little-endian. The Python side stores nonces in
    # BLAKE3-digest order, so we encode the reverse here.
    parts.append(bytes(reversed(nonce)))
    return "0x" + b"".join(parts).hex()


def test_decode_winning_solution_none():
    assert _decode_winning_solution_with_nonce("0x00") is None


def test_decode_winning_solution_round_trip():
    miner = bytes.fromhex("aa" * 32)
    salt = bytes.fromhex("bb" * 32)
    nonce = bytes.fromhex("cc" * 32)
    last_proof_block_hash = bytes.fromhex("dd" * 32)
    difficulty = SubstrateDifficulty(
        min_solutions=5,
        max_energy_milli=-4_100_000,
        min_diversity_milli=150,
    )
    encoded = _build_winning_solution_hex(
        miner=miner,
        salt=salt,
        energy_milli=-4_250_000,
        reward=10_000_000_000_000_000_000,  # > u64 — exercises u128 decode
        submitted_at=42,
        difficulty=difficulty,
        last_proof_block_hash=last_proof_block_hash,
        nonce=nonce,
    )

    view = _decode_winning_solution_with_nonce(encoded)

    assert view == WinningSolutionWithNonce(
        solution=WinningSolution(
            miner=miner,
            salt=salt,
            energy_milli=-4_250_000,
            reward=10_000_000_000_000_000_000,
            submitted_at=42,
            difficulty=difficulty,
            last_proof_block_hash=last_proof_block_hash,
        ),
        nonce=nonce,
    )


def test_decode_winning_solution_trailing_bytes_rejected():
    miner = b"\x00" * 32
    salt = b"\x01" * 32
    nonce = b"\x02" * 32
    encoded = _build_winning_solution_hex(
        miner=miner,
        salt=salt,
        energy_milli=0,
        reward=0,
        submitted_at=1,
        difficulty=SubstrateDifficulty(1, 0, 0),
        last_proof_block_hash=b"\x03" * 32,
        nonce=nonce,
    )
    with pytest.raises(ValueError, match="tail"):
        _decode_winning_solution_with_nonce(encoded + "deadbeef")


def test_decode_winning_solution_short_read_surfaces_field_name():
    # Truncate inside the nonce — the field-tagged decoder should surface
    # which field ran out of bytes.
    miner = b"\x00" * 32
    salt = b"\x01" * 32
    encoded = _build_winning_solution_hex(
        miner=miner,
        salt=salt,
        energy_milli=0,
        reward=0,
        submitted_at=1,
        difficulty=SubstrateDifficulty(1, 0, 0),
        last_proof_block_hash=b"\x03" * 32,
        nonce=b"\x02" * 32,
    )
    # Chop the last 32 hex chars (16 bytes) — last_proof_block_hash is fully
    # present but only 16 bytes of nonce remain. The length-branch fires
    # before reading the nonce, reporting an unknown tail size.
    truncated = encoded[: -32]
    with pytest.raises(ValueError, match="tail"):
        _decode_winning_solution_with_nonce(truncated)


def test_decode_difficulty_config_matches_storage_shape():
    """The runtime API returns a bare DifficultyConfig — same field layout as
    the inline one inside MiningSnapshot. Locks down that interface."""
    expected = SubstrateDifficulty(
        min_solutions=7,
        max_energy_milli=-3_500_000,
        min_diversity_milli=200,
    )
    data = ScaleBytes("0x" + _encode_difficulty(expected).hex())
    assert _decode_difficulty_config(data) == expected
    assert data.get_remaining_length() == 0


def _difficulty() -> SubstrateDifficulty:
    return SubstrateDifficulty(
        min_solutions=3,
        max_energy_milli=-1_000_000,
        min_diversity_milli=200,
    )


def test_decode_spec111_shape_reads_topology_and_device_time():
    encoded = _build_winning_solution_hex(
        miner=b"\x11" * 32,
        salt=b"\x22" * 32,
        energy_milli=-42_000,
        reward=50,
        submitted_at=7,
        difficulty=_difficulty(),
        last_proof_block_hash=b"\x33" * 32,
        topology_hash=b"\x44" * 32,
        device_access_time_us=1_234_567,
        nonce=b"\x55" * 32,
    )
    ws = _decode_winning_solution_with_nonce(encoded)
    assert ws.solution.topology_hash == b"\x44" * 32
    assert ws.solution.device_access_time_us == 1_234_567
    assert ws.nonce == b"\x55" * 32
    assert ws.solution.energy_milli == -42_000


def test_decode_legacy_110_shape_leaves_new_fields_none():
    encoded = _build_winning_solution_hex(
        miner=b"\x11" * 32,
        salt=b"\x22" * 32,
        energy_milli=-42_000,
        reward=50,
        submitted_at=7,
        difficulty=_difficulty(),
        last_proof_block_hash=b"\x33" * 32,
        nonce=b"\x55" * 32,
    )
    ws = _decode_winning_solution_with_nonce(encoded)
    assert ws.solution.topology_hash is None
    assert ws.solution.device_access_time_us is None
    assert ws.nonce == b"\x55" * 32


def test_decode_rejects_unknown_tail_length():
    encoded = _build_winning_solution_hex(
        miner=b"\x11" * 32,
        salt=b"\x22" * 32,
        energy_milli=-42_000,
        reward=50,
        submitted_at=7,
        difficulty=_difficulty(),
        last_proof_block_hash=b"\x33" * 32,
        nonce=b"\x55" * 32,
    ) + "ff"  # one stray trailing byte -> neither 32 nor 72 remaining
    with pytest.raises(ValueError, match="tail"):
        _decode_winning_solution_with_nonce(encoded)
