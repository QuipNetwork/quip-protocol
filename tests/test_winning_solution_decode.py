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

from shared.substrate_client import (
    _decode_difficulty_config,
    _decode_winning_solution_with_nonce,
)
from shared.substrate_types import (
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
    last_winning_hash: bytes,
    nonce: bytes,
    is_some: bool = True,
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
    parts.append(last_winning_hash)
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
    last_winning_hash = bytes.fromhex("dd" * 32)
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
        last_winning_hash=last_winning_hash,
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
            last_winning_hash=last_winning_hash,
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
        last_winning_hash=b"\x03" * 32,
        nonce=nonce,
    )
    with pytest.raises(ValueError, match="trailing bytes"):
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
        last_winning_hash=b"\x03" * 32,
        nonce=b"\x02" * 32,
    )
    # Chop the last 16 hex bytes (half the U256 nonce — last_winning_hash
    # is fully present, so the decoder fails at `nonce`).
    truncated = encoded[: -32]
    with pytest.raises(ValueError, match="nonce"):
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
