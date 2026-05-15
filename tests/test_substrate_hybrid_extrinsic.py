"""Unit tests for the hybrid-extrinsic builder helpers in `substrate_client`.

These cover the SCALE compact encoders (`_encode_compact_u32` /
`_encode_compact_u128`) at every mode boundary, the terminal-failure
status set, and the unknown-URI rejection in `miner_bootstrap`. The
on-chain wire-format parity test against a Rust-side fixture is a
deliberate follow-on — see MR !82 description #1.
"""
from __future__ import annotations

import pytest

from shared.miner_bootstrap import DEV_HYBRID_SEEDS, _resolve_dev_signer
from shared.substrate_client import (
    _HYBRID_TERMINAL_FAILURES,
    _encode_compact_u128,
    _encode_compact_u32,
)


# ----------------------------------------------------------------------
# _encode_compact_u32: mode boundaries
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, b"\x00"),
        (1, b"\x04"),
        (63, b"\xfc"),                  # last single-byte
        (64, (64 << 2 | 0b01).to_bytes(2, "little")),       # first 2-byte
        (16_383, (16_383 << 2 | 0b01).to_bytes(2, "little")),  # last 2-byte
        (16_384, (16_384 << 2 | 0b10).to_bytes(4, "little")),  # first 4-byte
        (2**30 - 1, ((2**30 - 1) << 2 | 0b10).to_bytes(4, "little")),
    ],
)
def test_encode_compact_u32_boundaries(value: int, expected: bytes):
    assert _encode_compact_u32(value) == expected


def test_encode_compact_u32_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _encode_compact_u32(-1)


def test_encode_compact_u32_rejects_big_int_mode():
    with pytest.raises(NotImplementedError, match="big-int"):
        _encode_compact_u32(2**30)


# ----------------------------------------------------------------------
# _encode_compact_u128: shares u32 modes plus big-int mode
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, b"\x00"),
        (63, b"\xfc"),
        (64, (64 << 2 | 0b01).to_bytes(2, "little")),
        (16_384, (16_384 << 2 | 0b10).to_bytes(4, "little")),
    ],
)
def test_encode_compact_u128_low_modes_match_u32(value: int, expected: bytes):
    assert _encode_compact_u128(value) == expected


def test_encode_compact_u128_big_int_mode_layout():
    """Big-int mode prefix: top 6 bits = (n_bytes - 4), low 2 bits = 0b11.

    `2**32` needs 5 bytes; the leading mode byte should be
    `((5 - 4) << 2) | 0b11 = 0b00000111 = 0x07`, followed by the LE bytes.
    """
    value = 2**32
    encoded = _encode_compact_u128(value)
    assert encoded[0] == 0x07
    assert int.from_bytes(encoded[1:], "little") == value
    assert len(encoded) == 1 + 5


def test_encode_compact_u128_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _encode_compact_u128(-1)


def test_encode_compact_u128_overflow_message_uses_67_limit():
    """Big-int mode caps at 67 raw bytes (mode-byte top 6 bits = n_bytes - 4,
    max value 63 → 67). The earlier OverflowError message said "64-byte" by
    mistake; pin the corrected text."""
    huge = 2 ** (67 * 8 + 1)
    with pytest.raises(OverflowError, match="67-byte"):
        _encode_compact_u128(huge)


# ----------------------------------------------------------------------
# Terminal-failure status set
# ----------------------------------------------------------------------


def test_hybrid_terminal_failures_covers_all_dead_states():
    """Every terminal state of substrate's `TransactionStatus` enum must be
    enumerated so the result_handler raises instead of hanging."""
    expected = {"dropped", "invalid", "usurped", "retracted", "finalitytimeout"}
    assert _HYBRID_TERMINAL_FAILURES == expected


# ----------------------------------------------------------------------
# _resolve_dev_signer rejection
# ----------------------------------------------------------------------


def test_resolve_dev_signer_rejects_unknown_uri():
    """A URI outside the precomputed DEV_HYBRID_SEEDS table must raise with
    a message that lists the known URIs — operators relying on this for
    sudo signing should not silently fall back to anything else."""
    with pytest.raises(ValueError, match="unknown dev URI"):
        _resolve_dev_signer("//Bogus")


def test_resolve_dev_signer_accepts_alice():
    signer = _resolve_dev_signer("//Alice")
    # Determinism: same URI → same account_id.
    assert _resolve_dev_signer("//Alice").account_id_bytes() == signer.account_id_bytes()


def test_dev_hybrid_seeds_contains_canonical_uris():
    """Pin the table's URI set so a typo or accidental removal fails loud."""
    assert set(DEV_HYBRID_SEEDS) == {"//Alice", "//Bob", "//Alice//stash"}
