"""Unit tests for the SCALE decoder in `substrate_client._decode_mining_snapshot`.

These don't need a live chain — they hand-build the SCALE bytes and check that
the decoder yields the expected `SubstrateMiningContext`. They cover both the
`None` (option-tag 0x00) case and the populated case, including the
milli-precision difficulty fields and the variable-length nodes/edges vecs.
"""
from __future__ import annotations

import pytest

from shared.allowed_value_spec import (
    AllowedValueContinuousRange,
    AllowedValueIntegerRange,
    AllowedValueSet,
)
from substrate.client import SubstrateClient, _decode_mining_snapshot
from substrate.types import SubstrateDifficulty


def _encode_set(values: list[int]) -> bytes:
    out = b"\x00" + _encode_compact_u32(len(values))
    for v in values:
        out += int(v).to_bytes(4, "little", signed=True)
    return out


def _encode_integer_range(lo: int, hi: int) -> bytes:
    return (
        b"\x01"
        + int(lo).to_bytes(4, "little", signed=True)
        + int(hi).to_bytes(4, "little", signed=True)
    )


def _encode_continuous_range(lo: int, hi: int) -> bytes:
    return (
        b"\x02"
        + int(lo).to_bytes(4, "little", signed=True)
        + int(hi).to_bytes(4, "little", signed=True)
    )


def _encode_compact_u32(n: int) -> bytes:
    """Reference SCALE compact encoder for u32. Mirrors the decoder under test."""
    if n < 0:
        raise ValueError(f"compact u32 must be non-negative, got {n}")
    if n < 0x40:
        return bytes([n << 2])
    if n < 0x4000:
        return ((n << 2) | 0b01).to_bytes(2, "little")
    if n < 0x4000_0000:
        return ((n << 2) | 0b10).to_bytes(4, "little")
    raise NotImplementedError("big-integer compact mode not exercised by these tests")


def _build_snapshot_hex(
    *,
    last_proof_block_hash: bytes,
    difficulty: SubstrateDifficulty,
    topology_hash: bytes,
    nodes: list[int],
    edges: list[tuple[int, int]],
    is_some: bool = True,
    spec_h: bytes | None = None,
    spec_j: bytes | None = None,
    spec_spin: bytes | None = None,
) -> str:
    parts: list[bytes] = []
    parts.append(b"\x01" if is_some else b"\x00")
    if not is_some:
        return "0x" + b"".join(parts).hex()
    parts.append(last_proof_block_hash)
    parts.append(difficulty.min_solutions.to_bytes(4, "little"))
    parts.append(difficulty.max_energy_milli.to_bytes(8, "little", signed=True))
    parts.append(difficulty.min_diversity_milli.to_bytes(4, "little"))
    parts.append(topology_hash)
    parts.append(_encode_compact_u32(len(nodes)))
    for n in nodes:
        parts.append(n.to_bytes(4, "little"))
    parts.append(_encode_compact_u32(len(edges)))
    for u, v in edges:
        parts.append(u.to_bytes(4, "little"))
        parts.append(v.to_bytes(4, "little"))
    parts.append(spec_h if spec_h is not None else _encode_set([-1000, 0, 1000]))
    parts.append(spec_j if spec_j is not None else _encode_set([-1000, 1000]))
    parts.append(spec_spin if spec_spin is not None else _encode_set([-1000, 1000]))
    return "0x" + b"".join(parts).hex()


def test_decode_none():
    assert _decode_mining_snapshot("0x00") is None


def test_decode_populated():
    last_proof_block_hash = bytes.fromhex("ab" * 32)
    topology_hash = bytes.fromhex("cd" * 32)
    difficulty = SubstrateDifficulty(
        min_solutions=5,
        max_energy_milli=-4100_000,
        min_diversity_milli=150,
    )
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3)]
    encoded = _build_snapshot_hex(
        last_proof_block_hash=last_proof_block_hash,
        difficulty=difficulty,
        topology_hash=topology_hash,
        nodes=nodes,
        edges=edges,
    )

    decoded = _decode_mining_snapshot(encoded)

    assert decoded is not None
    assert decoded["last_proof_block_hash"] == last_proof_block_hash
    assert decoded["topology_hash"] == topology_hash
    assert decoded["nodes"] == nodes
    assert decoded["edges"] == edges
    assert decoded["difficulty"] == difficulty
    assert decoded["allowed_h_values"] == AllowedValueSet((-1000, 0, 1000))
    assert decoded["allowed_j_values"] == AllowedValueSet((-1000, 1000))
    assert decoded["allowed_spin_values"] == AllowedValueSet((-1000, 1000))


def test_decode_mixed_spec_variants():
    """Decoder correctly handles a mix of Set / IntegerRange / ContinuousRange."""
    encoded = _build_snapshot_hex(
        last_proof_block_hash=b"\x00" * 32,
        difficulty=SubstrateDifficulty(1, -100, 0),
        topology_hash=b"\xcd" * 32,
        nodes=[0, 1],
        edges=[(0, 1)],
        spec_h=_encode_integer_range(-3, 3),
        spec_j=_encode_continuous_range(-1000, 1000),
        spec_spin=_encode_set([-1000, 1000]),
    )
    decoded = _decode_mining_snapshot(encoded)
    assert decoded["allowed_h_values"] == AllowedValueIntegerRange(min=-3, max=3)
    assert decoded["allowed_j_values"] == AllowedValueContinuousRange(
        min=-1000, max=1000
    )
    assert decoded["allowed_spin_values"] == AllowedValueSet((-1000, 1000))


def test_decode_zephyr_sized_graph():
    """The default Z(9,2) topology has 1368 nodes / 7692 edges — make sure the
    compact-length decoder handles vec lengths above the 1-byte boundary."""
    nodes = list(range(1368))
    edges = [(i, i + 1) for i in range(1367)]
    encoded = _build_snapshot_hex(
        last_proof_block_hash=b"\x10" * 32,
        difficulty=SubstrateDifficulty(5, -4_100_000, 150),
        topology_hash=b"\x20" * 32,
        nodes=nodes,
        edges=edges,
    )

    decoded = _decode_mining_snapshot(encoded)

    assert decoded is not None
    assert len(decoded["nodes"]) == 1368
    assert len(decoded["edges"]) == 1367
    assert decoded["nodes"][-1] == 1367
    assert decoded["edges"][-1] == (1366, 1367)


def test_decode_rejects_trailing_bytes():
    """A trailing byte after a valid Option::None tag must fail loud — a
    future runtime upgrade that appends a field would otherwise be silently
    dropped."""
    with pytest.raises(ValueError, match="trailing bytes"):
        _decode_mining_snapshot("0x00ff")


def test_decode_field_error_includes_field_name():
    """Truncating mid-decode reports the failing field, not a generic
    decoding error."""
    # Option::Some tag, then only a partial last_proof_block_hash. Error
    # message should name the failing field.
    truncated = "0x01" + ("ab" * 10)
    with pytest.raises(ValueError, match="last_proof_block_hash"):
        _decode_mining_snapshot(truncated)


# ----------------------------------------------------------------------
# get_mining_snapshot — the round-seed contract means there is no
# `at`-vs-head offset anymore. The seed is stable across the round.
# ----------------------------------------------------------------------


async def _make_snapshot_client_with_fake_rpc(encoded_hex: str) -> SubstrateClient:
    """Build a SubstrateClient whose `_iface.rpc_request` returns a known
    snapshot, bypassing connect() / websocket setup."""
    client = SubstrateClient.__new__(SubstrateClient)
    client.url = "ws://test.invalid:0"
    client._iface = None
    client._lock = None  # type: ignore[assignment]

    class _FakeIface:
        def rpc_request(self, method, params):  # noqa: D401
            return {"result": encoded_hex}

    client._iface = _FakeIface()  # type: ignore[assignment]

    async def _direct_run(fn):
        return fn()

    client._run = _direct_run  # type: ignore[assignment]
    return client


async def test_get_mining_snapshot_returns_last_proof_block_hash():
    """The returned context carries the snapshot's `last_proof_block_hash`
    verbatim — no chain-state lookup, no offset, no override from `at`."""
    seed = b"\xab" * 32
    encoded = _build_snapshot_hex(
        last_proof_block_hash=seed,
        difficulty=SubstrateDifficulty(1, -1000, 0),
        topology_hash=b"\xcd" * 32,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
    )
    client = await _make_snapshot_client_with_fake_rpc(encoded)

    ctx = await client.get_mining_snapshot(
        at=b"\x77" * 32,  # head hash — irrelevant to last proof block hash
        topology_hash=None,
        miner_account_bytes=b"\x42" * 32,
    )

    assert ctx is not None
    assert ctx.last_proof_block_hash == seed


async def test_get_mining_snapshot_at_none_returns_same_seed():
    """State-probe callers (e.g. bootstrap idempotency check) pass `at=None`.
    The last proof block hash is the same value either way — no offset branch."""
    seed = b"\x99" * 32
    encoded = _build_snapshot_hex(
        last_proof_block_hash=seed,
        difficulty=SubstrateDifficulty(1, -1000, 0),
        topology_hash=b"\xcd" * 32,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
    )
    client = await _make_snapshot_client_with_fake_rpc(encoded)

    ctx = await client.get_mining_snapshot(
        at=None,
        topology_hash=None,
        miner_account_bytes=b"\x42" * 32,
    )

    assert ctx is not None
    assert ctx.last_proof_block_hash == seed
