"""Unit tests for the SCALE decoder in `substrate_client._decode_mining_snapshot`.

These don't need a live chain — they hand-build the SCALE bytes and check that
the decoder yields the expected `SubstrateMiningContext`. They cover both the
`None` (option-tag 0x00) case and the populated case, including the
milli-precision difficulty fields and the variable-length nodes/edges vecs.

Phase 2's chain bootstrap will exercise the same path against real on-chain
data; this test exists so the decoder is locked down before then.
"""
from __future__ import annotations

import pytest

from shared.substrate_client import SubstrateClient, _decode_mining_snapshot
from shared.substrate_types import SubstrateDifficulty


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
    block_number: int,
    parent_hash: bytes,
    difficulty: SubstrateDifficulty,
    topology_hash: bytes,
    nodes: list[int],
    edges: list[tuple[int, int]],
    is_some: bool = True,
) -> str:
    parts: list[bytes] = []
    parts.append(b"\x01" if is_some else b"\x00")
    if not is_some:
        return "0x" + b"".join(parts).hex()
    parts.append(block_number.to_bytes(4, "little"))
    parts.append(parent_hash)
    parts.append(difficulty.min_solutions.to_bytes(4, "little"))
    parts.append(difficulty.max_energy_milli.to_bytes(8, "little", signed=True))
    parts.append(difficulty.min_diversity_milli.to_bytes(4, "little"))
    parts.append(difficulty.min_quality_milli.to_bytes(4, "little"))
    parts.append(topology_hash)
    parts.append(_encode_compact_u32(len(nodes)))
    for n in nodes:
        parts.append(n.to_bytes(4, "little"))
    parts.append(_encode_compact_u32(len(edges)))
    for u, v in edges:
        parts.append(u.to_bytes(4, "little"))
        parts.append(v.to_bytes(4, "little"))
    return "0x" + b"".join(parts).hex()


def test_decode_none():
    assert _decode_mining_snapshot("0x00") is None


def test_decode_populated():
    parent_hash = bytes.fromhex("ab" * 32)
    topology_hash = bytes.fromhex("cd" * 32)
    difficulty = SubstrateDifficulty(
        min_solutions=5,
        max_energy_milli=-4100_000,
        min_diversity_milli=150,
        min_quality_milli=900,
    )
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3)]
    encoded = _build_snapshot_hex(
        block_number=42,
        parent_hash=parent_hash,
        difficulty=difficulty,
        topology_hash=topology_hash,
        nodes=nodes,
        edges=edges,
    )

    decoded = _decode_mining_snapshot(encoded)

    assert decoded is not None
    assert decoded["block_number"] == 42
    assert decoded["parent_hash"] == parent_hash
    assert decoded["topology_hash"] == topology_hash
    assert decoded["nodes"] == nodes
    assert decoded["edges"] == edges
    assert decoded["difficulty"] == difficulty


def test_decode_zephyr_sized_graph():
    """The default Z(9,2) topology has 1368 nodes / 7692 edges — make sure the
    compact-length decoder handles vec lengths above the 1-byte boundary."""
    nodes = list(range(1368))
    edges = [(i, i + 1) for i in range(1367)]
    encoded = _build_snapshot_hex(
        block_number=12345,
        parent_hash=b"\x10" * 32,
        difficulty=SubstrateDifficulty(5, -4_100_000, 150, 900),
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
    # Option::Some tag, valid block_number, then ran out of bytes for
    # parent_hash. Error message should name the failing field.
    truncated = "0x01" + (5).to_bytes(4, "little").hex() + "ab" * 10
    with pytest.raises(ValueError, match="parent_hash"):
        _decode_mining_snapshot(truncated)


# ----------------------------------------------------------------------
# get_mining_snapshot off-by-one regression test (P1 review finding)
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

    # _run wraps blocking calls in a thread; for tests we'd rather execute
    # synchronously to avoid bringing up an executor. Monkey-patch on the
    # instance so the production async path stays untouched.
    async def _direct_run(fn):
        return fn()

    client._run = _direct_run  # type: ignore[assignment]
    return client


@pytest.mark.parametrize(
    "raw_block, at_hash, expected_block, expected_parent",
    [
        # When `at` is set (mining path), the controller validates against
        # the block where the extrinsic LANDS — that's raw_block + 1, with
        # parent_hash = at (the hash of raw_block).
        (
            42,
            b"\xab" * 32,
            43,
            b"\xab" * 32,
        ),
        (
            12_345,
            b"\x10" * 32,
            12_346,
            b"\x10" * 32,
        ),
        (
            0,
            b"\xff" * 32,
            1,
            b"\xff" * 32,
        ),
    ],
)
async def test_get_mining_snapshot_applies_plus_one_offset_when_at_set(
    raw_block, at_hash, expected_block, expected_parent
):
    """Regression: `submit_proof` validates against `System::block_number()`
    at the block where the extrinsic lands, not the head we queried at.
    Without the +1 offset every proof was rejected with `InvalidNonce`.
    """
    encoded = _build_snapshot_hex(
        block_number=raw_block,
        parent_hash=b"\x99" * 32,  # decoder-emitted; should be OVERWRITTEN by `at`
        difficulty=SubstrateDifficulty(1, -1000, 0, 0),
        topology_hash=b"\xcd" * 32,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
    )
    client = await _make_snapshot_client_with_fake_rpc(encoded)

    ctx = await client.get_mining_snapshot(
        at=at_hash,
        topology_hash=None,
        miner_account_bytes=b"\x42" * 32,
    )

    assert ctx is not None
    assert ctx.block_number == expected_block
    assert ctx.parent_hash == expected_parent


async def test_get_mining_snapshot_no_offset_when_at_none():
    """State-probe callers (e.g. bootstrap idempotency check) pass `at=None`
    and expect the raw snapshot values, not the +1-offset version."""
    encoded = _build_snapshot_hex(
        block_number=42,
        parent_hash=b"\x99" * 32,
        difficulty=SubstrateDifficulty(1, -1000, 0, 0),
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
    assert ctx.block_number == 42  # raw, not 43
    assert ctx.parent_hash == b"\x99" * 32  # decoded, not overwritten
