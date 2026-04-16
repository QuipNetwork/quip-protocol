"""Tests for Node.chain_by_hash index and build_locator helper.

Exercises the hash-indexed chain helpers in isolation from the full
``Node.__init__`` path (miner setup, crypto keys, multiprocess logging).
Node instances are constructed via ``object.__new__`` with only the
attributes the code under test reads.
"""

from types import SimpleNamespace

import pytest

from shared.node import Node


def _hash(i: int) -> bytes:
    """Deterministic 32-byte block hash for index ``i`` (big-endian)."""
    return i.to_bytes(32, "big")


def _index_from_hash(h: bytes) -> int:
    """Recover the synthetic index from a hash produced by ``_hash``."""
    return int.from_bytes(h, "big")


def _fake_block(index: int, block_hash: bytes | None):
    """A minimal stand-in for ``Block`` that exposes ``.header.index`` and ``.hash``."""
    return SimpleNamespace(header=SimpleNamespace(index=index), hash=block_hash)


def _new_node() -> Node:
    """Construct a Node shell with empty chain and hash index."""
    node = object.__new__(Node)
    node.chain = []
    node.chain_by_hash = {}
    return node


def _node_with_genesis(genesis_hash: bytes = _hash(0)) -> Node:
    node = _new_node()
    node._index_append(_fake_block(0, genesis_hash))
    return node


# ---------------------------------------------------------------------------
# get_block_by_hash
# ---------------------------------------------------------------------------


def test_get_block_by_hash_returns_indexed_block():
    node = _node_with_genesis()
    genesis_hash = _hash(0)
    result = node.get_block_by_hash(genesis_hash)
    assert result is not None
    assert result.header.index == 0


def test_get_block_by_hash_unknown_hash_returns_none():
    node = _node_with_genesis()
    assert node.get_block_by_hash(_hash(999)) is None


# ---------------------------------------------------------------------------
# _index_append invariants
# ---------------------------------------------------------------------------


def test_index_append_updates_both_chain_and_hash_index():
    node = _node_with_genesis()
    b1 = _fake_block(1, _hash(1))
    node._index_append(b1)
    assert len(node.chain) == 2
    assert node.chain[1] is b1
    assert node.get_block_by_hash(_hash(1)) is b1


def test_index_append_skips_hash_index_for_unfinalized_block():
    """Blocks with no hash are appended to ``chain`` but not indexed."""
    node = _node_with_genesis()
    unfinalized = _fake_block(1, None)
    node._index_append(unfinalized)
    assert len(node.chain) == 2
    assert node.chain[1] is unfinalized
    # The hash index only holds the genesis entry.
    assert len(node.chain_by_hash) == 1


# ---------------------------------------------------------------------------
# _index_truncate invariants
# ---------------------------------------------------------------------------


def test_index_truncate_evicts_dropped_blocks_from_hash_index():
    node = _node_with_genesis()
    for i in range(1, 5):
        node._index_append(_fake_block(i, _hash(i)))
    assert len(node.chain) == 5
    assert node.get_block_by_hash(_hash(3)) is not None

    node._index_truncate(2)  # keep genesis + block 1

    assert len(node.chain) == 2
    assert node.get_block_by_hash(_hash(0)) is not None
    assert node.get_block_by_hash(_hash(1)) is not None
    for dropped in (2, 3, 4):
        assert node.get_block_by_hash(_hash(dropped)) is None


def test_index_truncate_noop_when_new_length_exceeds_chain():
    node = _node_with_genesis()
    node._index_append(_fake_block(1, _hash(1)))
    original_chain = list(node.chain)
    original_index = dict(node.chain_by_hash)

    node._index_truncate(100)  # larger than current chain

    assert node.chain == original_chain
    assert node.chain_by_hash == original_index


def test_index_truncate_to_zero_clears_index():
    node = _node_with_genesis()
    for i in range(1, 4):
        node._index_append(_fake_block(i, _hash(i)))
    node._index_truncate(0)
    assert node.chain == []
    assert node.chain_by_hash == {}


# ---------------------------------------------------------------------------
# build_locator
# ---------------------------------------------------------------------------


def test_build_locator_empty_chain_returns_empty():
    node = _new_node()
    assert node.build_locator() == []


def test_build_locator_genesis_only_returns_single_entry():
    genesis_hash = _hash(42)
    node = _new_node()
    node._index_append(_fake_block(0, genesis_hash))
    assert node.build_locator() == [genesis_hash]


def test_build_locator_short_chain_is_contiguous_to_genesis():
    """A chain shorter than the 10-entry linear prefix is walked fully."""
    node = _node_with_genesis()
    for i in range(1, 6):
        node._index_append(_fake_block(i, _hash(i)))
    locator = node.build_locator()
    # tip first, working backward to genesis
    assert locator == [_hash(5), _hash(4), _hash(3), _hash(2), _hash(1), _hash(0)]


def test_build_locator_first_eleven_entries_are_contiguous():
    """First 11 entries walk the chain one block at a time from the tip."""
    node = _node_with_genesis()
    for i in range(1, 50):
        node._index_append(_fake_block(i, _hash(i)))
    locator = node.build_locator()
    indices = [_index_from_hash(h) for h in locator]
    assert indices[:11] == list(range(49, 38, -1))


def test_build_locator_long_chain_gaps_grow_then_ends_at_genesis():
    """After the linear prefix the step size doubles, and genesis is last."""
    node = _node_with_genesis()
    for i in range(1, 101):
        node._index_append(_fake_block(i, _hash(i)))
    locator = node.build_locator()
    indices = [_index_from_hash(h) for h in locator]

    # First entry is the tip; last entry is genesis.
    assert indices[0] == 100
    assert indices[-1] == 0

    # Each gap is monotonically non-decreasing up to the final jump to genesis.
    # The final gap (the jump into genesis) is allowed to be smaller than the
    # preceding one because of max(0, index - step) clamping.
    gaps = [indices[i] - indices[i + 1] for i in range(len(indices) - 1)]
    for i in range(1, len(gaps) - 1):
        assert gaps[i] >= gaps[i - 1], (
            f"locator gaps should be non-decreasing in the middle: "
            f"gaps[{i}]={gaps[i]} < gaps[{i - 1}]={gaps[i - 1]}"
        )

    # Length is ~11 + log2(100) ≈ 18 entries.
    assert 14 <= len(locator) <= 22


def test_build_locator_skips_unfinalized_middle_blocks():
    """A block with no hash is omitted from the locator but doesn't break it."""
    node = _node_with_genesis()
    node._index_append(_fake_block(1, _hash(1)))
    node._index_append(_fake_block(2, None))  # unfinalized, hash=None
    node._index_append(_fake_block(3, _hash(3)))
    locator = node.build_locator()
    # Index 2 is skipped; tip and genesis remain.
    assert _hash(3) in locator
    assert _hash(1) in locator
    assert _hash(0) in locator
    assert None not in locator
