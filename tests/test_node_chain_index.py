"""Tests for Node.chain_by_hash index and build_locator helper."""

from types import SimpleNamespace

from shared.node import Node


def _hash(i: int) -> bytes:
    return i.to_bytes(32, "big")


def _block(index: int, block_hash: bytes | None):
    return SimpleNamespace(header=SimpleNamespace(index=index), hash=block_hash)


def _node(genesis_hash: bytes = _hash(0)) -> Node:
    node = object.__new__(Node)
    node.chain = []
    node.chain_by_hash = {}
    node._index_append(_block(0, genesis_hash))
    return node


def test_get_block_by_hash_hits_and_misses():
    node = _node()
    assert node.get_block_by_hash(_hash(0)).header.index == 0
    assert node.get_block_by_hash(_hash(999)) is None


def test_index_append_updates_both_structures():
    node = _node()
    node._index_append(_block(1, _hash(1)))
    node._index_append(_block(2, None))  # unfinalized — chain only, not index
    assert [b.header.index for b in node.chain] == [0, 1, 2]
    assert set(node.chain_by_hash) == {_hash(0), _hash(1)}


def test_index_truncate_evicts_and_is_noop_for_oversize():
    node = _node()
    for i in range(1, 4):
        node._index_append(_block(i, _hash(i)))
    # No-op when new_length >= len(chain).
    node._index_truncate(999)
    assert len(node.chain) == 4
    # Drop blocks 2 and 3.
    node._index_truncate(2)
    assert [b.header.index for b in node.chain] == [0, 1]
    assert set(node.chain_by_hash) == {_hash(0), _hash(1)}


def test_build_locator_empty_and_genesis_only():
    empty = object.__new__(Node)
    empty.chain = []
    empty.chain_by_hash = {}
    assert empty.build_locator() == []
    assert _node().build_locator() == [_hash(0)]


def test_build_locator_linear_then_doubling_ends_at_genesis():
    """First 11 entries are contiguous from the tip; then gaps grow; genesis last."""
    node = _node()
    for i in range(1, 101):
        node._index_append(_block(i, _hash(i)))

    locator = node.build_locator()
    indices = [int.from_bytes(h, "big") for h in locator]

    # Contiguous prefix from the tip and genesis at the end.
    assert indices[:11] == list(range(100, 89, -1))
    assert indices[-1] == 0

    # Middle gaps are non-decreasing up to the final jump into genesis.
    gaps = [indices[i] - indices[i + 1] for i in range(len(indices) - 1)]
    for i in range(1, len(gaps) - 1):
        assert gaps[i] >= gaps[i - 1]
