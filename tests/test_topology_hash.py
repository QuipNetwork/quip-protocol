"""Unit tests for shared.topology_hash.

Verifies the post-MR-!20 contract that the topology hash binds the three
:class:`shared.allowed_value_spec.AllowedValueSpec` instances in addition
to the (nodes, edges) graph structure.
"""
from __future__ import annotations

from shared.allowed_value_spec import (
    AllowedValueContinuousRange,
    AllowedValueIntegerRange,
    AllowedValueSet,
)
from shared.topology_hash import topology_hash


_BIN = AllowedValueSet((-1000, 1000))
_TER = AllowedValueSet((-1000, 0, 1000))


def test_returns_thirty_two_bytes():
    h = topology_hash([0, 1, 2], [(0, 1), (1, 2)], _TER, _BIN, _BIN)
    assert isinstance(h, bytes)
    assert len(h) == 32


def test_node_order_invariant():
    h1 = topology_hash([0, 1, 2], [(0, 1), (1, 2)], _TER, _BIN, _BIN)
    h2 = topology_hash([2, 0, 1], [(0, 1), (1, 2)], _TER, _BIN, _BIN)
    assert h1 == h2


def test_edge_order_and_endpoint_swap_invariant():
    h1 = topology_hash([0, 1, 2], [(0, 1), (1, 2)], _TER, _BIN, _BIN)
    h2 = topology_hash([0, 1, 2], [(1, 2), (1, 0)], _TER, _BIN, _BIN)
    assert h1 == h2


def test_allowed_h_change_perturbs_hash():
    h1 = topology_hash([0, 1, 2], [(0, 1), (1, 2)], _TER, _BIN, _BIN)
    h2 = topology_hash([0, 1, 2], [(0, 1), (1, 2)], _BIN, _BIN, _BIN)
    assert h1 != h2


def test_allowed_j_change_perturbs_hash():
    h1 = topology_hash([0, 1, 2], [(0, 1), (1, 2)], _TER, _BIN, _BIN)
    h2 = topology_hash([0, 1, 2], [(0, 1), (1, 2)], _TER, _TER, _BIN)
    assert h1 != h2


def test_allowed_spin_change_perturbs_hash():
    h1 = topology_hash([0, 1, 2], [(0, 1), (1, 2)], _TER, _BIN, _BIN)
    h2 = topology_hash(
        [0, 1, 2],
        [(0, 1), (1, 2)],
        _TER,
        _BIN,
        AllowedValueIntegerRange(min=-1, max=1),
    )
    assert h1 != h2


def test_continuous_spec_distinguishes_from_set():
    h1 = topology_hash(
        [0, 1, 2],
        [(0, 1), (1, 2)],
        _TER,
        _BIN,
        AllowedValueSet((-1000, 1000)),
    )
    h2 = topology_hash(
        [0, 1, 2],
        [(0, 1), (1, 2)],
        _TER,
        _BIN,
        AllowedValueContinuousRange(min=-1000, max=1000),
    )
    assert h1 != h2


def test_set_element_order_invariant_within_spec():
    h1 = topology_hash(
        [0, 1, 2],
        [(0, 1), (1, 2)],
        AllowedValueSet((-1000, 0, 1000)),
        _BIN,
        _BIN,
    )
    h2 = topology_hash(
        [0, 1, 2],
        [(0, 1), (1, 2)],
        AllowedValueSet((1000, -1000, 0)),
        _BIN,
        _BIN,
    )
    assert h1 == h2
