# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""generate_ising_arrays_from_nonce must equal the scalar dict generator.

The vectorized generator (ChaCha8 keystream as one numpy array + vectorized
sampling) is the feeder's throughput fix, but it derives the SAME problem the
chain validator re-derives — so every value must match the proven scalar path
exactly. These compare the two directly, on a subgraph and the full topology.
"""

from __future__ import annotations

from dwave_topologies import DEFAULT_TOPOLOGY as _T
from shared.allowed_value_spec import AllowedValueSet
from shared.quantum_proof_of_work import (
    DEFAULT_ALLOWED_H,
    DEFAULT_ALLOWED_J,
    derive_nonce,
    generate_ising_arrays_from_nonce,
    generate_ising_model_from_nonce,
)

_NODES = list(range(12))
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (2, 8),
          (8, 9), (9, 10), (10, 11), (4, 11), (1, 7), (3, 9), (0, 6)]


def _assert_identical(nonce, nodes, edges, allowed_h, allowed_j):
    h, j = generate_ising_model_from_nonce(nonce, nodes, edges, allowed_h, allowed_j)
    h_arr, j_arr = generate_ising_arrays_from_nonce(
        nonce, nodes, edges, allowed_h, allowed_j,
    )
    assert h_arr.shape == (len(nodes),)
    assert j_arr.shape == (len(edges),)
    for i, n in enumerate(nodes):
        assert h_arr[i] == h[int(n)], f"h mismatch at node {n}"
    for k, (u, v) in enumerate(edges):
        assert j_arr[k] == j[(int(u), int(v))], f"j mismatch at edge {(u, v)}"


def test_default_specs_byte_identical():
    for s in range(6):
        nonce = derive_nonce(bytes([s]) * 32, b"\x02" * 32, b"\x03" * 32)
        _assert_identical(nonce, _NODES, _EDGES, DEFAULT_ALLOWED_H, DEFAULT_ALLOWED_J)


def test_zero_field_h_byte_identical():
    # The deployed topology registers zero-field h (J-only problem class).
    nonce = derive_nonce(b"\x11" * 32, b"\x22" * 32, b"\x33" * 32)
    _assert_identical(nonce, _NODES, _EDGES, AllowedValueSet((0,)), DEFAULT_ALLOWED_J)


def test_full_topology_byte_identical():
    nodes = list(_T.nodes)
    edges = list(_T.edges)
    nonce = derive_nonce(b"\x44" * 32, b"\x55" * 32, b"\x66" * 32)
    _assert_identical(nonce, nodes, edges, AllowedValueSet((0,)), DEFAULT_ALLOWED_J)
