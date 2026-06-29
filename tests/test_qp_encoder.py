# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""QPEncoder must be byte-identical to the SDK's encode_problem_as_qp.

The vectorized fast-submit path bypasses dimod + sample_bqm and builds the SAPI
``qp`` wire data straight from numpy arrays. If a single byte differs from what
the SDK would have produced, the QPU would solve a different problem and break
consensus — so these tests compare the two encoders directly (no connection)
across the tricky cases: inactive (NaN) qubits, active-but-non-edge (0) couplers,
couplers touching a defective qubit (excluded), and the defect-clamp path.
"""

from __future__ import annotations

from types import SimpleNamespace

import dimod
from dwave.cloud.coders import encode_problem_as_qp

from QPU.dwave_sampler import QPEncoder
from shared.problem_prep import live_topology, rebuild_ising, reduce_to_arrays

# Hardware-ish encoding order: qubit 4 is never in our problem (tests NaN);
# coupler (0,2) is active-active but not an edge (tests 0); (3,4) touches the
# inactive qubit 4 (tests exclusion).
_ENC_QUBITS = [0, 1, 2, 3, 4]
_ENC_COUPLERS = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (3, 4)]

_NODES = [0, 1, 2, 3]
_EDGES = [(0, 1), (1, 2), (2, 3), (0, 3)]
_H = {0: 0.5, 1: -1.0, 2: 0.3, 3: 0.8}
_J = {(0, 1): 1.0, (1, 2): -0.5, (2, 3): 0.7, (0, 3): -0.2}
_NONCE = (7).to_bytes(32, "big")


def _sdk_encode(h_vec, j_vec, live_nodes, live_edges, offset=0.0):
    """What the dimod path produces: arrays -> dicts -> dimod -> encode_problem_as_qp."""
    solver = SimpleNamespace(
        _encoding_qubits=_ENC_QUBITS, _encoding_couplers=_ENC_COUPLERS,
    )
    h, J = rebuild_ising(h_vec, j_vec, live_nodes, live_edges)
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    return encode_problem_as_qp(
        solver, dict(bqm.linear), dict(bqm.quadratic), offset,
        undirected_biases=True,
    )


def _assert_identical(dq, de):
    live_nodes, live_edges = live_topology(_NODES, _EDGES, dq, de)
    h_vec, j_vec, _ = reduce_to_arrays(_H, _J, _NONCE, dq, de, live_nodes, live_edges)

    sdk = _sdk_encode(h_vec, j_vec, live_nodes, live_edges)
    fast = QPEncoder(_ENC_QUBITS, _ENC_COUPLERS, live_nodes, live_edges).encode(
        h_vec, j_vec, 0.0,
    )

    assert fast["format"] == sdk["format"]
    assert fast["offset"] == sdk["offset"]
    assert fast["lin"] == sdk["lin"], "linear wire bytes differ from the SDK encoder"
    assert fast["quad"] == sdk["quad"], "quadratic wire bytes differ from the SDK encoder"


def test_byte_identical_no_defects():
    # qubit 4 inactive (NaN), coupler (0,2) active-non-edge (0), (3,4) excluded.
    _assert_identical([], set())


def test_byte_identical_with_defective_qubit():
    # Clamp qubit 2: it becomes inactive (NaN in lin) and its couplers drop out;
    # neighbor h-fields are adjusted by the clamp — both encoders must still agree.
    _assert_identical([2], set())


def test_byte_identical_with_defective_coupler():
    # (0,3) coupler offline between two live qubits: removed from the submitted
    # problem (encoded 0), preserved for reconstruction — encoders must agree.
    _assert_identical([], {(0, 3)})
