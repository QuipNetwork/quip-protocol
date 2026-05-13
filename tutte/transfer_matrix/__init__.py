"""Transfer Matrix module for computing Tutte polynomials of lattice strip graphs.

Detects periodic lattice strips (grid, triangular, honeycomb, square-octagon,
elongated triangular) and computes
their Tutte polynomials via transfer-matrix multiplication over non-crossing
partition states. Includes a C-accelerated path (cffi) with CRT fallback
for large coefficients.

Pipeline: lattice detection -> transfer matrix build -> matrix-vector sweep
-> FK extraction -> Tutte polynomial.

Public API:
    compute_tutte_via_transfer_matrix(graph) -> Optional[TuttePolynomial]
    detect_periodic_strip(graph, fp) -> Optional[StripProperties]
"""

from __future__ import annotations

from typing import Optional

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .core import (
    build_transfer_matrix,
    enumerate_noncrossing_partitions,
    partition_index_map,
    CATALAN_NUMBERS,
    MAX_TRANSFER_MATRIX_WIDTH,
)
from .sweep import direct_multiply, build_initial_vector
from .extraction import extract_tutte_polynomial, _convert_ab_to_xy
from .lattice_recognition import detect_periodic_strip
from ..family_recognition import compute_structural_fingerprint

__all__ = [
    'compute_tutte_via_transfer_matrix',
    'detect_periodic_strip',
    'build_transfer_matrix',
    'enumerate_noncrossing_partitions',
    'partition_index_map',
    'CATALAN_NUMBERS',
    'MAX_TRANSFER_MATRIX_WIDTH',
    'direct_multiply',
    'build_initial_vector',
    'extract_tutte_polynomial',
]


def compute_tutte_via_transfer_matrix(
    graph: Graph,
) -> Optional[TuttePolynomial]:
    """Compute the Tutte polynomial of a lattice strip via the transfer-matrix pipeline.

    Detects whether the graph is a periodic lattice strip (grid, triangular,
    honeycomb, or square-octagon). If so, builds transfer matrices from the unit cell edge
    pattern, runs matrix-vector multiplication across all columns, and
    extracts the Tutte polynomial from the final FK state vector.

    Tries a C-accelerated path first (full sweep in C, binomial conversion
    in Python). Falls back to pure Python if the C extension is unavailable.

    Args:
        graph: Input graph (simple, undirected).

    Returns:
        TuttePolynomial if the graph is a recognized lattice strip,
        None otherwise

    Time complexity: O(V + E) for detection, O((length - 1) * c_m^2 * P)
        for computation, where c_m = Catalan(width).
    """

    fp = compute_structural_fingerprint(graph)
    strip = detect_periodic_strip(graph, fp)

    if strip is None:
        return None

    width, length, transition_patterns, num_vertices, first_col_edges = strip

    # transition_patterns is a list of edge-pattern lists from detect_periodic_strip.
    # Single-element for grid/triangular, two-element for honeycomb.
    is_single_period = len(transition_patterns) == 1

    # Try C-accelerated path.
    try:
        if is_single_period:
            from ._c_extension import c_transfer_matrix_sweep
            ab_result = c_transfer_matrix_sweep(
                width, length, transition_patterns[0], num_vertices
            )
        else:
            from ._c_extension import c_transfer_matrix_sweep_multi
            fc = first_col_edges if first_col_edges is not None else [
                (i, i + 1) for i in range(width - 1)
            ]
            ab_result = c_transfer_matrix_sweep_multi(
                width, length, transition_patterns, fc, num_vertices
            )
        if ab_result is not None:
            return _convert_ab_to_xy(ab_result)
    except Exception:
        pass

    # Pure Python fallback.
    if is_single_period:
        transfer_mat = build_transfer_matrix(width, transition_patterns[0])
        final_vector = direct_multiply(
            transfer_mat, width, length,
            first_col_edges=first_col_edges,
        )
    else:
        matrices = [build_transfer_matrix(width, p) for p in transition_patterns]
        final_vector = direct_multiply(
            matrices[0], width, length,
            first_col_edges=first_col_edges,
            additional_matrices=matrices[1:],
        )

    return extract_tutte_polynomial(final_vector, width, num_vertices)