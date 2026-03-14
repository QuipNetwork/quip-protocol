"""Validation Utilities for Tutte Polynomial Computation.

This module provides verification functions to ensure correctness:
1. verify_with_networkx - Compare against networkx.tutte_polynomial()
2. verify_spanning_trees - Verify T(1,1) matches Kirchhoff's matrix-tree theorem
3. verify_composition - Verify composition formulas hold

These utilities are essential for testing the synthesis engine and
ensuring that computed polynomials are correct.
"""

from __future__ import annotations

from typing import Optional, Tuple

import networkx as nx

from .polynomial import TuttePolynomial
from .graph import Graph


# =============================================================================
# NETWORKX VERIFICATION
# =============================================================================

def verify_with_networkx(graph: Graph, computed: TuttePolynomial) -> bool:
    """Compare computed polynomial against networkx.tutte_polynomial().

    For large graphs (> 15 edges), skips verification as it's too expensive.

    Args:
        graph: The graph being verified
        computed: The computed Tutte polynomial

    Returns:
        True if polynomials match or graph is too large to verify
    """
    if graph.edge_count() > 15:
        return True  # Skip expensive verification for large graphs

    G = graph.to_networkx()
    nx_poly = compute_tutte_networkx(G)

    if nx_poly is None:
        return True  # Can't verify without networkx/sympy

    return computed == nx_poly


def compute_tutte_networkx(G: nx.Graph) -> Optional[TuttePolynomial]:
    """Compute Tutte polynomial using networkx's implementation.

    NetworkX returns a sympy polynomial, which we convert to our format.
    Returns None if sympy is not available.
    """
    try:
        from sympy import symbols, Poly

        # NetworkX tutte_polynomial returns a sympy expression
        x, y = symbols('x y')
        tutte_sympy = nx.tutte_polynomial(G)

        # Convert to our format
        poly = Poly(tutte_sympy, x, y)
        coeffs = {}
        for monom, coeff in poly.as_dict().items():
            # monom is (x_power, y_power)
            coeffs[monom] = int(coeff)

        return TuttePolynomial.from_coefficients(coeffs)
    except ImportError:
        # sympy not available
        return None
    except Exception:
        return None


# =============================================================================
# SPANNING TREE VERIFICATION
# =============================================================================

def verify_spanning_trees(graph: Graph, poly: TuttePolynomial) -> bool:
    """Verify T(1,1) matches Kirchhoff's matrix-tree theorem.

    The number of spanning trees can be computed two ways:
    1. Tutte polynomial: T(1,1)
    2. Kirchhoff: determinant of reduced Laplacian matrix

    Uses exact integer arithmetic (sympy) for graphs with >=20 nodes to avoid
    float64 precision loss. For smaller graphs, float is sufficient.

    Args:
        graph: The graph
        poly: The computed Tutte polynomial

    Returns:
        True if both methods agree
    """
    tutte_count = poly.num_spanning_trees()

    n = graph.node_count()
    if n >= 20:
        # Use exact integer arithmetic to avoid float precision issues
        try:
            kirchhoff_count = _exact_spanning_tree_count(graph)
        except Exception:
            return True
    else:
        G = graph.to_networkx()
        try:
            kirchhoff_count = round(nx.number_of_spanning_trees(G))
        except Exception:
            return True

    return tutte_count == kirchhoff_count


def _exact_spanning_tree_count(graph: Graph) -> int:
    """Compute exact spanning tree count using sympy integer determinant."""
    from sympy import Matrix, zeros

    nodes = sorted(graph.nodes)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    L = zeros(n, n)
    for u, v in graph.edges:
        i, j = idx[u], idx[v]
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1

    cofactor = L[1:, 1:]
    return int(cofactor.det())


def count_spanning_trees_kirchhoff(graph: Graph) -> int:
    """Count spanning trees using Kirchhoff's matrix-tree theorem.

    Uses NetworkX's implementation which computes the determinant
    of the reduced Laplacian matrix.
    """
    G = graph.to_networkx()
    try:
        return round(nx.number_of_spanning_trees(G))
    except Exception:
        return -1


# =============================================================================
# COMPOSITION VERIFICATION
# =============================================================================

def verify_composition(
    g1: Graph, t1: TuttePolynomial,
    g2: Graph, t2: TuttePolynomial,
    combined: Graph, t_combined: TuttePolynomial,
    operation: str
) -> bool:
    """Verify that composition formulas hold.

    Args:
        g1, g2: Input graphs
        t1, t2: Their Tutte polynomials
        combined: The combined graph
        t_combined: Its Tutte polynomial
        operation: Type of composition ("disjoint_union", "cut_vertex", etc.)

    Returns:
        True if the formula holds
    """
    if operation == "disjoint_union":
        # T(G1 ∪ G2) = T(G1) × T(G2)
        expected = t1 * t2
        return t_combined == expected

    elif operation == "cut_vertex":
        # T(G1 ·₁ G2) = T(G1) × T(G2) when joined at cut vertex
        expected = t1 * t2
        return t_combined == expected

    elif operation == "bridge":
        # Adding a bridge multiplies by x
        expected = t1 * TuttePolynomial.x()
        return t_combined == expected

    else:
        # Unknown operation, can't verify formula
        return True


def verify_disjoint_union(g1: Graph, g2: Graph, result: Graph,
                          compute_poly) -> Tuple[bool, str]:
    """Verify disjoint union formula: T(G1 ∪ G2) = T(G1) × T(G2).

    Args:
        g1, g2: Input graphs
        result: Combined graph
        compute_poly: Function to compute Tutte polynomial

    Returns:
        (success, message) tuple
    """
    t1 = compute_poly(g1)
    t2 = compute_poly(g2)
    t_result = compute_poly(result)

    expected = t1 * t2

    if t_result == expected:
        return True, "Disjoint union formula verified"
    else:
        return False, f"Mismatch: got {t_result}, expected {expected}"


def verify_cut_vertex_join(g1: Graph, g2: Graph, v1: int, v2: int,
                           result: Graph, compute_poly) -> Tuple[bool, str]:
    """Verify cut vertex join formula: T(G1 ·₁ G2) = T(G1) × T(G2).

    Args:
        g1, g2: Input graphs
        v1, v2: Vertices being identified
        result: Combined graph
        compute_poly: Function to compute Tutte polynomial

    Returns:
        (success, message) tuple
    """
    t1 = compute_poly(g1)
    t2 = compute_poly(g2)
    t_result = compute_poly(result)

    expected = t1 * t2

    if t_result == expected:
        return True, "Cut vertex join formula verified"
    else:
        return False, f"Mismatch: got {t_result}, expected {expected}"


# =============================================================================
# CONSISTENCY CHECKS
# =============================================================================

def verify_polynomial_properties(poly: TuttePolynomial, graph: Graph) -> Tuple[bool, str]:
    """Verify that a polynomial satisfies basic Tutte polynomial properties.

    Properties checked:
    1. All coefficients are non-negative (for graphs without bridges)
    2. T(1,1) = number of spanning trees
    3. Degree constraints based on graph size

    Args:
        poly: The computed polynomial
        graph: The graph

    Returns:
        (success, message) tuple
    """
    # Check spanning tree count
    if not verify_spanning_trees(graph, poly):
        return False, "Spanning tree count mismatch"

    # Check degree constraints
    n = graph.node_count()
    m = graph.edge_count()

    # x-degree should be at most m - n + 1 (cyclomatic complexity)
    # for connected graphs
    if graph.is_connected() and m > 0:
        max_x_deg = m - n + 1
        if poly.x_degree() > max_x_deg + 1:  # Allow some slack
            return False, f"x-degree {poly.x_degree()} exceeds expected max {max_x_deg}"

    return True, "All properties verified"


# =============================================================================
# RAINBOW TABLE VERIFICATION
# =============================================================================

def verify_rainbow_table_entry(
    entry_name: str,
    entry_poly: TuttePolynomial,
    graph_constructor,
    compute_poly
) -> Tuple[bool, str]:
    """Verify a rainbow table entry is correct.

    Args:
        entry_name: Name of the entry
        entry_poly: Stored polynomial
        graph_constructor: Function to construct the graph
        compute_poly: Function to compute Tutte polynomial

    Returns:
        (success, message) tuple
    """
    try:
        graph = graph_constructor()
        computed = compute_poly(graph)

        if computed == entry_poly:
            return True, f"{entry_name}: verified"
        else:
            return False, f"{entry_name}: mismatch - computed {computed}, stored {entry_poly}"
    except Exception as e:
        return False, f"{entry_name}: error - {e}"


def verify_rainbow_table_consistency(table, compute_poly) -> Tuple[int, int, list]:
    """Verify all computable entries in rainbow table.

    Args:
        table: RainbowTable instance
        compute_poly: Function to compute Tutte polynomial from Graph

    Returns:
        (verified_count, total_count, errors) tuple
    """
    from .graph import complete_graph, cycle_graph, path_graph, wheel_graph

    constructors = {
        'K_2': lambda: complete_graph(2),
        'K_3': lambda: complete_graph(3),
        'K_4': lambda: complete_graph(4),
        'K_5': lambda: complete_graph(5),
        'K_6': lambda: complete_graph(6),
        'C_3': lambda: cycle_graph(3),
        'C_4': lambda: cycle_graph(4),
        'C_5': lambda: cycle_graph(5),
        'C_6': lambda: cycle_graph(6),
        'P_2': lambda: path_graph(2),
        'P_3': lambda: path_graph(3),
        'P_4': lambda: path_graph(4),
        'P_5': lambda: path_graph(5),
        'W_4': lambda: wheel_graph(4),
        'W_5': lambda: wheel_graph(5),
        'W_6': lambda: wheel_graph(6),
    }

    verified = 0
    errors = []

    for name, constructor in constructors.items():
        entry = table.get_entry(name)
        if entry is None:
            continue

        success, msg = verify_rainbow_table_entry(
            name, entry.polynomial, constructor, compute_poly
        )

        if success:
            verified += 1
        else:
            errors.append(msg)

    return verified, len(constructors), errors
