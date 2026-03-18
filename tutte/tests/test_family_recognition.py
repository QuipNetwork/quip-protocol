"""Test suite for the family recognition module.

Tests recognize_family() on all supported graph families, verifying:
  1. Recognition succeeds (returns a polynomial, not None)
  2. Correctness via Kirchhoff: T(1,1) = spanning tree count
  3. Correctness via T(2,2) = 2^|E|

Families tested:
  - Tier 1 (closed-form): tree/path, cycle, wheel, fan, pan, sunlet, helm, book
  - Tier 2 (recurrence): ladder, gear, prism, Möbius, grid

Usage:
    pytest tests/tutte/test_family_recognition.py -v
"""

import networkx as nx
import pytest

from tutte.graph import Graph
from tutte.family_recognition import recognize_family
from tutte.validation import count_spanning_trees_kirchhoff


# ===========================================================================
# Graph builders
# ===========================================================================

def _build_gear(k: int) -> nx.Graph:
    """Build gear graph: hub + k rim vertices + k subdivision vertices."""
    G = nx.Graph()
    hub = 0
    rim = list(range(1, k + 1))
    sub = list(range(k + 1, 2 * k + 1))
    for i in range(k):
        G.add_edge(hub, rim[i])
        G.add_edge(rim[i], sub[i])
        G.add_edge(sub[i], rim[(i + 1) % k])
    return G


def _build_helm(k: int) -> nx.Graph:
    """Build helm: wheel W_k + pendant at each rim vertex."""
    G = nx.wheel_graph(k + 1)
    for i in range(1, k + 1):
        G.add_edge(i, k + i)
    return G


def _build_book(k: int) -> nx.Graph:
    """Build book: k triangles sharing edge (0, 1)."""
    G = nx.Graph()
    G.add_edge(0, 1)
    for i in range(k):
        v = i + 2
        G.add_edge(0, v)
        G.add_edge(1, v)
    return G


def _build_pan(cycle_size: int) -> nx.Graph:
    """Build pan: cycle of size cycle_size + one pendant edge."""
    G = nx.cycle_graph(cycle_size)
    G.add_edge(0, cycle_size)
    return G


def _build_sunlet(k: int) -> nx.Graph:
    """Build sunlet: C_k with pendant at each vertex."""
    G = nx.cycle_graph(k)
    for i in range(k):
        G.add_edge(i, k + i)
    return G


def _build_mobius(k: int) -> nx.Graph:
    """Build Möbius ladder: 2k-cycle with k rungs connecting v_i to v_{i+k}."""
    G = nx.cycle_graph(2 * k)
    for i in range(k):
        G.add_edge(i, i + k)
    return G


# ===========================================================================
# Verification helper
# ===========================================================================

def _verify_recognition(G_nx: nx.Graph, family_name: str, expected_recognized: bool = True):
    """Recognize the graph and verify the polynomial is correct."""
    graph = Graph.from_networkx(G_nx)
    poly = recognize_family(graph)

    if not expected_recognized:
        assert poly is None, f"{family_name}: should NOT be recognized but got polynomial"
        return

    assert poly is not None, f"{family_name}: not recognized (returned None)"

    # Kirchhoff: T(1,1) = spanning tree count
    trees = poly.num_spanning_trees()
    kirchhoff = count_spanning_trees_kirchhoff(graph)
    assert trees == kirchhoff, (
        f"{family_name}: T(1,1)={trees} != Kirchhoff={kirchhoff}"
    )

    # T(2,2) = 2^|E|
    e = G_nx.number_of_edges()
    t22 = poly.evaluate(2, 2)
    assert t22 == 2 ** e, (
        f"{family_name}: T(2,2)={t22} != 2^{e}={2 ** e}"
    )


# ===========================================================================
# Tier 1 — Closed-form families
# ===========================================================================

class TestTier1ClosedForm:
    """Families with O(1) polynomial computation after detection."""

    @pytest.mark.parametrize("n", range(3, 40))
    def test_path(self, n):
        """Path P_n (tree): T = x^(n-1)."""
        _verify_recognition(nx.path_graph(n), f"P_{n}")

    @pytest.mark.parametrize("n", range(3, 40))
    def test_cycle(self, n):
        """Cycle C_n: T = x^(n-1) + ... + x + y."""
        _verify_recognition(nx.cycle_graph(n), f"C_{n}")

    @pytest.mark.parametrize("k", range(3, 20))
    def test_wheel(self, k):
        """Wheel W_k (k+1 vertices)."""
        _verify_recognition(nx.wheel_graph(k + 1), f"W_{k}")

    @pytest.mark.parametrize("k", range(3, 20))
    def test_fan(self, k):
        """Fan F_k (k+1 vertices)."""
        G = nx.Graph()
        # Apex 0 connected to path 1..k
        for i in range(1, k + 1):
            G.add_edge(0, i)
        for i in range(1, k):
            G.add_edge(i, i + 1)
        _verify_recognition(G, f"F_{k}")

    @pytest.mark.parametrize("cycle_size", range(4, 30))
    def test_pan(self, cycle_size):
        """Pan: C_n with one pendant."""
        _verify_recognition(_build_pan(cycle_size), f"Pan_{cycle_size}")

    @pytest.mark.parametrize("k", range(3, 20))
    def test_sunlet(self, k):
        """Sunlet: C_k with pendant at each vertex."""
        _verify_recognition(_build_sunlet(k), f"Sunlet_{k}")

    @pytest.mark.parametrize("k", range(3, 15))
    def test_helm(self, k):
        """Helm: W_k with pendant at each rim vertex."""
        _verify_recognition(_build_helm(k), f"Helm_{k}")

    @pytest.mark.parametrize("k", range(1, 20))
    def test_book(self, k):
        """Book: k triangles sharing one edge."""
        _verify_recognition(_build_book(k), f"Book_{k}")


# ===========================================================================
# Tier 2 — Recurrence-based families
# ===========================================================================

class TestTier2Recurrence:
    """Families with O(k) polynomial computation after detection."""

    @pytest.mark.parametrize("k", range(2, 20))
    def test_ladder(self, k):
        """Ladder P_k x P_2."""
        _verify_recognition(nx.ladder_graph(k), f"Ladder_{k}")

    @pytest.mark.parametrize("k", range(3, 15))
    def test_gear(self, k):
        """Gear: wheel with subdivided rim edges."""
        _verify_recognition(_build_gear(k), f"Gear_{k}")

    @pytest.mark.parametrize("k", range(3, 15))
    def test_prism(self, k):
        """Prism (circular ladder) C_k x K_2."""
        _verify_recognition(nx.circular_ladder_graph(k), f"Prism_{k}")

    @pytest.mark.parametrize("k", range(3, 15))
    def test_mobius(self, k):
        """Möbius ladder: 2k-cycle + k rungs connecting v_i to v_{i+k}."""
        _verify_recognition(_build_mobius(k), f"Mobius_{k}")

    @pytest.mark.parametrize("m,n", [
        (1, n) for n in range(2, 10)
    ] + [
        (2, n) for n in range(2, 10)
    ])
    def test_grid(self, m, n):
        """Grid P_m x P_n."""
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, n))
        _verify_recognition(G, f"Grid_{m}x{n}")


# ===========================================================================
# Negative cases — should NOT be recognized
# ===========================================================================

class TestNotRecognized:
    """Graphs that should NOT be recognized as any family."""

    def test_petersen(self):
        """Petersen graph: 3-regular but not prism/Möbius."""
        _verify_recognition(nx.petersen_graph(), "Petersen", expected_recognized=False)

    def test_complete_k5(self):
        """K_5: not a recognized family."""
        _verify_recognition(nx.complete_graph(5), "K_5", expected_recognized=False)

    def test_complete_bipartite_k33(self):
        """K_{3,3}: this is actually M_3 (Möbius ladder), so it SHOULD be recognized."""
        # K_{3,3} is the Möbius ladder M_3 — verify it IS recognized
        _verify_recognition(
            nx.complete_bipartite_graph(3, 3), "K_{3,3}", expected_recognized=True
        )

    def test_complete_bipartite_k44(self):
        """K_{4,4}: 4-regular, not a recognized family."""
        _verify_recognition(
            nx.complete_bipartite_graph(4, 4), "K_{4,4}", expected_recognized=False
        )
