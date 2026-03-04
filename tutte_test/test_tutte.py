"""Tutte polynomial test suite.

Parametrized correctness tests validating synthesis against Kirchhoff's theorem
and NetworkX. Run with: python -m pytest tutte_test/test_tutte.py -v

Sections:
    A. Spanning tree verification (Kirchhoff)
    B. Cross-validation against NetworkX
    C. Rainbow table minor finding
    D. Graph atlas coverage
    E. D-Wave hardware topologies
    F. Composition formulas
    G. Performance regression
"""

import json
import os
import time

import networkx as nx
import pytest
from tutte_test.graph import (Graph, complete_graph, cut_vertex_join,
                              cycle_graph, disjoint_union, grid_graph,
                              path_graph, petersen_graph, wheel_graph)
from tutte_test.polynomial import TuttePolynomial
from tutte_test.synthesis import SynthesisEngine
from tutte_test.validation import (compute_tutte_networkx,
                                   count_spanning_trees_kirchhoff,
                                   verify_spanning_trees)

# =============================================================================
# A. SPANNING TREE VERIFICATION (Kirchhoff)
# =============================================================================

STANDARD_GRAPHS = [
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("K_5", lambda: complete_graph(5)),
    ("K_6", lambda: complete_graph(6)),
    ("K_7", lambda: complete_graph(7)),
    ("C_5", lambda: cycle_graph(5)),
    ("C_8", lambda: cycle_graph(8)),
    ("C_12", lambda: cycle_graph(12)),
    ("P_4", lambda: path_graph(4)),
    ("P_8", lambda: path_graph(8)),
    ("W_5", lambda: wheel_graph(5)),
    ("W_7", lambda: wheel_graph(7)),
    ("Petersen", lambda: petersen_graph()),
    ("K_3,3", lambda: Graph.from_networkx(nx.complete_bipartite_graph(3, 3))),
    ("Grid_3x3", lambda: grid_graph(3, 3)),
]


@pytest.mark.parametrize("name,builder", STANDARD_GRAPHS, ids=[g[0] for g in STANDARD_GRAPHS])
def test_spanning_trees(name, builder, engine, benchmark_collector):
    """T(1,1) must equal Kirchhoff spanning tree count."""
    import time

    graph = builder()
    kirchhoff = count_spanning_trees_kirchhoff(graph)
    assert kirchhoff > 0, f"Kirchhoff failed for {name}"

    t0 = time.perf_counter()
    result = engine.synthesize(graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    tutte_trees = result.polynomial.num_spanning_trees()
    assert tutte_trees == kirchhoff, (
        f"{name}: T(1,1)={tutte_trees} != Kirchhoff={kirchhoff}"
    )

    benchmark_collector.record(
        name=name,
        nodes=graph.node_count(),
        edges=graph.edge_count(),
        spanning_trees=kirchhoff,
        timings_ms={"synthesis_cej": round(elapsed_ms, 2)},
    )


# =============================================================================
# B. CROSS-VALIDATION AGAINST NETWORKX
# =============================================================================

SMALL_GRAPHS = [
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("K_5", lambda: complete_graph(5)),
    ("C_4", lambda: cycle_graph(4)),
    ("C_5", lambda: cycle_graph(5)),
    ("C_6", lambda: cycle_graph(6)),
    ("P_3", lambda: path_graph(3)),
    ("P_5", lambda: path_graph(5)),
    ("W_5", lambda: wheel_graph(5)),
    ("Petersen", lambda: petersen_graph()),
]


@pytest.mark.parametrize("name,builder", SMALL_GRAPHS, ids=[g[0] for g in SMALL_GRAPHS])
def test_tutte_matches_networkx(name, builder, engine):
    """Our polynomial must match NetworkX for graphs with <=15 edges."""
    graph = builder()
    if graph.edge_count() > 15:
        pytest.skip(f"{name} has {graph.edge_count()} edges, too slow for NetworkX")

    result = engine.synthesize(graph)
    nx_poly = compute_tutte_networkx(graph.to_networkx())
    if nx_poly is None:
        pytest.skip("sympy not available")

    assert result.polynomial == nx_poly, (
        f"{name}: our polynomial != NetworkX polynomial"
    )


# =============================================================================
# C. RAINBOW TABLE MINOR FINDING
# =============================================================================


def test_minor_k4_contains_k3(default_table):
    """K_4 should contain K_3 as a minor."""
    g = complete_graph(4)
    minors = default_table.find_minors_of(g)
    minor_names = {m.name for m in minors}
    assert "K_3" in minor_names


def test_minor_k5_contains_k4(default_table):
    """K_5 should contain K_4 as a minor."""
    g = complete_graph(5)
    minors = default_table.find_minors_of(g)
    minor_names = {m.name for m in minors}
    assert "K_4" in minor_names
    assert "K_3" in minor_names


def test_minor_petersen_contains_c5(default_table):
    """Petersen graph should contain C_5 as a minor."""
    g = petersen_graph()
    minors = default_table.find_minors_of(g)
    minor_names = {m.name for m in minors}
    assert "C_5" in minor_names


# =============================================================================
# D. GRAPH ATLAS COVERAGE
# =============================================================================


def _atlas_graphs():
    """Generate (index, graph) for connected atlas graphs with >= 1 edge."""
    for i in range(1, 1253):
        try:
            G = nx.graph_atlas(i)
        except Exception:
            continue
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue
        if not nx.is_connected(G):
            continue
        yield i, G


@pytest.mark.slow
@pytest.mark.parametrize(
    "atlas_idx,nx_graph",
    list(_atlas_graphs()),
    ids=[f"atlas_{i}" for i, _ in _atlas_graphs()],
)
def test_graph_atlas_spanning_trees(atlas_idx, nx_graph, engine):
    """Every connected atlas graph must have T(1,1) == Kirchhoff count."""
    graph = Graph.from_networkx(nx_graph)
    result = engine.synthesize(graph)
    kirchhoff = round(nx.number_of_spanning_trees(nx_graph))
    tutte_trees = result.polynomial.num_spanning_trees()
    assert tutte_trees == kirchhoff, (
        f"Atlas #{atlas_idx}: T(1,1)={tutte_trees} != Kirchhoff={kirchhoff}"
    )

    # Cross-check with NetworkX for small graphs
    if nx_graph.number_of_edges() <= 15:
        nx_poly = compute_tutte_networkx(nx_graph)
        if nx_poly is not None:
            assert result.polynomial == nx_poly, (
                f"Atlas #{atlas_idx}: polynomial mismatch with NetworkX"
            )


# =============================================================================
# E. D-WAVE HARDWARE TOPOLOGIES
# =============================================================================

# Graphs whose Tutte polynomials have been solved.  Parametrized tests below
# will skip any graph not in this set (unsolved — too many edges).
_SOLVED_DWAVE = {
    "Cm1",   # Chimera(1):  8 nodes,  16 edges
    "Z1_1",  # Zephyr(1,1): 12 nodes, 22 edges
}


def _dwave_graph(kind, *args):
    """Build a D-Wave graph, skipping if dwave-networkx is unavailable."""
    dnx = pytest.importorskip("dwave_networkx")
    builders = {
        "chimera": dnx.chimera_graph,
        "pegasus": dnx.pegasus_graph,
        "zephyr": dnx.zephyr_graph,
    }
    G = builders[kind](*args)
    if G.number_of_nodes() == 0:
        pytest.skip(f"{kind}({','.join(str(a) for a in args)}) is degenerate")
    return Graph.from_networkx(G), G


# --- Chimera C1–C16 ---------------------------------------------------------

CHIMERA_PARAMS = list(range(1, 17))


@pytest.mark.parametrize("m", CHIMERA_PARAMS, ids=[f"Cm{m}" for m in CHIMERA_PARAMS])
def test_chimera(m, engine):
    """Chimera(m) synthesis — skips unsolved topologies."""
    graph, G = _dwave_graph("chimera", m)
    tag = f"Cm{m}"

    if tag not in _SOLVED_DWAVE:
        pytest.skip(f"Cm{m} unsolved ({graph.node_count()}n, {graph.edge_count()}e)")

    result = engine.synthesize(graph)
    kirchhoff = round(nx.number_of_spanning_trees(G))
    assert result.polynomial.num_spanning_trees() == kirchhoff, (
        f"Cm{m}: T(1,1)={result.polynomial.num_spanning_trees()} != Kirchhoff={kirchhoff}"
    )


# --- Pegasus P1–P16 ---------------------------------------------------------

PEGASUS_PARAMS = list(range(1, 17))


@pytest.mark.parametrize("m", PEGASUS_PARAMS, ids=[f"Pm{m}" for m in PEGASUS_PARAMS])
def test_pegasus(m, engine):
    """Pegasus(m) synthesis — skips unsolved topologies."""
    graph, G = _dwave_graph("pegasus", m)
    tag = f"Pm{m}"

    if tag not in _SOLVED_DWAVE:
        pytest.skip(f"Pm{m} unsolved ({graph.node_count()}n, {graph.edge_count()}e)")

    result = engine.synthesize(graph)
    kirchhoff = round(nx.number_of_spanning_trees(G))
    assert result.polynomial.num_spanning_trees() == kirchhoff, (
        f"Pm{m}: T(1,1)={result.polynomial.num_spanning_trees()} != Kirchhoff={kirchhoff}"
    )


# --- Zephyr Z(m,t) ----------------------------------------------------------

_MAX_ZEPHYR_M = int(os.environ.get("TUTTE_MAX_ZEPHYR_M", "12"))
_MAX_ZEPHYR_T = int(os.environ.get("TUTTE_MAX_ZEPHYR_T", "4"))

ZEPHYR_PARAMS = [
    (m, t)
    for m in range(1, _MAX_ZEPHYR_M + 1)
    for t in range(1, _MAX_ZEPHYR_T + 1)
]


@pytest.mark.parametrize("m,t", ZEPHYR_PARAMS, ids=[f"Z{m}_{t}" for m, t in ZEPHYR_PARAMS])
def test_zephyr(m, t, engine):
    """Zephyr Z(m,t) synthesis — skips unsolved topologies."""
    graph, G = _dwave_graph("zephyr", m, t)
    tag = f"Z{m}_{t}"

    if tag not in _SOLVED_DWAVE:
        pytest.skip(f"Z({m},{t}) unsolved ({graph.node_count()}n, {graph.edge_count()}e)")

    result = engine.synthesize(graph)
    kirchhoff = round(nx.number_of_spanning_trees(G))
    assert result.polynomial.num_spanning_trees() == kirchhoff, (
        f"Z({m},{t}): T(1,1)={result.polynomial.num_spanning_trees()} != Kirchhoff={kirchhoff}"
    )


# =============================================================================
# F. COMPOSITION FORMULAS
# =============================================================================


def test_disjoint_union_formula(engine):
    """T(G1 ∪ G2) = T(G1) × T(G2)."""
    g1 = complete_graph(3)
    g2 = cycle_graph(4)
    g_union = disjoint_union(g1, g2)

    t1 = engine.synthesize(g1).polynomial
    t2 = engine.synthesize(g2).polynomial
    t_union = engine.synthesize(g_union).polynomial

    assert t_union == t1 * t2


def test_cut_vertex_k3_c4(engine):
    """T(K3 · C4) = T(K3) × T(C4) at cut vertex."""
    g1 = complete_graph(3)
    g2 = cycle_graph(4)
    joined = cut_vertex_join(g1, 0, g2, 0)

    t1 = engine.synthesize(g1).polynomial
    t2 = engine.synthesize(g2).polynomial
    t_joined = engine.synthesize(joined).polynomial

    assert t_joined == t1 * t2


def test_cut_vertex_k3_k4(engine):
    """T(K3 · K4) = T(K3) × T(K4) at cut vertex."""
    g1 = complete_graph(3)
    g2 = complete_graph(4)
    joined = cut_vertex_join(g1, 0, g2, 0)

    t1 = engine.synthesize(g1).polynomial
    t2 = engine.synthesize(g2).polynomial
    t_joined = engine.synthesize(joined).polynomial

    assert t_joined == t1 * t2


def test_cut_vertex_c4_c5(engine):
    """T(C4 · C5) = T(C4) × T(C5) at cut vertex."""
    g1 = cycle_graph(4)
    g2 = cycle_graph(5)
    joined = cut_vertex_join(g1, 0, g2, 0)

    t1 = engine.synthesize(g1).polynomial
    t2 = engine.synthesize(g2).polynomial
    t_joined = engine.synthesize(joined).polynomial

    assert t_joined == t1 * t2


# =============================================================================
# G. PERFORMANCE REGRESSION
# =============================================================================


def _load_benchmark_baseline():
    """Load baseline timings from benchmark_results.json.

    Returns dict of {name: {"synthesis_cej": ms, "synthesis_hybrid": ms}}.
    Returns empty dict if file is missing.
    """
    path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    baseline = {}
    for r in data.get("results", []):
        timings = r.get("timings_ms", {})
        baseline[r["name"]] = {
            "synthesis_cej": timings.get("synthesis_cej"),
            "synthesis_hybrid": timings.get("synthesis_hybrid"),
        }
    return baseline


def _build_dwave_graph(kind, *args):
    """Build a D-Wave graph, skipping if dwave-networkx unavailable."""
    dnx = pytest.importorskip("dwave_networkx")
    builders = {"chimera": dnx.chimera_graph, "zephyr": dnx.zephyr_graph}
    return Graph.from_networkx(builders[kind](*args))


PERF_GRAPHS = [
    ("Petersen", lambda: petersen_graph()),
    ("K_6", lambda: complete_graph(6)),
    ("Grid_3x3", lambda: grid_graph(3, 3)),
    ("Cm1", lambda: _build_dwave_graph("chimera", 1)),
    ("Z1_1", lambda: _build_dwave_graph("zephyr", 1, 1)),
]


@pytest.mark.perf
@pytest.mark.parametrize(
    "name,builder",
    PERF_GRAPHS,
    ids=[g[0] for g in PERF_GRAPHS],
)
def test_performance_regression(name, builder, engine):
    """Synthesis must not regress >10% vs benchmark_results.json baseline."""
    baseline = _load_benchmark_baseline()
    if not baseline:
        pytest.skip("no benchmark_results.json baseline file")

    if name not in baseline:
        pytest.skip(f"no baseline for {name} in benchmark_results.json")

    baseline_ms = baseline[name].get("synthesis_cej")
    if baseline_ms is None:
        pytest.skip(f"no CEJ timing for {name} in baseline")

    graph = builder()
    kirchhoff = count_spanning_trees_kirchhoff(graph)

    t0 = time.perf_counter()
    result = engine.synthesize(graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Correctness check
    assert result.polynomial.num_spanning_trees() == kirchhoff, (
        f"{name}: T(1,1)={result.polynomial.num_spanning_trees()} != Kirchhoff={kirchhoff}"
    )

    # Performance check: no more than 10% regression
    threshold_ms = baseline_ms * 1.10
    assert elapsed_ms <= threshold_ms, (
        f"{name}: {elapsed_ms:.1f}ms > {threshold_ms:.1f}ms "
        f"(baseline {baseline_ms:.1f}ms + 10%)"
    )


# =============================================================================
# H. MINOR VERIFICATION
# =============================================================================


def test_star_not_minor_of_cycle():
    """S_3 (star with 3 leaves) is NOT a graph minor of C_5.

    Stars need a degree-3+ node, but contracting a cycle can only produce
    degree-2 nodes or (at most) degree-2 after merging — never degree-3+
    without adding edges.
    """
    from tutte_test.graph import star_graph
    from tutte_test.rainbow_table import is_graph_minor

    s3 = star_graph(3)   # 4 nodes, 3 edges; center has degree 3
    c5 = cycle_graph(5)  # 5 nodes, 5 edges; all degree 2

    result = is_graph_minor(c5, s3)
    assert result is False, "S_3 should NOT be a minor of C_5"


def test_path_is_minor_of_cycle():
    """P_4 IS a graph minor of C_5 (delete one edge from cycle)."""
    from tutte_test.rainbow_table import is_graph_minor

    p4 = path_graph(4)   # 4 nodes, 3 edges
    c5 = cycle_graph(5)  # 5 nodes, 5 edges

    result = is_graph_minor(c5, p4)
    assert result is True, "P_4 should be a minor of C_5"


def test_k3_is_minor_of_k4():
    """K_3 IS a graph minor of K_4 (delete one vertex)."""
    from tutte_test.rainbow_table import is_graph_minor

    k3 = complete_graph(3)
    k4 = complete_graph(4)

    result = is_graph_minor(k4, k3)
    assert result is True, "K_3 should be a minor of K_4"


def test_high_degree_tree_minor():
    """K_{1,4} (star with 4 leaves) IS a minor of Petersen graph.

    Requires contraction: Petersen is 3-regular, but contracting one edge
    creates a degree-4 vertex that hosts the star center.
    """
    from tutte_test.graph import star_graph
    from tutte_test.rainbow_table import is_graph_minor

    s4 = star_graph(4)          # 5 nodes, 4 edges; center has degree 4
    petersen = petersen_graph()  # 10 nodes, 15 edges; 3-regular

    result = is_graph_minor(petersen, s4)
    assert result is True, "K_{1,4} should be a minor of Petersen"


# =============================================================================
# I. BINARY ROUNDTRIP
# =============================================================================


def test_binary_roundtrip():
    """Encode→decode preserves all entries and polynomials."""
    from tutte_test.rainbow_table import (RainbowTable,
                                          encode_rainbow_table_binary,
                                          decode_rainbow_table_binary)

    table = RainbowTable()
    k3 = complete_graph(3)
    c5 = cycle_graph(5)

    k3_poly = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    c5_coeffs = {(i, 0): 1 for i in range(1, 5)}
    c5_coeffs[(0, 1)] = 1
    c5_poly = TuttePolynomial.from_coefficients(c5_coeffs)

    table.add(k3, "K_3", k3_poly)
    table.add(c5, "C_5", c5_poly)

    data = encode_rainbow_table_binary(table)
    decoded = decode_rainbow_table_binary(data)

    assert len(decoded) == 2
    assert decoded.lookup_by_name("K_3") == k3_poly
    assert decoded.lookup_by_name("C_5") == c5_poly

    # Verify metadata
    k3_entry = decoded.get_entry("K_3")
    assert k3_entry.node_count == 3
    assert k3_entry.edge_count == 3
    assert k3_entry.spanning_trees == 3


def test_binary_roundtrip_with_minors():
    """Minor relationships survive binary encode→decode roundtrip."""
    from tutte_test.rainbow_table import (RainbowTable,
                                          encode_rainbow_table_binary,
                                          decode_rainbow_table_binary)

    table = RainbowTable()
    k3 = complete_graph(3)
    k4 = complete_graph(4)
    p2 = path_graph(2)

    k3_poly = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    k4_poly = TuttePolynomial.from_coefficients(
        {(3, 0): 1, (2, 0): 3, (1, 1): 4, (1, 0): 2, (0, 1): 2, (0, 2): 3, (0, 3): 1}
    )
    p2_poly = TuttePolynomial.x(1)

    table.add(p2, "P_2", p2_poly)
    table.add(k3, "K_3", k3_poly)
    table.add(k4, "K_4", k4_poly)

    # Manually set minor relationships
    k3_key = k3.canonical_key()
    k4_key = k4.canonical_key()
    p2_key = p2.canonical_key()
    table.minor_relationships[k4_key] = [k3_key, p2_key]
    table.minor_relationships[k3_key] = [p2_key]
    table._structural_minors_computed = True

    data = encode_rainbow_table_binary(table)
    decoded = decode_rainbow_table_binary(data)

    assert decoded._structural_minors_computed is True
    assert k4_key in decoded.minor_relationships
    assert set(decoded.minor_relationships[k4_key]) == {k3_key, p2_key}
    assert k3_key in decoded.minor_relationships
    assert decoded.minor_relationships[k3_key] == [p2_key]
