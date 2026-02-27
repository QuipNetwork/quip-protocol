"""
Demo: embed small Ising-model graphs into D-Wave topologies and render to Graphviz DOT.

The Ising models used here each have a clear physical/combinatorial meaning.
Run with:
    python -m topo_alloc.demo_embedding

Output DOT files are written next to this script.  Render with:
    dot -Tsvg demo_k5_chimera.dot -o demo_k5_chimera.svg
    dot -Tsvg demo_frustrated_zephyr.dot -o demo_frustrated_zephyr.svg
    ...

-----------------------------------------------------------------------
Ising-model examples and their meaning
-----------------------------------------------------------------------

1. K_5  --  "All-to-all 5-spin Ising ferromagnet"
   Spins: s_0 ... s_4 ∈ {-1, +1}
   Energy: H = -J ∑_{i<j} s_i s_j   (J > 0)
   The ground state has all spins aligned (+1,+1,+1,+1,+1) or all −1.
   K_5 is the *smallest* complete graph that is NOT planar (by Kuratowski's
   theorem), so it requires at least one vertex-model of size > 1 on both
   Chimera and Pegasus (neither contains K_5 as a unit-cell subgraph).
   This makes it a good stress-test for the embedder.

2. Frustrated triangle with pendant  --  "Frustrated Ising triangle"
   Spins: A, B, C (triangle) + D attached to A.
   Couplings: A-B = −1 (ferro), B-C = −1 (ferro), A-C = +1 (antiferro),
              A-D = −1 (ferro).
   The triangle A-B-C has competing interactions: B-C and A-B want
   ferromagnetic order, but A-C wants antiferromagnetic order.
   No assignment of {±1} can simultaneously satisfy all three edges —
   this is *geometric frustration*, the core difficulty of spin glasses.
   The pendant spin D is "dragged along" by A, illustrating how frustration
   propagates.  The graph itself is K_3 + a leaf (4 nodes).

3. Ising ring (C_8)  --  "Antiferromagnetic spin ring / QUBO parity check"
   8 spins on a ring, all couplings J = +1 (antiferromagnetic).
   Even-length antiferromagnetic rings are unfrustrated: the ground state
   simply alternates +-+-+- around the ring.
   In QUBO form this is equivalent to an 8-bit parity-check constraint,
   where adjacent bits want to differ.  C_8 is bipartite (all even cycles
   are), so it embeds trivially into Chimera (which is itself bipartite).

4. MAX-CUT 3-regular graph (K_{3,3})  --  "Bipartite MAX-CUT / MAX-2-SAT"
   The complete bipartite graph K_{3,3} with all antiferromagnetic couplings
   is the canonical MAX-CUT instance on 6 nodes.  MAX-CUT on K_{3,3} is
   trivially solvable (just cut along the bipartition), but embedding K_{3,3}
   into non-bipartite topologies (Zephyr) exercises the chain-stitching logic.
   Note: K_4 is a minor of K_{3,3}, making this a richer target than it looks.

5. "Chimera unit cell"  --  K_{4,4} sub-problem
   8 spins arranged in a complete bipartite graph K_{4,4}.
   This is literally *one unit cell* of the Chimera topology, so the embedding
   into Chimera(4) must succeed with unit chain length.  On Zephyr the same
   graph tests how the richer connectivity is exploited.
-----------------------------------------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path

import networkx as nx

from topo_alloc.graphviz_render import (
    EmbeddingStats,
    embedding_stats,
    format_stats_table,
    render_embedding,
)
from topo_alloc.minor_alloc import EmbedOption, find_embedding

# ---------------------------------------------------------------------------
# Ising-model graph definitions
# ---------------------------------------------------------------------------


def make_k5() -> nx.Graph:
    """K_5: all-to-all ferromagnetic 5-spin Ising model."""
    return nx.complete_graph(5)


def make_frustrated_triangle() -> nx.Graph:
    """Frustrated Ising triangle with one pendant spin."""
    g = nx.Graph()
    # Triangle
    g.add_edge("A", "B", J=-1)  # ferromagnetic
    g.add_edge("B", "C", J=-1)  # ferromagnetic
    g.add_edge("A", "C", J=+1)  # antiferromagnetic  → frustration
    # Pendant
    g.add_edge("A", "D", J=-1)  # drags D along with A
    return g


def make_ising_ring() -> nx.Graph:
    """C_8: 8-spin antiferromagnetic ring (unfrustrated even cycle)."""
    return nx.cycle_graph(8)


def make_k33() -> nx.Graph:
    """K_{3,3}: bipartite MAX-CUT / parity-check Ising model."""
    return nx.complete_bipartite_graph(3, 3)


def make_k44() -> nx.Graph:
    """K_{4,4}: one Chimera unit cell as an Ising model."""
    return nx.complete_bipartite_graph(4, 4)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def chimera_small() -> nx.Graph:
    """Chimera(4,4,4): small but rich enough for all demo models."""
    import dwave_networkx as dnx

    return dnx.chimera_graph(4, 4, 4)


def chimera_c16() -> nx.Graph:
    """Chimera C(16): the D-Wave 2000Q topology."""
    import dwave_networkx as dnx

    return dnx.chimera_graph(16)


def zephyr_small() -> nx.Graph:
    """Zephyr Z(3,2): smallest non-trivial Zephyr topology."""
    import dwave_networkx as dnx

    return dnx.zephyr_graph(3, 2)


def pegasus_small() -> nx.Graph:
    """Pegasus P(4): small Pegasus topology."""
    import dwave_networkx as dnx

    return dnx.pegasus_graph(4)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def seeded_rng(seed: int):
    def factory():
        return random.Random(seed)

    return factory


CASES: list[tuple[str, nx.Graph, str, nx.Graph]] = [
    # (label, source_graph, topology_name, target_graph)
    ("k5_chimera4", make_k5(), "Chimera(4)", chimera_small()),
    ("k5_zephyr3", make_k5(), "Zephyr(3,2)", zephyr_small()),
    ("k5_pegasus4", make_k5(), "Pegasus(4)", pegasus_small()),
    ("frustrated_chimera4", make_frustrated_triangle(), "Chimera(4)", chimera_small()),
    ("frustrated_zephyr3", make_frustrated_triangle(), "Zephyr(3,2)", zephyr_small()),
    ("ring8_chimera4", make_ising_ring(), "Chimera(4)", chimera_small()),
    ("ring8_zephyr3", make_ising_ring(), "Zephyr(3,2)", zephyr_small()),
    ("k33_chimera4", make_k33(), "Chimera(4)", chimera_small()),
    ("k33_zephyr3", make_k33(), "Zephyr(3,2)", zephyr_small()),
    ("k44_chimera4", make_k44(), "Chimera(4)", chimera_small()),
    ("k44_zephyr3", make_k44(), "Zephyr(3,2)", zephyr_small()),
    ("k44_pegasus4", make_k44(), "Pegasus(4)", pegasus_small()),
]

OUT_DIR = Path(__file__).parent

_COL_W = 28  # column width for comparison table


def _run_case(
    label: str,
    source: nx.Graph,
    topo_name: str,
    target: nx.Graph,
    order_by_degree: bool,
    seed: int = 42,
) -> tuple[str, tuple[EmbeddingStats, str] | None]:
    """Run one embedding attempt; return (suffix, stats_line | None)."""
    suffix = "degree" if order_by_degree else "random"
    embedding = find_embedding(
        source,
        target,
        rng_factory=seeded_rng(seed),
        tries=50,
        refinment_constant=20,
        overlap_penalty=2.0,
        options=EmbedOption.ORDER_BY_DEGREE if order_by_degree else EmbedOption(0),
    )
    if embedding is None:
        return suffix, None

    stats = embedding_stats(embedding)

    n_src = source.number_of_nodes()
    title = (
        f"{label} [{suffix}]\\nsource nodes: {n_src}"
        f"  target: {topo_name} ({target.number_of_nodes()} nodes)"
    )
    dot = render_embedding(target, embedding, title=title)
    out_path = OUT_DIR / f"demo_{label}_{suffix}.dot"
    dot.save(str(out_path))
    return suffix, (stats, str(out_path))


def _fmt(val) -> str:
    if val is None:
        return "FAILED"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def main() -> None:
    print("=" * 70)
    print("Ising-model minor-embedding demo  (random  vs  degree-ordered)")
    print("=" * 70)

    hdr = f"{'metric':<20}  {'random':>{_COL_W}}  {'degree-order':>{_COL_W}}"
    sep = "-" * len(hdr)

    for label, source, topo_name, target in CASES:
        n_src = source.number_of_nodes()
        n_tgt = target.number_of_nodes()
        print(
            f"\n-- {label}  (source: {n_src} nodes  →  target: {topo_name} {n_tgt} nodes)"
        )

        results: dict[str, object] = {}
        for use_degree in (False, True):
            suffix, payload = _run_case(label, source, topo_name, target, use_degree)
            results[suffix] = payload

        r_payload = results["random"]
        d_payload = results["degree"]

        # Pretty-print comparison table
        print(hdr)
        print(sep)

        def row(name: str, r_val, d_val) -> None:
            print(f"  {name:<18}  {_fmt(r_val):>{_COL_W}}  {_fmt(d_val):>{_COL_W}}")

        if r_payload is None and d_payload is None:
            print("  Both strategies FAILED to find an embedding.")
        else:
            r_stats = r_payload[0] if r_payload else None  # pyright: ignore[reportIndexIssue]
            d_stats = d_payload[0] if d_payload else None  # pyright: ignore[reportIndexIssue]
            row(
                "nodes used",
                r_stats.nodes_used if r_stats else None,
                d_stats.nodes_used if d_stats else None,
            )
            row(
                "chain avg",
                r_stats.chain_avg if r_stats else None,
                d_stats.chain_avg if d_stats else None,
            )
            row(
                "chain max",
                r_stats.chain_max if r_stats else None,
                d_stats.chain_max if d_stats else None,
            )
            row(
                "chain min",
                r_stats.chain_min if r_stats else None,
                d_stats.chain_min if d_stats else None,
            )

            for suffix, payload in [("random", r_payload), ("degree", d_payload)]:
                if payload:
                    print(f"    DOT [{suffix}] → {payload[1]}")

        # Also print per-strategy formatted stats for detail
        for suffix, payload in [("random", r_payload), ("degree", d_payload)]:
            if payload:
                print(f"\n  [{suffix}] full stats:")
                print(format_stats_table(payload[0]))  # pyright: ignore[reportIndexIssue]

    print("\n" + "=" * 70)
    print("Done.  Render with:  dot -Tsvg demo_<name>_<strategy>.dot -o out.svg")
    print("=" * 70)


if __name__ == "__main__":
    main()
