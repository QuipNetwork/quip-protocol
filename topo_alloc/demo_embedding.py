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

import graphviz as gv
import networkx as nx

from topo_alloc.minor_alloc import find_embedding, is_valid_embedding

# ---------------------------------------------------------------------------
# Palette (same as embed_cli.py for visual consistency)
# ---------------------------------------------------------------------------
_COLOURS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


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
# Graphviz rendering (adapted from embed_cli._render_graphviz)
# ---------------------------------------------------------------------------


def render_graphviz(
    target: nx.Graph,
    embedding: dict,
    title: str = "",
) -> gv.Graph:
    node_to_src: dict = {}
    for src_node, model in embedding.items():
        for g in model:
            node_to_src[g] = src_node

    src_nodes = list(embedding.keys())
    colour_of = {s: _COLOURS[i % len(_COLOURS)] for i, s in enumerate(src_nodes)}

    dot = gv.Graph(
        "embedding",
        graph_attr={
            "bgcolor": "white",
            "label": title,
            "labelloc": "t",
            "fontsize": "14",
            "fontname": "Helvetica",
        },
        node_attr={
            "style": "filled",
            "fontname": "Helvetica",
            "width": "0.3",
            "height": "0.3",
        },
        edge_attr={"penwidth": "1.0"},
    )

    for i, src_node in enumerate(src_nodes):
        colour = colour_of[src_node]
        with dot.subgraph(name=f"cluster_{i}") as sub:  # pyright: ignore[reportOptionalContextManager]
            sub.attr(
                label=str(src_node),
                style="filled",
                fillcolor=f"{colour}22",
                color=colour,
                penwidth="2",
            )
            for g in sorted(embedding[src_node], key=str):
                sub.node(str(g), fillcolor=colour, fontcolor="white")

    unassigned = [g for g in target.nodes if g not in node_to_src]
    if unassigned:
        with dot.subgraph(name="cluster_unassigned") as sub:  # pyright: ignore[reportOptionalContextManager]
            sub.attr(label="", style="invis")
            for g in sorted(unassigned, key=str):
                sub.node(str(g), fillcolor="#dddddd", fontcolor="#888888")

    seen: set[frozenset] = set()
    for u, v in target.edges:
        key = frozenset([u, v])
        if key in seen:
            continue
        seen.add(key)
        src_u = node_to_src.get(u)
        src_v = node_to_src.get(v)
        if src_u is not None and src_v is not None and src_u != src_v:
            dot.edge(str(u), str(v), penwidth="2.5", style="bold", color="#cc0000")
        else:
            dot.edge(str(u), str(v), color="#aaaaaa")

    return dot


# ---------------------------------------------------------------------------
# Embedding statistics
# ---------------------------------------------------------------------------


def chain_lengths(embedding: dict) -> dict:
    return {k: len(v) for k, v in embedding.items()}


def report(
    name: str, source: nx.Graph, target: nx.Graph, embedding: dict | None
) -> None:
    if embedding is None:
        print(f"  [{name}]  FAILED — no embedding found")
        return

    valid = is_valid_embedding(source, target, embedding)
    chains = chain_lengths(embedding)
    avg = sum(chains.values()) / len(chains)
    mx = max(chains.values())
    print(
        f"  [{name}]  valid={valid}  "
        f"nodes used={sum(chains.values())}  "
        f"avg_chain={avg:.2f}  max_chain={mx}"
    )


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


def main() -> None:
    print("=" * 70)
    print("Ising-model minor-embedding demo")
    print("=" * 70)

    for label, source, topo_name, target in CASES:
        src_name = source.__class__.__name__
        n_src = source.number_of_nodes()
        n_tgt = target.number_of_nodes()
        print(
            f"\n-- {label}  (source: {n_src} nodes  →  target: {topo_name} {n_tgt} nodes)"
        )

        embedding = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(42),
            tries=50,
            refinment_constant=20,
            overlap_penalty=2.0,
        )

        report(label, source, target, embedding)

        if embedding is not None:
            title = (
                f"{label}\\nsource nodes: {n_src}  target: {topo_name} ({n_tgt} nodes)"
            )
            dot = render_graphviz(target, embedding, title=title)
            out_path = OUT_DIR / f"demo_{label}.dot"
            dot.save(str(out_path))
            print(f"    DOT written → {out_path}")

    print("\n" + "=" * 70)
    print("Done.  Render with:  dot -Tsvg demo_<name>.dot -o demo_<name>.svg")
    print("=" * 70)


if __name__ == "__main__":
    main()
