"""
Graphviz visualisation for minor-embeddings.

Shared between embed_cli and demo_embedding.
"""

from __future__ import annotations

import graphviz as gv
import networkx as nx

# Palette for vertex-model colouring.  Cycles when there are more source nodes
# than colours.
COLOURS = [
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


def render_embedding(
    target: nx.Graph,
    embedding: dict,  # source_node -> frozenset | set | list of target nodes
    title: str = "",
) -> gv.Graph:
    """
    Build a graphviz.Graph visualising a minor-embedding on the target graph.

    Nodes are coloured by the source-node cluster they belong to and laid out
    according to the target graph's own edge structure — no ``cluster_``
    subgraphs are used, so Graphviz does not relocate or group nodes spatially.
    Cross-model edges (witnessing source edges) are drawn bold in red;
    intra-model and unassigned edges are thin grey.  Unassigned target nodes
    are rendered in light grey.

    Parameters
    ----------
    target:
        The hardware topology graph.
    embedding:
        Mapping from source node to the collection of target nodes that form
        its vertex-model.
    title:
        Optional graph title rendered at the top of the diagram.

    Returns
    -------
    A ``graphviz.Graph`` whose ``.source`` is the DOT representation.
    """
    # Build reverse map: target_node -> source_node
    node_to_src: dict = {}
    for src_node, model in embedding.items():
        for g in model:
            node_to_src[g] = src_node

    src_nodes = list(embedding.keys())
    colour_of = {s: COLOURS[i % len(COLOURS)] for i, s in enumerate(src_nodes)}

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

    # Emit every target node coloured by its cluster membership.
    for g in target.nodes:
        src = node_to_src.get(g)
        if src is not None:
            colour = colour_of[src]
            dot.node(
                str(g),
                fillcolor=colour,
                fontcolor="white",
                tooltip=str(src),
            )
        else:
            dot.node(str(g), fillcolor="#dddddd", fontcolor="#888888")

    # Edges
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


__all__ = ["COLOURS", "render_embedding"]
