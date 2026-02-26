"""
Graphviz visualisation and statistics for minor-embeddings.

Shared between embed_cli and demo_embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _stats_html_label(title: str, stats: EmbeddingStats) -> str:
    """
    Build a Graphviz HTML-like label combining an optional title with a stats
    table.  The returned string starts with ``<`` and ends with ``>`` so the
    graphviz library passes it through as an HTML label without escaping.
    """
    title_row = (
        f'<TR><TD COLSPAN="2" ALIGN="CENTER"><B>{title}</B></TD></TR>' if title else ""
    )

    def row(label: str, value: str) -> str:
        return f'<TR><TD ALIGN="LEFT">{label}</TD><TD ALIGN="RIGHT">{value}</TD></TR>'

    sep = '<TR><TD COLSPAN="2" BORDER="0"> </TD></TR>'
    inner = "".join(
        [
            title_row,
            row("source nodes", str(stats.num_source_nodes)),
            row("physical nodes used", str(stats.nodes_used)),
            sep,
            row("chain min", str(stats.chain_min)),
            row("chain max", str(stats.chain_max)),
            row("chain avg", f"{stats.chain_avg:.2f}"),
        ]
    )
    return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">{inner}</TABLE>>'


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
    Edge styles by category:
    - intra-cluster (chain edges): black, bold
    - cross-cluster (witness source edges): red, bold
    - unassigned: thin grey
    Unassigned target nodes are rendered in light grey.

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

    stats = embedding_stats(embedding)
    dot = gv.Graph(
        "embedding",
        graph_attr={
            "bgcolor": "white",
            "label": _stats_html_label(title, stats),
            "labelloc": "t",
            "fontname": "Helvetica",
            "layout": "fdp",
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
            # Cross-cluster edge: witnesses a source edge
            dot.edge(str(u), str(v), penwidth="2.5", style="bold", color="#cc0000")
        elif src_u is not None and src_u == src_v:
            # Intra-cluster edge: chain edge within a vertex-model
            dot.edge(str(u), str(v), penwidth="2.5", color="#000000")
        else:
            dot.edge(str(u), str(v), color="#aaaaaa")

    return dot


@dataclass
class EmbeddingStats:
    """Summary statistics for a minor-embedding."""

    num_source_nodes: int
    nodes_used: int  # total physical nodes across all chains
    chain_min: int
    chain_max: int
    chain_avg: float
    cluster_min: int  # same values, named for source-node perspective
    cluster_max: int
    cluster_avg: float


def embedding_stats(embedding: dict) -> EmbeddingStats:
    """
    Compute summary statistics for a minor-embedding.

    Parameters
    ----------
    embedding:
        Mapping from source node to the collection of target nodes that form
        its vertex-model.

    Returns
    -------
    An ``EmbeddingStats`` dataclass.
    """
    chains = [len(model) for model in embedding.values()]
    n = len(chains)
    total = sum(chains)
    return EmbeddingStats(
        num_source_nodes=n,
        nodes_used=total,
        chain_min=min(chains),
        chain_max=max(chains),
        chain_avg=total / n,
        cluster_min=min(chains),
        cluster_max=max(chains),
        cluster_avg=total / n,
    )


def format_stats_table(stats: EmbeddingStats) -> str:
    """
    Format an ``EmbeddingStats`` as a compact plain-text table.

    Example output::

        ┌─────────────────────┬───────┐
        │ source nodes        │     5 │
        │ physical nodes used │     8 │
        ├─────────────────────┼───────┤
        │ chain length  min   │     1 │
        │               max   │     2 │
        │               avg   │  1.60 │
        └─────────────────────┴───────┘
    """
    rows = [
        ("source nodes", f"{stats.num_source_nodes:>5}"),
        ("physical nodes used", f"{stats.nodes_used:>5}"),
        None,  # separator
        ("chain length  min", f"{stats.chain_min:>5}"),
        ("              max", f"{stats.chain_max:>5}"),
        ("              avg", f"{stats.chain_avg:>5.2f}"),
    ]
    col1 = max(len(r[0]) for r in rows if r is not None)
    col2 = max(len(r[1]) for r in rows if r is not None)
    h_top = f"┌{'─' * (col1 + 2)}┬{'─' * (col2 + 2)}┐"
    h_sep = f"├{'─' * (col1 + 2)}┼{'─' * (col2 + 2)}┤"
    h_bot = f"└{'─' * (col1 + 2)}┴{'─' * (col2 + 2)}┘"
    lines = [h_top]
    for row in rows:
        if row is None:
            lines.append(h_sep)
        else:
            label, value = row
            lines.append(f"│ {label:<{col1}} │ {value:>{col2}} │")
    lines.append(h_bot)
    return "\n".join(lines)


__all__ = [
    "COLOURS",
    "EmbeddingStats",
    "embedding_stats",
    "format_stats_table",
    "render_embedding",
]
