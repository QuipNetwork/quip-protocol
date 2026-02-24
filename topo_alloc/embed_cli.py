"""
CLI utility for running find_embedding on JSON-serialized graphs.

Usage:
    python -m topo_alloc.embed_cli --source source.json --target target.json
    python -m topo_alloc.embed_cli --source source.json --target target.json \\
        --tries 50 --refinement-constant 10 --overlap-penalty 3.0 --seed 42
"""

from __future__ import annotations

import json
import sys

import click
import graphviz as gv
import networkx as nx
from networkx.readwrite import json_graph

from topo_alloc.minor_alloc import find_embedding

# Palette for vertex-model colouring in the graphviz render.
# Cycles through these when there are more source nodes than colours.
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


def _load_graph(path: str) -> nx.Graph:
    with open(path) as f:
        data = json.load(f)
    # Support node-link format (default from nx.node_link_data) and
    # adjacency format (from nx.adjacency_data).
    if "nodes" in data and "links" in data:
        return json_graph.node_link_graph(data, edges="links")
    if "nodes" in data and "adjacency" in data:
        return json_graph.adjacency_graph(data)
    raise click.BadParameter(
        f"Unrecognised graph format in '{path}'. "
        "Expected node-link (keys: nodes, links) or adjacency (keys: nodes, adjacency) JSON."
    )


def _render_graphviz(
    target: nx.Graph,
    embedding: dict,  # source_node -> frozenset[target_node]
) -> gv.Graph:
    """
    Build a graphviz.Graph visualising the embedding on the target graph.

    Each target node is drawn inside a cluster subgraph corresponding to the
    source node whose vertex-model it belongs to.  Target edges that connect
    two different vertex-models are drawn bold (they witness source edges);
    intra-model and background edges are thin.  Unassigned target nodes are
    placed in a grey background subgraph.
    """
    # Build reverse map: target_node -> source_node
    node_to_src: dict = {}
    for src_node, model in embedding.items():
        for g in model:
            node_to_src[g] = src_node

    src_nodes = list(embedding.keys())
    colour_of = {s: _COLOURS[i % len(_COLOURS)] for i, s in enumerate(src_nodes)}

    dot = gv.Graph(
        "embedding",
        graph_attr={"bgcolor": "white"},
        node_attr={"style": "filled", "fontname": "Helvetica"},
        edge_attr={"penwidth": "1.0"},
    )

    # One cluster subgraph per source node
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

    # Unassigned target nodes
    unassigned = [g for g in target.nodes if g not in node_to_src]
    if unassigned:
        with dot.subgraph(name="cluster_unassigned") as sub:  # pyright: ignore[reportOptionalContextManager]
            sub.attr(label="", style="invis")
            for g in sorted(unassigned, key=str):
                sub.node(str(g), fillcolor="#dddddd")

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
            dot.edge(str(u), str(v), penwidth="2.5", style="bold")
        else:
            dot.edge(str(u), str(v))

    return dot


@click.command()
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the source graph JSON file (node-link or adjacency format).",
)
@click.option(
    "--target",
    "-t",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the target (hardware topology) graph JSON file.",
)
@click.option(
    "--tries",
    default=30,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of independent embedding attempts.",
)
@click.option(
    "--refinement-constant",
    default=20,
    show_default=True,
    type=click.IntRange(min=1),
    help="Refinement iterations = k * |V(source)|.",
)
@click.option(
    "--overlap-penalty",
    default=2.0,
    show_default=True,
    type=float,
    help="Penalty weight for edges leading into another vertex-model.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Fixed RNG seed for reproducibility (omit for non-deterministic).",
)
@click.option(
    "--output",
    "-o",
    default="-",
    type=click.Path(dir_okay=False, writable=True),
    help="Output file for the embedding JSON (default: stdout).",
)
@click.option(
    "--graphviz",
    "-g",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Also write a DOT-language visualisation of the embedding to this file (use '-' for stdout).",
)
def main(
    source: str,
    target: str,
    tries: int,
    refinement_constant: int,
    overlap_penalty: float,
    seed: int | None,
    output: str,
    graphviz: str | None,
) -> None:
    """Find a minor-embedding of SOURCE into TARGET and print the result as JSON.

    Each key in the output object is a source node; the value is the list of
    target nodes that form its vertex-model.  Exits with code 1 if no embedding
    is found.
    """
    import random as _rng

    source_graph = _load_graph(source)
    target_graph = _load_graph(target)

    rng_factory = (
        (lambda s: lambda: _rng.Random(s))(seed) if seed is not None else _rng.Random
    )

    embedding = find_embedding(
        source_graph,
        target_graph,
        rng_factory=rng_factory,
        tries=tries,
        refinment_constant=refinement_constant,
        overlap_penalty=overlap_penalty,
    )

    if embedding is None:
        click.echo("No embedding found.", err=True)
        sys.exit(1)

    result = {str(k): sorted(v) for k, v in embedding.items()}
    serialized = json.dumps(result, indent=2)

    if output == "-":
        click.echo(serialized)
    else:
        with open(output, "w") as f:
            f.write(serialized + "\n")
        click.echo(f"Embedding written to {output}", err=True)

    if graphviz is not None:
        dot = _render_graphviz(target_graph, embedding)
        if graphviz == "-":
            click.echo(dot.source)
        else:
            dot.save(graphviz)
            click.echo(f"Graphviz DOT written to {graphviz}", err=True)


if __name__ == "__main__":
    main()
