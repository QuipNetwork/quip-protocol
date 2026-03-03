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
import networkx as nx
from networkx.readwrite import json_graph

from topo_alloc.graphviz_render import (
    embedding_stats,
    format_stats_table,
    render_embedding,
)
from topo_alloc.minor_alloc import find_embedding


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
    default=2,
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
    overlap_penalty: int,
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

    click.echo(format_stats_table(embedding_stats(embedding)), err=True)

    result = {str(k): sorted(v) for k, v in embedding.items()}
    serialized = json.dumps(result, indent=2)

    if output == "-":
        click.echo(serialized)
    else:
        with open(output, "w") as f:
            f.write(serialized + "\n")
        click.echo(f"Embedding written to {output}", err=True)

    if graphviz is not None:
        dot = render_embedding(target_graph, embedding)
        if graphviz == "-":
            click.echo(dot.source)
        else:
            dot.save(graphviz)
            click.echo(f"Graphviz DOT written to {graphviz}", err=True)


if __name__ == "__main__":
    main()
