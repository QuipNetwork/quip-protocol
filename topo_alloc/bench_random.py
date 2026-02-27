"""
Benchmark minor-embedding on randomly generated source graphs.

Generates a configurable number of random source graphs, embeds each into a
chosen target topology, and prints per-sample and aggregate statistics.  Two
ordering strategies from find_embedding can be compared side-by-side.

Usage examples
--------------
# 20 Erdős-Rényi graphs (n=8, p=0.5) into a Chimera(4) target, both strategies:
    python -m topo_alloc.bench_random --graph-model er --nodes 8 --er-p 0.5 \\
        --topology chimera --topology-size 4 --samples 20 --strategy both

# Barabási-Albert graphs, degree-order strategy only:
    python -m topo_alloc.bench_random --graph-model ba --nodes 10 --ba-m 3 \\
        --topology zephyr --topology-size 3 --samples 15 --strategy degree

# Compare all strategies (random, degree, centrality, longest_chains, vertex_weights):
    python -m topo_alloc.bench_random --graph-model er --nodes 8 --er-p 0.5 \\
        --topology chimera --topology-size 4 --samples 20 --strategy all

# Vertex-weight strategy (Cai, Macready & Roy 2014):
    python -m topo_alloc.bench_random --graph-model er --nodes 8 --er-p 0.5 \\
        --topology chimera --topology-size 4 --samples 20 --strategy vertex_weights

# Random trees embedded into a Pegasus topology, CSV output:
    python -m topo_alloc.bench_random --graph-model tree --nodes 12 \\
        --topology pegasus --topology-size 4 --samples 30 --strategy both --csv

# Use a custom target loaded from a JSON file:
    python -m topo_alloc.bench_random --graph-model er --nodes 6 --er-p 0.4 \\
        --target-json my_target.json --samples 10 --strategy both
"""

from __future__ import annotations

import csv
import dataclasses
import random
import statistics
import sys
import time
from typing import Literal

import click
import networkx as nx

from topo_alloc.graphviz_render import EmbeddingStats, embedding_stats
from topo_alloc.minor_alloc import EmbedOption, find_embedding

# ---------------------------------------------------------------------------
# Graph-model generators
# ---------------------------------------------------------------------------

GraphModel = Literal["er", "ba", "tree"]


def _gen_er(n: int, p: float, seed: int) -> nx.Graph:
    """Erdős-Rényi G(n, p) with isolates removed."""
    g = nx.gnp_random_graph(n, p, seed=seed)
    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def _gen_ba(n: int, m: int, seed: int) -> nx.Graph:
    """Barabási-Albert preferential-attachment graph."""
    return nx.barabasi_albert_graph(n, m, seed=seed)


def _gen_tree(n: int, seed: int) -> nx.Graph:
    """Random labeled tree (connected, n-1 edges)."""
    return nx.random_labeled_tree(n, seed=seed)


# ---------------------------------------------------------------------------
# Target topology builders
# ---------------------------------------------------------------------------


def _build_target(topology: str, size: int) -> nx.Graph:
    import dwave_networkx as dnx

    if topology == "chimera":
        return dnx.chimera_graph(size)
    if topology == "zephyr":
        return dnx.zephyr_graph(size)
    if topology == "pegasus":
        return dnx.pegasus_graph(size)
    raise ValueError(f"Unknown topology: {topology!r}")


# ---------------------------------------------------------------------------
# Per-sample result
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SampleResult:
    sample_id: int
    seed: int
    source_nodes: int
    source_edges: int
    strategy: str
    success: bool
    nodes_used: int | None
    chain_min: int | None
    chain_max: int | None
    chain_avg: float | None
    elapsed_s: float


def _run_sample(
    sample_id: int,
    source: nx.Graph,
    target: nx.Graph,
    strategy: str,
    seed: int,
    tries: int,
    refinement_constant: int,
    overlap_penalty: float,
) -> SampleResult:
    def rng_factory():
        return random.Random(seed)

    options = EmbedOption(0)
    if strategy in ("degree", "longest_chains"):
        options |= EmbedOption.ORDER_BY_DEGREE
    if strategy == "centrality":
        options |= EmbedOption.ORDER_BY_CENTRALITY
    if strategy == "longest_chains":
        options |= EmbedOption.REFINE_LONGEST_CHAINS
    if strategy == "vertex_weights":
        options |= EmbedOption.USE_VERTEX_WEIGHTS
    t0 = time.perf_counter()
    embedding = find_embedding(
        source,
        target,
        rng_factory=rng_factory,
        tries=tries,
        refinment_constant=refinement_constant,
        overlap_penalty=overlap_penalty,
        options=options,
    )
    elapsed_s = time.perf_counter() - t0

    if embedding is None:
        return SampleResult(
            sample_id=sample_id,
            seed=seed,
            source_nodes=source.number_of_nodes(),
            source_edges=source.number_of_edges(),
            strategy=strategy,
            success=False,
            nodes_used=None,
            chain_min=None,
            chain_max=None,
            chain_avg=None,
            elapsed_s=elapsed_s,
        )

    stats: EmbeddingStats = embedding_stats(embedding)
    return SampleResult(
        sample_id=sample_id,
        seed=seed,
        source_nodes=source.number_of_nodes(),
        source_edges=source.number_of_edges(),
        strategy=strategy,
        success=True,
        nodes_used=stats.nodes_used,
        chain_min=stats.chain_min,
        chain_max=stats.chain_max,
        chain_avg=stats.chain_avg,
        elapsed_s=elapsed_s,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_W = 12  # numeric column width


def _fmt(val: object, width: int = _W) -> str:
    if val is None:
        return f"{'—':>{width}}"
    if isinstance(val, float):
        return f"{val:{width}.2f}"
    return f"{val!s:>{width}}"


def _elapsed_unit(mean_s: float) -> tuple[float, str]:
    """Return (scale, unit_label) so that mean_s * scale is in that unit."""
    if mean_s >= 1.0:
        return 1.0, "s"
    if mean_s >= 1e-3:
        return 1e3, "ms"
    if mean_s >= 1e-6:
        return 1e6, "µs"
    return 1e9, "ns"


def _agg_line(label: str, values: list[float | int], width: int = _W) -> str:
    if not values:
        return f"  {label:<22}{'—':>{width}}"
    return (
        f"  {label:<22}"
        f"{statistics.mean(values):{width}.2f}"
        f"{min(values):{width}.2f}"
        f"{max(values):{width}.2f}"
        f"{statistics.stdev(values) if len(values) > 1 else 0.0:{width}.2f}"
    )


def _print_aggregate(results: list[SampleResult], strategy: str) -> None:
    subset = [r for r in results if r.strategy == strategy]
    successes: list[SampleResult] = [r for r in subset if r.success]
    n = len(subset)
    s = len(successes)

    click.echo(f"\n  Strategy: {strategy}   ({s}/{n} succeeded)")
    header = f"  {'metric':<22}{'mean':>{_W}}{'min':>{_W}}{'max':>{_W}}{'stdev':>{_W}}"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    if not successes:
        click.echo("  No successful embeddings.")
        return

    elapsed_vals = [r.elapsed_s for r in subset]
    scale, unit = _elapsed_unit(statistics.mean(elapsed_vals))
    click.echo(_agg_line(f"elapsed ({unit})", [v * scale for v in elapsed_vals]))
    click.echo(_agg_line("nodes used", [r.nodes_used for r in successes]))  # pyright: ignore[reportArgumentType]
    click.echo(_agg_line("chain avg", [r.chain_avg for r in successes]))  # pyright: ignore[reportArgumentType]
    click.echo(_agg_line("chain max", [r.chain_max for r in successes]))  # pyright: ignore[reportArgumentType]
    click.echo(_agg_line("chain min", [r.chain_min for r in successes]))  # pyright: ignore[reportArgumentType]
    click.echo(_agg_line("source nodes", [r.source_nodes for r in successes]))
    click.echo(_agg_line("source edges", [r.source_edges for r in successes]))


def _print_sample_table(results: list[SampleResult]) -> None:
    col = 10
    mean_elapsed = statistics.mean(r.elapsed_s for r in results) if results else 0.0
    scale, unit = _elapsed_unit(mean_elapsed)
    elapsed_col = f"elapsed({unit})"
    header = (
        f"  {'id':>4}  {'seed':>8}  {'src_n':>5}  {'src_e':>5}  {'strat':>6}"
        f"  {'ok':>2}  {elapsed_col:>{col}}  {'n_used':>{col}}  {'ch_avg':>{col}}"
        f"  {'ch_min':>{col}}  {'ch_max':>{col}}"
    )
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for r in sorted(results, key=lambda x: (x.sample_id, x.strategy)):
        click.echo(
            f"  {r.sample_id:>4}  {r.seed:>8}  {r.source_nodes:>5}  {r.source_edges:>5}"
            f"  {r.strategy:>6}  {'Y' if r.success else 'N':>2}"
            f"  {_fmt(r.elapsed_s * scale, col)}  {_fmt(r.nodes_used, col)}  {_fmt(r.chain_avg, col)}"
            f"  {_fmt(r.chain_min, col)}  {_fmt(r.chain_max, col)}"
        )


def _write_csv(results: list[SampleResult], path: str) -> None:
    fields = [f.name for f in dataclasses.fields(SampleResult)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(dataclasses.asdict(r))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--graph-model",
    type=click.Choice(["er", "ba", "tree"]),
    default="er",
    show_default=True,
    help=(
        "Random graph family to generate as source.  "
        "'er' = Erdős-Rényi G(n,p),  "
        "'ba' = Barabási-Albert,  "
        "'tree' = random labeled tree."
    ),
)
@click.option(
    "--nodes",
    "-n",
    default=8,
    show_default=True,
    type=click.IntRange(min=2),
    help="Number of nodes in each generated source graph.",
)
@click.option(
    "--er-p",
    default=0.5,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Edge probability for Erdős--Rényi model (ignored for 'ba' and 'tree').",
)
@click.option(
    "--ba-m",
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of edges to attach per new node in Barabási--Albert model.",
)
@click.option(
    "--samples",
    default=20,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of random source graphs to generate and embed.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    type=int,
    help="Base RNG seed.  Each sample uses seed + sample_id for reproducibility.",
)
@click.option(
    "--topology",
    type=click.Choice(["chimera", "zephyr", "pegasus"]),
    default="chimera",
    show_default=True,
    help="D-Wave hardware topology to use as the embedding target.",
)
@click.option(
    "--topology-size",
    default=4,
    show_default=True,
    type=click.IntRange(min=1),
    help="Size parameter passed to the topology builder (e.g. 4 → Chimera(4)).",
)
@click.option(
    "--target-json",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Load the target graph from a JSON file instead of building a D-Wave topology.",
)
@click.option(
    "--strategy",
    type=click.Choice(
        ["random", "degree", "centrality", "longest_chains", "vertex_weights", "both", "all"]
    ),
    default="both",
    show_default=True,
    help=(
        "Ordering strategy for find_embedding.  "
        "'random' = uniform shuffle,  "
        "'degree' = descending source-degree first,  "
        "'centrality' = descending betweenness centrality first,  "
        "'longest_chains' = degree-order placement + longest-chain refinement,  "
        "'vertex_weights' = vertex-weight Dijkstra scheme (Cai et al. 2014),  "
        "'both' = random vs degree,  "
        "'all' = all five strategies."
    ),
)
@click.option(
    "--tries",
    default=30,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of embedding attempts per sample.",
)
@click.option(
    "--refinement-constant",
    default=20,
    show_default=True,
    type=click.IntRange(min=1),
    help="Refinement iterations = k × |V(source)|.",
)
@click.option(
    "--overlap-penalty",
    default=2.0,
    show_default=True,
    type=float,
    help="Penalty weight for edges leading into another vertex-model.",
)
@click.option(
    "--csv",
    "csv_path",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Write per-sample results to this CSV file.",
)
@click.option(
    "--no-detail",
    is_flag=True,
    default=False,
    help="Suppress the per-sample table; print only aggregate statistics.",
)
def main(
    graph_model: str,
    nodes: int,
    er_p: float,
    ba_m: int,
    samples: int,
    seed: int,
    topology: str,
    topology_size: int,
    target_json: str | None,
    strategy: str,
    tries: int,
    refinement_constant: int,
    overlap_penalty: float,
    csv_path: str | None,
    no_detail: bool,
) -> None:
    """Benchmark minor-embedding on randomly generated source graphs.

    Generates SAMPLES random source graphs, embeds each into the chosen target
    topology, and reports chain-length statistics broken down by strategy.
    """
    # Build target
    if target_json is not None:
        import json

        from networkx.readwrite import json_graph

        with open(target_json) as f:
            data = json.load(f)
        if "nodes" in data and "links" in data:
            target = json_graph.node_link_graph(data, edges="links")
        elif "nodes" in data and "adjacency" in data:
            target = json_graph.adjacency_graph(data)
        else:
            click.echo(
                "Unrecognised JSON graph format (expected node-link or adjacency).",
                err=True,
            )
            sys.exit(1)
        topo_label = target_json
    else:
        click.echo(f"Building target: {topology}({topology_size}) …", err=True)
        target = _build_target(topology, topology_size)
        topo_label = f"{topology}({topology_size})"

    if strategy == "both":
        strategies = ["random", "degree"]
    elif strategy == "all":
        strategies = ["random", "degree", "centrality", "longest_chains", "vertex_weights"]
    else:
        strategies = [strategy]

    click.echo(
        f"Graph model : {graph_model.upper()}"
        + (f"  n={nodes}  p={er_p}" if graph_model == "er" else "")
        + (f"  n={nodes}  m={ba_m}" if graph_model == "ba" else "")
        + (f"  n={nodes}" if graph_model == "tree" else ""),
        err=True,
    )
    click.echo(
        f"Target      : {topo_label} ({target.number_of_nodes()} nodes)", err=True
    )
    click.echo(
        f"Samples     : {samples}   base seed: {seed}   strategies: {', '.join(strategies)}",
        err=True,
    )
    click.echo("", err=True)

    all_results: list[SampleResult] = []

    for i in range(samples):
        sample_seed = seed + i

        # Generate source graph
        if graph_model == "er":
            source = _gen_er(nodes, er_p, sample_seed)
        elif graph_model == "ba":
            source = _gen_ba(nodes, ba_m, sample_seed)
        else:
            source = _gen_tree(nodes, sample_seed)

        if source.number_of_nodes() < 2:
            click.echo(
                f"  sample {i}: generated graph has <2 nodes, skipping.", err=True
            )
            continue

        for strat in strategies:
            result = _run_sample(
                sample_id=i,
                source=source,
                target=target,
                strategy=strat,
                seed=sample_seed,
                tries=tries,
                refinement_constant=refinement_constant,
                overlap_penalty=overlap_penalty,
            )
            all_results.append(result)

    # Per-sample table
    click.echo("=" * 78)
    click.echo("Per-sample results")
    click.echo("=" * 78)
    if not no_detail:
        _print_sample_table(all_results)
    else:
        click.echo("  (suppressed — use without --no-detail to see per-sample rows)")

    # Aggregate statistics
    click.echo("\n" + "=" * 78)
    click.echo("Aggregate statistics")
    click.echo("=" * 78)
    for strat in strategies:
        _print_aggregate(all_results, strat)

    # CSV output
    if csv_path:
        _write_csv(all_results, csv_path)
        click.echo(f"\nCSV written → {csv_path}", err=True)


if __name__ == "__main__":
    main()
