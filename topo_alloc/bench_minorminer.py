"""
Comparison benchmark: quip ``embed`` (balanced) vs D-Wave ``minorminer``.

Generates random source graphs and embeds each into a chosen D-Wave topology
using both solvers side-by-side.  Reports per-sample and aggregate statistics.

Requires ``minorminer`` to be installed::

    pip install minorminer

Usage examples
--------------
# Erdős-Rényi graphs (n=8, p=0.5) into Chimera(4), 20 samples:
    python -m topo_alloc.bench_minorminer --graph-model er --nodes 8 --er-p 0.5 \\
        --topology chimera --topology-size 4 --samples 20

# Barabási-Albert graphs into Zephyr(3):
    python -m topo_alloc.bench_minorminer --graph-model ba --nodes 10 --ba-m 3 \\
        --topology zephyr --topology-size 3 --samples 15

# Random trees into Pegasus(4), CSV output:
    python -m topo_alloc.bench_minorminer --graph-model tree --nodes 12 \\
        --topology pegasus --topology-size 4 --samples 30 --csv results.csv

# Suppress per-sample rows, show only aggregates:
    python -m topo_alloc.bench_minorminer --graph-model er --nodes 10 --er-p 0.6 \\
        --topology chimera --topology-size 4 --samples 50 --no-detail
"""

from __future__ import annotations

import csv
import dataclasses
import random
import statistics
import sys
import time

import click
import networkx as nx

from topo_alloc.graphviz_render import EmbeddingStats, embedding_stats
from topo_alloc.minor_alloc import embed

SOLVERS = ("quip", "minorminer")

# ---------------------------------------------------------------------------
# Graph-model generators
# ---------------------------------------------------------------------------


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
    solver: str
    success: bool
    nodes_used: int | None
    chain_min: int | None
    chain_max: int | None
    chain_avg: float | None
    elapsed_s: float


# ---------------------------------------------------------------------------
# Solver runners
# ---------------------------------------------------------------------------


def _run_quip(
    source: nx.Graph,
    target: nx.Graph,
    seed: int,
    tries: int,
    refinement_constant: int,
    overlap_penalty: int,
) -> tuple[dict | None, float]:
    def rng_factory():
        return random.Random(seed)

    t0 = time.perf_counter()
    embedding = embed(
        source,
        target,
        priority="balanced",
        rng_factory=rng_factory,
        tries=tries,
        refinment_constant=refinement_constant,
        overlap_penalty=overlap_penalty,
    )
    return embedding, time.perf_counter() - t0


def _run_minorminer(
    source: nx.Graph,
    target: nx.Graph,
    seed: int,
    tries: int,
) -> tuple[dict | None, float]:
    import minorminer

    t0 = time.perf_counter()
    embedding: dict = minorminer.find_embedding(
        source, target, random_seed=seed % (2**64), tries=tries
    )
    elapsed = time.perf_counter() - t0
    return (embedding if embedding else None), elapsed


def _make_result(
    sample_id: int,
    seed: int,
    source: nx.Graph,
    solver: str,
    embedding: dict | None,
    elapsed_s: float,
) -> SampleResult:
    if embedding is None:
        return SampleResult(
            sample_id=sample_id,
            seed=seed,
            source_nodes=source.number_of_nodes(),
            source_edges=source.number_of_edges(),
            solver=solver,
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
        solver=solver,
        success=True,
        nodes_used=stats.nodes_used,
        chain_min=stats.chain_min,
        chain_max=stats.chain_max,
        chain_avg=stats.chain_avg,
        elapsed_s=elapsed_s,
    )


def _run_sample(
    sample_id: int,
    source: nx.Graph,
    target: nx.Graph,
    seed: int,
    quip_tries: int,
    mm_tries: int,
    refinement_constant: int,
    overlap_penalty: int,
) -> list[SampleResult]:
    results: list[SampleResult] = []

    emb, elapsed = _run_quip(source, target, seed, quip_tries, refinement_constant, overlap_penalty)
    results.append(_make_result(sample_id, seed, source, "quip", emb, elapsed))

    emb, elapsed = _run_minorminer(source, target, seed, mm_tries)
    results.append(_make_result(sample_id, seed, source, "minorminer", emb, elapsed))

    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_W = 12


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


def _print_aggregate(results: list[SampleResult], solver: str) -> None:
    subset = [r for r in results if r.solver == solver]
    successes = [r for r in subset if r.success]
    n = len(subset)
    s = len(successes)

    click.echo(f"\n  Solver: {solver}   ({s}/{n} succeeded)")
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


def _print_sample_table(results: list[SampleResult]) -> None:
    col = 10
    mean_elapsed = statistics.mean(r.elapsed_s for r in results) if results else 0.0
    scale, unit = _elapsed_unit(mean_elapsed)
    elapsed_col = f"elapsed({unit})"
    header = (
        f"  {'id':>4}  {'seed':>8}  {'src_n':>5}  {'src_e':>5}  {'solver':>10}"
        f"  {'ok':>2}  {elapsed_col:>{col}}  {'n_used':>{col}}  {'ch_avg':>{col}}"
        f"  {'ch_min':>{col}}  {'ch_max':>{col}}"
    )
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for r in sorted(results, key=lambda x: (x.sample_id, x.solver)):
        click.echo(
            f"  {r.sample_id:>4}  {r.seed:>8}  {r.source_nodes:>5}  {r.source_edges:>5}"
            f"  {r.solver:>10}  {'Y' if r.success else 'N':>2}"
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
        "Random graph family to use as source.  "
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
    help="Edge probability for Erdős-Rényi model (ignored for 'ba' and 'tree').",
)
@click.option(
    "--ba-m",
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of edges to attach per new node in Barabási-Albert model.",
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
    "--quip-tries",
    default=50,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of independent embedding attempts for the quip solver.",
)
@click.option(
    "--mm-tries",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of restart attempts for minorminer (its default is 10).",
)
@click.option(
    "--refinement-constant",
    default=20,
    show_default=True,
    type=click.IntRange(min=1),
    help="quip refinement iterations = k × |V(source)|.",
)
@click.option(
    "--overlap-penalty",
    default=2,
    show_default=True,
    type=int,
    help="quip penalty weight for edges leading into another vertex-model.",
)
@click.option(
    "--warmup",
    default=3,
    show_default=True,
    type=click.IntRange(min=0),
    help=(
        "Number of warmup rounds to run before recording results.  "
        "Uses seeds just below the base seed so they never overlap with "
        "the measured samples.  Set to 0 to disable."
    ),
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
    quip_tries: int,
    mm_tries: int,
    refinement_constant: int,
    overlap_penalty: int,
    warmup: int,
    csv_path: str | None,
    no_detail: bool,
) -> None:
    """Compare quip (embed, balanced) against minorminer on random source graphs.

    Generates SAMPLES random source graphs and embeds each into the chosen D-Wave
    topology using both solvers, then reports chain-length and timing statistics.
    """
    try:
        import minorminer as _mm  # noqa: F401
    except ImportError:
        click.echo("Error: minorminer is not installed.  Run: pip install minorminer", err=True)
        sys.exit(1)

    click.echo(f"Building target: {topology}({topology_size}) …", err=True)
    target = _build_target(topology, topology_size)
    topo_label = f"{topology}({topology_size})"

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
        f"Samples     : {samples}   base seed: {seed}",
        err=True,
    )
    click.echo(
        f"quip tries  : {quip_tries}   mm tries: {mm_tries}",
        err=True,
    )
    click.echo("", err=True)

    def _make_source(s: int) -> nx.Graph:
        if graph_model == "er":
            return _gen_er(nodes, er_p, s)
        if graph_model == "ba":
            return _gen_ba(nodes, ba_m, s)
        return _gen_tree(nodes, s)

    # Warmup: discarded rounds to bring caches and JIT to steady state.
    # Seeds are offset below the base seed so they never collide with
    # the measured samples (which use seed, seed+1, …, seed+samples-1).
    if warmup > 0:
        click.echo(f"Warming up ({warmup} round(s)) …", err=True)
        for w in range(warmup):
            warmup_seed = seed - warmup + w
            src = _make_source(warmup_seed)
            if src.number_of_nodes() >= 2:
                _run_sample(
                    sample_id=-1,
                    source=src,
                    target=target,
                    seed=warmup_seed,
                    quip_tries=quip_tries,
                    mm_tries=mm_tries,
                    refinement_constant=refinement_constant,
                    overlap_penalty=overlap_penalty,
                )
        click.echo("Warmup done.  Starting timed benchmark …", err=True)
        click.echo("", err=True)

    all_results: list[SampleResult] = []

    for i in range(samples):
        sample_seed = seed + i
        source = _make_source(sample_seed)

        if source.number_of_nodes() < 2:
            click.echo(
                f"  sample {i}: generated graph has <2 nodes, skipping.", err=True
            )
            continue

        all_results.extend(
            _run_sample(
                sample_id=i,
                source=source,
                target=target,
                seed=sample_seed,
                quip_tries=quip_tries,
                mm_tries=mm_tries,
                refinement_constant=refinement_constant,
                overlap_penalty=overlap_penalty,
            )
        )

    # Per-sample table
    click.echo("=" * 78)
    click.echo("Per-sample results")
    click.echo("=" * 78)
    if not no_detail:
        _print_sample_table(all_results)
    else:
        click.echo("  (suppressed — run without --no-detail to see per-sample rows)")

    # Aggregate statistics
    click.echo("\n" + "=" * 78)
    click.echo("Aggregate statistics")
    click.echo("=" * 78)
    for solver in SOLVERS:
        _print_aggregate(all_results, solver)

    # CSV output
    if csv_path:
        _write_csv(all_results, csv_path)
        click.echo(f"\nCSV written → {csv_path}", err=True)


if __name__ == "__main__":
    main()
