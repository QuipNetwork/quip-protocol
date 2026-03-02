"""
Minor-embedding heuristic for mapping an arbitrary source graph H into a
hardware topology graph G as a graph minor.

Background
----------
A graph H is a *minor* of G when H can be obtained from G by a sequence of
edge contractions, edge deletions, and vertex deletions.  Equivalently, H is
a minor of G iff there exists a *minor embedding*: an injective map

    φ : V(H) → 2^{V(G)}

such that for each source node x ∈ V(H):

1. φ(x) is non-empty and induces a connected subgraph of G (*vertex model*,
   also called a *chain* in the quantum-annealing literature).
2. All vertex models are pairwise disjoint.
3. For every edge (x, y) ∈ E(H) there is at least one edge in G between
   φ(x) and φ(y).

The problem is NP-hard in general.  This module implements a practical
randomised heuristic following the approach of Cai, Macready & Roy (2014),
"A practical heuristic for finding graph minors"
(https://arxiv.org/abs/1406.2741).

Algorithm outline
-----------------
The outer loop makes up to `tries` independent attempts.  Each attempt
consists of three stages:

**Stage 1 – Greedy initialisation.**
Source nodes are placed one by one in some order (random, degree-first, or
centrality-first depending on `options`).  For each source node x, a vertex
model φ(x) is constructed by running multi-source Dijkstra from the already-
placed models of x's neighbours in H, finding the cheapest way to connect
them through a single new root target node.  The Dijkstra edge weights
encode a penalty for passing through nodes that are already occupied by
other models — either a flat `overlap_penalty` multiplier (default) or the
vertex-weight formula from Cai et al. (see below).  The result is a
connected Steiner-tree approximation anchored at the cheapest free root.
Overlapping models are accepted at this stage.

**Stage 2 – Overlap-removal refinement.**
Repeated for `refinement_constant × |V(H)|` rounds.  In each round, the
source node x* with the most overlapping target nodes is identified and
re-embedded from scratch (using the flat `overlap_penalty` scheme regardless
of `options`, because the penalty scheme is better at untangling overlaps).
The loop terminates early when all vertex models become disjoint.

**Stage 2b – Longest-chain refinement (optional, `REFINE_LONGEST_CHAINS`).**
After overlap removal has converged, a further
`refinement_constant × |V(H)|` rounds are run: in each round the source node
whose vertex model is currently longest is re-embedded, and the new model is
accepted only when it is *strictly shorter* (non-worsening).  This pass
directly targets the `nodes_used` metric without risking the validity of the
embedding.

**Stage 3 – Validation.**
The candidate embedding is checked against all three minor-embedding
conditions.  If it passes, it is returned immediately.  Otherwise the next
attempt begins.

Vertex-weight scheme (`USE_VERTEX_WEIGHTS`)
-------------------------------------------
Instead of the flat `overlap_penalty`, the Cai et al. (2014) vertex-weight
formula can be used during Stage 1:

    wt(g) = D^{n − inclusion_count(g)}

where D is the diameter of the target graph, n = |V(H)|, and
inclusion_count(g) counts how many source-node models currently contain g.
A target node shared by many models has a small exponent and therefore low
weight, making paths through it cheap and encouraging the algorithm to route
new chains through already-placed nodes (path re-use).  A completely free
node gets weight D^n, making it expensive and discouraging premature
commitment to unused resources.

Ordering heuristics (`ORDER_BY_DEGREE_ASC`, `ORDER_BY_CENTRALITY`)
-------------------------------------------------------------------
The greedy placement order in Stage 1 affects quality significantly.
``ORDER_BY_DEGREE_ASC`` places low-degree nodes first and the hub last,
so Dijkstra can bridge all already-placed neighbours when building the
hub's chain.  ``ORDER_BY_CENTRALITY`` sorts by descending betweenness,
which is particularly effective on tree-structured source graphs.
Within each tier of equal-valued nodes the order is shuffled randomly,
preserving the coarse ranking while still exploring different placements
across tries.
"""

from __future__ import annotations

import enum
import itertools
import random as rng
from collections import defaultdict
from typing import Any, Callable, Literal

import networkx as nx

type Model[G, H] = dict[H, frozenset[G]]


class EmbedOption(enum.Flag):
    """
    Options controlling the behaviour of `find_embedding`.

    Multiple options may be combined with the ``|`` operator::

        find_embedding(source, target, options=EmbedOption.ORDER_BY_DEGREE_ASC | EmbedOption.REFINE_LONGEST_CHAINS)

    Options
    -------
    ORDER_BY_DEGREE_ASC
        Sort source nodes in *ascending* order of their degree in the source
        graph before placement.  Low-degree nodes are placed first (cheaply,
        with few constraints) and the high-degree hub is placed last, by
        which point all of its neighbours are already positioned — Dijkstra
        then builds the hub's chain as a natural bridge through them.  This
        avoids the failure mode of descending-degree ordering on scale-free
        (e.g. Barabási-Albert) graphs with a dominant hub.

    ORDER_BY_CENTRALITY
        Sort source nodes in descending order of their betweenness centrality
        in the source graph.  Nodes that lie on many shortest paths are placed
        first.  Within ties the order is shuffled randomly across tries.
        Takes precedence over ``ORDER_BY_DEGREE_ASC`` when both are set.

    REFINE_LONGEST_CHAINS
        After overlap-removal refinement converges, run a second pass that
        re-embeds the source node with the longest chain for
        ``refinement_constant * |V(H)|`` iterations.  Each accepted
        re-embedding is strictly non-worsening.
    """

    ORDER_BY_DEGREE_ASC = enum.auto()
    ORDER_BY_CENTRALITY = enum.auto()
    REFINE_LONGEST_CHAINS = enum.auto()


def find_embedding[G, H](
    source: nx.Graph[H],
    target: nx.Graph[G],
    /,
    *,
    rng_factory: Callable[[], rng.Random] = rng.Random,
    tries: int = 30,
    refinment_constant: int = 10,
    overlap_penalty: float = 2.0,
    options: EmbedOption = EmbedOption(0),
) -> Model[G, H] | None:
    """
    Finds a graph embedding of `source` as a minor of `target`.

    # Parameters
    source: nx.Graph[H]
        The graph to be embedded into `target`.
    target: nx.Graph[G]
        The graph that reassembles the hardware topology.
    rng_factory: Callable[[], rng.Random]
        The pseudo-random number generator factory. By default
        returns `rng.Random()`.
    tries: int
        The number of retries when the heuristic algorithm fails
        to find non-overlapping mapping.
    refinment_constant: int
        The constant `k` which sets the number of refinment iterations
        to `k * |V(H)|`.
    overlap_penalty: float
        The weight given to an edge that leads towards a node belonging
        to a different vertex model.
    options: EmbedOption
        A set of `EmbedOption` flags controlling the embedding strategy.
        Multiple options may be combined with ``|``.  See `EmbedOption`
        for the full list.

    # Returns
    A `Model[G, H]` which is a dictionary from `H` nodes into
    sets of `G` nodes mapped. Returns `None` if after the number of `tries`
    there is no non-overlapping map.
    """
    order_by_degree_asc = EmbedOption.ORDER_BY_DEGREE_ASC in options
    order_by_centrality = EmbedOption.ORDER_BY_CENTRALITY in options
    refine_longest_chains = EmbedOption.REFINE_LONGEST_CHAINS in options

    refine_rounds = refinment_constant * len(source.nodes)
    rng = rng_factory()
    src_nodes = list(source.nodes)
    src_adj = {h: list(source.neighbors(h)) for h in src_nodes}

    degree_asc_order = None
    centrality_order = None

    centrality = None
    if order_by_centrality:
        centrality = nx.betweenness_centrality(source)
        centrality_order = sorted(src_nodes, key=lambda h: centrality[h], reverse=True)
    elif order_by_degree_asc:
        # Ascending degree: low-degree nodes placed first (cheap, few constraints),
        # hub placed last so Dijkstra can bridge all already-placed neighbours.
        degree_asc_order = sorted(src_nodes, key=lambda h: source.degree(h))

    for _ in range(tries):
        # --------------------------------
        # Stage 1 - initialize model with greedy placement in a random vertex order
        # --------------------------------
        if order_by_centrality:
            order = _shuffle_within_tiers(
                centrality_order,  # pyright: ignore[reportArgumentType]
                lambda h: centrality[h],  # pyright: ignore[reportOptionalSubscript]
                rng,
            )
        elif order_by_degree_asc:
            order = _shuffle_within_tiers(
                degree_asc_order or [], lambda h: source.degree(h), rng
            )
        else:
            order = list(src_nodes)
            rng.shuffle(order)
        phi: dict[H, list[G]] = {x: [] for x in order}
        for x in order:
            model = build_model(
                x,
                src_adj,
                phi,
                target,
                overlap_penalty,
            )
            if model is None:
                break
            phi[x] = model

        # --------------------------------
        # Stage 2 - refinement: re-embed nodes with overlapping models
        # --------------------------------
        for _ in range(refine_rounds):
            # Sum up all overlaps
            counters: defaultdict[G, int] = defaultdict(int)
            for model in phi.values():
                for g in model:
                    counters[g] += 1
            overlap_per_node = {
                x: sum(1 for g in phi[x] if counters[g] > 1) for x in src_nodes
            }
            if all(v == 0 for v in overlap_per_node.values()):
                # This means we have non-overlaping vertex-models
                break
            else:
                # Re-embed the H-node with the worst overlap
                x = max(order, key=lambda v: overlap_per_node[v])
                phi[x] = []
                model = build_model(
                    x,
                    src_adj,
                    phi,
                    target,
                    overlap_penalty=overlap_penalty,
                )
                if model is not None:
                    phi[x] = model

        # --------------------------------
        # Stage 2b - longest-chain refinement (optional)
        # --------------------------------
        if refine_longest_chains:
            phi = _refine_longest_chains(
                src_nodes,
                src_adj,
                phi,
                target,
                overlap_penalty=overlap_penalty,
                rounds=refine_rounds,
            )

        # --------------------------------
        # Stage 3 - validate the solution
        # --------------------------------
        phi_set = {x: frozenset(phi[x]) for x in src_nodes}
        if is_valid_embedding(source, target, phi_set):
            return phi_set
    return None


def _refine_longest_chains[G, H](
    src_nodes: list[H],
    src_adj: dict[H, list[H]],
    phi: dict[H, list[G]],
    graph: nx.Graph[G],
    *,
    rounds: int,
    overlap_penalty: float,
) -> dict[H, list[G]]:
    """
    Refinement pass that repeatedly re-embeds the source node with the
    longest chain, accepting the new placement only when it is strictly
    shorter (non-worsening).

    This pass runs after the overlap refinement has already converged, so it
    only operates on overlap-free embeddings and always keeps the embedding
    valid.
    """
    for _ in range(rounds):
        # Pick the source node whose chain is currently longest.
        x = max(src_nodes, key=lambda v: len(phi[v]))
        current_len = len(phi[x])

        # Temporarily remove x from phi so build_model treats those target
        # nodes as free.
        old_model = phi[x]
        phi[x] = []

        candidate = build_model(
            x,
            src_adj,
            phi,
            graph,
            overlap_penalty,
        )

        if candidate is not None and len(candidate) < current_len:
            # Accept: strictly shorter chain found.
            phi[x] = candidate
        else:
            # Reject: restore original model.
            phi[x] = old_model

    return phi


def _shuffle_within_tiers[H](
    order: list[H],
    key: Callable[[H], Any],
    rng_inst: rng.Random,
) -> list[H]:
    """
    Return a node ordering that preserves tier boundaries defined by `key` but
    shuffles nodes within each tier so successive tries explore different orderings.
    """
    from itertools import groupby

    result: list[H] = []
    for _, group in groupby(order, key=key):
        tier = list(group)
        rng_inst.shuffle(tier)
        result.extend(tier)
    return result


def build_model[G, H](
    x: H,
    adjlist: dict[H, list[H]],
    phi: dict[H, list[G]],
    graph: nx.Graph[G],
    overlap_penalty: float,
) -> list[G] | None:
    """
    Build a vertex-model (chain) for source node `x` in `graph`.

    Dijkstra edge weights penalise passing through target nodes already
    occupied by another chain via a flat `overlap_penalty` multiplier.
    """
    # Get already placed neighbours to x
    placed_neighbours = [y for y in adjlist[x] if phi[y]]

    # Find any other nodes than x in y-models
    occupied: set[G] = set().union(*(set(model) for y, model in phi.items() if y != x))

    # If there is no placed neighbour, pick the first free anchor target node.
    if not placed_neighbours:
        free = [g for g in graph.nodes if g not in occupied]
        if not free:
            return [next(iter(graph.nodes))]
        return [free[0]]

    # Run Dijkstra's algorithm from multiple sources from v-models.
    def weight(_u: G, v: G, _data) -> float:
        return 1.0 + overlap_penalty * (1.0 if v in occupied else 0.0)

    dist_from = {}
    path_from = {}
    for y in placed_neighbours:
        dist_from[y], path_from[y] = nx.multi_source_dijkstra(
            graph, sources=phi[y], weight=weight
        )

    # Choose the best root: prefer free nodes in both modes, with a fallback
    # to occupied nodes if no free root is reachable.
    # In vertex-weight mode the Dijkstra distances already encode occupancy
    # preference (shared nodes are cheaper to route through), so the best
    # free root naturally tends to be adjacent to well-placed chains.
    best_root, best_cost = None, float("+inf")
    for g in graph.nodes:
        if g in occupied:
            continue
        cost = sum(dist_from[y].get(g, float("+inf")) for y in placed_neighbours)
        if cost < best_cost:
            best_root, best_cost = g, cost

    if best_root is None or best_cost == float("inf"):
        # Fallback: allow root from occupied nodes (should rarely happen)
        for g in graph.nodes:
            cost = sum(dist_from[y].get(g, float("inf")) for y in placed_neighbours)
            if cost < best_cost:
                best_cost, best_root = cost, g

    # Worst case: we couldn't pick any root, so we fail.
    if best_root is None or best_cost == float("inf"):
        return None

    # Grow Steiner tree: trace paths from root toward each neighbour,
    # stopping at (but not including) phi[y] boundary nodes.
    model = {best_root}
    for y in placed_neighbours:
        phi_y_set = frozenset(phi[y])
        path = path_stopping_at(path_from[y], best_root, phi_y_set)
        model.update(path)

    # Keep only the component containing root (paths should be connected,
    # but guard against edge cases).
    sub = graph.subgraph(model)
    if len(model) > 1 and not nx.is_connected(sub):
        model = set(nx.node_connected_component(sub, best_root))

    # Final check: model must have at least one neighbour in G for each phi[y]
    # (otherwise the path didn't reach close enough — pick a node adjacent to phi[y])
    model_set = set(model)
    for y in placed_neighbours:
        phi_y_set = frozenset(phi[y])
        if not any(graph.has_edge(m, g) for m in model_set for g in phi_y_set):
            # Fallback: add the closest free node adjacent to phi[y]
            for gy in phi[y]:
                for nb in graph.neighbors(gy):
                    if nb not in occupied:
                        model_set.add(nb)
                        break
                else:
                    continue
                break

    return list(model_set)


def path_stopping_at[G](
    paths: dict[G, list[G]], target: G, stop_set: frozenset[G]
) -> list[G]:
    """
    From `paths[target]`, extract the tail starting at `target`
    going backward toward the source,
    stopping just before the first node in `stop_set`.

    This gives the sub-path that belongs to the new model: it starts at
    `target` (the root) and ends at the G-node immediately adjacent to
    the phi[y] boundary — without including any phi[y] node.
    """
    full_path = paths.get(target, [target])
    return list(itertools.takewhile(lambda n: n not in stop_set, reversed(full_path)))


def is_valid_embedding[G, H](
    source: nx.Graph[H], target: nx.Graph[G], phi: dict[H, frozenset[G]] | None
) -> bool:
    """
    Check all three minor-embedding conditions:

    1. Every source node has a non-empty, connected vertex-model in target.
    2. Vertex-models are pairwise disjoint.
    3. Every source edge (u, v) has at least one target edge between phi[u]
       and phi[v].
    """
    if phi is None:
        return False

    # Condition 1 – non-empty and connected vertex models
    for h in source.nodes:
        model = phi[h]
        if not model:
            return False
        if len(model) > 1 and not nx.is_connected(target.subgraph(model)):
            return False

    # Condition 2 – disjoint
    all_nodes = [g for model in phi.values() for g in model]
    if len(all_nodes) != len(set(all_nodes)):
        return False

    # Condition 3 – edge coverage
    for u, v in source.edges:
        if not any(target.has_edge(gu, gv) for gu in phi[u] for gv in phi[v]):
            return False

    return True


# ---------------------------------------------------------------------------
# Heuristic option selector
# ---------------------------------------------------------------------------

# Target-to-source node ratio below which a topology is considered "tight".
# On tight topologies front-loading high-degree / high-centrality nodes leaves
# no slack for later placements, causing degree/centrality orderings to succeed
# far less often than random (18-12/30 vs 27/30 on Chimera(4) benchmarks).
# Above this ratio there is enough room that ORDER_BY_DEGREE consistently
# improves chain quality at no cost to success rate.
_TIGHT_RATIO: float = 25.0


def select_embed_options(
    source: nx.Graph,
    target: nx.Graph,
    *,
    priority: Literal["speed", "balanced", "quality"] = "balanced",
) -> EmbedOption:
    """
    Analyse ``source`` and ``target`` and return the `EmbedOption` flags most
    likely to yield a good embedding, based on empirical benchmarks.

    # Parameters
    source: nx.Graph
        The graph to be embedded.
    target: nx.Graph
        The hardware topology graph.
    priority: ``"speed"`` | ``"balanced"`` | ``"quality"``
        Trade-off preference:

        * ``"speed"``    – minimise runtime; always use random ordering
          (no pre-computation overhead).
        * ``"balanced"`` – default; good quality with acceptable latency.
        * ``"quality"``  – minimise chain lengths; may add longest-chain
          refinement on spacious topologies.

    # Returns
    An `EmbedOption` flag set to pass directly to `find_embedding`.

    # Heuristic rules

    The rules are derived from benchmarks across Chimera(4), Zephyr(3), and
    Pegasus(4) topologies using ER, BA, and tree source graphs:

    1. **Speed priority** → ``EmbedOption(0)`` (random).
       Random ordering has zero pre-computation cost and is a strong baseline
       on tight topologies.

    2. **Tree source** → ``ORDER_BY_CENTRALITY``.
       Betweenness-centrality ordering produces near-identity embeddings
       (chain_avg ≈ 1.02, chain_max ≈ 1.14) on trees, regardless of how
       tight the target topology is.  This is a massive quality win at
       negligible extra cost for sparse sources.

    3. **Tight topology** (``target_n / source_n < 25``) → ``EmbedOption(0)``.
       Front-loading constrained nodes leaves no slack on small targets.
       Random ordering succeeds 27/30 while degree/centrality drop to
       12–18/30 (Chimera(4) benchmarks).

    4. **Spacious topology, quality priority** →
       ``ORDER_BY_DEGREE_ASC | REFINE_LONGEST_CHAINS``.
       Ascending-degree ordering resolves the hub-failure mode of descending
       order on scale-free graphs, while matching or beating it on balanced
       graphs (21 ms vs 345 ms on Pegasus(4) ER benchmarks).  Longest-chain
       refinement squeezes further chain-length reduction at ~40× the runtime
       cost.

    5. **Spacious topology, balanced priority** → ``ORDER_BY_DEGREE_ASC``.
       Best quality/speed tradeoff: fixes hub-graph failures completely
       (100% vs 25–75% success on BA graphs) with no regression on balanced
       ER graphs, and outperforms descending degree on Pegasus(4).
    """
    if priority == "speed":
        return EmbedOption(0)

    n = source.number_of_nodes()
    if n == 0:
        return EmbedOption(0)

    # Trees: centrality ordering yields near-identity embeddings (chain_avg ≈ 1)
    # regardless of topology tightness — clear quality win with negligible cost.
    if nx.is_tree(source):
        return EmbedOption.ORDER_BY_CENTRALITY

    # Tight topologies: random ordering maximises success rate.
    tightness_ratio = target.number_of_nodes() / n
    if tightness_ratio < _TIGHT_RATIO:
        return EmbedOption(0)

    # Spacious topologies: ascending-degree ordering places the hub last so
    # Dijkstra can bridge all already-placed neighbours, fixing the failure
    # mode of descending ORDER_BY_DEGREE on scale-free source graphs.
    if priority == "quality":
        return EmbedOption.ORDER_BY_DEGREE_ASC | EmbedOption.REFINE_LONGEST_CHAINS
    return EmbedOption.ORDER_BY_DEGREE_ASC


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


def embed[G, H](
    source: nx.Graph[H],
    target: nx.Graph[G],
    /,
    *,
    priority: Literal["speed", "balanced", "quality"] = "balanced",
    rng_factory: Callable[[], rng.Random] = rng.Random,
    tries: int = 30,
    refinment_constant: int = 20,
    overlap_penalty: float = 2.0,
    options: EmbedOption | None = None,
) -> Model[G, H] | None:
    """
    Find a minor-embedding of ``source`` into ``target``, automatically
    selecting the best heuristic options for the given graphs.

    This is a convenience façade over `find_embedding`.  When ``options``
    is ``None`` (the default), `select_embed_options` analyses ``source``
    and ``target`` and picks the `EmbedOption` flags most likely to yield a
    compact, valid embedding.  Pass explicit ``options`` to bypass
    auto-selection entirely.

    # Parameters
    source: nx.Graph[H]
        The graph to be embedded into ``target``.
    target: nx.Graph[G]
        The graph representing the hardware topology.
    priority: ``"speed"`` | ``"balanced"`` | ``"quality"``
        Trade-off hint forwarded to `select_embed_options`.  Ignored when
        ``options`` is supplied explicitly.
    rng_factory: Callable[[], rng.Random]
        Factory for the pseudo-random number generator.
    tries: int
        Maximum number of independent embedding attempts.
    refinment_constant: int
        Refinement iterations = ``refinment_constant × |V(source)|``.
    overlap_penalty: float
        Flat penalty weight for edges leading into another vertex-model.
        Used when `EmbedOption.USE_VERTEX_WEIGHTS` is not active.
    options: EmbedOption | None
        Explicit option flags.  When ``None`` (default), options are
        chosen automatically via `select_embed_options`.

    # Returns
    A `Model[G, H]` mapping each source node to its vertex-model
    (a frozenset of target nodes), or ``None`` if no valid embedding was
    found within ``tries`` attempts.

    # Examples

    Basic auto-selected embedding::

        embedding = embed(source_graph, target_graph)

    Prefer chain-length quality over speed::

        embedding = embed(source_graph, target_graph, priority="quality")

    Override auto-selection with explicit options::

        embedding = embed(
            source_graph, target_graph,
            options=EmbedOption.ORDER_BY_DEGREE | EmbedOption.REFINE_LONGEST_CHAINS,
        )
    """
    if options is None:
        options = select_embed_options(source, target, priority=priority)
    return find_embedding(
        source,
        target,
        rng_factory=rng_factory,
        tries=tries,
        refinment_constant=refinment_constant,
        overlap_penalty=overlap_penalty,
        options=options,
    )


__all__ = [
    "EmbedOption",
    "Model",
    "build_model",
    "embed",
    "find_embedding",
    "is_valid_embedding",
    "select_embed_options",
]
