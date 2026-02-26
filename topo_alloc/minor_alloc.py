"""
A graph allocator that sends some arbitrary Ising model
into a known architecture as a graph.

The whole allocator uses minor-embedding approach to
perform proper allocations.
"""

from __future__ import annotations

import itertools
import random as rng
from collections import defaultdict
from typing import Callable

import networkx as nx

type Model[G, H] = dict[H, frozenset[G]]


def find_embedding[G, H](
    source: nx.Graph[H],
    target: nx.Graph[G],
    /,
    *,
    rng_factory: Callable[[], rng.Random] = rng.Random,
    tries: int = 30,
    refinment_constant: int = 20,
    overlap_penalty: float = 2.0,
    order_by_degree: bool = False,
    order_by_centrality: bool = False,
    refine_longest_chains: bool = False,
    use_vertex_weights: bool = False,
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
        to different vertex model.  Only used when `use_vertex_weights`
        is False.
    order_by_degree: bool
        When True, the initial source node ordering is seeded by sorting
        nodes in descending order of their degree in the source graph,
        rather than a uniform random shuffle.  High-degree nodes (more
        constrained) are placed first, which can reduce chain lengths and
        improve embedding quality on structured topologies.
    order_by_centrality: bool
        When True, the initial source node ordering is seeded by sorting
        nodes in descending order of their betweenness centrality in the
        source graph.  Nodes that lie on many shortest paths (and are thus
        more structurally central) are placed first.  Within ties the order
        is shuffled randomly across tries.  Mutually exclusive with
        `order_by_degree`; if both are True, `order_by_centrality` takes
        precedence.
    refine_longest_chains: bool
        When True, after the overlap-removal refinement converges a second
        refinement pass is run: the source node whose vertex-model is
        currently longest is re-embedded, repeating for
        `refinment_constant * |V(H)|` iterations.  Each successful
        re-embedding is accepted only when it does not increase the chain
        length, so the pass is strictly non-worsening.  This directly
        targets the `nodes_used` metric.
    use_vertex_weights: bool
        When True, use the vertex-weight scheme from Cai, Macready & Roy
        (2014).  Each target node g receives weight
        wt(g) = D^{|{i : g ∉ φ(x_i)}|}, where D is the diameter of the
        target graph.  Nodes included in many existing vertex-models get
        low weight (encouraging path re-use through shared nodes), while
        free nodes get high weight (D^n).  This replaces the flat
        `overlap_penalty` approach.

    # Returns
    A `Model[G, H]` which is a dictionary from `H` nodes into
    sets of `G` nodes mapped. Returns `None` if after the number of `tries`
    there is no non-overlapping map.
    """
    refine_rounds = refinment_constant * len(source.nodes)
    rng = rng_factory()
    src_nodes = list(source.nodes)
    src_adj = {h: list(source.neighbors(h)) for h in src_nodes}
    degree_order = None
    centrality_order = None

    # Pre-compute target diameter for vertex-weight mode (Cai et al. 2014).
    # nx.diameter requires a connected graph; fall back to a safe lower-bound
    # of 2 on disconnected targets so the weight formula remains well-defined.
    # In the default overlap-penalty mode the diameter is unused (set to 1).
    target_diameter = (
        (nx.diameter(target) if nx.is_connected(target) else 2)
        if use_vertex_weights
        else 1
    )

    centrality = None
    if order_by_centrality:
        centrality = nx.betweenness_centrality(source)
        centrality_order = sorted(src_nodes, key=lambda h: centrality[h], reverse=True)
    elif order_by_degree:
        # Seed the order by descending source-graph degree so that the most
        # constrained nodes are placed first.  Ties are broken randomly.
        degree_order = sorted(src_nodes, key=lambda h: source.degree(h), reverse=True)

    for _ in range(tries):
        # --------------------------------
        # Stage 1 - initialize model with greedy placement in a random vertex order
        # --------------------------------
        if order_by_centrality:
            order = _shuffle_within_centrality_tiers(centrality_order, centrality, rng)  # pyright: ignore[reportArgumentType]
        elif order_by_degree:
            # Shuffle within each degree tier to keep randomness across tries
            # while preserving the coarse degree ordering.
            order = _shuffle_within_degree_tiers(degree_order, source, rng)
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
                use_vertex_weights=use_vertex_weights,
                target_diameter=target_diameter,
                num_source_nodes=len(src_nodes),
            )
            if model is None:
                break
            phi[x] = model

        # --------------------------------
        # Stage 2 - refinement: re-embed nodes with overlapping models
        # --------------------------------
        # Overlap resolution always uses the flat overlap_penalty regardless
        # of use_vertex_weights.  Vertex weights encourage path re-use and
        # intentionally create overlapping models; the penalty-based scheme is
        # better suited for untangling them because it actively steers away
        # from occupied nodes.
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
                    use_vertex_weights=False,
                    target_diameter=1,
                    num_source_nodes=len(src_nodes),
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
                use_vertex_weights=use_vertex_weights,
                target_diameter=target_diameter,
                num_source_nodes=len(src_nodes),
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
    use_vertex_weights: bool = False,
    target_diameter: int = 1,
    num_source_nodes: int = 0,
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
            use_vertex_weights=use_vertex_weights,
            target_diameter=target_diameter,
            num_source_nodes=num_source_nodes,
        )

        if candidate is not None and len(candidate) < current_len:
            # Accept: strictly shorter chain found.
            phi[x] = candidate
        else:
            # Reject: restore original model.
            phi[x] = old_model

    return phi


def _shuffle_within_degree_tiers[H](
    degree_order: list[H] | None,
    source: nx.Graph[H],
    rng_inst: rng.Random,
) -> list[H]:
    """
    Return a node ordering that respects descending degree tiers but shuffles
    nodes within each tier so successive tries explore different orderings.
    """
    from itertools import groupby

    if degree_order is None:
        return []

    result: list[H] = []
    for _, group in groupby(degree_order, key=lambda h: source.degree(h)):
        tier = list(group)
        rng_inst.shuffle(tier)
        result.extend(tier)
    return result


def _shuffle_within_centrality_tiers[H](
    centrality_order: list[H],
    centrality: dict[H, float],
    rng_inst: rng.Random,
) -> list[H]:
    """
    Return a node ordering that respects descending betweenness-centrality
    tiers but shuffles nodes within each tier so successive tries explore
    different orderings.

    Two nodes are in the same tier when their centrality values are equal
    (which is common for regular or symmetric graphs).
    """
    from itertools import groupby

    result: list[H] = []
    for _, group in groupby(centrality_order, key=lambda h: centrality[h]):
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
    *,
    use_vertex_weights: bool = False,
    target_diameter: int = 1,
    num_source_nodes: int = 0,
) -> list[G] | None:
    """
    Build a vertex-model (chain) for source node `x` in `graph`.

    When `use_vertex_weights` is True the Dijkstra edge weights are derived
    from the vertex-weight formula of Cai, Macready & Roy (2014):

        ``wt(g) = D^{|{i : g ∉ φ(x_i)}|}``

    where
        - D = diameter of the target topology; and
        - the exponent counts how many source nodes whose current model *does not*
          contain target node g.

    A node shared by many vertex-models has a small exponent → low weight,
    making paths through shared nodes cheap.  A completely free node has
    exponent n (number of source nodes) → weight D^n, making it expensive.
    This naturally guides Dijkstra toward compact re-use of already-placed
    nodes.

    When `use_vertex_weights` is False the original flat `overlap_penalty`
    scheme is used instead.
    """
    # Get already placed neighbours to x
    placed_neighbours = [y for y in adjlist[x] if phi[y]]

    # Find any other nodes than x in y-models
    occupied: set[G] = set().union(*(set(model) for y, model in phi.items() if y != x))

    # If there is no placed neighbour, just pick any free node.
    if not placed_neighbours:
        free = [g for g in graph.nodes if g not in occupied]
        return [free[0]] if free else [next(iter(graph.nodes))]

    # Run Dijkstra's algorithm from multiple sources from v-models.
    if use_vertex_weights:
        # Count how many source-node models contain each target node.
        # (Exclude x itself since its model is being rebuilt.)
        inclusion_count: dict[G, int] = {}
        for y, model in phi.items():
            if y != x:
                for g in model:
                    inclusion_count[g] = inclusion_count.get(g, 0) + 1

        n = num_source_nodes or len(phi)

        def weight(_u: G, v: G, _data) -> float:
            # exponent = #{i : v ∉ φ(x_i)} = n - #{i : v ∈ φ(x_i)}
            exponent = n - inclusion_count.get(v, 0)
            return 1.0 + float(target_diameter**exponent)
    else:

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


__all__ = [
    "Model",
    "build_model",
    "find_embedding",
    "is_valid_embedding",
]
