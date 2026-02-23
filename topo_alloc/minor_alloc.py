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
) -> Model[G, H] | None:
    """
    Finds a graph embedding of `source` as a minor of `target`.

    # Parameters
    `source: nx.Graph[G]`
        The graph to be embedded into `target`.
    `target: nx.Graph[H]`
        The graph that reassembles the hardware topology.
    `rng_factory: Callable[[], rng.Random]`
        The pseudo-random number generator factory. By default
        returns `rng.Random()`.
    `tries: int`
        The number of retries when the heuristic algorithm fails
        to find non-overlapping mapping.
    `refinment_constant: int`
        The constant `k` which sets the number of refinment iterations
        to `k * |V(H)|`.
    `overlap_penalty: float`
        The weight given to an edge that leads towards a node belonging
        to different vertex model.

    # Returns
    A `Model[G, H]` which is a dictionary from `H` nodes into
    sets of `G` nodes mapped. Returns `None` if after the number of `tries`
    there is no non-overlapping map.
    """
    refine_rounds = refinment_constant * len(source.nodes)
    rng = rng_factory()
    src_nodes = list(source.nodes)
    src_adj = {h: list(source.neighbors(h)) for h in src_nodes}

    for _ in range(tries):
        # --------------------------------
        # Stage 1 - initialize model with greedy placement in a random vertex order
        # --------------------------------
        order = list(src_nodes)
        phi: dict[H, list[G]] = {x: [] for x in order}
        rng.shuffle(order)
        for x in order:
            model = build_model(x, src_adj, phi, target, overlap_penalty)
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
                model = build_model(x, src_adj, phi, target, overlap_penalty)
                if model is not None:
                    phi[x] = model

        # --------------------------------
        # Stage 3 - validate the solution
        # --------------------------------
        phi_set = {x: frozenset(phi[x]) for x in src_nodes}
        if is_valid_embedding(source, target, phi_set):
            return phi_set
    return None


def build_model[G, H](
    x: H,
    adjlist: dict[H, list[H]],
    phi: dict[H, list[G]],
    graph: nx.Graph[G],
    overlap_penalty: float,
) -> list[G] | None:
    # Get already placed neighbours to x
    placed_neighbours = [y for y in adjlist[x] if phi[y]]

    # Find any other nodes than x in y-models
    occupied: set[G] = set().union(*(set(model) for y, model in phi.items() if y != x))

    # If there is no placed neighbour, just pick any free node.
    if not placed_neighbours:
        free = [g for g in graph.nodes if g not in occupied]
        return [free[0]] if free else [next(iter(graph.nodes))]

    # Run Dijkstra's algorithm from multiple sources from v-models.
    def weight(_u: G, v: G, _data) -> float:
        return 1.0 + overlap_penalty * (1.0 if v in occupied else 0.0)

    dist_from = {}
    path_from = {}
    for y in placed_neighbours:
        dist_from[y], path_from[y] = nx.multi_source_dijkstra(
            graph, sources=phi[y], weight=weight
        )

    # Choose the best root.
    #
    # Take `free` node from the target graph by minimising sum of
    # distances to all neighbour models.
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


def is_valid_embedding[G, H](source: nx.Graph[H], target: nx.Graph[G], phi: dict[H, frozenset[G]] | None) -> bool:
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
