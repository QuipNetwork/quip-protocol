"""
A graph allocator that sends some arbitrary Ising model
into a known architecture as a graph.

The whole allocator uses minor-embedding approach to
perform proper allocations.
"""

import random as rng
from collections import defaultdict
from typing import Callable, Dict, FrozenSet, List, Optional

import networkx as nx

type Model[G, H] = Dict[H, FrozenSet[G]]


def find_embedding[G, H](
    source: nx.Graph[H],
    target: nx.Graph[G],
    /,
    *,
    rng_factory: Callable[[], rng.Random] = rng.Random,
    tries: int = 30,
    refinment_constant: int = 20,
    overlap_penalty: float = 2.0,
) -> Optional[Model[G, H]]:
    """
    Finds a graph embedding of `source` as a minor of `target`.

    # Parameters
    `source: nx.Graph[H]`
        The graph to be embedded into `target`.
    `target: nx.Graph[G]`
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
        phi = {x: [] for x in order}
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
            counters = defaultdict(int)
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
        phi_image = sum((list(s) for s in phi_set.values()), [])
        if len(phi_image) != len(set(phi_image)):  # Models should not overlap
            continue
        if any(not phi_set[x] for x in src_nodes):  # V-models should be non-empty
            continue
        if any(
            len(phi_set[x]) > 1 and not nx.is_connected(target.subgraph(phi_set[x]))
            for x in src_nodes
        ):  # V-model should generate connected subgraph
            continue
        if any(
            not any(target.has_edge(gu, gv) for gu in phi_set[u] for gv in phi_set[v])
            for u, v in source.edges
        ):  # Source graph edges should be covered
            continue
        return phi_set
    return None


def build_model[G, H](
    x: H,
    adjlist: Dict[H, List[H]],
    phi: Dict[H, List[G]],
    target: nx.Graph[G],
    overlap_penalty: float,
) -> Optional[List[G]]:
    # Get already placed neighbours to x
    placed_neighbours = [y for y in adjlist[x] if phi[y]]

    # Find any other nodes than x in y-models
    occupied = set.union(set(), *(set(model) for y, model in phi.items() if y != x))

    # If there is no placed neighbour, just pick any free node.
    if not placed_neighbours:
        free = [g for g in target.nodes if g not in occupied]
        return [free[0]] if free else [next(iter(target.nodes))]

    # Run Dijkstra's algorithm from multiple sources from v-models.
    def weight(_u, v, _data):
        return 1.0 + overlap_penalty * (1.0 if v in occupied else 0.0)

    dist_from = {}
    pred_from = {}
    for y in placed_neighbours:
        dist_from[y], pred_from[y] = nx.multi_source_dijkstra(
            target, sources=phi[y], weight=weight
        )

    # Choose the best root.
    #
    # Take `free` node from the target graph by minimising sum of
    # distances to all neighbour models.
    best_root, best_cost = None, float("+inf")
    for g in target.nodes:
        if g in occupied:
            continue
        cost = sum(dist_from[y].get(g, float("+inf")) for y in placed_neighbours)
        if cost < best_cost:
            best_root, best_cost = g, cost

    if best_root is None or best_cost == float("+inf"):
        return None

    # Grow Steiner tree: trace paths from root toward each neighbour,
    # stopping at (but not including) phi[y] boundary nodes.
    model = {best_root}
    for y in placed_neighbours:
        phi_y_set = frozenset(phi[y])
        path = path_stopping_at(pred_from[y], best_root, phi_y_set)
        model.update(path)

    # Keep only the component containing root (paths should be connected,
    # but guard against edge cases).
    sub = target.subgraph(model)
    if len(model) > 1 and not nx.is_connected(sub):
        model = set(nx.node_connected_component(sub, best_root))

    # Final check: model must have at least one neighbour in G for each phi[y]
    # (otherwise the path didn't reach close enough — pick a node adjacent to phi[y])
    model_set = set(model)
    for y in placed_neighbours:
        phi_y_set = frozenset(phi[y])
        if not any(target.has_edge(m, g) for m in model_set for g in phi_y_set):
            # Fallback: add the closest free node adjacent to phi[y]
            for gy in phi[y]:
                for nb in target.neighbors(gy):
                    if nb not in occupied:
                        model_set.add(nb)
                        break
                else:
                    continue
                break

    return list(model_set)


def path_stopping_at(paths, target, stop_set):
    """
    From `paths[target]`, extract the tail starting at `target`
    going backward toward the source,
    stopping just before the first node in `stop_set`.

    This gives the sub-path that belongs to the new model: it starts at
    `target` (the root) and ends at the G-node immediately adjacent to
    the phi[y] boundary — without including any phi[y] node.
    """
    full_path = paths.get(target, [target])
    result = []
    for node in reversed(full_path):
        if node in stop_set:
            break
        result.append(node)
    return result


__all__ = [
    "Model",
    "build_model",
    "find_embedding",
]
