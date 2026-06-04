"""Shared helper: coerce topology node/edge lists to plain Python ints."""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def coerce_int_nodes_edges(
    nodelist: list,
    edgelist: list,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Validate and coerce topology nodes and edges to int.

    Both the CPU SA sampler and the Modal GPU sampler receive node/edge lists
    from a topology graph whose element type is ``Hashable``.  This helper
    enforces that every element is (or can be coerced to) a plain ``int``,
    raising ``ValueError`` with a consistent message if not.

    Args:
        nodelist: Sequence of node indices, expected to be ``int``.
        edgelist: Sequence of ``(i, j)`` edge tuples, expected to be pairs of
            ``int``.

    Returns:
        A tuple ``(nodes, edges)`` where every element has been converted to
        ``int``.

    Raises:
        ValueError: If any node is not an ``int``, or any edge is not a
            length-2 tuple of ``int``.
    """
    nodes: List[int] = []
    for node in nodelist:
        if not isinstance(node, int):
            raise ValueError(f"Expected node index to be int, got {type(node)}")
        nodes.append(int(node))

    edges: List[Tuple[int, int]] = []
    for edge in edgelist:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError(f"Expected edge to be tuple of length 2, got {edge}")
        if not isinstance(edge[0], int) or not isinstance(edge[1], int):
            raise ValueError(
                f"Expected edge indices to be int, got {type(edge[0])} and {type(edge[1])}"
            )
        edges.append((int(edge[0]), int(edge[1])))

    return nodes, edges
