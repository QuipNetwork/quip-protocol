"""Dynamic-programming methods for Tutte polynomial synthesis.

Houses DP-based syntheses that supplement the engine's general pipeline:
- Cell-quotient cycle DP (Phase 18.E.3.e/g): closed-form polynomial via
  rooted-Tutte composition for graphs with cell-quotient cycle topology.
- Cell-quotient grid DP (Phase 18.E.3.h, planned): extension to grid
  topologies (Cm3 and beyond).

All methods are generic over cell template + junction connectivity.
The directory consolidates DP methods that previously lived in separate
ad-hoc scripts; future moves of treewidth_dp, cotree_dp, etc. can land
here too.
"""

from .cell_quotient_cycle import compute_cycle_dp
from .cell_quotient_grid import (
    compute_grid_dp_grouped,
    is_grid_topology,
)
from .cell_quotient_path import compute_path_dp, compute_path_dp_grouped
from .cell_anchor_adapter import (
    CellAnchorGroups,
    CellGridSpec,
    CellRowSpec,
    detect_cell_anchor_groups,
    extract_grid_specs,
    extract_path_specs,
    normalize_cell_anchors_for_cycle,
)


def compute_cell_quotient_cycle_dp(graph, table) -> "Optional[TuttePolynomial]":
    """Top-level engine entry: cycle-topology cell-quotient DP.

    Detects whether `graph` decomposes hierarchically into cells whose
    cell-quotient is a SIMPLE CYCLE. If so, computes T(graph) via
    rooted-Tutte composition. Returns None if topology isn't a cycle
    or normalization fails.

    Generic dispatch — works for any cell template (e.g., K_{4,4}) and
    any junction connectivity (M_k matchings, K_{a,b} bipartite).
    """
    from ..graphs.covering import try_hierarchical_partition

    result = try_hierarchical_partition(graph, table)
    if result is None:
        return None
    cell_entry, partition, inter_info = result

    # Build cell template from the actual cell 0 induced subgraph
    # (cell_entry.graph may be None in the rainbow table).
    from ..graph import Graph
    cell0_nodes = sorted(partition[0])
    relabel = {v: i for i, v in enumerate(cell0_nodes)}
    cell0_edges = [
        (relabel[u], relabel[v]) for u, v in graph.edges
        if u in relabel and v in relabel
    ]
    cell_template = Graph(list(range(len(cell0_nodes))), cell0_edges)

    inter_edges = list(inter_info.edges)
    norm = normalize_cell_anchors_for_cycle(
        graph, partition, inter_edges, cell_template,
    )
    if norm is None:
        return None
    relabeled, cells_canonical, lefts, rights, junction_edges, cycle_order = norm

    if not junction_edges:
        return None
    first_junction = junction_edges[0]
    if not first_junction:
        return None

    a = len(lefts[0])
    b = len(rights[0])
    n_cells = len(cells_canonical)

    # Extract canonical cell template from the RELABELED graph's cell 0
    # (its bipartition may differ from cell_entry.graph's labeling).
    from ..graph import Graph
    cell0_set = set(cells_canonical[0])
    base0 = cells_canonical[0][0]
    cell0_edges_local = []
    for u, v in relabeled.edges:
        if u in cell0_set and v in cell0_set:
            cell0_edges_local.append((min(u, v) - base0, max(u, v) - base0))
    cell_template_canonical = Graph(list(range(a + b)), cell0_edges_local)

    # Cell anchors in canonical local positions.
    cell_left_local = [v - base0 for v in lefts[0]]
    cell_right_local = [v - base0 for v in rights[0]]

    # Build canonical junction template from junction 0's edges.
    base1 = cells_canonical[1][0]
    matching: dict = {}
    for u, v in first_junction:
        if u in cells_canonical[0]:
            r_local = u - base0  # in cell_right_local positions
            l_local = v - base1  # in cell_left_local positions
        else:
            r_local = v - base0
            l_local = u - base1
        # r_local should be in cell_right_local; map to its index.
        if r_local in cell_right_local and l_local in cell_left_local:
            matching[cell_right_local.index(r_local)] = cell_left_local.index(l_local)

    if len(matching) != b:
        return None  # junction structure didn't fit

    # Junction template uses canonical anchor positions [0..b-1] | [b..a+b-1].
    m_edges = [(i, b + matching[i]) for i in range(b)]
    junction_template = Graph(list(range(a + b)), m_edges)
    junction_anchors_A = list(range(b))
    junction_anchors_B = list(range(b, a + b))

    # Cell template uses [0..a-1] | [a..a+b-1] for left/right (canonical).
    # Need to translate cell_template_canonical's anchors to this scheme.
    # cell_left_local and cell_right_local are derived from the relabeled
    # graph; these ARE the anchors we want compute_cycle_dp to use.
    poly, _stats = compute_cycle_dp(
        cell_template_canonical, cell_left_local, cell_right_local,
        junction_template, junction_anchors_A, junction_anchors_B,
        n_cells=n_cells,
    )
    return poly


__all__ = [
    "compute_cycle_dp",
    "is_grid_topology",
    "compute_cell_quotient_cycle_dp",
    "normalize_cell_anchors_for_cycle",
    "detect_cell_anchor_groups",
    "CellAnchorGroups",
    "CellRowSpec",
    "CellGridSpec",
    "extract_path_specs",
    "extract_grid_specs",
    "compute_path_dp",
    "compute_path_dp_grouped",
    "compute_grid_dp_grouped",
]
