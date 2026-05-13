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

from __future__ import annotations

from typing import Dict, List

from .cell_quotient_cycle import compute_cycle_dp
from .cell_quotient_grid import (
    _grid_cell_layout,
    compute_grid_dp_grouped,
    compute_grid_dp_streamed_kab,
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


def compute_cell_quotient_grid_dp_streamed(graph, table) -> "Optional[TuttePolynomial]":
    """Top-level engine entry: 2D-grid cell-quotient DP via v5 streaming.

    Detects whether `graph` decomposes hierarchically into K_{a,b}-style
    cells whose cell-quotient is a 2D grid (no shared-anchor interior
    cells) AND whose inter-cell couplers are M_k matchings. If so, calls
    `compute_grid_dp_streamed_kab`. Returns None on any precondition
    mismatch (caller falls through to other engine steps).

    Phase B Round 6 (May 2026): introduced to beat the engine's
    `kmatching_formula` baseline on Cm₂ (~55 s → ~36 s, 1.5×).
    Does NOT yet handle Cm₃: the existing Cm₃ partition has interior
    cells with shared horizontal+vertical anchors, which this dispatch
    rejects via precondition.
    """
    import networkx as nx

    from ..graph import Graph
    from ..graphs.covering import try_hierarchical_partition

    result = try_hierarchical_partition(graph, table)
    if result is None:
        return None
    _cell_entry, partition, inter_info = result
    if not partition or not inter_info.edges:
        return None

    cag = detect_cell_anchor_groups(partition, inter_info.edges)
    if cag is None:
        return None

    adj: Dict[int, set] = {i: set() for i in range(len(partition))}
    for (a, b) in inter_info.cell_adjacencies:
        adj[a].add(b)
        adj[b].add(a)
    grid = is_grid_topology(adj, len(partition))
    if grid is None:
        return None
    rows, cols = grid
    if rows < 2 or cols < 2:
        return None
    layout = _grid_cell_layout(len(partition), rows, cols, adj)
    if layout is None:
        return None
    grid_specs = extract_grid_specs(cag, layout)
    if grid_specs is None:
        return None

    # Precondition: no shared anchors on any cell (Cm₂-style; Cm₃ has shared).
    for row in grid_specs:
        for spec in row:
            if spec.has_shared_horizontal or spec.has_shared_vertical:
                return None

    # Build cell_template + cell_anchor_groups from cell 0.
    cell_nodes = sorted(partition[0])
    cell_index = {v: i for i, v in enumerate(cell_nodes)}
    nx_cell = nx.Graph()
    nx_cell.add_nodes_from(range(len(cell_nodes)))
    for u in cell_nodes:
        for v in graph.neighbors(u):
            if v in cell_index and cell_index[u] < cell_index[v]:
                nx_cell.add_edge(cell_index[u], cell_index[v])
    if not nx.is_bipartite(nx_cell):
        return None
    cell_template = Graph.from_networkx(nx_cell)
    cell_anchor_groups: Dict[int, List[int]] = {}
    for grp_id, vertices in cag.cell_groups[0]:
        cell_anchor_groups[grp_id] = [cell_index[v] for v in vertices]

    # Verify junction templates are M_k matchings (one edge per anchor pair).
    # Use a generic M_k built off the cell anchor group sizes. Infer k_horiz
    # from a horizontal junction (any cell-pair with adjacent columns) and
    # k_vert from a vertical junction (adjacent rows).
    horiz_pair = None
    vert_pair = None
    for r in range(rows):
        for c in range(cols - 1):
            horiz_pair = (layout[r][c], layout[r][c + 1])
            break
        if horiz_pair is not None:
            break
    for r in range(rows - 1):
        for c in range(cols):
            vert_pair = (layout[r][c], layout[r + 1][c])
            break
        if vert_pair is not None:
            break
    if horiz_pair is None or vert_pair is None:
        return None

    def _junction_edge_count(cell_a: int, cell_b: int) -> int:
        cells_a = set(partition[cell_a])
        cells_b = set(partition[cell_b])
        return sum(
            1 for (u, v) in inter_info.edges
            if (u in cells_a and v in cells_b) or (u in cells_b and v in cells_a)
        )

    k_horiz = _junction_edge_count(*horiz_pair)
    k_vert = _junction_edge_count(*vert_pair)
    if k_horiz < 1 or k_vert < 1:
        return None

    # Canonical M_k template: vertices 0..2k-1, edges (i, k+i).
    horiz_junction = Graph(list(range(2 * k_horiz)),
                           [(i, k_horiz + i) for i in range(k_horiz)])
    horiz_junction_A = list(range(k_horiz))
    horiz_junction_B = list(range(k_horiz, 2 * k_horiz))
    vert_junction = Graph(list(range(2 * k_vert)),
                          [(i, k_vert + i) for i in range(k_vert)])
    vert_junction_A = list(range(k_vert))
    vert_junction_B = list(range(k_vert, 2 * k_vert))

    return compute_grid_dp_streamed_kab(
        cell_template=cell_template,
        cell_anchor_groups=cell_anchor_groups,
        horiz_junction_template=horiz_junction,
        horiz_junction_anchors_A=horiz_junction_A,
        horiz_junction_anchors_B=horiz_junction_B,
        vert_junction_template=vert_junction,
        vert_junction_anchors_A=vert_junction_A,
        vert_junction_anchors_B=vert_junction_B,
        grid_specs=grid_specs,
    )


def _build_cell_tree_spec_from_graph(
    graph, table, require_tree: bool = True,
):
    """Shared spec construction for tree DP + hybrid DP.

    Returns ``(spec, junctions, cell_topology)`` or ``None`` on failure.

    When ``require_tree=True``: returns None if cell-topology is not a
    tree (so the caller can fall through to a different dispatch).

    When ``require_tree=False``: also accepts cyclic cell-topologies; the
    caller (hybrid module) handles cycle-close.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher

    from ..graph import Graph
    from ..graphs.covering import (
        try_hierarchical_partition, detect_kmatching_topology,
    )
    from .cell_quotient_tree import CellTreeSpec

    result = try_hierarchical_partition(graph, table)
    if result is None:
        return None
    cell_entry, partition, inter_info = result
    if not partition or not inter_info.edges:
        return None

    junctions = detect_kmatching_topology(
        graph, partition, list(inter_info.edges),
    )
    if junctions is None or not junctions:
        return None

    cell_topology = nx.Graph()
    cell_topology.add_nodes_from(range(len(partition)))
    for j in junctions:
        cell_topology.add_edge(j.cell_i, j.cell_j)
    if require_tree and not nx.is_tree(cell_topology):
        return None

    cell0_nodes = sorted(partition[0])
    relabel0 = {v: i for i, v in enumerate(cell0_nodes)}
    cell0_edges = [
        (relabel0[u], relabel0[v]) for u, v in graph.edges
        if u in relabel0 and v in relabel0
    ]
    cell_template = Graph(list(range(len(cell0_nodes))), cell0_edges)
    cell_template_nx = nx.Graph()
    cell_template_nx.add_nodes_from(range(len(cell0_nodes)))
    cell_template_nx.add_edges_from(cell0_edges)

    isos: List[Dict[int, int]] = []
    for i, cell_nodes in enumerate(partition):
        if i == 0:
            isos.append(relabel0)
            continue
        cell_nx = nx.Graph()
        cell_nx.add_nodes_from(sorted(cell_nodes))
        for u, v in graph.edges:
            if u in cell_nodes and v in cell_nodes:
                cell_nx.add_edge(u, v)
        matcher = GraphMatcher(cell_nx, cell_template_nx)
        if not matcher.is_isomorphic():
            return None
        isos.append(matcher.mapping)

    cell_anchor_groups: Dict[int, Dict[int, List[int]]] = {
        i: {} for i in range(len(partition))
    }
    junction_k = junctions[0].k
    for j in junctions:
        if j.k != junction_k:
            return None
        anchors_i_local = [isos[j.cell_i][a] for a in j.anchors_i]
        anchors_j_local = [isos[j.cell_j][a] for a in j.anchors_j]
        if j.cell_j in cell_anchor_groups[j.cell_i]:
            return None
        cell_anchor_groups[j.cell_i][j.cell_j] = anchors_i_local
        cell_anchor_groups[j.cell_j][j.cell_i] = anchors_j_local

    junction_template = Graph(
        list(range(2 * junction_k)),
        [(i, i + junction_k) for i in range(junction_k)],
    )
    junction_anchors_A = list(range(junction_k))
    junction_anchors_B = list(range(junction_k, 2 * junction_k))

    leaf_candidates = [n for n, d in cell_topology.degree() if d == 1]
    root = leaf_candidates[0] if leaf_candidates else 0

    spec = CellTreeSpec(
        cell_template=cell_template,
        junction_template=junction_template,
        cell_tree=cell_topology,
        cell_anchor_groups=cell_anchor_groups,
        junction_anchors_A=junction_anchors_A,
        junction_anchors_B=junction_anchors_B,
        root=root,
    )
    return spec, junctions, cell_topology, isos, partition


def compute_cell_quotient_tree_dp(graph, table) -> "Optional[TuttePolynomial]":
    """Top-level engine entry: tree-topology cell-quotient DP.

    Detects whether `graph` decomposes hierarchically into cells whose
    cell-quotient is a TREE (n cells, n-1 junctions, no cycles).
    Builds a `CellTreeSpec` from the partition + k-matching junctions
    and dispatches to `compute_tree_dp_recursive` with per-cell
    compression enabled.

    Returns None if:
    - hierarchical partition fails,
    - junctions aren't k-matchings,
    - cell-topology has a cycle (use cycle DP for that),
    - cells aren't isomorphic to a single template,
    - building the spec fails for any other reason.

    Generic over cell template + junction k. The combined-aut path
    (`tutte/research/data/combined_aut_findings.md`) handles
    keep_shared / fully-consumed cases that the original per-cell
    fallback couldn't.
    """
    from .cell_quotient_tree import compute_tree_dp_recursive

    built = _build_cell_tree_spec_from_graph(graph, table, require_tree=True)
    if built is None:
        return None
    spec, _junctions, _cell_topology, _isos, _partition = built
    try:
        return compute_tree_dp_recursive(spec, enable_per_cell_compression=True)
    except Exception:
        return None


__all__ = [
    "compute_cycle_dp",
    "is_grid_topology",
    "compute_cell_quotient_cycle_dp",
    "compute_cell_quotient_tree_dp",
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
