"""Graph algorithms and operations subpackage."""

from .series_parallel import (
    SPNode,
    is_series_parallel,
    decompose_series_parallel,
    compute_sp_tutte,
    compute_sp_tutte_if_applicable,
    compute_sp_chi_coeffs,
    compute_contraction_chi,
)
from .k_join import (
    tutte_k,
    k_sum,
    k_sum_multiple,
    k_join_divide,
    polynomial_divmod,
    polynomial_divide,
    TileJoin,
    compute_cover_polynomial,
    divide_out_fringe,
)
from .minor import is_graph_minor
from .covering import (
    Tile,
    Cover,
    Fringe,
    InterCellInfo,
    find_subgraph_isomorphisms,
    find_edge_mapping,
    find_disjoint_cover,
    compute_fringe,
    compute_inter_tile_edges,
    analyze_tile_connections,
    find_cell_candidates,
    partition_into_cells,
    verify_cell_partition,
    analyze_inter_cell_edges,
    try_hierarchical_partition,
)
