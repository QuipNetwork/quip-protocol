"""Matroid infrastructure subpackage."""

from .core import (
    GraphicMatroid,
    FlatLattice,
    Edge,
    enumerate_flats,
    enumerate_flats_with_hasse,
    is_cyclic_flat,
    enumerate_cyclic_flats,
    MAX_FLATS_FOR_LATTICE,
)
from .parallel_connection import (
    BivariateLaurentPoly,
    theorem6_parallel_connection,
    precompute_contractions,
    build_extended_cell_graph,
)
