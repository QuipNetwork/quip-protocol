"""Rainbow table lookup system subpackage."""

from .core import (
    MinorEntry,
    FlatLatticeData,
    GCDMinorIndex,
    RainbowTable,
    load_default_table,
)
from .binary import (
    encode_rainbow_table_binary,
    decode_rainbow_table_binary,
    load_binary_rainbow_table,
    save_binary_rainbow_table,
)
from .bootstrap import (
    build_basic_table,
    sympy_to_tutte,
)
# Re-export is_graph_minor here for backward compat (was in rainbow_table)
from ..graphs.minor import is_graph_minor
