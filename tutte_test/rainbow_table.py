"""Rainbow Table for Tutte Polynomial Lookup.

This module provides:
1. RainbowTable class for loading/saving the polynomial lookup table
2. Minor indexing for finding graph minors
3. Lookup by canonical key or graph name

The rainbow table stores known Tutte polynomials indexed by their
canonical graph key (SHA256 of graph6 encoding). This enables:
- O(1) lookup of known polynomials
- Finding all known minors of a graph
- Building graphs from smaller known components
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from math import gcd as math_gcd
from typing import Dict, List, Optional, Set, Tuple

from .polynomial import TuttePolynomial, encode_varuint
from .graph import Graph


# =============================================================================
# MINOR ENTRY DATA CLASS
# =============================================================================

@dataclass
class MinorEntry:
    """An entry in the rainbow table."""
    name: str
    polynomial: TuttePolynomial
    node_count: int
    edge_count: int
    canonical_key: str
    spanning_trees: int
    num_terms: int
    graph: Optional['Graph'] = None  # Stored graph for tiling reconstruction

    @property
    def complexity(self) -> int:
        """Complexity score for sorting (higher = more complex)."""
        # Prioritize by edge count, then node count, then spanning trees
        return self.edge_count * 1000 + self.node_count * 100 + self.spanning_trees

    def __hash__(self) -> int:
        return hash(self.canonical_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MinorEntry):
            return NotImplemented
        return self.canonical_key == other.canonical_key


# =============================================================================
# GCD-BASED MINOR INDEX
# =============================================================================

@dataclass
class GCDMinorIndex:
    """Index of GCD-based relationships between polynomial entries.

    This structure enables efficient lookup of polynomials that share
    common factors, which indicates structural relationships between
    the corresponding graphs.

    Attributes:
        shared_factor_graph: Maps graph name -> set of names sharing a factor
        factor_to_graphs: Maps factor polynomial key -> set of graph names
        gcd_cache: Cache of computed GCDs between pairs
    """
    shared_factor_graph: Dict[str, Set[str]] = field(default_factory=dict)
    factor_to_graphs: Dict[str, Set[str]] = field(default_factory=dict)
    gcd_cache: Dict[Tuple[str, str], TuttePolynomial] = field(default_factory=dict)

    def get_gcd(self, name1: str, name2: str) -> Optional[TuttePolynomial]:
        """Get cached GCD between two entries."""
        key = (name1, name2) if name1 < name2 else (name2, name1)
        return self.gcd_cache.get(key)

    def graphs_sharing_factor_with(self, name: str) -> Set[str]:
        """Get all graphs sharing a polynomial factor with the given graph."""
        return self.shared_factor_graph.get(name, set())

    def graphs_with_factor(self, factor: TuttePolynomial) -> Set[str]:
        """Get all graphs containing the given polynomial as a factor."""
        factor_key = str(factor)
        return self.factor_to_graphs.get(factor_key, set())

    def largest_shared_factor(self, name: str) -> Optional[Tuple[str, TuttePolynomial]]:
        """Find the entry sharing the largest factor with the given entry.

        Returns (related_name, gcd) or None if no shared factors.
        """
        related = self.shared_factor_graph.get(name, set())
        if not related:
            return None

        best = None
        best_degree = -1

        for other_name in related:
            gcd = self.get_gcd(name, other_name)
            if gcd is not None:
                degree = gcd.total_degree()
                if degree > best_degree:
                    best = (other_name, gcd)
                    best_degree = degree

        return best


# =============================================================================
# RAINBOW TABLE CLASS
# =============================================================================

@dataclass
class RainbowTable:
    """Lookup table for Tutte polynomials with minor indexing."""

    entries: Dict[str, MinorEntry] = field(default_factory=dict)  # canonical_key -> entry
    name_index: Dict[str, str] = field(default_factory=dict)      # name -> canonical_key

    # Minor relationships: maps name -> list of names that are minors
    minor_relationships: Dict[str, List[str]] = field(default_factory=dict)

    # Sorted by complexity for largest-first lookup
    _sorted_by_complexity: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: str) -> 'RainbowTable':
        """Load rainbow table from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        table = cls()

        # Load entries
        graphs = data.get('graphs', {})
        for key, entry_data in graphs.items():
            # Parse coefficients from string keys "i,j" to tuple keys
            coeffs = {}
            for coeff_key, coeff_val in entry_data.get('coefficients', {}).items():
                i, j = map(int, coeff_key.split(','))
                coeffs[(i, j)] = coeff_val

            polynomial = TuttePolynomial.from_coefficients(coeffs)

            entry = MinorEntry(
                name=entry_data.get('name', 'unknown'),
                polynomial=polynomial,
                node_count=entry_data.get('nodes', 0),
                edge_count=entry_data.get('edges', 0),
                canonical_key=key,
                spanning_trees=entry_data.get('spanning_trees', 0),
                num_terms=entry_data.get('num_terms', len(coeffs)),
            )

            table.entries[key] = entry
            table.name_index[entry.name] = key

        # Load minor relationships if present
        table.minor_relationships = data.get('minor_relationships', {})

        # Sort by complexity
        table._sort_by_complexity()

        return table

    def save(self, path: str) -> None:
        """Save rainbow table to JSON file."""
        graphs = {}
        for key, entry in self.entries.items():
            # Convert coefficients to string keys
            coeffs = {}
            for (i, j), c in entry.polynomial.to_coefficients().items():
                coeffs[f"{i},{j}"] = c

            graphs[key] = {
                'name': entry.name,
                'nodes': entry.node_count,
                'edges': entry.edge_count,
                'spanning_trees': entry.spanning_trees,
                'x_degree': entry.polynomial.x_degree(),
                'y_degree': entry.polynomial.y_degree(),
                'num_terms': entry.num_terms,
                'polynomial_str': str(entry.polynomial),
                'coefficients': coeffs,
            }

        data = {
            'description': 'Tutte Polynomial Rainbow Table',
            'total_entries': len(self.entries),
            'note': 'Coefficients stored as "i,j": coefficient for x^i * y^j',
            'graphs': graphs,
            'minor_relationships': self.minor_relationships,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _sort_by_complexity(self) -> None:
        """Sort entries by complexity (descending) for largest-first lookup."""
        self._sorted_by_complexity = sorted(
            self.entries.keys(),
            key=lambda k: self.entries[k].complexity,
            reverse=True
        )

    def lookup(self, graph: Graph) -> Optional[TuttePolynomial]:
        """Look up polynomial for a graph by its canonical key."""
        key = graph.canonical_key()
        entry = self.entries.get(key)
        return entry.polynomial if entry else None

    def lookup_by_name(self, name: str) -> Optional[TuttePolynomial]:
        """Look up polynomial by graph name."""
        key = self.name_index.get(name)
        if key is None:
            return None
        entry = self.entries.get(key)
        return entry.polynomial if entry else None

    def get_entry(self, name: str) -> Optional[MinorEntry]:
        """Get full entry by name."""
        key = self.name_index.get(name)
        if key is None:
            return None
        return self.entries.get(key)

    def get_entry_by_key(self, key: str) -> Optional[MinorEntry]:
        """Get full entry by canonical key."""
        return self.entries.get(key)

    def find_minors_of(self, graph: Graph) -> List[MinorEntry]:
        """Find all table entries that are minors of the given graph.

        A polynomial P1 is considered a "minor" of P2 if:
        - P2 - P1 has all non-negative coefficients
        - P1's node count <= graph's node count
        - P1's edge count <= graph's edge count

        Returns entries sorted by complexity (descending).
        """
        target_key = graph.canonical_key()
        target_nodes = graph.node_count()
        target_edges = graph.edge_count()

        # Check if graph is in table and has stored relationships
        if target_key in self.entries:
            entry = self.entries[target_key]
            stored_minors = self.minor_relationships.get(entry.name, [])
            if stored_minors:
                # Use stored relationships
                minors = [entry]
                for minor_name in stored_minors:
                    minor_entry = self.get_entry(minor_name)
                    if minor_entry:
                        minors.append(minor_entry)
                return sorted(minors, key=lambda e: e.complexity, reverse=True)

        # Find minors by size comparison (graphs smaller than target)
        candidates = []
        for key in self._sorted_by_complexity:
            entry = self.entries[key]

            # Quick filters by size
            if entry.node_count > target_nodes:
                continue
            if entry.edge_count > target_edges:
                continue

            candidates.append(entry)

        return candidates

    def largest_minor_of(self, graph: Graph) -> Optional[MinorEntry]:
        """Return the largest minor by edge count.

        This is useful for the creation-expansion-join algorithm
        to find the best building block for a graph.
        """
        minors = self.find_minors_of(graph)
        if not minors:
            return None

        # Return largest by edge count (already sorted by complexity)
        for entry in minors:
            # Don't return the graph itself
            if entry.canonical_key != graph.canonical_key():
                return entry

        return None

    def add(self, graph: Graph, name: str, polynomial: TuttePolynomial) -> None:
        """Add a new entry to the table."""
        key = graph.canonical_key()

        entry = MinorEntry(
            name=name,
            polynomial=polynomial,
            node_count=graph.node_count(),
            edge_count=graph.edge_count(),
            canonical_key=key,
            spanning_trees=polynomial.num_spanning_trees(),
            num_terms=polynomial.num_terms(),
            graph=graph,
        )

        self.entries[key] = entry
        self.name_index[name] = key

        # Re-sort
        self._sort_by_complexity()

    def add_from_networkx(self, G, name: str, polynomial: TuttePolynomial) -> None:
        """Add entry from NetworkX graph."""
        graph = Graph.from_networkx(G)
        self.add(graph, name, polynomial)

    def find_by_polynomial(self, polynomial: TuttePolynomial) -> Optional[MinorEntry]:
        """Find entry with matching polynomial."""
        for entry in self.entries.values():
            if entry.polynomial == polynomial:
                return entry
        return None

    def find_factors(self, target: TuttePolynomial) -> List[Tuple[MinorEntry, TuttePolynomial]]:
        """Find table entries whose polynomial divides the target.

        Returns list of (entry, quotient) tuples where entry.polynomial * quotient = target.
        """
        results = []
        target_trees = target.num_spanning_trees()

        for entry in self.entries.values():
            # Quick filter by spanning trees
            if target_trees % entry.spanning_trees != 0:
                continue

            # Try to divide
            quotient = _try_divide(target, entry.polynomial)
            if quotient is not None:
                results.append((entry, quotient))

        return results

    def compute_minor_relationships(self) -> Dict[str, List[str]]:
        """Compute minor relationships between all entries.

        A polynomial P1 is a minor of P2 if P2 - P1 has all non-negative coefficients.
        """
        relationships = {}

        entries_list = list(self.entries.values())

        for entry in entries_list:
            minors = []
            for other in entries_list:
                if entry.canonical_key == other.canonical_key:
                    continue
                if other.node_count > entry.node_count:
                    continue
                if other.edge_count > entry.edge_count:
                    continue

                # Check polynomial relationship
                diff = entry.polynomial - other.polynomial
                coeffs = diff.to_coefficients()
                if all(c >= 0 for c in coeffs.values()):
                    minors.append(other.name)

            if minors:
                relationships[entry.name] = minors

        self.minor_relationships = relationships
        return relationships

    def compute_gcd_relationships(self) -> 'GCDMinorIndex':
        """Compute GCD-based relationships between all polynomial entries.

        Two polynomials are related if they share a non-trivial common factor
        (GCD ≠ 1). This is a symmetric relationship that captures structural
        similarities between graphs.

        Returns:
            GCDMinorIndex containing all relationships
        """
        from .factorization import polynomial_gcd, has_common_factor

        index = GCDMinorIndex()

        entries_list = list(self.entries.values())
        n = len(entries_list)

        # Compare all pairs
        for i in range(n):
            entry_i = entries_list[i]
            name_i = entry_i.name
            poly_i = entry_i.polynomial
            trees_i = entry_i.spanning_trees

            for j in range(i + 1, n):
                entry_j = entries_list[j]
                name_j = entry_j.name
                poly_j = entry_j.polynomial
                trees_j = entry_j.spanning_trees

                # Quick pre-filter: if spanning tree counts are coprime,
                # polynomials cannot share a factor
                if trees_i > 0 and trees_j > 0:
                    if math_gcd(trees_i, trees_j) == 1:
                        continue

                # Check if they share a factor
                if has_common_factor(poly_i, poly_j):
                    # Compute and cache the GCD
                    gcd = polynomial_gcd(poly_i, poly_j)

                    # Add to shared factor graph
                    if name_i not in index.shared_factor_graph:
                        index.shared_factor_graph[name_i] = set()
                    if name_j not in index.shared_factor_graph:
                        index.shared_factor_graph[name_j] = set()

                    index.shared_factor_graph[name_i].add(name_j)
                    index.shared_factor_graph[name_j].add(name_i)

                    # Cache the GCD
                    pair_key = (name_i, name_j) if name_i < name_j else (name_j, name_i)
                    index.gcd_cache[pair_key] = gcd

                    # Index by factor polynomial (use string repr as key)
                    factor_key = str(gcd)
                    if factor_key not in index.factor_to_graphs:
                        index.factor_to_graphs[factor_key] = set()
                    index.factor_to_graphs[factor_key].add(name_i)
                    index.factor_to_graphs[factor_key].add(name_j)

        return index

    def find_gcd_related(self, name: str) -> List[Tuple[str, TuttePolynomial]]:
        """Find all entries sharing a polynomial factor with the given entry.

        Args:
            name: Name of entry to find relations for

        Returns:
            List of (related_name, gcd) tuples
        """
        entry = self.get_entry(name)
        if entry is None:
            return []

        from .factorization import polynomial_gcd, has_common_factor

        results = []
        target_poly = entry.polynomial
        target_trees = entry.spanning_trees

        for other_entry in self.entries.values():
            if other_entry.name == name:
                continue

            other_trees = other_entry.spanning_trees

            # Quick pre-filter
            if target_trees > 0 and other_trees > 0:
                if math_gcd(target_trees, other_trees) == 1:
                    continue

            if has_common_factor(target_poly, other_entry.polynomial):
                gcd = polynomial_gcd(target_poly, other_entry.polynomial)
                results.append((other_entry.name, gcd))

        # Sort by GCD complexity (prefer larger shared factors)
        results.sort(key=lambda x: x[1].total_degree(), reverse=True)

        return results

    def find_factors_of(self, target: TuttePolynomial) -> List[Tuple[MinorEntry, TuttePolynomial]]:
        """Find table entries whose polynomial divides the target (with quotient).

        This extends find_factors() with proper polynomial division.

        Args:
            target: Target polynomial to factor

        Returns:
            List of (entry, quotient) tuples where entry.polynomial * quotient = target
        """
        from .k_join import polynomial_divmod

        results = []
        target_trees = target.num_spanning_trees()

        for entry in self.entries.values():
            entry_trees = entry.spanning_trees

            # Quick filter by spanning tree divisibility
            if entry_trees > 0 and target_trees % entry_trees != 0:
                continue

            # Try polynomial division
            try:
                quotient, remainder = polynomial_divmod(target, entry.polynomial)

                if remainder.is_zero() and not quotient.is_zero():
                    results.append((entry, quotient))
            except (ValueError, ZeroDivisionError):
                continue

        # Sort by factor size (prefer larger factors)
        results.sort(key=lambda x: x[0].complexity, reverse=True)

        return results

    def _entry_to_polynomial(self, entry_data: dict) -> TuttePolynomial:
        """Convert entry data dict to polynomial (for compatibility)."""
        coeffs = {}
        for coeff_key, coeff_val in entry_data.get('coefficients', {}).items():
            if isinstance(coeff_key, str):
                i, j = map(int, coeff_key.split(','))
            else:
                i, j = coeff_key
            coeffs[(i, j)] = coeff_val
        return TuttePolynomial.from_coefficients(coeffs)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _try_divide(dividend: TuttePolynomial, divisor: TuttePolynomial) -> Optional[TuttePolynomial]:
    """Try polynomial division. Returns quotient if exact, None otherwise.

    Currently only handles monomial divisors for simplicity.
    """
    divisor_coeffs = divisor.to_coefficients()

    # Simple case: divisor is monomial
    if len(divisor_coeffs) == 1:
        (a, b), c = next(iter(divisor_coeffs.items()))
        if c != 1:
            return None  # Only handle coefficient 1

        quotient_coeffs = {}
        for (i, j), coeff in dividend.to_coefficients().items():
            if i >= a and j >= b:
                quotient_coeffs[(i - a, j - b)] = coeff
            else:
                return None  # Division not exact

        if not quotient_coeffs:
            return None

        return TuttePolynomial.from_coefficients(quotient_coeffs)

    # Check if divisor * something could equal dividend
    # by verifying degrees are compatible
    if divisor.x_degree() > dividend.x_degree():
        return None
    if divisor.y_degree() > dividend.y_degree():
        return None

    # For non-monomial divisors, we'd need actual polynomial division
    # which is more complex. Return None for now.
    return None


def load_default_table() -> RainbowTable:
    """Load the default rainbow table from the package directory."""
    table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
    if os.path.exists(table_path):
        return RainbowTable.load(table_path)
    return RainbowTable()


# =============================================================================
# BINARY ENCODING
# =============================================================================

def encode_rainbow_table_binary(table: RainbowTable) -> bytes:
    """Encode rainbow table to compact binary format.

    Format:
        Header:
            [magic: 4 bytes] = "RTBL"
            [version: 1 byte] = 1
            [num_entries: varuint]

        For each entry:
            [name_len: varuint]
            [name: bytes]
            [node_count: varuint]
            [edge_count: varuint]
            [spanning_trees: varuint]
            [polynomial_bytes_len: varuint]
            [polynomial_bytes: bytes]
    """
    result = bytearray()

    # Magic header
    result.extend(b"RTBL")
    result.append(1)  # version

    # Number of entries
    result.extend(encode_varuint(len(table.entries)))

    # Each entry
    for key, entry in table.entries.items():
        # Name
        name_bytes = entry.name.encode('utf-8')
        result.extend(encode_varuint(len(name_bytes)))
        result.extend(name_bytes)

        # Metadata
        result.extend(encode_varuint(entry.node_count))
        result.extend(encode_varuint(entry.edge_count))
        result.extend(encode_varuint(entry.spanning_trees))

        # Polynomial as bitstring
        poly_bytes = entry.polynomial.to_bytes()
        result.extend(encode_varuint(len(poly_bytes)))
        result.extend(poly_bytes)

    return bytes(result)


def save_binary_rainbow_table(table: RainbowTable, path: str) -> int:
    """Save rainbow table to binary format, return size in bytes."""
    data = encode_rainbow_table_binary(table)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def create_minimal_json(table: RainbowTable) -> dict:
    """Create minimal JSON representation (no redundant fields)."""
    graphs = {}
    for key, entry in table.entries.items():
        # Only store what's needed to reconstruct
        coeffs = {}
        for (i, j), c in entry.polynomial.to_coefficients().items():
            coeffs[f"{i},{j}"] = c

        graphs[key] = {
            'n': entry.name,
            'v': entry.node_count,
            'e': entry.edge_count,
            'c': coeffs
        }

    return {'g': graphs}


def analyze_binary_breakdown(table: RainbowTable) -> Dict[str, int]:
    """Analyze where bytes go in binary format. Returns size breakdown dict."""
    header_size = 5  # magic + version
    entry_count_size = len(encode_varuint(len(table.entries)))

    name_bytes = 0
    metadata_bytes = 0
    polynomial_bytes = 0

    for entry in table.entries.values():
        name_b = entry.name.encode('utf-8')
        name_bytes += len(encode_varuint(len(name_b))) + len(name_b)

        metadata_bytes += len(encode_varuint(entry.node_count))
        metadata_bytes += len(encode_varuint(entry.edge_count))
        metadata_bytes += len(encode_varuint(entry.spanning_trees))

        poly_b = entry.polynomial.to_bytes()
        polynomial_bytes += len(encode_varuint(len(poly_b))) + len(poly_b)

    total = header_size + entry_count_size + name_bytes + metadata_bytes + polynomial_bytes

    return {
        'header': header_size + entry_count_size,
        'names': name_bytes,
        'metadata': metadata_bytes,
        'polynomials': polynomial_bytes,
        'total': total,
    }


# =============================================================================
# SYMPY CONVERSION
# =============================================================================

def sympy_to_tutte(nx_poly) -> TuttePolynomial:
    """Convert networkx/sympy polynomial to TuttePolynomial."""
    coeffs = {}

    # Handle different sympy types
    if hasattr(nx_poly, 'as_dict'):
        poly_dict = nx_poly.as_dict()
    elif hasattr(nx_poly, 'as_poly'):
        poly_dict = nx_poly.as_poly().as_dict()
    else:
        # Try to convert to Poly first
        from sympy import Poly, symbols
        x, y = symbols('x y')
        try:
            poly = Poly(nx_poly, x, y)
            poly_dict = poly.as_dict()
        except Exception:
            # Single term or constant
            poly_dict = {(0, 0): int(nx_poly)}

    for monom, coeff in poly_dict.items():
        if len(monom) >= 2:
            coeffs[(monom[0], monom[1])] = int(coeff)
        elif len(monom) == 1:
            coeffs[(monom[0], 0)] = int(coeff)
        else:
            coeffs[(0, 0)] = int(coeff)

    return TuttePolynomial.from_coefficients(coeffs)


# =============================================================================
# BASIC TABLE BUILDER
# =============================================================================

def build_basic_table() -> RainbowTable:
    """Build rainbow table using closed-form formulas and self-synthesis.

    Phase 1: Seed with closed-form polynomials (no computation needed)
      - Complete graphs K_2 through K_5
      - Cycle graphs C_3 through C_12
      - Path graphs P_2 through P_10
      - Star graphs S_3 through S_8

    Phase 2: Use the synthesis engine with the seeded table to compute
      - Complete graphs K_6 through K_8
      - Wheel graphs W_4 through W_8
      - Petersen graph
      - Small grid graphs
    """
    import networkx as nx
    from .graph import (complete_graph, cycle_graph, path_graph,
                        wheel_graph, star_graph)

    table = RainbowTable()

    # === Phase 1: Closed-form polynomials ===

    # Complete graphs K_n: well-known polynomials
    k_polys = {
        2: {(1, 0): 1},  # x
        3: {(2, 0): 1, (1, 0): 1, (0, 1): 1},  # x^2 + x + y
        4: {(3, 0): 1, (2, 0): 3, (1, 1): 4, (1, 0): 2, (0, 1): 2, (0, 2): 3, (0, 3): 1},
        5: {(4, 0): 1, (3, 0): 6, (2, 1): 10, (2, 0): 11, (1, 1): 20, (1, 2): 15,
            (1, 3): 5, (1, 0): 6, (0, 1): 6, (0, 2): 15, (0, 3): 15, (0, 4): 10,
            (0, 5): 4, (0, 6): 1},
    }

    for n in range(2, 6):
        g = complete_graph(n)
        poly = TuttePolynomial.from_coefficients(k_polys[n])
        table.add(g, f"K_{n}", poly)

    # Cycle graphs C_n: T(C_n) = x^{n-1} + x^{n-2} + ... + x + y
    for n in range(3, 13):
        g = cycle_graph(n)
        coeffs = {(i, 0): 1 for i in range(1, n)}
        coeffs[(0, 1)] = 1
        poly = TuttePolynomial.from_coefficients(coeffs)
        table.add(g, f"C_{n}", poly)

    # Path graphs P_n: T(P_n) = x^{n-1}
    for n in range(2, 11):
        g = path_graph(n)
        poly = TuttePolynomial.x(n - 1)
        table.add(g, f"P_{n}", poly)

    # Star graphs S_n: T(S_n) = x^n (n bridges)
    for n in range(3, 9):
        g = star_graph(n)
        poly = TuttePolynomial.x(n)
        table.add(g, f"S_{n}", poly)

    # === Phase 2: Synthesize remaining graphs using the seeded table ===

    from .synthesis import SynthesisEngine
    engine = SynthesisEngine(table=table, verbose=False)

    # Complete graphs K_6, K_7, K_8
    for n in range(6, 9):
        g = complete_graph(n)
        result = engine.synthesize(g)
        table.add(g, f"K_{n}", result.polynomial)

    # Wheel graphs W_n (n spokes + hub)
    for n in range(4, 9):
        G_nx = nx.wheel_graph(n)
        g = Graph.from_networkx(G_nx)
        result = engine.synthesize(g)
        table.add(g, f"W_{n}", result.polynomial)

    # Petersen graph
    g_pet = Graph.from_networkx(nx.petersen_graph())
    result = engine.synthesize(g_pet)
    table.add(g_pet, "Petersen", result.polynomial)

    # Small grid graphs
    for rows in range(2, 4):
        for cols in range(2, 5):
            G_grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols))
            g_grid = Graph.from_networkx(G_grid)
            if G_grid.number_of_edges() <= 12:
                result = engine.synthesize(g_grid)
                table.add(g_grid, f"Grid_{rows}x{cols}", result.polynomial)

    return table
