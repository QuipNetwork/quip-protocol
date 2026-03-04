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

from ..graph import CellSignature, Graph, compute_signature
from ..polynomial import TuttePolynomial, encode_varuint, decode_varuint


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
    signature: Optional['CellSignature'] = None  # Cached signature for fast matching

    def get_signature(self) -> Optional['CellSignature']:
        """Get or compute the cell signature for this entry."""
        if self.signature is not None:
            return self.signature
        if self.graph is not None:
            return compute_signature(self.graph)
        return None

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

    # Minor relationships: maps canonical_key -> list of canonical_keys that are minors
    minor_relationships: Dict[str, List[str]] = field(default_factory=dict)

    # True after compute_minor_relationships() has run (comprehensive structural check).
    # When False, minor_relationships only contains synthesis-tracked minors.
    _structural_minors_computed: bool = field(default=False)

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
        table._structural_minors_computed = data.get('structural_minors_computed', False)

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
            'structural_minors_computed': self._structural_minors_computed,
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
        """Find all table entries that are structural minors of the given graph.

        If comprehensive structural minor relationships have been computed (via
        compute_minor_relationships()), uses them for O(1) lookup.
        Otherwise falls back to size-based filtering.

        Returns entries sorted by complexity (descending).
        """
        target_key = graph.canonical_key()

        # Use pre-computed structural relationships only if comprehensive
        # (synthesis-tracked minors are incomplete — they only include what
        # was used during synthesis, not all structural minors)
        if self._structural_minors_computed and target_key in self.minor_relationships:
            minor_keys = self.minor_relationships[target_key]
            entries = [self.entries[k] for k in minor_keys if k in self.entries]
            entries.sort(key=lambda e: e.complexity, reverse=True)
            return entries

        # Fallback: size-based filtering
        target_nodes = graph.node_count()
        target_edges = graph.edge_count()

        candidates = []
        for key in self._sorted_by_complexity:
            entry = self.entries[key]
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

    def add(self, graph: Graph, name: str, polynomial: TuttePolynomial,
            minors_used: Optional[Set[str]] = None) -> None:
        """Add a new entry to the table.

        Args:
            graph: The graph to add
            name: Human-readable name for the entry
            polynomial: Computed Tutte polynomial
            minors_used: Set of canonical keys of table entries used during
                         synthesis. Transitive minors are inherited automatically.
        """
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
            signature=compute_signature(graph),
        )

        self.entries[key] = entry
        self.name_index[name] = key

        # Build transitive minor closure (Merkle-tree style)
        if minors_used:
            all_minors = set()
            for minor_key in minors_used:
                if minor_key == key:
                    continue  # Don't list self
                if minor_key in self.entries:
                    all_minors.add(minor_key)
                    # Inherit transitive minors from each direct minor
                    if minor_key in self.minor_relationships:
                        all_minors |= set(self.minor_relationships[minor_key])
            if all_minors:
                self.minor_relationships[key] = sorted(all_minors)

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

    def compute_minor_relationships(self, verify: bool = True,
                                     max_contractions: int = 5) -> Dict[str, List[str]]:
        """Compute minor relationships between all entries using coefficient domination.

        Uses the Tutte polynomial monotonicity property: if H is a graph minor
        of G, then every coefficient of T(H) is <= the corresponding coefficient
        of T(G). This is a known theorem (no false negatives).

        The converse is not always true (~12% false positive rate on small graphs),
        but the false positives are well-characterized (mainly Tutte-equivalent
        trees like P_n vs S_{n-1}) and harmless for synthesis purposes.

        When ``verify=True``, suspicious candidate pairs are checked with
        ``is_graph_minor()`` to filter false positives (requires stored graphs).

        Additional filter: Tutte-equivalent non-isomorphic graphs (same polynomial,
        different canonical key) are excluded since neither can be a minor of the
        other when they have the same edge count.

        Keys and values are canonical keys for O(1) lookup.
        """
        from ..graphs.minor import is_graph_minor

        entries_list = list(self.entries.values())

        # Phase 1: coefficient domination (no false negatives)
        # Merge new minors into existing synthesis-tracked relationships
        relationships: Dict[str, set] = {}
        for key, minor_list in self.minor_relationships.items():
            relationships[key] = set(minor_list)

        for entry in entries_list:
            existing = relationships.get(entry.canonical_key, set())

            for other in entries_list:
                if entry.canonical_key == other.canonical_key:
                    continue
                if other.canonical_key in existing:
                    continue  # Already known minor
                if other.node_count > entry.node_count:
                    continue
                if other.edge_count > entry.edge_count:
                    continue

                # Coefficient domination check: T(major) - T(minor) >= 0
                diff = entry.polynomial - other.polynomial
                coeffs = diff.to_coefficients()
                if not all(c >= 0 for c in coeffs.values()):
                    continue

                # Exclude Tutte-equivalent non-isomorphic graphs (same polynomial,
                # different structure). Neither can be a minor of the other when
                # they have the same number of edges.
                if entry.polynomial == other.polynomial:
                    continue

                existing.add(other.canonical_key)

            if existing:
                relationships[entry.canonical_key] = existing

        # Phase 2: verify suspicious pairs with actual graph minor check
        if verify:
            import sys as _sys
            import time as _time
            from networkx.algorithms.isomorphism import GraphMatcher as _GM

            def _progress(msg):
                print(msg, flush=True)

            verified = 0
            removed = 0
            skipped = 0

            # Pre-build NetworkX graph cache (avoids repeated conversion)
            nx_cache: Dict[str, object] = {}
            for key, entry in self.entries.items():
                if entry.graph is not None:
                    nx_cache[key] = entry.graph.to_networkx()

            # Count total suspicious pairs for progress reporting
            total_suspicious = 0
            for major_key, minor_keys in relationships.items():
                major_entry = self.entries.get(major_key)
                if major_entry is None or major_key not in nx_cache:
                    continue
                for minor_key in minor_keys:
                    minor_entry = self.entries.get(minor_key)
                    if minor_entry is None or minor_key not in nx_cache:
                        continue
                    is_minor_tree = (minor_entry.spanning_trees == 1
                                     and minor_entry.edge_count == minor_entry.node_count - 1)
                    major_is_tree = (major_entry.spanning_trees == 1
                                     and major_entry.edge_count == major_entry.node_count - 1)
                    same_nodes = (major_entry.node_count == minor_entry.node_count
                                  and major_entry.edge_count != minor_entry.edge_count)
                    if (is_minor_tree and not major_is_tree) or same_nodes:
                        total_suspicious += 1

            _progress(f"  Verifying {total_suspicious} suspicious pairs "
                      f"({len(nx_cache)} graphs cached)...")

            checked = 0
            t_start = _time.perf_counter()

            for major_key, minor_keys in list(relationships.items()):
                major_entry = self.entries.get(major_key)
                G_major = nx_cache.get(major_key)
                if major_entry is None or G_major is None:
                    continue

                to_remove = set()
                for minor_key in list(minor_keys):
                    minor_entry = self.entries.get(minor_key)
                    G_minor = nx_cache.get(minor_key)
                    if minor_entry is None or G_minor is None:
                        continue

                    # Determine if this pair is "suspicious" (likely false positive)
                    is_minor_tree = (minor_entry.spanning_trees == 1
                                     and minor_entry.edge_count == minor_entry.node_count - 1)
                    major_is_tree = (major_entry.spanning_trees == 1
                                     and major_entry.edge_count == major_entry.node_count - 1)
                    same_nodes = (major_entry.node_count == minor_entry.node_count
                                  and major_entry.edge_count != minor_entry.edge_count)

                    suspicious = (is_minor_tree and not major_is_tree) or same_nodes
                    if not suspicious:
                        continue

                    checked += 1
                    if total_suspicious > 1000 and checked % 25000 == 0:
                        elapsed = _time.perf_counter() - t_start
                        rate = checked / elapsed if elapsed > 0 else 0
                        remaining = (total_suspicious - checked) / rate if rate > 0 else 0
                        _progress(f"    {checked}/{total_suspicious} verified "
                                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

                    needed = major_entry.node_count - minor_entry.node_count
                    if needed == 0:
                        # Same node count: subgraph monomorphism IS the full test
                        # (no contractions needed — edge deletion only)
                        if _GM(G_major, G_minor).subgraph_is_monomorphic():
                            verified += 1
                        else:
                            to_remove.add(minor_key)
                            removed += 1
                    else:
                        # Different node counts: structural rules + BFS contraction
                        result = is_graph_minor(major_entry.graph, minor_entry.graph,
                                               max_contractions=max_contractions)
                        if result is False:
                            to_remove.add(minor_key)
                            removed += 1
                        elif result is True:
                            verified += 1
                        else:
                            skipped += 1

                minor_keys -= to_remove

            elapsed = _time.perf_counter() - t_start
            _progress(f"  Minor verification: {verified} confirmed, {removed} false positives removed, "
                      f"{skipped} inconclusive ({elapsed:.1f}s)")

        # Convert sets to sorted lists
        final: Dict[str, List[str]] = {}
        for key, minor_set in relationships.items():
            if minor_set:
                final[key] = sorted(minor_set)

        self.minor_relationships = final
        self._structural_minors_computed = True
        return final

    def compute_gcd_relationships(self) -> 'GCDMinorIndex':
        """Compute GCD-based relationships between all polynomial entries.

        Two polynomials are related if they share a non-trivial common factor
        (GCD != 1). This is a symmetric relationship that captures structural
        similarities between graphs.

        Returns:
            GCDMinorIndex containing all relationships
        """
        from ..factorization import has_common_factor, polynomial_gcd

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

        from ..factorization import has_common_factor, polynomial_gcd

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
        from ..graphs.k_join import polynomial_divmod

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
    """Load the default rainbow table from the package directory.

    Tries binary format first (faster, includes minor relationships),
    falls back to JSON.
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    bin_path = os.path.join(base_dir, 'lookup_table.bin')
    json_path = os.path.join(base_dir, 'lookup_table.json')

    if os.path.exists(bin_path):
        try:
            from .binary import load_binary_rainbow_table
            return load_binary_rainbow_table(bin_path)
        except Exception:
            pass  # Fall through to JSON

    if os.path.exists(json_path):
        return RainbowTable.load(json_path)

    return RainbowTable()
