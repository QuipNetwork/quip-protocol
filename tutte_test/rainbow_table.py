"""
Tutte Polynomial Rainbow Table

Pre-compute and store Tutte polynomials for graph motifs.
Use for rapid lookup when analyzing larger graphs.

Usage:
    # Build and save table
    python -m tutte_test.rainbow_table

    # Use in code
    from tutte_test.rainbow_table import RainbowTable
    table = RainbowTable.load('tutte_test/tutte_rainbow_table.json')
    poly = table.lookup_by_name('K_4')
"""

import sys
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

import networkx as nx

# Ensure imports work when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(200000)

from tutte_test.tutte_to_ising import (
    TuttePolynomial,
    GraphBuilder,
    compute_tutte_polynomial,
)
from tutte_test.tutte_utils import (
    networkx_to_graphbuilder,
    graph_to_canonical_key,
)

# Check D-Wave availability
try:
    import dwave_networkx as dnx
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False


@dataclass
class RainbowTable:
    """
    Rainbow table for Tutte polynomial lookup.

    Stores pre-computed polynomials indexed by canonical graph representation.
    Supports both build-time computation and runtime lookup.
    """
    # Main storage: canonical_key -> entry dict
    entries: Dict[str, Dict] = field(default_factory=dict)

    # Name index for lookup by name
    name_index: Dict[str, str] = field(default_factory=dict)

    # Statistics
    stats: Dict[str, int] = field(default_factory=lambda: {
        'hits': 0, 'misses': 0, 'computations': 0
    })

    def add(self, G: nx.Graph, name: str,
            polynomial: Optional[TuttePolynomial] = None) -> TuttePolynomial:
        """
        Add a graph to the table, computing its Tutte polynomial if needed.

        Args:
            G: NetworkX graph
            name: Human-readable name for the graph
            polynomial: Pre-computed polynomial (optional)

        Returns:
            The Tutte polynomial for the graph
        """
        key = graph_to_canonical_key(G)

        # Check if already exists
        if key in self.entries:
            self.stats['hits'] += 1
            return self._entry_to_polynomial(self.entries[key])

        self.stats['misses'] += 1

        # Compute if not provided
        if polynomial is None:
            self.stats['computations'] += 1
            gb = networkx_to_graphbuilder(G)
            polynomial = compute_tutte_polynomial(gb)

        # Store entry
        coeffs = {f'{i},{j}': c for (i, j), c in polynomial.coefficients.items()}
        self.entries[key] = {
            'name': name,
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'spanning_trees': polynomial.num_spanning_trees(),
            'x_degree': polynomial.x_degree(),
            'y_degree': polynomial.y_degree(),
            'num_terms': len(polynomial.coefficients),
            'polynomial_str': str(polynomial),
            'coefficients': coeffs,
        }

        # Update name index
        self.name_index[name] = key

        return polynomial

    def lookup(self, G: nx.Graph) -> Optional[TuttePolynomial]:
        """
        Look up polynomial for a graph by structure.

        Args:
            G: NetworkX graph

        Returns:
            TuttePolynomial if found, None otherwise
        """
        key = graph_to_canonical_key(G)
        if key in self.entries:
            self.stats['hits'] += 1
            return self._entry_to_polynomial(self.entries[key])
        self.stats['misses'] += 1
        return None

    def lookup_by_name(self, name: str) -> Optional[TuttePolynomial]:
        """
        Look up polynomial by graph name.

        Args:
            name: Name of the graph (e.g., 'K_4', 'C_5')

        Returns:
            TuttePolynomial if found, None otherwise
        """
        if name in self.name_index:
            key = self.name_index[name]
            return self._entry_to_polynomial(self.entries[key])
        return None

    def get_entry(self, name: str) -> Optional[Dict]:
        """Get full entry dict by name."""
        if name in self.name_index:
            return self.entries[self.name_index[name]]
        return None

    def _entry_to_polynomial(self, entry: Dict) -> TuttePolynomial:
        """Convert stored entry back to TuttePolynomial."""
        coeffs = {
            tuple(map(int, k.split(','))): v
            for k, v in entry['coefficients'].items()
        }
        return TuttePolynomial.from_dict(coeffs)

    def save(self, filepath: str) -> int:
        """
        Save table to JSON file.

        Returns:
            File size in bytes
        """
        output = {
            'description': 'Tutte Polynomial Rainbow Table',
            'total_entries': len(self.entries),
            'note': 'Coefficients stored as "i,j": coefficient for x^i * y^j',
            'graphs': self.entries,
        }
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        return os.path.getsize(filepath)

    @classmethod
    def load(cls, filepath: str) -> 'RainbowTable':
        """
        Load table from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            RainbowTable instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        table = cls()
        table.entries = data.get('graphs', {})

        # Rebuild name index
        for key, entry in table.entries.items():
            if 'name' in entry:
                table.name_index[entry['name']] = key

        return table

    def __len__(self) -> int:
        return len(self.entries)


def build_standard_graphs(table: RainbowTable, verbose: bool = True):
    """Add standard graph families to the table."""

    if verbose:
        print("\n--- Standard Graph Families ---")

    # Complete graphs K_n
    if verbose:
        print("Complete graphs K_n:")
    for n in range(2, 9):
        G = nx.complete_graph(n)
        if G.number_of_edges() <= 28:
            t = table.add(G, f'K_{n}')
            if verbose:
                print(f"  K_{n}: {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")

    # Cycles C_n
    if verbose:
        print("Cycle graphs C_n: C_3 to C_24")
    for n in range(3, 25):
        G = nx.cycle_graph(n)
        table.add(G, f'C_{n}')

    # Paths P_n
    if verbose:
        print("Path graphs P_n: P_2 to P_24")
    for n in range(2, 25):
        G = nx.path_graph(n)
        table.add(G, f'P_{n}')

    # Complete bipartite K_{m,n}
    if verbose:
        print("Complete bipartite K_{m,n}:")
    for m in range(2, 6):
        for n in range(m, 6):
            G = nx.complete_bipartite_graph(m, n)
            if G.number_of_edges() <= 25:
                t = table.add(G, f'K_{m},{n}')
                if verbose:
                    print(f"  K_{m},{n}: {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")

    # Wheel graphs W_n
    if verbose:
        print("Wheel graphs W_n:")
    for n in range(3, 12):
        G = nx.wheel_graph(n)
        if G.number_of_edges() <= 22:
            t = table.add(G, f'W_{n}')
            if verbose:
                print(f"  W_{n}: {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")

    # Grid graphs
    if verbose:
        print("Grid graphs:")
    for m in range(2, 6):
        for n in range(2, 6):
            G = nx.grid_2d_graph(m, n)
            if G.number_of_edges() <= 22:
                t = table.add(G, f'Grid_{m}x{n}')
                if verbose:
                    print(f"  Grid_{m}x{n}: {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")

    # Hypercubes Q_n
    if verbose:
        print("Hypercube graphs Q_n:")
    for n in range(1, 5):
        G = nx.hypercube_graph(n)
        if G.number_of_edges() <= 22:
            t = table.add(G, f'Q_{n}')
            if verbose:
                print(f"  Q_{n}: {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")

    # Special graphs
    if verbose:
        print("Special graphs:")

    G = nx.petersen_graph()
    t = table.add(G, 'Petersen')
    if verbose:
        print(f"  Petersen: {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")

    G = nx.moebius_kantor_graph()
    if G.number_of_edges() <= 24:
        t = table.add(G, 'Moebius-Kantor')
        if verbose:
            print(f"  Moebius-Kantor: {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")

    # Ladder graphs
    if verbose:
        print("Ladder graphs:")
    for n in range(2, 12):
        G = nx.ladder_graph(n)
        if G.number_of_edges() <= 22:
            t = table.add(G, f'Ladder_{n}')
            if verbose:
                print(f"  Ladder_{n}: {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")

    # Circulant graphs
    if verbose:
        print("Circulant graphs:")
    for n in range(5, 12):
        G = nx.circulant_graph(n, [1, 2])
        if G.number_of_edges() <= 22:
            t = table.add(G, f'Circ_{n}_1_2')
            if verbose:
                print(f"  Circ({n},[1,2]): {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")


def build_zephyr_motifs(table: RainbowTable, verbose: bool = True):
    """Extract and catalog motifs from Zephyr topology."""

    if not DWAVE_AVAILABLE:
        if verbose:
            print("\n--- Zephyr Motifs: SKIPPED (dwave_networkx not available) ---")
        return

    if verbose:
        print("\n--- Zephyr Motifs from Z(2,2) ---")

    G_zephyr = dnx.zephyr_graph(2, 2)
    if verbose:
        print(f"Source: Z(2,2) with {G_zephyr.number_of_nodes()} nodes, {G_zephyr.number_of_edges()} edges")

    motif_counts = defaultdict(int)
    nodes_list = list(G_zephyr.nodes())

    # Sample starting points and grow subgraphs
    for start_idx in range(min(30, len(nodes_list))):
        start_node = nodes_list[start_idx]

        for target_size in range(3, 9):
            # BFS to grow subgraph
            visited = {start_node}
            frontier = [start_node]

            while len(visited) < target_size and frontier:
                next_frontier = []
                for node in frontier:
                    for neighbor in G_zephyr.neighbors(node):
                        if neighbor not in visited and len(visited) < target_size:
                            visited.add(neighbor)
                            next_frontier.append(neighbor)
                frontier = next_frontier

            if len(visited) >= 3:
                subgraph = G_zephyr.subgraph(visited).copy()
                if subgraph.number_of_edges() <= 22:
                    key = graph_to_canonical_key(subgraph)
                    if key not in table.entries:
                        try:
                            name = f'Zephyr_motif_{len(visited)}n_{motif_counts[len(visited)]}'
                            table.add(subgraph, name)
                            motif_counts[len(visited)] += 1
                        except Exception:
                            pass

    if verbose:
        print("Zephyr motifs discovered:")
        for size in sorted(motif_counts.keys()):
            print(f"  {size} nodes: {motif_counts[size]} unique motifs")


def build_zephyr_z11(table: RainbowTable, verbose: bool = True):
    """Add the Z(1,1) Zephyr topology."""

    if not DWAVE_AVAILABLE:
        if verbose:
            print("\n--- Z(1,1): SKIPPED (dwave_networkx not available) ---")
        return

    if verbose:
        print("\n--- Z(1,1) Zephyr Topology ---")

    G = dnx.zephyr_graph(1, 1)
    t = table.add(G, 'Z(1,1)')

    if verbose:
        print(f"Z(1,1): {G.number_of_edges()} edges, T(1,1)={t.num_spanning_trees()}")


def build_pegasus_motifs(table: RainbowTable, verbose: bool = True):
    """
    Extract and catalog motifs from Pegasus topology.

    Pegasus graphs are too large to compute full Tutte polynomials:
    - P(2): 40 nodes, 164 edges
    - P(3): 128 nodes, 704 edges

    Instead, we extract small connected subgraphs (motifs) that capture
    the local structure of Pegasus topology.
    """

    if not DWAVE_AVAILABLE:
        if verbose:
            print("\n--- Pegasus Motifs: SKIPPED (dwave_networkx not available) ---")
        return

    if verbose:
        print("\n--- Pegasus Motifs from P(2) ---")

    G_pegasus = dnx.pegasus_graph(2)
    if verbose:
        print(f"Source: P(2) with {G_pegasus.number_of_nodes()} nodes, {G_pegasus.number_of_edges()} edges")

    motif_counts = defaultdict(int)
    nodes_list = list(G_pegasus.nodes())

    # Sample starting points and grow subgraphs
    for start_idx in range(min(40, len(nodes_list))):
        start_node = nodes_list[start_idx]

        for target_size in range(3, 10):
            # BFS to grow subgraph
            visited = {start_node}
            frontier = [start_node]

            while len(visited) < target_size and frontier:
                next_frontier = []
                for node in frontier:
                    for neighbor in G_pegasus.neighbors(node):
                        if neighbor not in visited and len(visited) < target_size:
                            visited.add(neighbor)
                            next_frontier.append(neighbor)
                frontier = next_frontier

            if len(visited) >= 3:
                subgraph = G_pegasus.subgraph(visited).copy()
                if subgraph.number_of_edges() <= 24:
                    key = graph_to_canonical_key(subgraph)
                    if key not in table.entries:
                        try:
                            name = f'Pegasus_motif_{len(visited)}n_{motif_counts[len(visited)]}'
                            table.add(subgraph, name)
                            motif_counts[len(visited)] += 1
                        except Exception:
                            pass

    if verbose:
        print("Pegasus motifs discovered:")
        for size in sorted(motif_counts.keys()):
            print(f"  {size} nodes: {motif_counts[size]} unique motifs")


def build_full_table(verbose: bool = True) -> RainbowTable:
    """
    Build complete rainbow table with all graph families.

    Returns:
        RainbowTable with standard graphs, Zephyr motifs, Z(1,1), and Pegasus motifs
    """
    table = RainbowTable()

    build_standard_graphs(table, verbose=verbose)
    build_zephyr_motifs(table, verbose=verbose)
    build_zephyr_z11(table, verbose=verbose)
    build_pegasus_motifs(table, verbose=verbose)

    return table


def main():
    """Build and save the rainbow table."""
    print("=" * 70)
    print("BUILDING TUTTE POLYNOMIAL RAINBOW TABLE")
    print("=" * 70)

    table = build_full_table(verbose=True)

    print("\n" + "=" * 70)
    print(f"TOTAL ENTRIES: {len(table)}")
    print("=" * 70)

    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'tutte_rainbow_table.json')
    file_size = table.save(output_path)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size:,} bytes")

    return table


if __name__ == "__main__":
    main()
