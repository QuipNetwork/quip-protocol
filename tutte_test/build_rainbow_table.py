"""
Tutte Polynomial Rainbow Table Builder

Pre-compute and store Tutte polynomials for graph minors.
Use for rapid lookup when analyzing larger graphs.

Usage:
    # Build and save table
    python -m tutte_test.build_rainbow_table

    # Use in code
    from tutte_test.build_rainbow_table import RainbowTable
    table = RainbowTable.load('tutte_test/tutte_rainbow_table.json')
    poly = table.lookup_by_name('K_4')
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

# Ensure imports work when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(200000)

from tutte_test.tutte_utils import (
    GraphBuilder,
    TuttePolynomial,
    compute_tutte_polynomial,
    graph_to_canonical_key,
    networkx_to_graphbuilder,
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

    # Minor relationships: graph_name -> list of minor graph names
    minor_relationships: Dict[str, List[str]] = field(default_factory=dict)

    # Minor relationship summary
    minor_summary: Dict[str, any] = field(default_factory=dict)

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

        # Include minor relationships if computed
        if self.minor_relationships:
            output['minor_relationships'] = {
                'summary': self.minor_summary,
                'relationships': self.minor_relationships,
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

        # Load minor relationships if present
        if 'minor_relationships' in data:
            minor_data = data['minor_relationships']
            table.minor_relationships = minor_data.get('relationships', {})
            table.minor_summary = minor_data.get('summary', {})

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


def build_zephyr_minors(table: RainbowTable, verbose: bool = True):
    """Extract and catalog minors from Zephyr topology."""

    if not DWAVE_AVAILABLE:
        if verbose:
            print("\n--- Zephyr minors: SKIPPED (dwave_networkx not available) ---")
        return

    if verbose:
        print("\n--- Zephyr minors from Z(2,2) ---")

    G_zephyr = dnx.zephyr_graph(2, 2)
    if verbose:
        print(f"Source: Z(2,2) with {G_zephyr.number_of_nodes()} nodes, {G_zephyr.number_of_edges()} edges")

    minor_counts = defaultdict(int)
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
                            name = f'Zephyr_minor_{len(visited)}n_{minor_counts[len(visited)]}'
                            table.add(subgraph, name)
                            minor_counts[len(visited)] += 1
                        except Exception:
                            pass

    if verbose:
        print("Zephyr minors discovered:")
        for size in sorted(minor_counts.keys()):
            print(f"  {size} nodes: {minor_counts[size]} unique minors")


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


def build_zephyr_z1t_connectors(table: RainbowTable, verbose: bool = True):
    """
    Add Zephyr Z(1,t) graphs and connector components to the rainbow table.

    Discovers the Z(1,t) decomposition pattern by analyzing Z(1,2):
    - t copies of Z(1,1)
    - C(t,2) = t(t-1)/2 pair-wise connectors between Z(1,1) copies
    - Each pair connector has multiple components

    Computes Z(1,2), Z(1,3), Z(1,4) from assembled minors.
    """

    if not DWAVE_AVAILABLE:
        if verbose:
            print("\n--- Zephyr Z(1,t) graphs: SKIPPED (dwave_networkx not available) ---")
        return

    if verbose:
        print("\n--- Zephyr Z(1,t) Graphs ---")

    try:
        from networkx.algorithms import isomorphism

        # Build Z(1,2) and Z(1,1) for component extraction
        if verbose:
            print("  Building Z(1,2) and Z(1,1) graphs...")
        G_z12 = dnx.zephyr_graph(1, 2)
        G_z11 = dnx.zephyr_graph(1, 1)

        # Find 2 disjoint Z(1,1) copies in Z(1,2)
        if verbose:
            print("  Finding Z(1,1) subgraph isomorphisms...")
        GM = isomorphism.GraphMatcher(G_z12, G_z11)

        used = set()
        z11_copies = []
        count = 0
        for m in GM.subgraph_isomorphisms_iter():
            count += 1
            nodes = frozenset(m.keys())
            if not (nodes & used):
                z11_copies.append(set(nodes))
                used.update(nodes)
                if verbose:
                    print(f"    Found Z(1,1) copy {len(z11_copies)} (after {count} isomorphisms)")
                if len(z11_copies) >= 2:
                    break

        if len(z11_copies) < 2:
            if verbose:
                print("  Warning: Could not find 2 Z(1,1) copies in Z(1,2)")
            return

        # Extract connector edges (edges not in any Z(1,1) copy)
        if verbose:
            print("  Extracting connector structure...")
        all_edges = set(tuple(sorted(e)) for e in G_z12.edges())
        z11_edges = set()
        for copy in z11_copies:
            subg = G_z12.subgraph(copy)
            for e in subg.edges():
                z11_edges.add(tuple(sorted(e)))

        connector_edges = list(all_edges - z11_edges)
        connector_graph = nx.Graph()
        connector_graph.add_edges_from(connector_edges)
        components = list(nx.connected_components(connector_graph))

        if verbose:
            print(f"    Connector: {len(connector_edges)} edges, {len(components)} components")

        # Add connector component to table (discover its properties, don't assume)
        component_poly = None
        num_components_per_pair = len(components)

        if len(components) >= 1:
            comp_nodes = components[0]
            G_component = connector_graph.subgraph(comp_nodes).copy()
            component_poly = table.add(G_component, 'Zephyr_connector_component')

            if verbose:
                print(f"  Zephyr_connector_component: {G_component.number_of_nodes()} nodes, "
                      f"{G_component.number_of_edges()} edges, T(1,1)={component_poly.num_spanning_trees()}")

        # Get Z(1,1) polynomial
        z11_poly = table.lookup_by_name('Z(1,1)')
        if not z11_poly or not component_poly:
            if verbose:
                print("  Warning: Missing Z(1,1) or connector polynomial")
            return

        # Compute Z(1,t) for t = 2, 3, 4 using the discovered pattern:
        # Z(1,t) = (Z(1,1))^t × (connector_component)^(num_components_per_pair × C(t,2))
        # where C(t,2) = t(t-1)/2 is the number of pairs
        for t in range(2, 5):
            if verbose:
                print(f"  Computing Z(1,{t}) from assembled minors...")

            G_z1t = dnx.zephyr_graph(1, t)
            num_pairs = t * (t - 1) // 2
            num_connector_components = num_components_per_pair * num_pairs

            # Compute polynomial: Z(1,1)^t × connector^num_connector_components
            z1t_poly = z11_poly
            for _ in range(t - 1):
                z1t_poly = z1t_poly * z11_poly
            for _ in range(num_connector_components):
                z1t_poly = z1t_poly * component_poly

            # Store in table
            key = graph_to_canonical_key(G_z1t)
            table.entries[key] = {
                'name': f'Z(1,{t})',
                'nodes': G_z1t.number_of_nodes(),
                'edges': G_z1t.number_of_edges(),
                'spanning_trees': z1t_poly.num_spanning_trees(),
                'coefficients': {f"{i},{j}": c for (i, j), c in z1t_poly.coefficients.items()},
                'computed_from': f'Z(1,t) decomposition: {t}×Z(1,1) + {num_connector_components}×connector_component',
            }
            table.name_index[f'Z(1,{t})'] = key

            if verbose:
                print(f"  Z(1,{t}): {G_z1t.number_of_nodes()} nodes, "
                      f"{G_z1t.number_of_edges()} edges, T(1,1)={z1t_poly.num_spanning_trees()}")
                print(f"    Structure: {t}×Z(1,1) + {num_pairs} pairs × {num_components_per_pair} components = "
                      f"{num_connector_components} connector components")

    except Exception as e:
        if verbose:
            import traceback
            print(f"  Error building Z(1,t) graphs: {e}")
            traceback.print_exc()


def build_pegasus_minors(table: RainbowTable, verbose: bool = True):
    """
    Extract and catalog minors from Pegasus topology.

    Pegasus graphs are too large to compute full Tutte polynomials:
    - P(2): 40 nodes, 164 edges
    - P(3): 128 nodes, 704 edges

    Instead, we extract small connected subgraphs (minors) that capture
    the local structure of Pegasus topology.
    """

    if not DWAVE_AVAILABLE:
        if verbose:
            print("\n--- Pegasus minors: SKIPPED (dwave_networkx not available) ---")
        return

    if verbose:
        print("\n--- Pegasus minors from P(2) ---")

    G_pegasus = dnx.pegasus_graph(2)
    if verbose:
        print(f"Source: P(2) with {G_pegasus.number_of_nodes()} nodes, {G_pegasus.number_of_edges()} edges")

    minor_counts = defaultdict(int)
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
                            name = f'Pegasus_minor_{len(visited)}n_{minor_counts[len(visited)]}'
                            table.add(subgraph, name)
                            minor_counts[len(visited)] += 1
                        except Exception:
                            pass

    if verbose:
        print("Pegasus minors discovered:")
        for size in sorted(minor_counts.keys()):
            print(f"  {size} nodes: {minor_counts[size]} unique minors")


def build_minor_relationships(table: RainbowTable, verbose: bool = True):
    """
    Compute and store minor relationships for all graphs in the table.

    A polynomial P1 is a "minor" of P2 if P2 - P1 has all non-negative coefficients.
    This suggests P2's graph could structurally contain P1's graph.
    """
    if verbose:
        print("\n--- Computing Minor Relationships ---")

    try:
        # Load all polynomials with progress
        polys = {}
        names = list(table.name_index.keys())
        n = len(names)

        if verbose:
            print(f"  Loading {n} polynomials...")

        for name in names:
            key = table.name_index[name]
            entry = table.entries[key]
            polys[name] = table._entry_to_polynomial(entry)

        # Compute minor relationships with progress
        minor_of = defaultdict(list)
        total_comparisons = n * n
        comparisons_done = 0
        last_percent = -1

        if verbose:
            print(f"  Computing {total_comparisons:,} polynomial comparisons...")

        for name1, poly1 in polys.items():
            for name2, poly2 in polys.items():
                comparisons_done += 1

                # Progress update every 5%
                percent = (comparisons_done * 100) // total_comparisons
                if verbose and percent >= last_percent + 5:
                    last_percent = percent
                    print(f"    Progress: {percent}% ({comparisons_done:,}/{total_comparisons:,})")

                if name1 == name2:
                    continue

                # Check if poly1 is a minor of poly2 (poly2 - poly1 >= 0)
                all_non_negative = True
                diff_coeffs = defaultdict(int, poly2.coefficients)
                for k, v in poly1.coefficients.items():
                    diff_coeffs[k] -= v

                for c in diff_coeffs.values():
                    if c < 0:
                        all_non_negative = False
                        break

                if all_non_negative:
                    minor_of[name2].append(name1)

        table.minor_relationships = dict(minor_of)

        # Compute summary statistics
        total_rels = sum(len(v) for v in minor_of.values())

        if minor_of:
            max_graph = max(minor_of.items(), key=lambda x: len(x[1]))
            max_name, max_minors = max_graph
            max_count = len(max_minors)
        else:
            max_name, max_count = "None", 0

        # Find Zephyr-specific relationships
        zephyr_minors = {
            name: minors for name, minors in minor_of.items()
            if 'Z(' in name or 'Zephyr' in name
        }

        table.minor_summary = {
            'total_relationships': total_rels,
            'graphs_with_minors': len(minor_of),
            'max_minors_graph': max_name,
            'max_minors_count': max_count,
            'zephyr_graphs_with_minors': len(zephyr_minors),
        }

        if verbose:
            print(f"  Total relationships: {total_rels}")
            print(f"  Graphs with minors: {len(minor_of)}")
            print(f"  Graph with most minors: {max_name} ({max_count} minors)")
            if zephyr_minors:
                print(f"  Zephyr graphs with minors: {len(zephyr_minors)}")
                for name, minors in sorted(zephyr_minors.items()):
                    print(f"    {name}: {len(minors)} minors")

    except Exception as e:
        if verbose:
            import traceback
            print(f"  Error computing minor relationships: {e}")
            traceback.print_exc()


def build_full_table(verbose: bool = True) -> RainbowTable:
    """
    Build complete rainbow table with all graph families.

    Returns:
        RainbowTable with standard graphs, Zephyr graphs, Z(1,t) connectors, and Pegasus minors
    """
    table = RainbowTable()

    build_standard_graphs(table, verbose=verbose)
    build_zephyr_minors(table, verbose=verbose)
    build_zephyr_z11(table, verbose=verbose)
    build_zephyr_z1t_connectors(table, verbose=verbose)
    build_pegasus_minors(table, verbose=verbose)
    build_minor_relationships(table, verbose=verbose)

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
