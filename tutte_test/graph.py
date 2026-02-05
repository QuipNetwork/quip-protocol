"""Graph Representation for Tutte Polynomial Computation.

This module provides:
1. Graph - Immutable graph for hashing and storage
2. MutableGraph - Mutable graph for incremental construction
3. NetworkX conversion utilities
4. Canonical key generation for isomorphism-invariant identification
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, FrozenSet, Optional, Iterator

import networkx as nx


# =============================================================================
# IMMUTABLE GRAPH CLASS
# =============================================================================

@dataclass(frozen=True)
class Graph:
    """Immutable graph representation for hashing and storage.

    Edges are stored as frozenset of (min, max) tuples to ensure
    canonical representation regardless of edge direction.
    """
    nodes: FrozenSet[int]
    edges: FrozenSet[Tuple[int, int]]

    def __post_init__(self):
        # Validate edges reference valid nodes
        for u, v in self.edges:
            if u not in self.nodes or v not in self.nodes:
                raise ValueError(f"Edge ({u}, {v}) references invalid node")

    @classmethod
    def from_networkx(cls, G: nx.Graph) -> 'Graph':
        """Convert NetworkX graph to immutable Graph.

        Node labels are converted to sequential integers.
        """
        node_map = {n: i for i, n in enumerate(G.nodes())}
        nodes = frozenset(node_map.values())
        edges = frozenset(
            (min(node_map[u], node_map[v]), max(node_map[u], node_map[v]))
            for u, v in G.edges()
            if u != v  # Skip self-loops in edge representation
        )
        return cls(nodes=nodes, edges=edges)

    @classmethod
    def from_edge_list(cls, edge_list: List[Tuple[int, int]]) -> 'Graph':
        """Create graph from edge list, inferring nodes."""
        nodes = set()
        edges = set()
        for u, v in edge_list:
            nodes.add(u)
            nodes.add(v)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return cls(nodes=frozenset(nodes), edges=frozenset(edges))

    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return G

    def canonical_key(self) -> str:
        """Create a canonical string key for the graph (isomorphism-invariant).

        Uses SHA256 hash of the graph6 format for compact, deterministic keys.
        Falls back to hash of sorted edge list for graphs that can't be encoded.
        """
        G = self.to_networkx()
        G_relabeled = nx.convert_node_labels_to_integers(G)
        try:
            canonical_bytes = nx.to_graph6_bytes(G_relabeled, header=False)
        except Exception:
            edges = sorted(self.edges)
            canonical_bytes = str(edges).encode('utf-8')
        return hashlib.sha256(canonical_bytes).hexdigest()

    def node_count(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def degree(self, node: int) -> int:
        """Get degree of a node."""
        return sum(1 for u, v in self.edges if u == node or v == node)

    def neighbors(self, node: int) -> Set[int]:
        """Get neighbors of a node."""
        result = set()
        for u, v in self.edges:
            if u == node:
                result.add(v)
            elif v == node:
                result.add(u)
        return result

    def subgraph(self, nodes: Set[int]) -> 'Graph':
        """Return induced subgraph on given nodes."""
        new_nodes = frozenset(nodes & self.nodes)
        new_edges = frozenset(
            (u, v) for u, v in self.edges
            if u in new_nodes and v in new_nodes
        )
        return Graph(nodes=new_nodes, edges=new_edges)

    def edge_induced_subgraph(self, edges: Set[Tuple[int, int]]) -> 'Graph':
        """Return edge-induced subgraph."""
        normalized_edges = set()
        for u, v in edges:
            normalized_edges.add((min(u, v), max(u, v)))

        new_edges = frozenset(self.edges & normalized_edges)
        new_nodes = set()
        for u, v in new_edges:
            new_nodes.add(u)
            new_nodes.add(v)
        return Graph(nodes=frozenset(new_nodes), edges=new_edges)

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        if not self.nodes:
            return True
        if not self.edges:
            return len(self.nodes) <= 1

        visited = set()
        start = next(iter(self.nodes))
        stack = [start]
        visited.add(start)

        while stack:
            node = stack.pop()
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return len(visited) == len(self.nodes)

    def connected_components(self) -> List['Graph']:
        """Return list of connected component subgraphs."""
        if not self.nodes:
            return []

        visited = set()
        components = []

        for start in self.nodes:
            if start in visited:
                continue

            component_nodes = set()
            stack = [start]

            while stack:
                node = stack.pop()
                if node in component_nodes:
                    continue
                component_nodes.add(node)
                visited.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor not in component_nodes:
                        stack.append(neighbor)

            components.append(self.subgraph(component_nodes))

        return components

    def is_simple(self) -> bool:
        """True if no parallel edges or loops (always True for Graph)."""
        return True

    def add_edge(self, u: int, v: int) -> 'Graph':
        """Return new graph with an additional edge."""
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must exist")
        if u == v:
            raise ValueError("Graph does not support self-loops")
        edge = (min(u, v), max(u, v))
        if edge in self.edges:
            raise ValueError(f"Edge {edge} already exists (use MultiGraph for parallel edges)")
        return Graph(nodes=self.nodes, edges=self.edges | {edge})

    def remove_edge(self, u: int, v: int) -> 'Graph':
        """Return new graph with an edge removed."""
        edge = (min(u, v), max(u, v))
        if edge not in self.edges:
            raise ValueError(f"Edge {edge} not found")
        return Graph(nodes=self.nodes, edges=self.edges - {edge})

    def has_cut_vertex(self) -> Optional[int]:
        """Find a cut vertex (articulation point) if one exists.

        A cut vertex is a vertex whose removal disconnects the graph.
        Returns the first cut vertex found, or None if none exist.
        """
        if len(self.nodes) <= 2:
            return None

        # Use DFS-based articulation point algorithm
        visited = set()
        disc = {}  # Discovery time
        low = {}   # Lowest reachable discovery time
        parent = {}
        ap = set()  # Articulation points
        time = [0]

        def dfs(u: int):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1

            for v in self.neighbors(u):
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # u is articulation point if:
                    # 1) u is root of DFS tree and has 2+ children
                    # 2) u is not root and low[v] >= disc[u]
                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        # Start DFS from first node
        start = next(iter(self.nodes))
        dfs(start)

        # Return first articulation point, or None
        return next(iter(ap)) if ap else None

    def find_all_cut_vertices(self) -> Set[int]:
        """Find all cut vertices (articulation points).

        Returns set of all cut vertices in the graph.
        """
        if len(self.nodes) <= 2:
            return set()

        visited = set()
        disc = {}
        low = {}
        parent = {}
        ap = set()
        time = [0]

        def dfs(u: int):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1

            for v in self.neighbors(u):
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        start = next(iter(self.nodes))
        dfs(start)
        return ap

    def split_at_cut_vertex(self, v: int) -> List['Graph']:
        """Split graph into components at cut vertex.

        Returns list of subgraphs, each containing the cut vertex.
        The cut vertex appears in all returned subgraphs.
        """
        if v not in self.nodes:
            raise ValueError(f"Node {v} not in graph")

        # Find components when v is removed
        other_nodes = self.nodes - {v}
        if not other_nodes:
            return [self]

        # Build adjacency for nodes other than v
        visited = set()
        components = []

        for start in other_nodes:
            if start in visited:
                continue

            component_nodes = {v}  # Include cut vertex in each component
            stack = [start]

            while stack:
                node = stack.pop()
                if node in visited or node == v:
                    continue
                visited.add(node)
                component_nodes.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor not in visited and neighbor != v:
                        stack.append(neighbor)

            # Get induced subgraph
            component_edges = frozenset(
                (u, w) for u, w in self.edges
                if u in component_nodes and w in component_nodes
            )
            components.append(Graph(
                nodes=frozenset(component_nodes),
                edges=component_edges
            ))

        return components if len(components) > 1 else [self]

    def merge_nodes(self, u: int, v: int) -> 'MultiGraph':
        """Merge nodes u and v, creating a MultiGraph.

        This operation identifies u and v:
        - Edges from both u and v to the same node w become parallel edges
        - The edge (u, v) if it exists becomes a loop
        - All nodes connected to v become connected to u

        Returns a MultiGraph (which may have parallel edges or loops).
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must exist")
        if u == v:
            return MultiGraph.from_graph(self)

        # The surviving node is the smaller one (for consistency)
        survivor = min(u, v)
        removed = max(u, v)

        # Build new node set
        new_nodes = frozenset(n if n != removed else survivor for n in self.nodes) - {removed}

        # Build edge counts (tracking parallel edges and loops)
        edge_counts: Dict[Tuple[int, int], int] = {}
        loop_counts: Dict[int, int] = {}

        for a, b in self.edges:
            # Remap removed node to survivor
            new_a = survivor if a == removed else a
            new_b = survivor if b == removed else b

            if new_a == new_b:
                # This edge becomes a loop
                loop_counts[new_a] = loop_counts.get(new_a, 0) + 1
            else:
                edge = (min(new_a, new_b), max(new_a, new_b))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        return MultiGraph(
            nodes=new_nodes,
            edge_counts=edge_counts,
            loop_counts=loop_counts
        )


# =============================================================================
# MULTIGRAPH CLASS (supports parallel edges and loops)
# =============================================================================

@dataclass(frozen=True)
class MultiGraph:
    """Immutable multigraph that supports parallel edges and loops.

    This class is used when merging nodes creates parallel edges.
    For simple graphs (no parallel edges or loops), use Graph instead.
    """
    nodes: FrozenSet[int]
    edge_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)  # edge -> count
    loop_counts: Dict[int, int] = field(default_factory=dict)  # node -> loop count

    def __post_init__(self):
        # Validate edges reference valid nodes
        for (u, v), count in self.edge_counts.items():
            if u not in self.nodes or v not in self.nodes:
                raise ValueError(f"Edge ({u}, {v}) references invalid node")
            if count <= 0:
                raise ValueError(f"Edge count must be positive")
        for node, count in self.loop_counts.items():
            if node not in self.nodes:
                raise ValueError(f"Loop at node {node} references invalid node")
            if count <= 0:
                raise ValueError(f"Loop count must be positive")

    @classmethod
    def from_graph(cls, g: Graph) -> 'MultiGraph':
        """Convert simple Graph to MultiGraph."""
        edge_counts = {e: 1 for e in g.edges}
        return cls(nodes=g.nodes, edge_counts=edge_counts, loop_counts={})

    def is_simple(self) -> bool:
        """True if no parallel edges or loops."""
        if self.loop_counts:
            return False
        return all(c == 1 for c in self.edge_counts.values())

    def to_simple_graph(self) -> Optional[Graph]:
        """Convert to simple Graph if possible (no parallel edges or loops)."""
        if not self.is_simple():
            return None
        return Graph(nodes=self.nodes, edges=frozenset(self.edge_counts.keys()))

    def node_count(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Total number of edges (counting multiplicities)."""
        return sum(self.edge_counts.values()) + sum(self.loop_counts.values())

    def unique_edge_count(self) -> int:
        """Number of unique edges (not counting multiplicities)."""
        return len(self.edge_counts) + len(self.loop_counts)

    def total_loop_count(self) -> int:
        """Total number of loops."""
        return sum(self.loop_counts.values())

    def degree(self, node: int) -> int:
        """Get degree of a node (loops count as 2 each)."""
        deg = 0
        for (u, v), count in self.edge_counts.items():
            if u == node or v == node:
                deg += count
        deg += 2 * self.loop_counts.get(node, 0)
        return deg

    def neighbors(self, node: int) -> Set[int]:
        """Get neighbors of a node (not counting multiplicities)."""
        result = set()
        for (u, v) in self.edge_counts.keys():
            if u == node:
                result.add(v)
            elif v == node:
                result.add(u)
        return result

    def edge_multiplicity(self, u: int, v: int) -> int:
        """Get number of edges between u and v."""
        if u == v:
            return self.loop_counts.get(u, 0)
        edge = (min(u, v), max(u, v))
        return self.edge_counts.get(edge, 0)

    def has_cut_vertex(self) -> Optional[int]:
        """Find a cut vertex if one exists (ignoring multiplicities)."""
        if len(self.nodes) <= 2:
            return None

        # Convert to simple adjacency for cut vertex detection
        visited = set()
        disc = {}
        low = {}
        parent = {}
        ap = set()
        time = [0]

        def dfs(u: int):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1

            for v in self.neighbors(u):
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        if self.nodes:
            start = next(iter(self.nodes))
            dfs(start)

        return next(iter(ap)) if ap else None

    def split_at_cut_vertex(self, v: int) -> List['MultiGraph']:
        """Split multigraph into components at cut vertex."""
        if v not in self.nodes:
            raise ValueError(f"Node {v} not in graph")

        other_nodes = self.nodes - {v}
        if not other_nodes:
            return [self]

        visited = set()
        components = []

        for start in other_nodes:
            if start in visited:
                continue

            component_nodes = {v}
            stack = [start]

            while stack:
                node = stack.pop()
                if node in visited or node == v:
                    continue
                visited.add(node)
                component_nodes.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor not in visited and neighbor != v:
                        stack.append(neighbor)

            # Get induced subgraph edges
            component_edge_counts = {
                (u, w): c for (u, w), c in self.edge_counts.items()
                if u in component_nodes and w in component_nodes
            }
            component_loop_counts = {
                n: c for n, c in self.loop_counts.items()
                if n in component_nodes
            }
            components.append(MultiGraph(
                nodes=frozenset(component_nodes),
                edge_counts=component_edge_counts,
                loop_counts=component_loop_counts
            ))

        return components if len(components) > 1 else [self]

    def remove_loops(self) -> 'MultiGraph':
        """Return new multigraph with all loops removed."""
        return MultiGraph(
            nodes=self.nodes,
            edge_counts=dict(self.edge_counts),
            loop_counts={}
        )

    def is_connected(self) -> bool:
        """Check if the multigraph is connected."""
        if not self.nodes:
            return True
        if not self.edge_counts and not self.loop_counts:
            return len(self.nodes) <= 1

        visited = set()
        start = next(iter(self.nodes))
        stack = [start]
        visited.add(start)

        while stack:
            node = stack.pop()
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return len(visited) == len(self.nodes)

    def merge_nodes(self, u: int, v: int) -> 'MultiGraph':
        """Merge nodes u and v."""
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must exist")
        if u == v:
            return self

        survivor = min(u, v)
        removed = max(u, v)

        new_nodes = frozenset(n if n != removed else survivor for n in self.nodes) - {removed}

        new_edge_counts: Dict[Tuple[int, int], int] = {}
        new_loop_counts: Dict[int, int] = dict(self.loop_counts)

        # Handle loops at removed node
        if removed in new_loop_counts:
            new_loop_counts[survivor] = new_loop_counts.get(survivor, 0) + new_loop_counts.pop(removed)

        for (a, b), count in self.edge_counts.items():
            new_a = survivor if a == removed else a
            new_b = survivor if b == removed else b

            if new_a == new_b:
                new_loop_counts[new_a] = new_loop_counts.get(new_a, 0) + count
            else:
                edge = (min(new_a, new_b), max(new_a, new_b))
                new_edge_counts[edge] = new_edge_counts.get(edge, 0) + count

        return MultiGraph(
            nodes=new_nodes,
            edge_counts=new_edge_counts,
            loop_counts=new_loop_counts
        )

    def is_just_parallel_edges(self) -> bool:
        """True if graph is just parallel edges between 2 nodes (no loops)."""
        if self.loop_counts:
            return False
        if len(self.nodes) != 2:
            return False
        if len(self.edge_counts) != 1:
            return False
        return True

    def parallel_edge_count(self) -> int:
        """For a graph that is just parallel edges, return the count."""
        if not self.is_just_parallel_edges():
            return 0
        return next(iter(self.edge_counts.values()))

    def canonical_key(self) -> str:
        """Create a canonical key for the multigraph."""
        # Include edge multiplicities in the key
        parts = [
            f"n{len(self.nodes)}",
            "e" + ",".join(f"{u}-{v}x{c}" for (u, v), c in sorted(self.edge_counts.items())),
            "l" + ",".join(f"{n}x{c}" for n, c in sorted(self.loop_counts.items()))
        ]
        key_str = "|".join(parts)
        return hashlib.sha256(key_str.encode()).hexdigest()


# =============================================================================
# MUTABLE GRAPH CLASS
# =============================================================================

@dataclass
class MutableGraph:
    """Mutable graph for incremental construction.

    Supports adding nodes, edges, and loops. Can be frozen into
    an immutable Graph for storage and hashing.
    """
    nodes: Set[int] = field(default_factory=set)
    edges: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    loops: Dict[int, int] = field(default_factory=dict)  # edge_id -> node_id
    _next_node_id: int = 0
    _next_edge_id: int = 0

    def add_node(self) -> int:
        """Add a new isolated node. Returns node ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes.add(node_id)
        return node_id

    def add_nodes(self, count: int) -> List[int]:
        """Add multiple nodes. Returns list of node IDs."""
        return [self.add_node() for _ in range(count)]

    def add_edge(self, u: int, v: int) -> int:
        """Add an edge between two existing nodes. Returns edge ID."""
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must exist")
        if u == v:
            return self.add_loop(u)

        edge_id = self._next_edge_id
        self._next_edge_id += 1
        self.edges[edge_id] = (min(u, v), max(u, v))
        return edge_id

    def add_loop(self, node: int) -> int:
        """Add a self-loop at a node. Returns edge ID."""
        if node not in self.nodes:
            raise ValueError(f"Node {node} must exist")
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        self.loops[edge_id] = node
        return edge_id

    def remove_edge(self, edge_id: int) -> None:
        """Remove an edge by ID."""
        if edge_id in self.edges:
            del self.edges[edge_id]
        elif edge_id in self.loops:
            del self.loops[edge_id]
        else:
            raise ValueError(f"Edge {edge_id} not found")

    def remove_node(self, node: int) -> None:
        """Remove a node and all incident edges."""
        if node not in self.nodes:
            raise ValueError(f"Node {node} not found")

        # Remove incident edges
        edges_to_remove = [
            eid for eid, (u, v) in self.edges.items()
            if u == node or v == node
        ]
        for eid in edges_to_remove:
            del self.edges[eid]

        # Remove loops
        loops_to_remove = [
            eid for eid, n in self.loops.items()
            if n == node
        ]
        for eid in loops_to_remove:
            del self.loops[eid]

        self.nodes.remove(node)

    def contract_edge(self, edge_id: int) -> int:
        """Contract an edge, merging its endpoints. Returns surviving node."""
        if edge_id not in self.edges:
            raise ValueError(f"Edge {edge_id} not found")

        u, v = self.edges[edge_id]
        del self.edges[edge_id]

        # Redirect all edges from v to u
        new_edges = {}
        for eid, (a, b) in self.edges.items():
            if a == v:
                a = u
            if b == v:
                b = u
            if a == b:
                # Edge becomes a loop
                self.loops[eid] = a
            else:
                new_edges[eid] = (min(a, b), max(a, b))
        self.edges = new_edges

        # Redirect loops
        for eid in self.loops:
            if self.loops[eid] == v:
                self.loops[eid] = u

        self.nodes.remove(v)
        return u

    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def num_edges(self) -> int:
        """Number of edges (including loops)."""
        return len(self.edges) + len(self.loops)

    def degree(self, node: int) -> int:
        """Get degree of a node (loops count as 2)."""
        deg = sum(1 for u, v in self.edges.values() if u == node or v == node)
        deg += 2 * sum(1 for n in self.loops.values() if n == node)
        return deg

    def neighbors(self, node: int) -> Set[int]:
        """Get neighbors of a node."""
        result = set()
        for u, v in self.edges.values():
            if u == node:
                result.add(v)
            elif v == node:
                result.add(u)
        return result

    def copy(self) -> 'MutableGraph':
        """Create a deep copy."""
        g = MutableGraph()
        g.nodes = set(self.nodes)
        g.edges = dict(self.edges)
        g.loops = dict(self.loops)
        g._next_node_id = self._next_node_id
        g._next_edge_id = self._next_edge_id
        return g

    def freeze(self) -> Graph:
        """Convert to immutable Graph.

        Note: Loops are not included in the frozen representation
        as the immutable Graph class does not support them.
        """
        return Graph(
            nodes=frozenset(self.nodes),
            edges=frozenset(self.edges.values())
        )

    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges.values())
        for node in self.loops.values():
            G.add_edge(node, node)
        return G

    @classmethod
    def from_networkx(cls, G: nx.Graph) -> 'MutableGraph':
        """Create from NetworkX graph."""
        g = cls()
        node_map = {}
        for n in G.nodes():
            node_map[n] = g.add_node()
        for u, v in G.edges():
            if u == v:
                g.add_loop(node_map[u])
            else:
                g.add_edge(node_map[u], node_map[v])
        return g

    @classmethod
    def from_graph(cls, graph: Graph) -> 'MutableGraph':
        """Create from immutable Graph."""
        g = cls()
        node_map = {}
        for n in graph.nodes:
            node_map[n] = g.add_node()
        for u, v in graph.edges:
            g.add_edge(node_map[u], node_map[v])
        return g


# =============================================================================
# GRAPH BUILDERS FOR COMMON STRUCTURES
# =============================================================================

def complete_graph(n: int) -> Graph:
    """Create complete graph K_n."""
    if n <= 0:
        return Graph(nodes=frozenset(), edges=frozenset())
    nodes = frozenset(range(n))
    edges = frozenset(
        (i, j) for i in range(n) for j in range(i + 1, n)
    )
    return Graph(nodes=nodes, edges=edges)


def cycle_graph(n: int) -> Graph:
    """Create cycle graph C_n."""
    if n <= 0:
        return Graph(nodes=frozenset(), edges=frozenset())
    if n == 1:
        return Graph(nodes=frozenset([0]), edges=frozenset())
    if n == 2:
        return Graph(nodes=frozenset([0, 1]), edges=frozenset([(0, 1)]))
    nodes = frozenset(range(n))
    edges = frozenset(
        (i, (i + 1) % n) if i < (i + 1) % n else ((i + 1) % n, i)
        for i in range(n)
    )
    return Graph(nodes=nodes, edges=edges)


def path_graph(n: int) -> Graph:
    """Create path graph P_n with n nodes."""
    if n <= 0:
        return Graph(nodes=frozenset(), edges=frozenset())
    if n == 1:
        return Graph(nodes=frozenset([0]), edges=frozenset())
    nodes = frozenset(range(n))
    edges = frozenset((i, i + 1) for i in range(n - 1))
    return Graph(nodes=nodes, edges=edges)


def star_graph(n: int) -> Graph:
    """Create star graph S_n (center + n leaves)."""
    if n <= 0:
        return Graph(nodes=frozenset([0]), edges=frozenset())
    nodes = frozenset(range(n + 1))
    edges = frozenset((0, i) for i in range(1, n + 1))
    return Graph(nodes=nodes, edges=edges)


def wheel_graph(n: int) -> Graph:
    """Create wheel graph W_n (center + n-cycle rim)."""
    if n < 3:
        raise ValueError("Wheel graph requires at least 3 rim nodes")
    nodes = frozenset(range(n + 1))  # 0 is center, 1..n are rim
    # Rim edges
    rim_edges = frozenset(
        (i, i + 1) if i < n else (1, n)
        for i in range(1, n + 1)
    )
    # Spoke edges
    spoke_edges = frozenset((0, i) for i in range(1, n + 1))
    return Graph(nodes=nodes, edges=rim_edges | spoke_edges)


def grid_graph(m: int, n: int) -> Graph:
    """Create m x n grid graph."""
    if m <= 0 or n <= 0:
        return Graph(nodes=frozenset(), edges=frozenset())

    def node_id(i, j):
        return i * n + j

    nodes = frozenset(node_id(i, j) for i in range(m) for j in range(n))
    edges = set()

    # Horizontal edges
    for i in range(m):
        for j in range(n - 1):
            edges.add((node_id(i, j), node_id(i, j + 1)))

    # Vertical edges
    for i in range(m - 1):
        for j in range(n):
            edges.add((node_id(i, j), node_id(i + 1, j)))

    return Graph(nodes=nodes, edges=frozenset(edges))


def petersen_graph() -> Graph:
    """Create the Petersen graph."""
    G = nx.petersen_graph()
    return Graph.from_networkx(G)


def disjoint_union(g1: Graph, g2: Graph) -> Graph:
    """Create disjoint union of two graphs.

    Nodes in g2 are relabeled to avoid collision.
    T(G1 ∪ G2) = T(G1) × T(G2)
    """
    # Relabel g2 nodes
    offset = max(g1.nodes) + 1 if g1.nodes else 0
    g2_nodes = frozenset(n + offset for n in g2.nodes)
    g2_edges = frozenset((u + offset, v + offset) for u, v in g2.edges)

    return Graph(
        nodes=g1.nodes | g2_nodes,
        edges=g1.edges | g2_edges
    )


def cut_vertex_join(g1: Graph, v1: int, g2: Graph, v2: int) -> Graph:
    """Join two graphs at a cut vertex (1-sum).

    Identifies vertex v1 in g1 with vertex v2 in g2.
    T(G) = T(G1) × T(G2) when the shared vertex is a cut vertex.
    """
    if v1 not in g1.nodes or v2 not in g2.nodes:
        raise ValueError("Vertices must exist in their respective graphs")

    # Relabel g2 nodes, mapping v2 to v1
    offset = max(g1.nodes) + 1 if g1.nodes else 0
    g2_node_map = {v2: v1}
    for n in g2.nodes:
        if n != v2:
            g2_node_map[n] = n + offset

    g2_nodes = frozenset(g2_node_map[n] for n in g2.nodes)
    g2_edges = frozenset(
        (min(g2_node_map[u], g2_node_map[v]), max(g2_node_map[u], g2_node_map[v]))
        for u, v in g2.edges
    )

    return Graph(
        nodes=g1.nodes | g2_nodes,
        edges=g1.edges | g2_edges
    )
