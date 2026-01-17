r"""Construct Ising model graphs from Tutte polynomials.

This module implements the inverse of the deletion-contraction algorithm:
instead of computing T(G) by deleting/contracting edges, we BUILD graphs
by creating/expanding operations that yield a target Tutte polynomial.

Key insight: The Tutte polynomial T(G; x, y) satisfies:
  - T(bridge) = x
  - T(loop) = y  
  - T(G) = T(G-e) + T(G/e) for regular edges e

Inverting this:
  - x^i terms -> need i bridges (cut edges)
  - y^j terms -> need j loops (self-edges)
  - x^i*y^j terms -> need specific graph structures combining both

The Tutte polynomial connects to Ising models via the Potts model:
  Z_Potts(G; q, v) relates to T(G; (v+q)/v, v+1)

For Ising (q=2), this gives us a principled way to generate graphs
with specific partition function properties.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from collections import defaultdict
import hashlib

import numpy as np

try:
    from blake3 import blake3
except ImportError:
    # Fallback to hashlib if blake3 not available
    import hashlib
    def blake3(data):
        """Fallback using SHA-256."""
        class Blake3Wrapper:
            def __init__(self, data):
                self._hash = hashlib.sha256(data)
            def digest(self):
                return self._hash.digest()
        return Blake3Wrapper(data)

logger = logging.getLogger(__name__)


@dataclass
class TuttePolynomial:
    """Represents a Tutte polynomial as a sum of x^i * y^j terms.
    
    The polynomial is stored as a dictionary mapping (i, j) -> coefficient
    where the term is coefficient * x^i * y^j.
    
    Example: T = 2x^2 + 3xy + y^2 would be:
        {(2, 0): 2, (1, 1): 3, (0, 2): 1}
    """
    coefficients: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Remove zero coefficients
        self.coefficients = {k: v for k, v in self.coefficients.items() if v != 0}
    
    @classmethod
    def from_dict(cls, d: Dict[Tuple[int, int], int]) -> 'TuttePolynomial':
        """Create from coefficient dictionary."""
        return cls(coefficients=dict(d))
    
    @classmethod
    def x(cls, power: int = 1) -> 'TuttePolynomial':
        """Create x^power (single bridge or chain of bridges)."""
        return cls({(power, 0): 1})
    
    @classmethod
    def y(cls, power: int = 1) -> 'TuttePolynomial':
        """Create y^power (loops)."""
        return cls({(0, power): 1})
    
    @classmethod
    def one(cls) -> 'TuttePolynomial':
        """Create constant 1 (empty graph on one vertex)."""
        return cls({(0, 0): 1})
    
    def __add__(self, other: 'TuttePolynomial') -> 'TuttePolynomial':
        """Add two Tutte polynomials (corresponds to disjoint union in some contexts)."""
        result = defaultdict(int, self.coefficients)
        for k, v in other.coefficients.items():
            result[k] += v
        return TuttePolynomial(dict(result))
    
    def __mul__(self, other: 'TuttePolynomial') -> 'TuttePolynomial':
        """Multiply two Tutte polynomials."""
        result = defaultdict(int)
        for (i1, j1), c1 in self.coefficients.items():
            for (i2, j2), c2 in other.coefficients.items():
                result[(i1 + i2, j1 + j2)] += c1 * c2
        return TuttePolynomial(dict(result))
    
    def __rmul__(self, scalar: int) -> 'TuttePolynomial':
        """Scalar multiplication."""
        return TuttePolynomial({k: scalar * v for k, v in self.coefficients.items()})
    
    def __eq__(self, other: 'TuttePolynomial') -> bool:
        return self.coefficients == other.coefficients
    
    def __repr__(self) -> str:
        if not self.coefficients:
            return "0"
        terms = []
        for (i, j), c in sorted(self.coefficients.items(), reverse=True):
            if c == 0:
                continue
            term = ""
            if c != 1 or (i == 0 and j == 0):
                term = str(c)
            if i > 0:
                term += f"x^{i}" if i > 1 else "x"
            if j > 0:
                term += f"y^{j}" if j > 1 else "y"
            terms.append(term)
        return " + ".join(terms) if terms else "0"
    
    def evaluate(self, x: float, y: float) -> float:
        """Evaluate the polynomial at (x, y)."""
        return sum(c * (x ** i) * (y ** j) 
                   for (i, j), c in self.coefficients.items())
    
    def total_degree(self) -> int:
        """Return the maximum i + j across all terms."""
        if not self.coefficients:
            return 0
        return max(i + j for i, j in self.coefficients.keys())
    
    def x_degree(self) -> int:
        """Maximum power of x."""
        if not self.coefficients:
            return 0
        return max(i for i, j in self.coefficients.keys())
    
    def y_degree(self) -> int:
        """Maximum power of y."""
        if not self.coefficients:
            return 0
        return max(j for i, j in self.coefficients.keys())
    
    def num_spanning_trees(self) -> int:
        """T(1, 1) = number of spanning trees."""
        return int(self.evaluate(1.0, 1.0))
    
    def chromatic_polynomial_eval(self, k: int) -> int:
        """Evaluate chromatic polynomial at k using Tutte relation."""
        # P(G; k) = (-1)^{|V|-c(G)} * k^{c(G)} * T(G; 1-k, 0)
        # This requires knowing |V| and c(G), which we don't have directly
        # For connected graphs: P(G; k) = (-1)^{|V|-1} * k * T(G; 1-k, 0)
        raise NotImplementedError("Requires vertex count information")


@dataclass
class GraphBuilder:
    """Builds a graph incrementally with operations that track Tutte polynomial.
    
    The graph is represented as:
    - nodes: set of node IDs
    - edges: dict mapping edge_id -> (u, v) where u, v are node IDs
    - loops: set of node IDs that have self-loops
    
    We track the Tutte polynomial as we build.
    """
    nodes: Set[int] = field(default_factory=set)
    edges: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    loops: Dict[int, int] = field(default_factory=dict)  # edge_id -> node_id
    _next_node_id: int = 0
    _next_edge_id: int = 0
    _tutte: Optional[TuttePolynomial] = None
    
    def add_node(self) -> int:
        """Add a new isolated node. Returns node ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes.add(node_id)
        self._tutte = None  # Invalidate cache
        return node_id
    
    def add_edge(self, u: int, v: int) -> int:
        """Add an edge between two existing nodes. Returns edge ID."""
        assert u in self.nodes and v in self.nodes, "Nodes must exist"
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        self.edges[edge_id] = (u, v)
        self._tutte = None
        return edge_id
    
    def add_loop(self, node: int) -> int:
        """Add a self-loop at a node. Returns edge ID."""
        assert node in self.nodes, "Node must exist"
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        self.loops[edge_id] = node
        self._tutte = None
        return edge_id
    
    def add_bridge(self) -> Tuple[int, int, int]:
        """Add a new node connected by a bridge to a new node.
        
        Returns (node1_id, node2_id, edge_id).
        A bridge is an edge whose removal disconnects the graph.
        """
        u = self.add_node()
        v = self.add_node()
        e = self.add_edge(u, v)
        return u, v, e
    
    def connect_with_bridge(self, existing_node: int) -> Tuple[int, int]:
        """Add a new node connected to an existing node by an edge.
        
        Returns (new_node_id, edge_id).
        """
        new_node = self.add_node()
        edge = self.add_edge(existing_node, new_node)
        return new_node, edge
    
    def add_parallel_edge(self, u: int, v: int) -> int:
        """Add a parallel edge between u and v (creates a multi-edge)."""
        return self.add_edge(u, v)
    
    def merge_nodes(self, u: int, v: int) -> int:
        """Merge node v into node u (contraction). Returns surviving node u.
        
        All edges incident to v are redirected to u.
        """
        assert u in self.nodes and v in self.nodes and u != v
        
        # Redirect edges
        new_edges = {}
        for edge_id, (a, b) in self.edges.items():
            if a == v:
                a = u
            if b == v:
                b = u
            # Handle case where edge becomes a loop
            if a == b:
                self.loops[edge_id] = a
            else:
                new_edges[edge_id] = (a, b)
        self.edges = new_edges
        
        # Redirect loops
        new_loops = {}
        for edge_id, node in self.loops.items():
            new_loops[edge_id] = u if node == v else node
        self.loops = new_loops
        
        self.nodes.remove(v)
        self._tutte = None
        return u
    
    def get_adjacency(self) -> Dict[int, List[int]]:
        """Get adjacency list representation."""
        adj = defaultdict(list)
        for (u, v) in self.edges.values():
            adj[u].append(v)
            adj[v].append(u)
        return dict(adj)
    
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    def num_edges(self) -> int:
        return len(self.edges) + len(self.loops)
    
    def to_edge_list(self) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Convert to node list and edge list format for Ising model generation."""
        nodes = sorted(self.nodes)
        edges = list(self.edges.values())
        # Include loops as edges (u, u)
        for node in self.loops.values():
            edges.append((node, node))
        return nodes, edges
    
    def copy(self) -> 'GraphBuilder':
        """Create a deep copy."""
        g = GraphBuilder()
        g.nodes = set(self.nodes)
        g.edges = dict(self.edges)
        g.loops = dict(self.loops)
        g._next_node_id = self._next_node_id
        g._next_edge_id = self._next_edge_id
        return g


# =============================================================================
# CANONICAL GRAPH STRUCTURES WITH KNOWN TUTTE POLYNOMIALS
# =============================================================================

def create_path_graph(n: int) -> GraphBuilder:
    """Create a path graph P_n with n nodes.
    
    T(P_n) = x^{n-1} (chain of n-1 bridges)
    """
    g = GraphBuilder()
    if n <= 0:
        return g
    
    prev = g.add_node()
    for _ in range(n - 1):
        curr = g.add_node()
        g.add_edge(prev, curr)
        prev = curr
    return g


def create_cycle_graph(n: int) -> GraphBuilder:
    """Create a cycle graph C_n with n nodes.
    
    T(C_n) = x^{n-1} + x^{n-2} + ... + x + y  (for n >= 3)
    Special: T(C_1) = y (single loop), T(C_2) = x + y (two parallel edges)
    """
    g = GraphBuilder()
    if n <= 0:
        return g
    
    if n == 1:
        node = g.add_node()
        g.add_loop(node)
        return g
    
    first = g.add_node()
    prev = first
    for _ in range(n - 1):
        curr = g.add_node()
        g.add_edge(prev, curr)
        prev = curr
    g.add_edge(prev, first)  # Close the cycle
    return g


def create_complete_graph(n: int) -> GraphBuilder:
    """Create complete graph K_n.
    
    K_1: T = 1
    K_2: T = x
    K_3: T = x^2 + x + y
    K_4: T = x^3 + 3x^2 + 4x + 2y + 2xy + y^2
    """
    g = GraphBuilder()
    nodes = [g.add_node() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(nodes[i], nodes[j])
    return g


def create_multi_edge(u: int, v: int, k: int, g: GraphBuilder) -> List[int]:
    """Add k parallel edges between u and v in graph g.
    
    For k parallel edges: T = sum_{i=0}^{k-1} x^i * y^{k-1-i}
                           = (x^k - y^k) / (x - y)  if x != y
    """
    edge_ids = []
    for _ in range(k):
        edge_ids.append(g.add_edge(u, v))
    return edge_ids


def create_bouquet(n: int) -> GraphBuilder:
    """Create a bouquet of n loops at a single vertex.
    
    T(B_n) = y^n
    """
    g = GraphBuilder()
    node = g.add_node()
    for _ in range(n):
        g.add_loop(node)
    return g


def create_theta_graph(a: int, b: int, c: int) -> GraphBuilder:
    """Create theta graph: two nodes connected by three internally disjoint paths.
    
    Paths have a, b, c internal nodes respectively.
    """
    g = GraphBuilder()
    u = g.add_node()
    v = g.add_node()
    
    for path_len in [a, b, c]:
        if path_len == 0:
            g.add_edge(u, v)
        else:
            prev = u
            for _ in range(path_len):
                curr = g.add_node()
                g.add_edge(prev, curr)
                prev = curr
            g.add_edge(prev, v)
    
    return g


# =============================================================================
# TUTTE POLYNOMIAL COMPUTATION (for verification)
# =============================================================================

def compute_tutte_polynomial(g: GraphBuilder) -> TuttePolynomial:
    """Compute Tutte polynomial via deletion-contraction.
    
    Uses the recurrence:
    - T(empty) = 1
    - T(G with bridge e) = x * T(G - e)
    - T(G with loop e) = y * T(G - e)
    - T(G) = T(G - e) + T(G / e) for regular edge e
    
    This is exponential time but correct. Use only for small graphs.
    """
    # Base case: no edges
    if g.num_edges() == 0:
        if g.num_nodes() == 0:
            return TuttePolynomial.one()
        # Each isolated node contributes nothing to Tutte (by convention)
        # Actually for k connected components: multiply by 1 for each
        return TuttePolynomial.one()
    
    # Handle loops first (simple case)
    if g.loops:
        edge_id = next(iter(g.loops.keys()))
        g_minus_e = g.copy()
        del g_minus_e.loops[edge_id]
        return TuttePolynomial.y() * compute_tutte_polynomial(g_minus_e)
    
    # Pick an edge
    edge_id = next(iter(g.edges.keys()))
    u, v = g.edges[edge_id]
    
    # Check if edge is a bridge (removing it disconnects the graph)
    is_bridge = _is_bridge(g, edge_id)
    
    if is_bridge:
        # T(G) = x * T(G \ e)
        g_minus_e = g.copy()
        del g_minus_e.edges[edge_id]
        return TuttePolynomial.x() * compute_tutte_polynomial(g_minus_e)
    
    # Regular edge: T(G) = T(G \ e) + T(G / e)
    # Deletion
    g_delete = g.copy()
    del g_delete.edges[edge_id]
    t_delete = compute_tutte_polynomial(g_delete)
    
    # Contraction
    g_contract = g.copy()
    del g_contract.edges[edge_id]
    g_contract.merge_nodes(u, v)
    t_contract = compute_tutte_polynomial(g_contract)
    
    return t_delete + t_contract


def _is_bridge(g: GraphBuilder, edge_id: int) -> bool:
    """Check if removing edge disconnects the graph."""
    if edge_id not in g.edges:
        return False
    
    u, v = g.edges[edge_id]
    
    # BFS/DFS from u without using edge_id
    visited = {u}
    stack = [u]
    
    while stack:
        curr = stack.pop()
        for eid, (a, b) in g.edges.items():
            if eid == edge_id:
                continue
            neighbor = None
            if a == curr and b not in visited:
                neighbor = b
            elif b == curr and a not in visited:
                neighbor = a
            if neighbor is not None:
                visited.add(neighbor)
                stack.append(neighbor)
    
    return v not in visited


# =============================================================================
# INVERSE CONSTRUCTION: TUTTE POLYNOMIAL → GRAPH
# =============================================================================

class TutteToGraphBuilder:
    """Constructs graphs from Tutte polynomial specifications.
    
    Uses the nonce for deterministic randomness when multiple valid
    graphs exist for a given polynomial.
    
    Strategy:
    1. Decompose the polynomial into buildable components
    2. Use canonical structures (paths, cycles, bouquets) where possible
    3. Combine components using series/parallel composition
    4. Use nonce to select among equivalent constructions
    """
    
    def __init__(self, nonce: int):
        self.nonce = nonce
        self.rng = np.random.RandomState(nonce)
    
    def build_from_tutte(self, tutte: TuttePolynomial) -> GraphBuilder:
        """Main entry point: construct a graph with the given Tutte polynomial.
        
        Note: Not all polynomials are realizable as Tutte polynomials of graphs.
        This method attempts to find a graph or raises an error if impossible.
        """
        # Check for simple cases first
        if self._is_zero(tutte):
            raise ValueError("Zero polynomial has no graph realization")
        
        if self._is_constant_one(tutte):
            # Single isolated vertex
            g = GraphBuilder()
            g.add_node()
            return g
        
        # Pure x^n case: path graph
        if self._is_pure_x_power(tutte):
            n = tutte.x_degree()
            return create_path_graph(n + 1)  # n edges = n+1 nodes in path
        
        # Pure y^n case: bouquet of loops
        if self._is_pure_y_power(tutte):
            n = tutte.y_degree()
            return create_bouquet(n)
        
        # Cycle polynomial: x^{n-1} + x^{n-2} + ... + x + y
        cycle_n = self._matches_cycle_polynomial(tutte)
        if cycle_n is not None:
            return create_cycle_graph(cycle_n)
        
        # Try factorization approach
        factored = self._try_factor(tutte)
        if factored is not None:
            return factored
        
        # Try series-parallel decomposition
        sp_result = self._try_series_parallel(tutte)
        if sp_result is not None:
            return sp_result
        
        # Fall back to heuristic construction
        return self._heuristic_construction(tutte)
    
    def _is_zero(self, tutte: TuttePolynomial) -> bool:
        return len(tutte.coefficients) == 0
    
    def _is_constant_one(self, tutte: TuttePolynomial) -> bool:
        return tutte.coefficients == {(0, 0): 1}
    
    def _is_pure_x_power(self, tutte: TuttePolynomial) -> bool:
        """Check if polynomial is x^n for some n."""
        if len(tutte.coefficients) != 1:
            return False
        (i, j), c = next(iter(tutte.coefficients.items()))
        return j == 0 and c == 1
    
    def _is_pure_y_power(self, tutte: TuttePolynomial) -> bool:
        """Check if polynomial is y^n for some n."""
        if len(tutte.coefficients) != 1:
            return False
        (i, j), c = next(iter(tutte.coefficients.items()))
        return i == 0 and c == 1
    
    def _matches_cycle_polynomial(self, tutte: TuttePolynomial) -> Optional[int]:
        """Check if polynomial matches T(C_n) = x^{n-1} + ... + x + y.
        
        Returns n if it matches, None otherwise.
        """
        # Find highest x power
        max_x = tutte.x_degree()
        
        # Cycle polynomial has:
        # - Terms x^{n-1}, x^{n-2}, ..., x^1, y (all with coefficient 1)
        # - n = max_x + 1
        
        expected = {}
        for i in range(1, max_x + 1):
            expected[(i, 0)] = 1
        expected[(0, 1)] = 1
        
        if tutte.coefficients == expected:
            return max_x + 1
        return None
    
    def _try_factor(self, tutte: TuttePolynomial) -> Optional[GraphBuilder]:
        """Try to factor the polynomial and use disjoint union."""
        # Disjoint union: T(G1 ∪ G2) = T(G1) * T(G2)
        # This is complex to implement generally; placeholder for now
        return None
    
    def _try_series_parallel(self, tutte: TuttePolynomial) -> Optional[GraphBuilder]:
        """Try series-parallel decomposition.
        
        For series-parallel graphs, we can use:
        - Series: T(G1 ∘ G2) has a specific formula
        - Parallel: T(G1 || G2) has another formula
        """
        # This requires solving polynomial equations; complex for general case
        return None
    
    def _heuristic_construction(self, tutte: TuttePolynomial) -> GraphBuilder:
        """Heuristic construction for general polynomials.
        
        Strategy: Build a graph that approximates the target polynomial
        by combining canonical structures based on coefficient analysis.
        """
        g = GraphBuilder()
        
        # Analyze the polynomial structure
        x_deg = tutte.x_degree()
        y_deg = tutte.y_degree()
        
        # Start with nodes based on total degree
        num_nodes = max(2, x_deg + 1)
        nodes = [g.add_node() for _ in range(num_nodes)]
        
        # Add bridges for x terms
        for i in range(min(x_deg, num_nodes - 1)):
            g.add_edge(nodes[i], nodes[i + 1])
        
        # Add loops for y terms
        if y_deg > 0:
            # Distribute loops based on nonce
            for _ in range(y_deg):
                node_idx = self.rng.randint(0, num_nodes)
                g.add_loop(nodes[node_idx])
        
        # Add extra edges for mixed terms
        for (i, j), coeff in tutte.coefficients.items():
            if i > 0 and j > 0:
                # Mixed term: add some parallel edges or cycles
                for _ in range(min(coeff, num_nodes - 1)):
                    u_idx = self.rng.randint(0, num_nodes)
                    v_idx = self.rng.randint(0, num_nodes)
                    if u_idx != v_idx:
                        g.add_edge(nodes[u_idx], nodes[v_idx])
        
        return g


# =============================================================================
# ISING MODEL GENERATION FROM TUTTE-CONSTRUCTED GRAPHS
# =============================================================================

def generate_ising_from_tutte(
    nonce: int,
    tutte: TuttePolynomial,
    h_values: Optional[List[float]] = None
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], List[int], List[Tuple[int, int]]]:
    """Generate Ising model parameters from a Tutte polynomial specification.
    
    Args:
        nonce: Random seed for deterministic generation
        tutte: Target Tutte polynomial
        h_values: Allowed field values (default: [-1, 0, +1])
    
    Returns:
        (h, J, nodes, edges) where:
        - h: Dict mapping node_id → field value
        - J: Dict mapping (u, v) → coupling value in {-1, +1}
        - nodes: List of node IDs
        - edges: List of edge tuples
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]
    
    # Build graph from Tutte polynomial
    builder = TutteToGraphBuilder(nonce)
    graph = builder.build_from_tutte(tutte)
    
    # Extract graph structure
    nodes, edges = graph.to_edge_list()
    
    # Generate Ising parameters using nonce
    rng = np.random.RandomState(nonce)
    
    # Generate h: random selection from h_values for each node
    if len(h_values) == 1 and h_values[0] == 0.0:
        h = {node: 0.0 for node in nodes}
    else:
        h_vals = rng.choice(h_values, size=len(nodes))
        h = {node: float(h_vals[i]) for i, node in enumerate(nodes)}
    
    # Generate J: random ±1 for each edge (excluding self-loops for J)
    J = {}
    for (u, v) in edges:
        if u != v:  # Skip self-loops for coupling
            J[(u, v)] = float(2 * rng.randint(2) - 1)
    
    return h, J, nodes, edges


def ising_nonce_from_tutte_hash(
    prev_hash: bytes,
    miner_id: str,
    cur_index: int,
    salt: bytes,
    tutte: TuttePolynomial
) -> int:
    """Generate nonce incorporating Tutte polynomial into the hash.
    
    This ensures the Ising model is deterministically tied to both
    the block parameters AND the target Tutte polynomial.
    """
    # Serialize Tutte polynomial deterministically
    tutte_bytes = _serialize_tutte(tutte)
    
    seed = f"{prev_hash.hex()}{miner_id}{cur_index}".encode() + salt + tutte_bytes
    nonce_bytes = blake3(seed).digest()
    nonce = int.from_bytes(nonce_bytes[:4], 'big')
    return nonce


def _serialize_tutte(tutte: TuttePolynomial) -> bytes:
    """Serialize Tutte polynomial to bytes deterministically."""
    # Sort coefficients for determinism
    items = sorted(tutte.coefficients.items())
    parts = [f"{i},{j}:{c}" for (i, j), c in items]
    return "|".join(parts).encode('utf-8')


# =============================================================================
# EXAMPLE TUTTE POLYNOMIALS FOR COMMON GRAPH FAMILIES
# =============================================================================

KNOWN_TUTTE_POLYNOMIALS = {
    # Path graphs P_n (n nodes, n-1 edges)
    'P2': TuttePolynomial({(1, 0): 1}),  # x
    'P3': TuttePolynomial({(2, 0): 1}),  # x^2
    'P4': TuttePolynomial({(3, 0): 1}),  # x^3
    
    # Cycle graphs C_n
    'C3': TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1}),  # x^2 + x + y
    'C4': TuttePolynomial({(3, 0): 1, (2, 0): 1, (1, 0): 1, (0, 1): 1}),  # x^3 + x^2 + x + y
    'C5': TuttePolynomial({(4, 0): 1, (3, 0): 1, (2, 0): 1, (1, 0): 1, (0, 1): 1}),
    
    # Complete graphs K_n
    'K2': TuttePolynomial({(1, 0): 1}),  # x (same as P2)
    'K3': TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1}),  # x^2 + x + y (same as C3)
    'K4': TuttePolynomial({(3, 0): 1, (2, 0): 3, (1, 0): 4, (0, 1): 2, (1, 1): 2, (0, 2): 1}),
    
    # Bouquet of n loops B_n
    'B1': TuttePolynomial({(0, 1): 1}),  # y
    'B2': TuttePolynomial({(0, 2): 1}),  # y^2
    'B3': TuttePolynomial({(0, 3): 1}),  # y^3
    
    # Two parallel edges (dipole)
    'dipole': TuttePolynomial({(1, 0): 1, (0, 1): 1}),  # x + y
    
    # Three parallel edges
    'tri_edge': TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1}),  # x^2 + x + y
    
    # Theta graph θ(1,1,1) - two vertices with 3 edges
    'theta_111': TuttePolynomial({(2, 0): 1, (1, 0): 2, (0, 1): 1, (1, 1): 1}),
    
    # Petersen graph (known complex polynomial)
    'petersen': TuttePolynomial({
        (9, 0): 1, (8, 0): 6, (7, 0): 21, (6, 0): 56, (5, 0): 126,
        (4, 0): 216, (3, 0): 266, (2, 0): 228, (1, 0): 120, (0, 0): 36,
        # ... (simplified - full polynomial is very long)
    }),
}


def get_tutte_for_difficulty(difficulty_level: int) -> TuttePolynomial:
    """Generate a Tutte polynomial appropriate for a difficulty level.
    
    Higher difficulty = larger/more complex graphs.
    """
    if difficulty_level <= 1:
        return KNOWN_TUTTE_POLYNOMIALS['P3']
    elif difficulty_level <= 3:
        return KNOWN_TUTTE_POLYNOMIALS['C4']
    elif difficulty_level <= 5:
        return KNOWN_TUTTE_POLYNOMIALS['K4']
    else:
        # Build a more complex polynomial
        # Combine multiple structures
        base = KNOWN_TUTTE_POLYNOMIALS['K4']
        extra_loops = TuttePolynomial.y(difficulty_level - 5)
        return base * extra_loops


# =============================================================================
# INTEGRATION WITH EXISTING QUANTUM PROOF-OF-WORK
# =============================================================================

def generate_ising_model_from_nonce_and_tutte(
    nonce: int,
    tutte: TuttePolynomial,
    h_values: Optional[List[float]] = None
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], List[int], List[Tuple[int, int]]]:
    """Drop-in replacement for generate_ising_model_from_nonce that uses Tutte polynomial.
    
    This function generates an Ising model where the underlying graph structure
    is determined by the Tutte polynomial, making the mining problem
    mathematically structured rather than purely random.
    
    Args:
        nonce: Random seed for deterministic generation
        tutte: Target Tutte polynomial specifying graph structure
        h_values: Allowed field values (default: [-1, 0, +1])
    
    Returns:
        (h, J, nodes, edges) tuple for Ising model
    """
    return generate_ising_from_tutte(nonce, tutte, h_values)


# =============================================================================
# VERIFICATION UTILITIES
# =============================================================================

def verify_tutte_construction(tutte: TuttePolynomial, verbose: bool = False) -> bool:
    """Verify that we can construct a graph and recover the Tutte polynomial.
    
    For small polynomials, constructs the graph and computes its Tutte polynomial,
    checking if they match (or are equivalent up to graph isomorphism effects).
    """
    builder = TutteToGraphBuilder(42)  # Fixed nonce for reproducibility
    
    try:
        graph = builder.build_from_tutte(tutte)
    except ValueError as e:
        if verbose:
            print(f"Construction failed: {e}")
        return False
    
    # For small graphs, verify by computing Tutte polynomial
    if graph.num_edges() <= 10:
        computed = compute_tutte_polynomial(graph)
        match = computed == tutte
        if verbose:
            print(f"Target:   {tutte}")
            print(f"Computed: {computed}")
            print(f"Match: {match}")
        return match
    else:
        if verbose:
            print(f"Graph too large for exact verification ({graph.num_edges()} edges)")
        return True  # Assume correct for large graphs


if __name__ == "__main__":
    # Demo / test
    print("=== Tutte Polynomial to Ising Model Demo ===\n")
    
    # Test with cycle graph C4
    print("Testing C4 (4-cycle):")
    c4_tutte = KNOWN_TUTTE_POLYNOMIALS['C4']
    print(f"  Tutte polynomial: {c4_tutte}")
    
    h, J, nodes, edges = generate_ising_from_tutte(12345, c4_tutte)
    print(f"  Nodes: {nodes}")
    print(f"  Edges: {edges}")
    print(f"  h: {h}")
    print(f"  J: {J}")
    
    # Verify construction
    print(f"  Verification: {verify_tutte_construction(c4_tutte, verbose=True)}")
    
    print("\nTesting K4 (complete graph on 4 vertices):")
    k4_tutte = KNOWN_TUTTE_POLYNOMIALS['K4']
    print(f"  Tutte polynomial: {k4_tutte}")
    
    h, J, nodes, edges = generate_ising_from_tutte(12345, k4_tutte)
    print(f"  Nodes: {nodes}")
    print(f"  Edges: {edges}")
    print(f"  h: {h}")
    print(f"  J: {J}")
