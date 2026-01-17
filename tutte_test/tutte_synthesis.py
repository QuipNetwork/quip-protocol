"""
K-Cut and K-Join Operations for Tutte Polynomial Graph Synthesis.

This module implements a framework for:
1. Analyzing the "difference" between target and source Tutte polynomials
2. Inferring k-join operations needed to transform source graphs toward target graphs
3. Building target graphs from smaller motifs using polynomial-guided synthesis

Key Concepts:
- k-cut: Removing k edges/vertices that disconnect graph components
- k-join: The inverse - connecting components to increase connectivity
- Polynomial difference: T(target) - T(source) tells us what structure is missing

Example:
    Target K_3: T(x,y) = x^2 + x + y
    Source K_2: T(x,y) = x

    Two disjoint K_2 graphs: T = x * x = x^2
    Remaining: (x^2 + x + y) - x^2 = x + y

    Interpretation: Need one edge (x) that also closes a cycle (y)
    Solution: Connect the third K_2 to join the two open nodes

Theory:
- Pure x^k terms indicate bridges/cut edges needed
- Pure y^k terms indicate cycles/loops needed
- Mixed x^i*y^j terms indicate specific graph structures
- Coefficients indicate multiplicities or alternative constructions
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Union
from enum import Enum
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutte_test.tutte_utils import (
    TuttePolynomial,
    GraphBuilder,
    compute_tutte_polynomial,
    create_path_graph,
    create_cycle_graph,
    create_complete_graph,
)
from tutte_test.graph_composition import (
    disjoint_union,
    cut_vertex_join,
    parallel_connection,
    two_sum,
    clique_sum,
    identify_vertices,
    find_cut_vertices,
    find_bridges,
    analyze_k_cuts,
    get_edge_connectivity,
    create_edge,
    create_k3,
)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class JoinOperation(Enum):
    """Types of k-join operations for graph synthesis."""
    EDGE_ADD = "edge_add"           # Add single edge between existing nodes
    BRIDGE_ADD = "bridge_add"       # Add edge that creates a bridge
    CYCLE_CLOSE = "cycle_close"     # Add edge that closes a cycle
    VERTEX_JOIN = "vertex_join"     # Identify two vertices (1-join)
    EDGE_JOIN = "edge_join"         # Glue on shared edge (2-join)
    TRIANGLE_JOIN = "triangle_join" # Glue on shared triangle (3-join)
    K_JOIN = "k_join"               # General k-clique join
    PARALLEL_ADD = "parallel_add"   # Add parallel edge to existing edge


@dataclass
class PolynomialTerm:
    """A single term in a Tutte polynomial: coeff * x^i * y^j."""
    x_power: int
    y_power: int
    coefficient: int

    @property
    def is_pure_x(self) -> bool:
        """Is this a pure x^k term?"""
        return self.y_power == 0 and self.x_power > 0

    @property
    def is_pure_y(self) -> bool:
        """Is this a pure y^k term?"""
        return self.x_power == 0 and self.y_power > 0

    @property
    def is_mixed(self) -> bool:
        """Is this a mixed x^i*y^j term?"""
        return self.x_power > 0 and self.y_power > 0

    @property
    def total_degree(self) -> int:
        return self.x_power + self.y_power

    def __repr__(self) -> str:
        if self.coefficient == 0:
            return "0"
        parts = []
        if self.coefficient != 1 or (self.x_power == 0 and self.y_power == 0):
            parts.append(str(self.coefficient))
        if self.x_power > 0:
            parts.append(f"x^{self.x_power}" if self.x_power > 1 else "x")
        if self.y_power > 0:
            parts.append(f"y^{self.y_power}" if self.y_power > 1 else "y")
        return "".join(parts) if parts else str(self.coefficient)


@dataclass
class JoinInstruction:
    """An instruction for a k-join operation."""
    operation: JoinOperation
    k: int  # The 'k' in k-join (number of shared vertices/edges)
    source_nodes: List[int] = field(default_factory=list)  # Nodes to join from
    target_nodes: List[int] = field(default_factory=list)  # Nodes to join to
    polynomial_contribution: Optional[TuttePolynomial] = None
    description: str = ""

    def __repr__(self) -> str:
        return f"{self.operation.value}(k={self.k}): {self.description}"


@dataclass
class SynthesisPlan:
    """A plan for synthesizing a target graph from source components."""
    target_polynomial: TuttePolynomial
    source_components: List[Tuple[GraphBuilder, TuttePolynomial]]
    join_instructions: List[JoinInstruction]
    remaining_polynomial: TuttePolynomial
    is_complete: bool = False

    def __repr__(self) -> str:
        lines = [
            f"Target: {self.target_polynomial}",
            f"Sources: {len(self.source_components)} components",
            f"Instructions: {len(self.join_instructions)}",
            f"Remaining: {self.remaining_polynomial}",
            f"Complete: {self.is_complete}",
        ]
        return "\n".join(lines)


@dataclass
class PolynomialDifference:
    """Analysis of the difference between target and source polynomials."""
    target: TuttePolynomial
    source: TuttePolynomial
    difference: TuttePolynomial

    # Classified terms
    pure_x_terms: List[PolynomialTerm] = field(default_factory=list)
    pure_y_terms: List[PolynomialTerm] = field(default_factory=list)
    mixed_terms: List[PolynomialTerm] = field(default_factory=list)
    negative_terms: List[PolynomialTerm] = field(default_factory=list)

    # Inferred operations
    inferred_joins: List[JoinInstruction] = field(default_factory=list)

    def total_x_degree_needed(self) -> int:
        """Sum of x powers across positive terms."""
        total = 0
        for term in self.pure_x_terms + self.mixed_terms:
            if term.coefficient > 0:
                total += term.x_power * term.coefficient
        return total

    def total_y_degree_needed(self) -> int:
        """Sum of y powers across positive terms."""
        total = 0
        for term in self.pure_y_terms + self.mixed_terms:
            if term.coefficient > 0:
                total += term.y_power * term.coefficient
        return total


# =============================================================================
# POLYNOMIAL ARITHMETIC AND ANALYSIS
# =============================================================================

def subtract_polynomials(p1: TuttePolynomial, p2: TuttePolynomial) -> TuttePolynomial:
    """Compute p1 - p2."""
    result = defaultdict(int, p1.coefficients)
    for k, v in p2.coefficients.items():
        result[k] -= v
    return TuttePolynomial({k: v for k, v in result.items() if v != 0})


def polynomial_to_terms(p: TuttePolynomial) -> List[PolynomialTerm]:
    """Convert polynomial to list of terms."""
    return [
        PolynomialTerm(x_power=i, y_power=j, coefficient=c)
        for (i, j), c in p.coefficients.items()
    ]


def analyze_polynomial_difference(
    target: TuttePolynomial,
    source: TuttePolynomial
) -> PolynomialDifference:
    """
    Analyze the difference between target and source polynomials.

    Classifies terms by type and infers required join operations.
    """
    diff = subtract_polynomials(target, source)
    terms = polynomial_to_terms(diff)

    analysis = PolynomialDifference(
        target=target,
        source=source,
        difference=diff
    )

    # Classify terms
    for term in terms:
        if term.coefficient < 0:
            analysis.negative_terms.append(term)
        elif term.is_pure_x:
            analysis.pure_x_terms.append(term)
        elif term.is_pure_y:
            analysis.pure_y_terms.append(term)
        elif term.is_mixed:
            analysis.mixed_terms.append(term)

    # Infer join operations from terms
    analysis.inferred_joins = infer_joins_from_difference(analysis)

    return analysis


def infer_joins_from_difference(analysis: PolynomialDifference) -> List[JoinInstruction]:
    """
    Infer k-join operations needed based on polynomial difference.

    Interpretation rules:
    - x^k with coeff c: Need c bridges contributing x^k total
    - y^k with coeff c: Need c cycle-closing edges
    - x*y with coeff c: Need c edges that are both bridge and cycle-closer
    - x^i*y^j: Need structure with i bridges and j cycles intertwined
    """
    instructions = []

    # Handle pure x terms (bridges/paths needed)
    for term in analysis.pure_x_terms:
        if term.coefficient > 0:
            # Each x represents a bridge
            # x^k means k bridges in series, or the polynomial of a path
            instructions.append(JoinInstruction(
                operation=JoinOperation.BRIDGE_ADD,
                k=1,
                polynomial_contribution=TuttePolynomial({(term.x_power, 0): 1}),
                description=f"Add {term.coefficient} path(s) of length {term.x_power}"
            ))

    # Handle pure y terms (cycles/loops needed)
    for term in analysis.pure_y_terms:
        if term.coefficient > 0:
            # y^k means k loops or the cycle contribution
            instructions.append(JoinInstruction(
                operation=JoinOperation.CYCLE_CLOSE,
                k=term.y_power,
                polynomial_contribution=TuttePolynomial({(0, term.y_power): 1}),
                description=f"Close {term.coefficient} cycle(s) of degree {term.y_power}"
            ))

    # Handle mixed terms (complex structures)
    for term in analysis.mixed_terms:
        if term.coefficient > 0:
            # x^i*y^j typically comes from graphs with both bridges and cycles
            # The exact structure depends on the specific term
            k_value = min(term.x_power, term.y_power) + 1  # Estimate join complexity
            instructions.append(JoinInstruction(
                operation=JoinOperation.K_JOIN,
                k=k_value,
                polynomial_contribution=TuttePolynomial({(term.x_power, term.y_power): 1}),
                description=f"Add structure for {term}: {term.coefficient} copies"
            ))

    return instructions


# =============================================================================
# K-JOIN OPERATION ALIASES
# =============================================================================

# These are aliases to graph_composition functions with k-join terminology
perform_1_join = cut_vertex_join  # 1-join = cut vertex join
perform_k_join = clique_sum       # k-join = k-clique sum


def perform_2_join(g1: GraphBuilder, e1: int, g2: GraphBuilder, e2: int,
                   delete_shared_edge: bool = True) -> GraphBuilder:
    """2-join: Glue on edge. If delete_shared_edge=True, uses 2-sum; else parallel connection."""
    return two_sum(g1, e1, g2, e2) if delete_shared_edge else parallel_connection(g1, e1, g2, e2)


def perform_3_join(g1: GraphBuilder, tri1: List[int], g2: GraphBuilder, tri2: List[int],
                   delete_triangle: bool = True) -> GraphBuilder:
    """3-join: Glue on triangle (3-clique sum)."""
    return clique_sum(g1, tri1, g2, tri2, delete_clique_edges=delete_triangle)


def add_edge_between_nodes(g: GraphBuilder, u: int, v: int) -> Tuple[GraphBuilder, int]:
    """
    Add a single edge between existing nodes.

    Returns the modified graph and the new edge ID.
    """
    result = g.copy()
    edge_id = result.add_edge(u, v)
    return result, edge_id


def close_cycle(g: GraphBuilder, u: int, v: int) -> Tuple[GraphBuilder, TuttePolynomial]:
    """
    Add an edge that closes a cycle (u and v must already be path-connected).

    Returns the modified graph and its new Tutte polynomial.
    """
    result = g.copy()
    result.add_edge(u, v)
    new_tutte = compute_tutte_polynomial(result)
    return result, new_tutte


# =============================================================================
# GRAPH SYNTHESIS FROM POLYNOMIAL SPECIFICATION
# =============================================================================

def find_open_nodes(g: GraphBuilder) -> List[int]:
    """
    Find nodes with degree < 2 that could be endpoints for joining.

    "Open" nodes are good candidates for k-join operations.
    """
    degree = defaultdict(int)
    for (u, v) in g.edges.values():
        degree[u] += 1
        degree[v] += 1

    return [n for n in g.nodes if degree[n] < 2]


def find_joinable_pairs(g1: GraphBuilder, g2: GraphBuilder) -> List[Tuple[int, int]]:
    """
    Find pairs of nodes (one from each graph) suitable for joining.

    Returns list of (v1, v2) pairs where v1 ∈ g1, v2 ∈ g2.
    """
    open1 = find_open_nodes(g1)
    open2 = find_open_nodes(g2)

    # If no open nodes, use any nodes
    if not open1:
        open1 = list(g1.nodes)[:2] if len(g1.nodes) >= 2 else list(g1.nodes)
    if not open2:
        open2 = list(g2.nodes)[:2] if len(g2.nodes) >= 2 else list(g2.nodes)

    return [(v1, v2) for v1 in open1 for v2 in open2]


def synthesize_from_k2_graphs(
    target_polynomial: TuttePolynomial,
    max_k2_count: int = 10
) -> Optional[SynthesisPlan]:
    """
    Attempt to synthesize a graph with target polynomial using K_2 building blocks.

    This implements the user's example:
    - Start with K_2 graphs (each has T = x)
    - Combine using disjoint union (multiplication) and joins
    - Track remaining polynomial to guide next steps

    Args:
        target_polynomial: The Tutte polynomial to achieve
        max_k2_count: Maximum number of K_2 graphs to use

    Returns:
        SynthesisPlan with instructions, or None if synthesis fails
    """
    # Start with empty plan
    plan = SynthesisPlan(
        target_polynomial=target_polynomial,
        source_components=[],
        join_instructions=[],
        remaining_polynomial=target_polynomial,
        is_complete=False
    )

    # Create K_2 component
    k2 = create_edge()
    k2_tutte = TuttePolynomial({(1, 0): 1})  # x

    # Build up using K_2 graphs
    current_graph = None
    current_tutte = TuttePolynomial({(0, 0): 1})  # Start with "1" (empty)

    for i in range(max_k2_count):
        # Check if we've reached the target
        if current_tutte == target_polynomial:
            plan.is_complete = True
            break

        # Analyze what's still needed
        diff = analyze_polynomial_difference(target_polynomial, current_tutte)

        # If no positive terms remain, we're done or stuck
        if not (diff.pure_x_terms or diff.pure_y_terms or diff.mixed_terms):
            if diff.negative_terms:
                # We've overshot - synthesis failed this way
                break
            plan.is_complete = True
            break

        # Decide operation based on remaining terms
        if diff.pure_x_terms and any(t.coefficient > 0 for t in diff.pure_x_terms):
            # Need more x terms - use disjoint union or vertex join
            new_k2 = create_edge()

            if current_graph is None:
                current_graph = new_k2
                current_tutte = k2_tutte
                plan.source_components.append((new_k2.copy(), k2_tutte))
                plan.join_instructions.append(JoinInstruction(
                    operation=JoinOperation.BRIDGE_ADD,
                    k=0,
                    description="Start with K_2 (contributes x)"
                ))
            else:
                # Check if we should do disjoint union (multiply) or join
                # Disjoint union: T(G1 ∪ G2) = T(G1) × T(G2)
                test_union = current_tutte * k2_tutte

                diff_after_union = analyze_polynomial_difference(target_polynomial, test_union)

                # If union gets us closer (or we need x^2 etc), do union
                if len(diff_after_union.pure_x_terms) < len(diff.pure_x_terms):
                    current_graph = disjoint_union(current_graph, new_k2)
                    current_tutte = test_union
                    plan.source_components.append((new_k2.copy(), k2_tutte))
                    plan.join_instructions.append(JoinInstruction(
                        operation=JoinOperation.BRIDGE_ADD,
                        k=0,
                        description=f"Disjoint union with K_2 (now T = {current_tutte})"
                    ))
                else:
                    # Try 1-join (vertex identification)
                    v1 = sorted(current_graph.nodes)[-1]
                    v2 = sorted(new_k2.nodes)[0]
                    current_graph = perform_1_join(current_graph, v1, new_k2, v2)
                    # For 1-join: T = T1 * T2
                    current_tutte = current_tutte * k2_tutte
                    plan.source_components.append((new_k2.copy(), k2_tutte))
                    plan.join_instructions.append(JoinInstruction(
                        operation=JoinOperation.VERTEX_JOIN,
                        k=1,
                        description=f"1-join with K_2 (now T = {current_tutte})"
                    ))

        elif diff.pure_y_terms or diff.mixed_terms:
            # Need y terms - this means closing cycles
            if current_graph is not None and len(current_graph.nodes) >= 2:
                # Find two nodes that aren't already connected
                nodes = sorted(current_graph.nodes)
                for u in nodes:
                    for v in nodes:
                        if u < v and not _are_adjacent(current_graph, u, v):
                            # Adding this edge will contribute to y terms
                            old_tutte = current_tutte
                            current_graph, edge_id = add_edge_between_nodes(current_graph, u, v)
                            current_tutte = compute_tutte_polynomial(current_graph)

                            plan.join_instructions.append(JoinInstruction(
                                operation=JoinOperation.CYCLE_CLOSE,
                                k=1,
                                description=f"Close cycle between {u}-{v} (T: {old_tutte} -> {current_tutte})"
                            ))
                            break
                    else:
                        continue
                    break
            else:
                # Add another K_2 first
                new_k2 = create_edge()
                if current_graph is None:
                    current_graph = new_k2
                else:
                    current_graph = disjoint_union(current_graph, new_k2)
                current_tutte = compute_tutte_polynomial(current_graph)
                plan.source_components.append((new_k2.copy(), k2_tutte))
                plan.join_instructions.append(JoinInstruction(
                    operation=JoinOperation.BRIDGE_ADD,
                    k=0,
                    description=f"Add K_2 to prepare for cycle (now T = {current_tutte})"
                ))

    # Update remaining polynomial
    plan.remaining_polynomial = subtract_polynomials(target_polynomial, current_tutte)

    # Check completion
    if current_tutte == target_polynomial:
        plan.is_complete = True

    return plan


def _are_adjacent(g: GraphBuilder, u: int, v: int) -> bool:
    """Check if u and v are adjacent in graph g."""
    for (a, b) in g.edges.values():
        if (a == u and b == v) or (a == v and b == u):
            return True
    return False


def synthesize_k3_from_k2() -> SynthesisPlan:
    """
    Synthesize K_3 (triangle) from K_2 graphs.

    This is the example from the user:
    - Target: K_3 with T(x,y) = x^2 + x + y
    - Start: K_2 with T(x,y) = x
    - Process:
      1. Take two K_2 graphs, disjoint union: T = x^2
      2. Remaining: x^2 + x + y - x^2 = x + y
      3. Connect endpoints to add edge (x) and close cycle (y)
    """
    # Target polynomial
    k3_tutte = TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1})

    # Use general synthesis
    plan = synthesize_from_k2_graphs(k3_tutte, max_k2_count=5)

    # If that didn't work, do it manually
    if not plan.is_complete:
        plan = _manual_k3_synthesis()

    return plan


def _manual_k3_synthesis() -> SynthesisPlan:
    """Manual synthesis of K_3 from K_2 graphs for demonstration."""
    k3_tutte = TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    k2_tutte = TuttePolynomial({(1, 0): 1})

    # Step 1: First K_2 (nodes 0, 1 with edge 0-1)
    k2_1 = create_edge()

    # Step 2: Second K_2, 1-join to first
    k2_2 = create_edge()
    # Join endpoint of k2_1 (node 1) to endpoint of k2_2 (node 0)
    # This creates a proper path: 0 -- 1 -- 2 (P_3)
    # Note: k2_1 has nodes {0,1}, k2_2 has nodes {0,1}
    # After join: k2_1's node 1 merges with k2_2's node 0
    # Result: nodes {0, 1, 2} with edges (0,1) and (1,2) = path P_3
    path = perform_1_join(k2_1, 1, k2_2, 0)
    path_tutte = compute_tutte_polynomial(path)  # Should be x^2

    # Step 3: Close the cycle by connecting the two endpoints (nodes 0 and 2)
    # This transforms P_3 into C_3 = K_3
    nodes = sorted(path.nodes)
    # Find the degree-1 nodes (endpoints of the path)
    degree = {n: 0 for n in nodes}
    for (u, v) in path.edges.values():
        degree[u] += 1
        degree[v] += 1
    endpoints = [n for n in nodes if degree[n] == 1]

    triangle = path.copy()
    triangle.add_edge(endpoints[0], endpoints[1])
    triangle_tutte = compute_tutte_polynomial(triangle)

    plan = SynthesisPlan(
        target_polynomial=k3_tutte,
        source_components=[
            (k2_1.copy(), k2_tutte),
            (k2_2.copy(), k2_tutte),
        ],
        join_instructions=[
            JoinInstruction(
                operation=JoinOperation.BRIDGE_ADD,
                k=0,
                description="Start with first K_2 (T = x)"
            ),
            JoinInstruction(
                operation=JoinOperation.VERTEX_JOIN,
                k=1,
                description=f"1-join with second K_2 -> P_3 (T = {path_tutte})"
            ),
            JoinInstruction(
                operation=JoinOperation.CYCLE_CLOSE,
                k=1,
                description=f"Close cycle to form K_3 (T = {triangle_tutte})"
            ),
        ],
        remaining_polynomial=subtract_polynomials(k3_tutte, triangle_tutte),
        is_complete=(triangle_tutte == k3_tutte)
    )

    return plan


# =============================================================================
# SPECIALIZED GRAPH SYNTHESIS
# =============================================================================

def synthesize_zephyr_z11() -> Tuple[GraphBuilder, TuttePolynomial, Dict]:
    """
    Synthesize Zephyr Z(1,1) from K_4 core + C_8 periphery + spoke connections.

    Z(1,1) can be constructed as a composition of:
    1. K_4 (complete graph on 4 nodes) as inner core
    2. C_8 (8-cycle) as outer periphery
    3. 8 spoke edges connecting pairs of periphery nodes to each core node

    Structure:
        Periphery: 0-1-2-3-4-5-6-7-0 (8-cycle)
        Core: 8-9-10-11 (K_4)
        Spokes: {0,1}->8, {2,3}->9, {4,5}->10, {6,7}->11

    This creates:
    - 12 nodes, 22 edges
    - Degree sequence: [3,3,3,3,3,3,3,3,5,5,5,5]
    - 69,360 spanning trees

    Returns:
        Tuple of (graph, tutte_polynomial, edge_info_dict)
    """
    g = GraphBuilder()
    nodes = [g.add_node() for _ in range(12)]

    edge_info = {
        'core': [],      # K_4 edges
        'cycle': [],     # C_8 edges
        'spokes': [],    # Connection edges
    }

    # K_4 core (nodes 8-11): 6 edges
    for i in range(8, 12):
        for j in range(i+1, 12):
            g.add_edge(nodes[i], nodes[j])
            edge_info['core'].append((i, j))

    # C_8 periphery (nodes 0-7): 8 edges
    for i in range(8):
        g.add_edge(nodes[i], nodes[(i+1) % 8])
        edge_info['cycle'].append((i, (i+1) % 8))

    # Spoke connections (periphery to core): 8 edges
    # Each core node connects to 2 adjacent periphery nodes
    spoke_pairs = [
        (0, 8), (1, 8),   # nodes 0,1 connect to core node 8
        (2, 9), (3, 9),   # nodes 2,3 connect to core node 9
        (4, 10), (5, 10), # nodes 4,5 connect to core node 10
        (6, 11), (7, 11), # nodes 6,7 connect to core node 11
    ]
    for u, v in spoke_pairs:
        g.add_edge(nodes[u], nodes[v])
        edge_info['spokes'].append((u, v))

    poly = compute_tutte_polynomial(g)

    return g, poly, edge_info


# =============================================================================
# RAINBOW TABLE INTEGRATION
# =============================================================================

def find_motif_for_polynomial(
    target: TuttePolynomial,
    rainbow_table: Optional['RainbowTable'] = None
) -> Optional[Tuple[str, TuttePolynomial]]:
    """
    Look up a motif in the rainbow table that matches or contributes to target.

    Returns (name, polynomial) if found, None otherwise.
    """
    if rainbow_table is None:
        # Try to load default table
        try:
            from tutte_test.build_rainbow_table import RainbowTable
            table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
            rainbow_table = RainbowTable.load(table_path)
        except Exception:
            return None

    # Direct match
    for key, entry in rainbow_table.entries.items():
        poly = rainbow_table._entry_to_polynomial(entry)
        if poly == target:
            return (entry['name'], poly)

    # Look for divisors (motifs whose polynomial divides target)
    # This is useful for factorization
    for key, entry in rainbow_table.entries.items():
        poly = rainbow_table._entry_to_polynomial(entry)
        quotient = _try_divide_polynomials(target, poly)
        if quotient is not None:
            return (entry['name'], poly)

    return None


def _try_divide_polynomials(
    dividend: TuttePolynomial,
    divisor: TuttePolynomial
) -> Optional[TuttePolynomial]:
    """
    Try to divide dividend by divisor.

    Returns quotient if exact division is possible, None otherwise.

    Note: This is a simplified check - full polynomial division is complex.
    We check if divisor terms appear in dividend with compatible coefficients.
    """
    # Simple case: divisor is monomial x^a * y^b
    if len(divisor.coefficients) == 1:
        (a, b), c = next(iter(divisor.coefficients.items()))
        if c != 1:
            return None  # Only handle coefficient 1 for now

        quotient_coeffs = {}
        for (i, j), coeff in dividend.coefficients.items():
            if i >= a and j >= b:
                quotient_coeffs[(i - a, j - b)] = coeff
            else:
                return None  # Division not exact

        return TuttePolynomial(quotient_coeffs) if quotient_coeffs else None

    # General case: too complex for simple implementation
    return None


def suggest_motifs_for_synthesis(
    target: TuttePolynomial,
    rainbow_table: Optional['RainbowTable'] = None
) -> List[Tuple[str, TuttePolynomial, str]]:
    """
    Suggest motifs from rainbow table that could help synthesize target.

    Returns list of (name, polynomial, reason) tuples.
    """
    suggestions = []

    if rainbow_table is None:
        try:
            from tutte_test.build_rainbow_table import RainbowTable
            table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
            rainbow_table = RainbowTable.load(table_path)
        except Exception:
            return suggestions

    target_x_deg = target.x_degree()
    target_y_deg = target.y_degree()

    for key, entry in rainbow_table.entries.items():
        poly = rainbow_table._entry_to_polynomial(entry)
        name = entry['name']

        # Exact match
        if poly == target:
            suggestions.append((name, poly, "Exact match"))
            continue

        # Divisor match
        if _try_divide_polynomials(target, poly):
            suggestions.append((name, poly, "Divisor (can use in product)"))
            continue

        # Degree compatibility
        if poly.x_degree() <= target_x_deg and poly.y_degree() <= target_y_deg:
            diff = subtract_polynomials(target, poly)
            if all(c >= 0 for c in diff.coefficients.values()):
                suggestions.append((name, poly, "Subtractable (leaves positive remainder)"))

    return suggestions[:10]  # Limit to top 10


# =============================================================================
# EFFICIENT SYNTHESIS ENGINE
# =============================================================================

@dataclass
class SynthesisResult:
    """Result of a graph synthesis attempt."""
    success: bool
    graph: Optional[GraphBuilder]
    polynomial: Optional[TuttePolynomial]
    operations: List[str]
    motifs_used: List[str]
    num_steps: int
    verified: bool = False

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"SynthesisResult({status}, steps={self.num_steps}, verified={self.verified})"


class SynthesisEngine:
    """
    Efficient graph synthesis engine using multiple strategies.

    Strategies:
    1. Direct lookup - Check if target is in rainbow table
    2. Factorization - Try to factor polynomial into products
    3. Subtractive - Find motifs that leave positive remainder
    4. Incremental - Build up from small motifs with k-joins
    """

    def __init__(self, rainbow_table=None, verbose: bool = False):
        self.verbose = verbose
        self._cache: Dict[str, SynthesisResult] = {}

        # Load rainbow table
        if rainbow_table is None:
            try:
                from tutte_test.build_rainbow_table import RainbowTable
                table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
                self.rainbow_table = RainbowTable.load(table_path)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load rainbow table: {e}")
                self.rainbow_table = None
        else:
            self.rainbow_table = rainbow_table

        # Build index of polynomials for fast lookup
        self._poly_to_name: Dict[str, str] = {}
        self._poly_to_graph_info: Dict[str, Dict] = {}
        if self.rainbow_table:
            for key, entry in self.rainbow_table.entries.items():
                poly = self.rainbow_table._entry_to_polynomial(entry)
                poly_key = self._poly_key(poly)
                self._poly_to_name[poly_key] = entry['name']
                self._poly_to_graph_info[poly_key] = entry

    def _poly_key(self, poly: TuttePolynomial) -> str:
        """Create hashable key for polynomial."""
        return str(sorted(poly.coefficients.items()))

    def _log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"  [Synth] {msg}")

    def synthesize(self, target: TuttePolynomial, max_depth: int = 10) -> SynthesisResult:
        """
        Main synthesis entry point. Tries multiple strategies.

        Args:
            target: Target Tutte polynomial
            max_depth: Maximum recursion depth for synthesis

        Returns:
            SynthesisResult with synthesized graph if successful
        """
        # Check cache
        cache_key = self._poly_key(target)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._log(f"Synthesizing: {target}")

        # Strategy 1: Direct lookup
        result = self._try_direct_lookup(target)
        if result.success:
            self._cache[cache_key] = result
            return result

        # Strategy 2: Factorization
        result = self._try_factorization(target, max_depth)
        if result.success:
            self._cache[cache_key] = result
            return result

        # Strategy 3: Subtractive approach
        result = self._try_subtractive(target, max_depth)
        if result.success:
            self._cache[cache_key] = result
            return result

        # Strategy 4: Incremental building
        result = self._try_incremental(target, max_depth)
        if result.success:
            self._cache[cache_key] = result
            return result

        # Failed
        return SynthesisResult(
            success=False, graph=None, polynomial=None,
            operations=["No valid synthesis found"],
            motifs_used=[], num_steps=0
        )

    def _try_direct_lookup(self, target: TuttePolynomial) -> SynthesisResult:
        """Strategy 1: Check if target polynomial is directly in rainbow table."""
        poly_key = self._poly_key(target)

        if poly_key in self._poly_to_name:
            name = self._poly_to_name[poly_key]
            info = self._poly_to_graph_info[poly_key]
            self._log(f"Direct lookup hit: {name}")

            # Reconstruct graph from known types
            graph = self._reconstruct_graph(name, info)

            return SynthesisResult(
                success=True,
                graph=graph,
                polynomial=target,
                operations=[f"Direct lookup: {name}"],
                motifs_used=[name],
                num_steps=1,
                verified=True  # Rainbow table is pre-verified
            )

        return SynthesisResult(
            success=False, graph=None, polynomial=None,
            operations=[], motifs_used=[], num_steps=0
        )

    def _reconstruct_graph(self, name: str, info: Dict) -> Optional[GraphBuilder]:
        """Reconstruct a graph from its name/info."""
        # Complete graphs
        if name.startswith('K_'):
            try:
                n = int(name[2:])
                return create_complete_graph(n)
            except:
                pass

        # Cycles
        if name.startswith('C_'):
            try:
                n = int(name[2:])
                return create_cycle_graph(n)
            except:
                pass

        # Paths
        if name.startswith('P_'):
            try:
                n = int(name[2:])
                return create_path_graph(n)
            except:
                pass

        # Wheels - W_n has n total nodes (1 hub + n-1 rim nodes)
        if name.startswith('W_'):
            try:
                n = int(name[2:])
                g = GraphBuilder()
                center = g.add_node()
                rim_size = n - 1  # n total nodes = 1 center + (n-1) rim
                rim = [g.add_node() for _ in range(rim_size)]
                for r in rim:
                    g.add_edge(center, r)
                for i in range(rim_size):
                    g.add_edge(rim[i], rim[(i+1) % rim_size])
                return g
            except:
                pass

        # For other graphs, we can't easily reconstruct
        # Return None and let synthesis continue
        return None

    def _try_factorization(self, target: TuttePolynomial, max_depth: int) -> SynthesisResult:
        """
        Strategy 2: Try to factor the polynomial.

        If T_target = T_1 * T_2, we can use:
        - Disjoint union: G = G_1 ∪ G_2
        - Or 1-join: G = G_1 ·₁ G_2
        """
        if max_depth <= 0:
            return SynthesisResult(success=False, graph=None, polynomial=None,
                                   operations=[], motifs_used=[], num_steps=0)

        self._log("Trying factorization...")

        # Try to find divisors from rainbow table
        if not self.rainbow_table:
            return SynthesisResult(success=False, graph=None, polynomial=None,
                                   operations=[], motifs_used=[], num_steps=0)

        # Sort potential divisors by size (prefer larger factors)
        candidates = []
        for key, entry in self.rainbow_table.entries.items():
            divisor = self.rainbow_table._entry_to_polynomial(entry)
            quotient = self._try_divide(target, divisor)
            if quotient is not None and len(quotient.coefficients) > 0:
                # Valid factorization found
                candidates.append((entry['name'], divisor, quotient, entry))

        # Sort by quotient complexity (prefer simpler quotients)
        candidates.sort(key=lambda x: len(x[2].coefficients))

        for name, divisor, quotient, info in candidates:
            self._log(f"  Trying factor: {name} (quotient: {quotient})")

            # Recursively synthesize quotient
            quotient_result = self.synthesize(quotient, max_depth - 1)

            if quotient_result.success:
                # Build factor graph
                factor_graph = self._reconstruct_graph(name, info)
                if factor_graph is None:
                    continue

                # Combine via disjoint union (T = T1 * T2)
                combined = disjoint_union(factor_graph, quotient_result.graph)

                # Verify
                combined_poly = compute_tutte_polynomial(combined)
                if combined_poly == target:
                    return SynthesisResult(
                        success=True,
                        graph=combined,
                        polynomial=combined_poly,
                        operations=[f"Factor: {name} × (quotient)"] + quotient_result.operations,
                        motifs_used=[name] + quotient_result.motifs_used,
                        num_steps=quotient_result.num_steps + 1,
                        verified=True
                    )

        return SynthesisResult(success=False, graph=None, polynomial=None,
                               operations=[], motifs_used=[], num_steps=0)

    def _try_divide(self, dividend: TuttePolynomial, divisor: TuttePolynomial) -> Optional[TuttePolynomial]:
        """Try polynomial division. Returns quotient if exact, None otherwise."""
        # Simple case: divisor is monomial
        if len(divisor.coefficients) == 1:
            (a, b), c = next(iter(divisor.coefficients.items()))
            if c != 1:
                return None

            quotient_coeffs = {}
            for (i, j), coeff in dividend.coefficients.items():
                if i >= a and j >= b:
                    quotient_coeffs[(i - a, j - b)] = coeff
                else:
                    return None
            return TuttePolynomial(quotient_coeffs) if quotient_coeffs else None

        # General polynomial division - more complex
        # Check if divisor * something = dividend
        # This is expensive, so we use heuristics

        # Quick check: degrees must be compatible
        if divisor.x_degree() > dividend.x_degree():
            return None
        if divisor.y_degree() > dividend.y_degree():
            return None

        # Try to find quotient by examining leading terms
        # This is a simplified approach - full polynomial division is complex
        return None

    def _try_subtractive(self, target: TuttePolynomial, max_depth: int) -> SynthesisResult:
        """
        Strategy 3: Find motifs whose polynomial can be "subtracted" from target.

        This doesn't directly work for Tutte polynomials (no additive decomposition),
        but we can use it to guide incremental construction.
        """
        # This strategy is subsumed by incremental for Tutte polynomials
        return SynthesisResult(success=False, graph=None, polynomial=None,
                               operations=[], motifs_used=[], num_steps=0)

    def _try_incremental(self, target: TuttePolynomial, max_depth: int) -> SynthesisResult:
        """
        Strategy 4: Build incrementally using k-join operations.

        Start with base motifs and iteratively add structure.
        """
        if max_depth <= 0:
            return SynthesisResult(success=False, graph=None, polynomial=None,
                                   operations=[], motifs_used=[], num_steps=0)

        self._log("Trying incremental construction...")

        # Analyze target structure
        x_deg = target.x_degree()
        y_deg = target.y_degree()
        num_trees = target.num_spanning_trees()

        self._log(f"  Target: x-deg={x_deg}, y-deg={y_deg}, trees={num_trees}")

        # Heuristic: estimate graph size
        # For connected graphs: |E| >= |V| - 1, and x-degree ~ |E| - |V| + 1
        # y-degree relates to number of independent cycles

        # Try building from complete graphs (good for dense targets)
        for n in range(2, min(x_deg + 3, 8)):
            kn = create_complete_graph(n)
            kn_poly = compute_tutte_polynomial(kn)

            if kn_poly == target:
                return SynthesisResult(
                    success=True, graph=kn, polynomial=kn_poly,
                    operations=[f"Build K_{n}"], motifs_used=[f"K_{n}"],
                    num_steps=1, verified=True
                )

            # Check if K_n is "smaller" than target and can be extended
            if self._poly_leq(kn_poly, target):
                self._log(f"  K_{n} is subset, trying to extend...")
                extended = self._try_extend(kn, kn_poly, target, max_depth - 1)
                if extended.success:
                    return extended

        # Try building from cycles
        for n in range(3, min(x_deg + 3, 12)):
            cn = create_cycle_graph(n)
            cn_poly = compute_tutte_polynomial(cn)

            if cn_poly == target:
                return SynthesisResult(
                    success=True, graph=cn, polynomial=cn_poly,
                    operations=[f"Build C_{n}"], motifs_used=[f"C_{n}"],
                    num_steps=1, verified=True
                )

        # Try building from paths + cycle closing
        for n in range(2, min(x_deg + 2, 10)):
            path = create_path_graph(n)
            path_poly = compute_tutte_polynomial(path)

            # Try adding edges to close cycles
            extended = self._try_add_edges(path, path_poly, target, max_depth - 1)
            if extended.success:
                return extended

        return SynthesisResult(success=False, graph=None, polynomial=None,
                               operations=[], motifs_used=[], num_steps=0)

    def _poly_leq(self, p1: TuttePolynomial, p2: TuttePolynomial) -> bool:
        """Check if p1 <= p2 coefficient-wise (all coeffs of p1 <= p2)."""
        for (i, j), c1 in p1.coefficients.items():
            c2 = p2.coefficients.get((i, j), 0)
            if c1 > c2:
                return False
        return True

    def _try_extend(self, graph: GraphBuilder, current_poly: TuttePolynomial,
                    target: TuttePolynomial, max_depth: int) -> SynthesisResult:
        """Try to extend a graph to reach target polynomial."""
        if max_depth <= 0:
            return SynthesisResult(success=False, graph=None, polynomial=None,
                                   operations=[], motifs_used=[], num_steps=0)

        # Try adding a new vertex connected to all existing vertices
        extended = graph.copy()
        new_v = extended.add_node()
        for v in list(extended.nodes)[:-1]:
            extended.add_edge(new_v, v)

        ext_poly = compute_tutte_polynomial(extended)
        if ext_poly == target:
            return SynthesisResult(
                success=True, graph=extended, polynomial=ext_poly,
                operations=["Add vertex connected to all"],
                motifs_used=[], num_steps=1, verified=True
            )

        # Try adding edges between existing vertices
        return self._try_add_edges(graph, current_poly, target, max_depth)

    def _try_add_edges(self, graph: GraphBuilder, current_poly: TuttePolynomial,
                       target: TuttePolynomial, max_depth: int) -> SynthesisResult:
        """Try adding edges (including parallel edges) to reach target polynomial."""
        if max_depth <= 0 or current_poly == target:
            if current_poly == target:
                return SynthesisResult(
                    success=True, graph=graph, polynomial=current_poly,
                    operations=[], motifs_used=[], num_steps=0, verified=True
                )
            return SynthesisResult(success=False, graph=None, polynomial=None,
                                   operations=[], motifs_used=[], num_steps=0)

        nodes = sorted(graph.nodes)

        # Count existing edges between each pair
        edge_counts = defaultdict(int)
        for (u, v) in graph.edges.values():
            edge_counts[(min(u, v), max(u, v))] += 1

        # Try adding edges (including parallel edges for multi-edge graphs)
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                pair = (u, v)

                # Allow up to 3 parallel edges (for theta graphs, etc.)
                if edge_counts[pair] < 3:
                    # Try adding this edge
                    test = graph.copy()
                    test.add_edge(u, v)
                    test_poly = compute_tutte_polynomial(test)

                    if test_poly == target:
                        edge_type = "parallel edge" if edge_counts[pair] > 0 else "edge"
                        return SynthesisResult(
                            success=True, graph=test, polynomial=test_poly,
                            operations=[f"Add {edge_type} ({u},{v})"],
                            motifs_used=[], num_steps=1, verified=True
                        )

                    # Check if we're getting closer
                    if self._poly_leq(test_poly, target):
                        # Recursively try to extend
                        result = self._try_add_edges(test, test_poly, target, max_depth - 1)
                        if result.success:
                            edge_type = "parallel edge" if edge_counts[pair] > 0 else "edge"
                            result.operations = [f"Add {edge_type} ({u},{v})"] + result.operations
                            result.num_steps += 1
                            return result

        # Also try adding loops for y terms
        if target.y_degree() > 0:
            for u in nodes:
                test = graph.copy()
                test.add_loop(u)
                test_poly = compute_tutte_polynomial(test)

                if test_poly == target:
                    return SynthesisResult(
                        success=True, graph=test, polynomial=test_poly,
                        operations=[f"Add loop at {u}"],
                        motifs_used=[], num_steps=1, verified=True
                    )

                if self._poly_leq(test_poly, target):
                    result = self._try_add_edges(test, test_poly, target, max_depth - 1)
                    if result.success:
                        result.operations = [f"Add loop at {u}"] + result.operations
                        result.num_steps += 1
                        return result

        return SynthesisResult(success=False, graph=None, polynomial=None,
                               operations=[], motifs_used=[], num_steps=0)

    def verify_synthesis(self, result: SynthesisResult, target: TuttePolynomial) -> bool:
        """Verify that a synthesis result actually produces the target polynomial."""
        if not result.success or result.graph is None:
            return False

        computed = compute_tutte_polynomial(result.graph)
        return computed == target


def synthesize_graph(target: TuttePolynomial, verbose: bool = False) -> SynthesisResult:
    """
    Convenience function to synthesize a graph for a target polynomial.

    Args:
        target: Target Tutte polynomial
        verbose: Print progress information

    Returns:
        SynthesisResult with synthesized graph
    """
    engine = SynthesisEngine(verbose=verbose)
    return engine.synthesize(target)


def benchmark_synthesis(test_cases: List[Tuple[str, TuttePolynomial]], verbose: bool = True):
    """
    Benchmark synthesis on multiple test cases.

    Args:
        test_cases: List of (name, polynomial) tuples
        verbose: Print detailed results
    """
    import time

    engine = SynthesisEngine(verbose=False)

    print("\n" + "=" * 80)
    print("SYNTHESIS BENCHMARK")
    print("=" * 80)
    print(f"\n{'Name':<20} {'Terms':<8} {'Trees':<10} {'Time(ms)':<10} {'Steps':<8} {'Status'}")
    print("-" * 80)

    total_time = 0
    successes = 0

    for name, target in test_cases:
        start = time.time()
        result = engine.synthesize(target)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        num_terms = len(target.coefficients)
        num_trees = target.num_spanning_trees()
        status = "OK" if result.success else "FAIL"

        if result.success:
            successes += 1
            # Verify
            if result.graph:
                verified = engine.verify_synthesis(result, target)
                status = "OK (verified)" if verified else "OK (unverified)"

        print(f"{name:<20} {num_terms:<8} {num_trees:<10} {elapsed:<10.2f} {result.num_steps:<8} {status}")

        if verbose and result.success:
            print(f"    Operations: {' -> '.join(result.operations[:3])}")
            if len(result.operations) > 3:
                print(f"                ... ({len(result.operations) - 3} more)")

    print("-" * 80)
    print(f"Total: {successes}/{len(test_cases)} successful, {total_time:.2f}ms total")


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def verify_against_rainbow_table():
    """
    Verify that our computed polynomials match the rainbow table.

    This ensures correctness of the computation algorithm.
    """
    print("=" * 70)
    print("VERIFICATION: Comparing computed polynomials to rainbow table")
    print("=" * 70)

    try:
        from tutte_test.build_rainbow_table import RainbowTable
        table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
        rainbow_table = RainbowTable.load(table_path)
    except Exception as e:
        print(f"Could not load rainbow table: {e}")
        return False

    all_match = True
    test_cases = [
        ('K_2', lambda: create_edge()),
        ('K_3', lambda: create_complete_graph(3)),
        ('K_4', lambda: create_complete_graph(4)),
        ('K_5', lambda: create_complete_graph(5)),
        ('C_3', lambda: create_cycle_graph(3)),
        ('C_4', lambda: create_cycle_graph(4)),
        ('C_5', lambda: create_cycle_graph(5)),
        ('P_3', lambda: create_path_graph(3)),
        ('P_4', lambda: create_path_graph(4)),
    ]

    print(f"\n{'Graph':<10} {'Computed':<40} {'Rainbow Table':<40} {'Match'}")
    print("-" * 100)

    for name, graph_fn in test_cases:
        # Compute polynomial
        g = graph_fn()
        computed = compute_tutte_polynomial(g)

        # Look up in rainbow table
        table_entry = rainbow_table.get_entry(name)
        if table_entry:
            table_poly = rainbow_table._entry_to_polynomial(table_entry)
            match = (computed == table_poly)

            # Truncate long polynomials for display
            comp_str = str(computed)[:38] + ".." if len(str(computed)) > 40 else str(computed)
            table_str = str(table_poly)[:38] + ".." if len(str(table_poly)) > 40 else str(table_poly)

            status = "✓" if match else "✗"
            print(f"{name:<10} {comp_str:<40} {table_str:<40} {status}")

            if not match:
                all_match = False
                print(f"  MISMATCH DETAILS:")
                print(f"    Computed:    {computed}")
                print(f"    Rainbow:     {table_poly}")
                print(f"    Computed T(1,1): {computed.num_spanning_trees()}")
                print(f"    Rainbow T(1,1):  {table_poly.num_spanning_trees()}")
        else:
            print(f"{name:<10} {str(computed)[:40]:<40} {'(not in table)':<40} -")

    print("-" * 100)
    if all_match:
        print("All computed polynomials match rainbow table!")
    else:
        print("Some polynomials do not match - investigation needed!")

    return all_match


def demo_k3_synthesis():
    """Demonstrate K_3 synthesis from K_2 graphs."""
    print("\n" + "=" * 70)
    print("DEMO: Synthesizing K_3 from K_2 graphs")
    print("=" * 70)

    # Target
    k3 = create_k3()
    k3_tutte = compute_tutte_polynomial(k3)
    print(f"\nTarget: K_3 (triangle)")
    print(f"  T(K_3) = {k3_tutte}")
    print(f"  Spanning trees: {k3_tutte.num_spanning_trees()}")

    # Source
    k2 = create_edge()
    k2_tutte = compute_tutte_polynomial(k2)
    print(f"\nSource: K_2 (single edge)")
    print(f"  T(K_2) = {k2_tutte}")

    # Synthesis
    print("\n" + "-" * 70)
    print("SYNTHESIS PROCESS:")
    print("-" * 70)

    plan = synthesize_k3_from_k2()

    print(f"\nSynthesis Plan:")
    print(f"  Components used: {len(plan.source_components)}")
    print(f"  Join operations: {len(plan.join_instructions)}")
    print(f"  Complete: {plan.is_complete}")
    print(f"  Remaining polynomial: {plan.remaining_polynomial}")

    print("\nStep-by-step:")
    for i, instr in enumerate(plan.join_instructions, 1):
        print(f"  {i}. {instr}")


def demo_polynomial_difference():
    """Demonstrate polynomial difference analysis."""
    print("\n" + "=" * 70)
    print("DEMO: Polynomial Difference Analysis")
    print("=" * 70)

    # K_4 vs K_3 - compute actual polynomials
    k4 = create_complete_graph(4)
    k3 = create_complete_graph(3)
    k4_tutte = compute_tutte_polynomial(k4)
    k3_tutte = compute_tutte_polynomial(k3)

    print(f"\nT(K_4) = {k4_tutte}")
    print(f"T(K_3) = {k3_tutte}")

    diff = analyze_polynomial_difference(k4_tutte, k3_tutte)

    print(f"\nDifference T(K_4) - T(K_3) = {diff.difference}")
    print(f"\nClassified terms:")
    print(f"  Pure x terms: {diff.pure_x_terms}")
    print(f"  Pure y terms: {diff.pure_y_terms}")
    print(f"  Mixed terms: {diff.mixed_terms}")
    print(f"  Negative terms: {diff.negative_terms}")

    print(f"\nInferred join operations:")
    for instr in diff.inferred_joins:
        print(f"  - {instr}")


def demo_k_cut_analysis():
    """Demonstrate k-cut analysis."""
    print("\n" + "=" * 70)
    print("DEMO: K-Cut Analysis")
    print("=" * 70)

    # Analyze K_4
    k4 = create_complete_graph(4)
    print(f"\nGraph: K_4 ({k4.num_nodes()} nodes, {k4.num_edges()} edges)")

    cuts = analyze_k_cuts(k4)

    print(f"\nK-cuts found:")
    for k in [1, 2, 3]:
        print(f"  {k}-cuts: {len(cuts[k])}")
        for cut in cuts[k][:3]:  # Show first 3
            print(f"    {cut}")

    # Analyze a path (should have many 1-cuts)
    path = create_path_graph(5)
    print(f"\nGraph: P_5 ({path.num_nodes()} nodes, {path.num_edges()} edges)")

    cuts = analyze_k_cuts(path)
    print(f"\nK-cuts found:")
    for k in [1, 2, 3]:
        print(f"  {k}-cuts: {len(cuts[k])}")


def demo_y_power_interpretation():
    """
    Demonstrate how y^k terms relate to k-join operations.

    Key insight: y^k in the polynomial difference indicates the need
    for structures that contribute cycles. Specifically:
    - y^1: Single cycle (or closing one cycle)
    - y^2: Two independent cycles, or a theta graph structure
    - y^3: Three cycles, possibly requiring a 3-join to create

    The relationship between y powers and joins:
    - y appears when closing a cycle (adding edge between path endpoints)
    - y^k from k independent loops
    - Mixed x^i*y^j from graphs with both bridge-like and cycle-like structure
    """
    print("\n" + "=" * 70)
    print("DEMO: Y^k Terms and K-Join Operations")
    print("=" * 70)

    # Example 1: Building a graph with y^2 (two cycles)
    print("\n--- Example 1: Graph with y^2 term ---")

    # Theta graph θ(0,0,0) = three edges between two vertices
    # T(θ) = x^2 + x + y (same as K_3 actually)
    # For y^2, we need something like K_4 minus edge, or two triangles sharing edge

    # Let's analyze K_4's polynomial
    k4 = create_complete_graph(4)
    k4_tutte = compute_tutte_polynomial(k4)
    print(f"T(K_4) = {k4_tutte}")
    print(f"  Note: y^2 term with coefficient {k4_tutte.coefficients.get((0,2), 0)}")

    # The y^2 in K_4 comes from having two independent cycles
    # When we remove spanning tree edges, we can create 2-cycle configurations

    # Example 2: Understanding y^3
    print("\n--- Example 2: Understanding y^3 ---")
    print("""
    A y^3 term in the difference polynomial indicates we need a structure
    that contributes 3 independent cycles. Options:

    1. Three independent loops (bouquet B_3): T = y^3
    2. Part of a more complex graph like K_5

    For a 3-join operation (opposite of 3-cut):
    - We identify triangles from two graphs
    - This can create/preserve cycle structures
    """)

    # Demonstrate with K_5
    k5 = create_complete_graph(5)
    k5_tutte = compute_tutte_polynomial(k5)
    print(f"T(K_5) = {k5_tutte}")
    y3_coeff = k5_tutte.coefficients.get((0, 3), 0)
    print(f"  y^3 coefficient: {y3_coeff}")

    # Example 3: Building K_4 from smaller pieces
    print("\n--- Example 3: Building K_4 from K_3 + operations ---")

    k3 = create_complete_graph(3)
    k3_tutte = compute_tutte_polynomial(k3)
    print(f"T(K_3) = {k3_tutte}")

    # Difference K_4 - K_3
    diff = analyze_polynomial_difference(k4_tutte, k3_tutte)
    print(f"\nT(K_4) - T(K_3) = {diff.difference}")
    print("\nTo transform K_3 -> K_4, we need to add:")
    for term in diff.pure_x_terms + diff.pure_y_terms + diff.mixed_terms:
        if term.coefficient > 0:
            print(f"  {term}")

    # Actually build K_4 from K_3
    print("\nSynthesis approach:")
    print("  1. Start with K_3")
    print("  2. Add a new vertex connected to all 3 vertices of K_3")
    print("  3. This adds 3 edges, creating 3 new triangles")

    # Manual construction
    result = k3.copy()
    # Add 4th vertex
    new_v = result.add_node()
    # Connect to all existing vertices
    for v in list(result.nodes)[:-1]:  # All except the new one
        result.add_edge(new_v, v)

    result_tutte = compute_tutte_polynomial(result)
    print(f"\nResult polynomial: {result_tutte}")
    print(f"Matches K_4: {result_tutte == k4_tutte}")


def demo_rainbow_table_synthesis():
    """Demonstrate using rainbow table for synthesis."""
    print("\n" + "=" * 70)
    print("DEMO: Rainbow Table Guided Synthesis")
    print("=" * 70)

    try:
        from tutte_test.build_rainbow_table import RainbowTable
        table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
        rainbow_table = RainbowTable.load(table_path)
        print(f"Loaded rainbow table with {len(rainbow_table)} entries")
    except Exception as e:
        print(f"Could not load rainbow table: {e}")
        return

    # Target: Build the Petersen graph polynomial
    petersen_entry = rainbow_table.get_entry('Petersen')
    if petersen_entry:
        petersen_tutte = rainbow_table._entry_to_polynomial(petersen_entry)
        print(f"\nTarget: Petersen graph")
        print(f"  Nodes: {petersen_entry['nodes']}, Edges: {petersen_entry['edges']}")
        print(f"  Spanning trees: {petersen_entry['spanning_trees']}")
        print(f"  Polynomial terms: {petersen_entry['num_terms']}")

        # Find motifs that could help
        suggestions = suggest_motifs_for_synthesis(petersen_tutte, rainbow_table)
        print(f"\nSuggested building blocks:")
        for name, poly, reason in suggestions[:5]:
            print(f"  {name}: {reason}")

    # Simpler example: Build K_4 using table lookup
    print("\n--- Building K_4 using rainbow table ---")
    k4_entry = rainbow_table.get_entry('K_4')
    if k4_entry:
        k4_tutte = rainbow_table._entry_to_polynomial(k4_entry)

        # Try to find factors or building blocks
        k3_entry = rainbow_table.get_entry('K_3')
        if k3_entry:
            k3_tutte = rainbow_table._entry_to_polynomial(k3_entry)

            diff = analyze_polynomial_difference(k4_tutte, k3_tutte)
            print(f"\nTo build K_4 from K_3:")
            print(f"  Need to add: {diff.difference}")

            # Check if any motif in table matches the difference
            match = find_motif_for_polynomial(diff.difference, rainbow_table)
            if match:
                print(f"  Found matching motif: {match[0]}")
            else:
                print("  No single motif matches - need composite construction")


def demo_synthesis_strategies():
    """
    Demonstrate different synthesis strategies based on polynomial structure.
    """
    print("\n" + "=" * 70)
    print("DEMO: Synthesis Strategies")
    print("=" * 70)

    print("""
    SYNTHESIS STRATEGY GUIDE:

    Given target polynomial T_target and source T_source:
    1. Compute difference D = T_target - T_source
    2. Analyze terms in D to determine operations:

    TERM TYPE           -> OPERATION
    ----------------------------------------
    x^k (positive)      -> Add k bridges (path extension)
    y^k (positive)      -> Add k cycle-closing edges or k-join
    c*x^i*y^j (mixed)   -> Complex structure, may need:
                           - Parallel edges
                           - Multiple cycle closures
                           - k-clique joins

    SPECIAL PATTERNS:
    - x + y             -> Single edge closing a cycle (common in triangles)
    - x^2 + x + y       -> Either path + cycle close, or triangle
    - 2xy               -> Two structures each contributing xy (parallel triangles?)
    - y^k alone         -> k independent loops, or k-join operation

    MULTIPLICATIVE DECOMPOSITION:
    If T_target = T_1 * T_2, then:
    - Use disjoint union: G_target = G_1 ∪ G_2
    - Or 1-join (cut vertex): G_1 ·₁ G_2
    """)

    # Example: Decompose a product polynomial
    print("\n--- Example: Factored polynomial ---")
    # T = x * (x + y) = x^2 + xy
    # This is T(P_2) * T(dipole)
    p2_tutte = TuttePolynomial({(1, 0): 1})  # x
    dipole_tutte = TuttePolynomial({(1, 0): 1, (0, 1): 1})  # x + y
    product = p2_tutte * dipole_tutte
    print(f"T(P_2) = {p2_tutte}")
    print(f"T(dipole) = {dipole_tutte}")
    print(f"Product = {product}")
    print("This can be realized as: P_2 ∪ dipole (disjoint) or P_2 ·₁ dipole (1-join)")


def demo_efficient_synthesis():
    """
    Demonstrate the efficient synthesis engine on complex graphs.
    """
    print("\n" + "=" * 80)
    print("DEMO: Efficient Synthesis Engine")
    print("=" * 80)

    # Load rainbow table for test targets
    try:
        from tutte_test.build_rainbow_table import RainbowTable
        table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
        rainbow_table = RainbowTable.load(table_path)
    except Exception as e:
        print(f"Could not load rainbow table: {e}")
        return

    # Build test cases from rainbow table
    test_cases = []

    # Complete graphs
    for n in range(2, 7):
        entry = rainbow_table.get_entry(f'K_{n}')
        if entry:
            poly = rainbow_table._entry_to_polynomial(entry)
            test_cases.append((f'K_{n}', poly))

    # Cycles
    for n in [4, 5, 6, 8]:
        entry = rainbow_table.get_entry(f'C_{n}')
        if entry:
            poly = rainbow_table._entry_to_polynomial(entry)
            test_cases.append((f'C_{n}', poly))

    # Wheels
    for n in [4, 5, 6]:
        entry = rainbow_table.get_entry(f'W_{n}')
        if entry:
            poly = rainbow_table._entry_to_polynomial(entry)
            test_cases.append((f'W_{n}', poly))

    # Special graphs
    for name in ['Petersen', 'Ladder_4', 'Grid_3x3']:
        entry = rainbow_table.get_entry(name)
        if entry:
            poly = rainbow_table._entry_to_polynomial(entry)
            test_cases.append((name, poly))

    # Run benchmark
    benchmark_synthesis(test_cases, verbose=True)

    # Detailed example: Synthesize K_5
    print("\n" + "-" * 80)
    print("DETAILED: Synthesizing K_5")
    print("-" * 80)

    k5_entry = rainbow_table.get_entry('K_5')
    if k5_entry:
        k5_target = rainbow_table._entry_to_polynomial(k5_entry)
        print(f"Target: T(K_5) = {k5_target}")
        print(f"Spanning trees: {k5_target.num_spanning_trees()}")

        engine = SynthesisEngine(verbose=True)
        result = engine.synthesize(k5_target)

        print(f"\nResult: {result}")
        if result.success and result.graph:
            print(f"Graph: {result.graph.num_nodes()} nodes, {result.graph.num_edges()} edges")
            print(f"Operations: {result.operations}")

            # Verify
            computed = compute_tutte_polynomial(result.graph)
            print(f"Verified: {computed == k5_target}")

    # Test incremental synthesis
    print("\n" + "-" * 80)
    print("DETAILED: Building W_5 (wheel with 5 spokes) incrementally")
    print("-" * 80)

    w5_entry = rainbow_table.get_entry('W_5')
    if w5_entry:
        w5_target = rainbow_table._entry_to_polynomial(w5_entry)
        print(f"Target: T(W_5)")
        print(f"Spanning trees: {w5_target.num_spanning_trees()}")

        engine = SynthesisEngine(verbose=True)
        result = engine.synthesize(w5_target)

        print(f"\nResult: {result}")
        if result.success and result.graph:
            print(f"Graph: {result.graph.num_nodes()} nodes, {result.graph.num_edges()} edges")

            # Verify
            computed = compute_tutte_polynomial(result.graph)
            match = computed == w5_target
            print(f"Verified: {match}")
            if not match:
                print(f"  Expected: {w5_target}")
                print(f"  Got:      {computed}")


def demo_novel_synthesis():
    """
    Test synthesis on polynomials NOT directly in the rainbow table.

    This exercises the factorization and incremental building algorithms.
    """
    print("\n" + "=" * 80)
    print("NOVEL SYNTHESIS TESTS (polynomials not in rainbow table)")
    print("=" * 80)

    engine = SynthesisEngine(verbose=True)

    # Test 1: Product of two different graphs (K_3 * K_2)
    print("\n--- Test 1: K_3 * K_2 (disjoint triangle + edge) ---")
    k3 = create_complete_graph(3)
    k2 = create_edge()
    k3_poly = compute_tutte_polynomial(k3)
    k2_poly = compute_tutte_polynomial(k2)
    target = k3_poly * k2_poly  # (x^2 + x + y) * x = x^3 + x^2 + xy
    print(f"Target: {target}")
    print(f"Expected: disjoint union of K_3 and K_2")

    result = engine.synthesize(target)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Steps: {result.num_steps}")
        print(f"Operations: {result.operations}")
        verified = engine.verify_synthesis(result, target)
        print(f"Verified: {verified}")
        if result.graph:
            print(f"Graph: {result.graph.num_nodes()} nodes, {result.graph.num_edges()} edges")

    # Test 2: Triple product (K_2 * K_2 * K_2 = x^3)
    print("\n--- Test 2: K_2 * K_2 * K_2 (three disjoint edges) ---")
    target = k2_poly * k2_poly * k2_poly  # x * x * x = x^3
    print(f"Target: {target}")
    print(f"This equals T(P_4), so should find path of length 3")

    result = engine.synthesize(target)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Operations: {result.operations}")
        verified = engine.verify_synthesis(result, target)
        print(f"Verified: {verified}")

    # Test 3: Product that creates a more complex graph
    print("\n--- Test 3: (K_3)^2 = two disjoint triangles ---")
    target = k3_poly * k3_poly
    print(f"Target: {target}")

    result = engine.synthesize(target)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Steps: {result.num_steps}")
        print(f"Operations: {result.operations}")
        verified = engine.verify_synthesis(result, target)
        print(f"Verified: {verified}")
        if result.graph:
            print(f"Graph: {result.graph.num_nodes()} nodes, {result.graph.num_edges()} edges")

    # Test 4: Build a graph not in the table - K_4 minus one edge (diamond)
    print("\n--- Test 4: Diamond graph (K_4 minus one edge) ---")
    diamond = GraphBuilder()
    nodes = [diamond.add_node() for _ in range(4)]
    # Add 5 edges (K_4 has 6)
    diamond.add_edge(nodes[0], nodes[1])
    diamond.add_edge(nodes[0], nodes[2])
    diamond.add_edge(nodes[0], nodes[3])
    diamond.add_edge(nodes[1], nodes[2])
    diamond.add_edge(nodes[2], nodes[3])
    # Missing edge: (1, 3)

    diamond_poly = compute_tutte_polynomial(diamond)
    print(f"Target: {diamond_poly}")
    print(f"Spanning trees: {diamond_poly.num_spanning_trees()}")

    result = engine.synthesize(diamond_poly)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Steps: {result.num_steps}")
        print(f"Operations: {result.operations}")
        verified = engine.verify_synthesis(result, diamond_poly)
        print(f"Verified: {verified}")

    # Test 5: A "theta graph" - two vertices connected by 3 paths
    print("\n--- Test 5: Theta graph (two vertices, three parallel paths) ---")
    theta = GraphBuilder()
    v1 = theta.add_node()
    v2 = theta.add_node()
    # Three edges between v1 and v2
    theta.add_edge(v1, v2)
    theta.add_edge(v1, v2)
    theta.add_edge(v1, v2)

    theta_poly = compute_tutte_polynomial(theta)
    print(f"Target: {theta_poly}")

    result = engine.synthesize(theta_poly)
    print(f"Success: {result.success}")
    if result.success:
        verified = engine.verify_synthesis(result, theta_poly)
        print(f"Verified: {verified}")


def demo_complex_synthesis():
    """
    Test synthesis on progressively more complex graphs.
    """
    print("\n" + "=" * 80)
    print("COMPLEX SYNTHESIS TESTS")
    print("=" * 80)

    engine = SynthesisEngine(verbose=False)

    # Test 1: Build K_4 from scratch
    print("\n--- Test 1: K_4 synthesis ---")
    k4 = create_complete_graph(4)
    k4_target = compute_tutte_polynomial(k4)
    print(f"Target: {k4_target}")

    result = engine.synthesize(k4_target)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Steps: {result.num_steps}")
        print(f"Verified: {engine.verify_synthesis(result, k4_target)}")

    # Test 2: Build a path and extend to cycle
    print("\n--- Test 2: C_6 synthesis (cycle) ---")
    c6 = create_cycle_graph(6)
    c6_target = compute_tutte_polynomial(c6)
    print(f"Target: {c6_target}")

    result = engine.synthesize(c6_target)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Steps: {result.num_steps}")
        print(f"Operations: {result.operations}")

    # Test 3: Product polynomial (two disjoint triangles)
    print("\n--- Test 3: Two disjoint K_3 (product) ---")
    k3 = create_complete_graph(3)
    k3_poly = compute_tutte_polynomial(k3)
    target = k3_poly * k3_poly  # T(K_3) * T(K_3)
    print(f"Target (K_3 x K_3): {target}")

    result = engine.synthesize(target)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Steps: {result.num_steps}")
        print(f"Operations: {result.operations}")
        print(f"Motifs: {result.motifs_used}")
        print(f"Verified: {engine.verify_synthesis(result, target)}")

    # Test 4: K_5 (more complex)
    print("\n--- Test 4: K_5 synthesis ---")
    k5 = create_complete_graph(5)
    k5_target = compute_tutte_polynomial(k5)
    print(f"Target spanning trees: {k5_target.num_spanning_trees()}")

    result = engine.synthesize(k5_target)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Steps: {result.num_steps}")
        verified = engine.verify_synthesis(result, k5_target)
        print(f"Verified: {verified}")

    # Test 5: Attempt Petersen graph (challenging)
    print("\n--- Test 5: Petersen graph synthesis ---")
    try:
        from tutte_test.build_rainbow_table import RainbowTable
        table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
        rt = RainbowTable.load(table_path)
        petersen_entry = rt.get_entry('Petersen')
        if petersen_entry:
            petersen_target = rt._entry_to_polynomial(petersen_entry)
            print(f"Target spanning trees: {petersen_target.num_spanning_trees()}")
            print(f"Target terms: {len(petersen_target.coefficients)}")

            result = engine.synthesize(petersen_target, max_depth=5)
            print(f"Success: {result.success}")
            if result.success:
                print(f"Steps: {result.num_steps}")
            else:
                print("Petersen graph is challenging - no simple construction found")
                print("(Would require specialized synthesis rules)")
    except Exception as e:
        print(f"Could not test Petersen: {e}")


if __name__ == "__main__":
    # First verify our computations match the rainbow table
    verify_against_rainbow_table()

    # Run novel synthesis tests (polynomials NOT in rainbow table)
    demo_novel_synthesis()

    # Run efficient synthesis tests
    demo_complex_synthesis()
    demo_efficient_synthesis()

    # Then run other demos (comment out for faster testing)
    # demo_k3_synthesis()
    # demo_polynomial_difference()
    # demo_k_cut_analysis()
    # demo_y_power_interpretation()
    # demo_rainbow_table_synthesis()
    # demo_synthesis_strategies()
