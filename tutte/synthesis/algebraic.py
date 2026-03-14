"""Algebraic Synthesis Engine for Tutte Polynomials.

This module implements the algebraic decomposition approach to computing
Tutte polynomials. Instead of graph-based tiling, it uses polynomial
algebra (GCD, division with remainder) to decompose target polynomials
into known components.

Core Algorithm:
1. Accept input graph → compute T(input) or use given polynomial
2. Add to lookup table with canonical key
3. Find all rainbow table entries sharing a polynomial factor with T(input)
4. Select largest minor by polynomial complexity
5. Construct algebraic cover: T(input) = T(minor) × Q + R
6. If R ≠ 0: recurse on remainder (fringe)
7. If R = 0: exact cover found
8. Verify T(1,1) matches Kirchhoff spanning tree count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..polynomial import TuttePolynomial
from ..graph import Graph
from ..lookup.core import RainbowTable, MinorEntry, GCDMinorIndex, load_default_table
from ..graphs.k_join import polynomial_divmod, polynomial_divide, tutte_k
from ..factorization import (
    polynomial_gcd, has_common_factor, monomial_content,
    primitive_part, find_divisibility_chain, try_factorize
)
from ..validation import verify_spanning_trees


# =============================================================================
# SYNTHESIS RESULT
# =============================================================================

@dataclass
class AlgebraicSynthesisResult:
    """Result of algebraic polynomial synthesis."""

    polynomial: TuttePolynomial
    decomposition: List[str] = field(default_factory=list)  # Names of components
    quotients: List[TuttePolynomial] = field(default_factory=list)  # Quotient polynomials
    remainder: TuttePolynomial = field(default_factory=TuttePolynomial.zero)
    recursion_depth: int = 0
    verified: bool = False  # T(1,1) check passed
    method: str = "algebraic"
    recipe: List[str] = field(default_factory=list)  # Human-readable steps

    def __repr__(self) -> str:
        status = "verified" if self.verified else "unverified"
        return f"AlgebraicSynthesisResult({self.polynomial}, {status}, depth={self.recursion_depth})"

    def to_expression(self) -> str:
        """Return algebraic expression for the decomposition."""
        if not self.decomposition:
            return str(self.polynomial)

        parts = []
        for i, name in enumerate(self.decomposition):
            if i < len(self.quotients) and not self.quotients[i].is_zero():
                parts.append(f"{name} × ({self.quotients[i]})")
            else:
                parts.append(name)

        expr = " + ".join(parts)
        if not self.remainder.is_zero():
            expr += f" + [{self.remainder}]"

        return expr


# =============================================================================
# ALGEBRAIC SYNTHESIS ENGINE
# =============================================================================

class AlgebraicSynthesisEngine:
    """Synthesis engine using algebraic decomposition.

    This engine computes Tutte polynomials by:
    1. Looking up known polynomials in rainbow table
    2. Finding polynomial factors shared with known graphs
    3. Decomposing via division: T = T(minor) × Q + R
    4. Recursively handling remainders
    """

    def __init__(
        self,
        table: Optional[RainbowTable] = None,
        verbose: bool = False,
        use_gcd_index: bool = True
    ):
        """Initialize algebraic synthesis engine.

        Args:
            table: Rainbow table for lookups (loads default if None)
            verbose: Print progress information
            use_gcd_index: Whether to use GCD-based minor relationships
        """
        self.table = table if table is not None else load_default_table()
        self.verbose = verbose
        self.use_gcd_index = use_gcd_index

        # Caches
        self._cache: Dict[str, AlgebraicSynthesisResult] = {}
        self._gcd_index: Optional[GCDMinorIndex] = None

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[AlgSynth] {msg}", flush=True)

    def _get_gcd_index(self) -> GCDMinorIndex:
        """Get or compute GCD index."""
        if self._gcd_index is None:
            self._log("Computing GCD index...")
            self._gcd_index = self.table.compute_gcd_relationships()
            self._log(f"GCD index has {len(self._gcd_index.shared_factor_graph)} entries")
        return self._gcd_index

    # =========================================================================
    # MAIN SYNTHESIS METHODS
    # =========================================================================

    def synthesize(
        self,
        graph: Graph,
        max_depth: int = 10
    ) -> AlgebraicSynthesisResult:
        """Main entry point: compute Tutte polynomial via algebraic decomposition.

        Args:
            graph: Graph to compute polynomial for
            max_depth: Maximum recursion depth

        Returns:
            AlgebraicSynthesisResult with computed polynomial
        """
        # Check cache
        cache_key = graph.canonical_key()
        if cache_key in self._cache:
            self._log(f"Cache hit: {cache_key[:16]}...")
            return self._cache[cache_key]

        self._log(f"Synthesizing graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

        # 1. Check rainbow table first
        cached = self.table.lookup(graph)
        if cached is not None:
            self._log("Direct rainbow table lookup")
            result = AlgebraicSynthesisResult(
                polynomial=cached,
                decomposition=["table_lookup"],
                verified=True,
                method="lookup",
                recipe=["Rainbow table lookup"]
            )
            self._cache[cache_key] = result
            return result

        # 2. Handle base cases
        if graph.edge_count() == 0:
            result = AlgebraicSynthesisResult(
                polynomial=TuttePolynomial.one(),
                decomposition=["empty"],
                verified=True,
                method="base_case",
                recipe=["Empty graph: T = 1"]
            )
            self._cache[cache_key] = result
            return result

        if graph.edge_count() == 1:
            result = AlgebraicSynthesisResult(
                polynomial=TuttePolynomial.x(),
                decomposition=["K_2"],
                verified=True,
                method="base_case",
                recipe=["Single edge: T = x"]
            )
            self._cache[cache_key] = result
            return result

        # 3. Check if graph is disconnected
        components = graph.connected_components()
        if len(components) > 1:
            result = self._synthesize_disconnected(components, max_depth)
            self._cache[cache_key] = result
            return result

        # 4. Use algebraic decomposition for connected graphs
        result = self._synthesize_algebraic(graph, max_depth)
        self._cache[cache_key] = result
        return result

    def synthesize_from_polynomial(
        self,
        target: TuttePolynomial,
        max_depth: int = 10
    ) -> AlgebraicSynthesisResult:
        """Decompose a known polynomial algebraically.

        This is useful when you already have the polynomial and want to
        find its decomposition in terms of known graph polynomials.

        Args:
            target: Polynomial to decompose
            max_depth: Maximum recursion depth

        Returns:
            AlgebraicSynthesisResult with decomposition
        """
        return self._decompose_algebraically(target, max_depth, [])

    # =========================================================================
    # DECOMPOSITION METHODS
    # =========================================================================

    def _synthesize_disconnected(
        self,
        components: List[Graph],
        max_depth: int
    ) -> AlgebraicSynthesisResult:
        """Synthesize polynomial for disconnected graph.

        For disconnected graphs: T(G₁ ∪ G₂ ∪ ...) = T(G₁) × T(G₂) × ...
        """
        self._log(f"Disconnected graph: {len(components)} components")

        poly = TuttePolynomial.one()
        decomposition = []
        recipe = [f"Disconnected: {len(components)} components"]

        for i, comp in enumerate(components):
            comp_result = self.synthesize(comp, max_depth)
            poly = poly * comp_result.polynomial
            decomposition.extend(comp_result.decomposition)
            recipe.append(f"  Component {i+1}: {comp_result.polynomial}")

        recipe.append(f"Product: {poly}")

        return AlgebraicSynthesisResult(
            polynomial=poly,
            decomposition=decomposition,
            verified=True,
            method="disjoint_union",
            recipe=recipe
        )

    def _synthesize_algebraic(
        self,
        graph: Graph,
        max_depth: int
    ) -> AlgebraicSynthesisResult:
        """Synthesize polynomial using algebraic decomposition.

        Algorithm:
        1. Compute polynomial via deletion-contraction (baseline)
        2. Find factors from rainbow table
        3. Express as product of known polynomials
        """
        # First, compute the polynomial using a fallback method
        # (We need the polynomial to perform algebraic decomposition)
        poly = self._compute_via_fallback(graph)

        # Now decompose it algebraically
        result = self._decompose_algebraically(poly, max_depth, [])

        # Verify against Kirchhoff
        result.verified = verify_spanning_trees(graph, result.polynomial)

        return result

    def _decompose_algebraically(
        self,
        target: TuttePolynomial,
        depth: int,
        recipe: List[str]
    ) -> AlgebraicSynthesisResult:
        """Recursively decompose polynomial via division.

        Algorithm:
        1. Find largest polynomial factor from rainbow table
        2. Divide: target = factor × quotient + remainder
        3. If remainder ≠ 0, recursively decompose remainder
        4. Express as sum of products
        """
        if depth <= 0:
            return AlgebraicSynthesisResult(
                polynomial=target,
                remainder=target,
                recursion_depth=0,
                recipe=recipe + ["Max depth reached"]
            )

        recipe = list(recipe)
        recipe.append(f"Decomposing: {target}")

        # Check if target is already in rainbow table
        existing = self.table.find_by_polynomial(target)
        if existing is not None:
            recipe.append(f"  = {existing.name}")
            return AlgebraicSynthesisResult(
                polynomial=target,
                decomposition=[existing.name],
                verified=True,
                recursion_depth=0,
                recipe=recipe
            )

        # Find the largest factor
        factor_result = self._find_largest_factor(target)

        if factor_result is None:
            # No factors found - target is the remainder
            recipe.append("  No known factors found")
            return AlgebraicSynthesisResult(
                polynomial=target,
                remainder=target,
                recursion_depth=0,
                recipe=recipe
            )

        entry, quotient = factor_result
        recipe.append(f"  Factor: {entry.name} ({entry.polynomial})")

        # Compute remainder
        quotient, remainder = polynomial_divmod(target, entry.polynomial)

        if remainder.is_zero():
            # Exact division - continue decomposing quotient if non-trivial
            recipe.append(f"  Quotient: {quotient}")
            recipe.append(f"  Remainder: 0 (exact)")

            if quotient == TuttePolynomial.one():
                return AlgebraicSynthesisResult(
                    polynomial=target,
                    decomposition=[entry.name],
                    quotients=[TuttePolynomial.one()],
                    verified=True,
                    recursion_depth=1,
                    recipe=recipe
                )

            # Recursively decompose quotient
            sub_result = self._decompose_algebraically(quotient, depth - 1, [])

            return AlgebraicSynthesisResult(
                polynomial=target,
                decomposition=[entry.name] + sub_result.decomposition,
                quotients=[quotient] + sub_result.quotients,
                remainder=sub_result.remainder,
                recursion_depth=sub_result.recursion_depth + 1,
                verified=sub_result.verified,
                recipe=recipe + sub_result.recipe
            )
        else:
            # Non-zero remainder - need to handle fringe
            recipe.append(f"  Quotient: {quotient}")
            recipe.append(f"  Remainder: {remainder}")

            # Handle the remainder recursively
            remainder_result = self._handle_remainder(remainder, depth - 1, [])

            return AlgebraicSynthesisResult(
                polynomial=target,
                decomposition=[entry.name] + remainder_result.decomposition,
                quotients=[quotient] + remainder_result.quotients,
                remainder=remainder_result.remainder,
                recursion_depth=max(1, remainder_result.recursion_depth + 1),
                recipe=recipe + remainder_result.recipe
            )

    def _find_largest_factor(
        self,
        target: TuttePolynomial
    ) -> Optional[Tuple[MinorEntry, TuttePolynomial]]:
        """Find the largest rainbow table entry that divides the target.

        Uses spanning tree count pre-filtering and polynomial division
        to find exact factors.

        Returns:
            (entry, quotient) if factor found, None otherwise
        """
        factors = self.table.find_factors_of(target)

        if not factors:
            return None

        # Return the largest factor (already sorted by complexity)
        return factors[0]

    def _find_largest_gcd_minor(
        self,
        target: TuttePolynomial
    ) -> Optional[Tuple[MinorEntry, TuttePolynomial]]:
        """Find the rainbow table entry sharing the largest factor with target.

        This finds entries where gcd(T(entry), target) is non-trivial,
        useful for finding structural relationships even when exact
        division isn't possible.

        Returns:
            (entry, gcd) if common factor found, None otherwise
        """
        target_trees = target.num_spanning_trees()
        best_entry = None
        best_gcd = None
        best_degree = 0

        for entry in self.table.entries.values():
            # Quick pre-filter
            if entry.spanning_trees > 0 and target_trees > 0:
                from math import gcd as math_gcd
                if math_gcd(entry.spanning_trees, target_trees) == 1:
                    continue

            if has_common_factor(entry.polynomial, target):
                gcd = polynomial_gcd(entry.polynomial, target)
                degree = gcd.total_degree()

                if degree > best_degree:
                    best_entry = entry
                    best_gcd = gcd
                    best_degree = degree

        if best_entry is not None:
            return (best_entry, best_gcd)
        return None

    def _handle_remainder(
        self,
        remainder: TuttePolynomial,
        depth: int,
        recipe: List[str]
    ) -> AlgebraicSynthesisResult:
        """Handle non-zero remainder from division.

        Strategies:
        1. Check if remainder is a known polynomial
        2. Try to decompose remainder recursively
        3. Accept remainder as irreducible fringe
        """
        recipe = list(recipe)
        recipe.append(f"Handling remainder: {remainder}")

        if remainder.is_zero():
            return AlgebraicSynthesisResult(
                polynomial=remainder,
                verified=True,
                recipe=recipe
            )

        # Check if remainder is known
        existing = self.table.find_by_polynomial(remainder)
        if existing is not None:
            recipe.append(f"  Remainder = {existing.name}")
            return AlgebraicSynthesisResult(
                polynomial=remainder,
                decomposition=[existing.name],
                verified=True,
                recipe=recipe
            )

        # Try to decompose remainder
        if depth > 0:
            sub_result = self._decompose_algebraically(remainder, depth - 1, [])
            if sub_result.decomposition:
                return AlgebraicSynthesisResult(
                    polynomial=remainder,
                    decomposition=sub_result.decomposition,
                    quotients=sub_result.quotients,
                    remainder=sub_result.remainder,
                    recursion_depth=sub_result.recursion_depth,
                    recipe=recipe + sub_result.recipe
                )

        # Remainder is irreducible
        recipe.append("  Remainder is irreducible")
        return AlgebraicSynthesisResult(
            polynomial=remainder,
            remainder=remainder,
            recipe=recipe
        )

    # =========================================================================
    # FALLBACK COMPUTATION
    # =========================================================================

    def _compute_via_fallback(self, graph: Graph) -> TuttePolynomial:
        """Compute polynomial using HybridSynthesisEngine.

        This delegates to the hybrid engine which uses spanning tree
        expansion + pattern recognition instead of deletion-contraction.
        """
        from .hybrid import HybridSynthesisEngine

        self._log(f"Computing via hybrid synthesis: {graph.edge_count()} edges")

        engine = HybridSynthesisEngine(table=self.table, verbose=self.verbose)
        return engine.synthesize(graph).polynomial


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def algebraic_synthesize(
    graph: Graph,
    verbose: bool = False
) -> AlgebraicSynthesisResult:
    """Convenience function to synthesize polynomial algebraically.

    Args:
        graph: Graph to compute polynomial for
        verbose: Print progress information

    Returns:
        AlgebraicSynthesisResult with computed polynomial
    """
    engine = AlgebraicSynthesisEngine(verbose=verbose)
    return engine.synthesize(graph)


def decompose_polynomial(
    polynomial: TuttePolynomial,
    verbose: bool = False
) -> AlgebraicSynthesisResult:
    """Decompose a known polynomial into algebraic factors.

    Args:
        polynomial: Polynomial to decompose
        verbose: Print progress information

    Returns:
        AlgebraicSynthesisResult with decomposition
    """
    engine = AlgebraicSynthesisEngine(verbose=verbose)
    return engine.synthesize_from_polynomial(polynomial)
