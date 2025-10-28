"""Energy adjustment utilities for quantum blockchain."""

import math
import numpy as np
from typing import Dict, Tuple, List, Any, Optional


def expected_solution_energy(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    c: float = 0.75,
    h_values: Optional[List[float]] = None
) -> float:
    """Calculate expected ground state energy for random Ising problems on a given topology.

    Based on empirical observations that expected energy density (GSE/N) scales with √(degree).
    This formula accounts for topology dimensionality/connectivity effects on achievable energy.

    Theory and Calibration:
    -----------------------
    For random Ising problems with J ∈ {-1, +1} couplings and h field params, the expected ground state
    energy (GSE) follows an empirical scaling law:

        GSE ≈ -c × √(avg_degree) × N

    Where:
        - N = number of nodes (qubits)
        - M = number of edges (couplings)
        - avg_degree = 2M / N (average node degree in the graph)
        - c = empirical constant representing connectivity effects (default 0.75)

    The √(degree) scaling captures how solution quality improves with connectivity, while
    the linear N scaling reflects extensive energy growth with system size.

    Example Calibration (Advantage2 topology):
    --------------------------------------------
    - N = 4,593 nodes
    - M = 41,796 edges
    - avg_degree = 2 × 41,796 / 4,593 ≈ 18.2
    - √(avg_degree) ≈ 4.27
    - Observed GSE ≈ -15,700
    - Implied c ≈ 15,700 / (4.27 × 4,593) ≈ 0.80

    Using c = 0.75 yields: -0.75 × 4.27 × 4,593 ≈ -14,709

    Statistical Variation:
    ---------------------
    Individual nonces will fluctuate around this expectation by approximately ±√M due to
    central limit theorem effects on independent random couplings. For Advantage2, this
    means ±√41,796 ≈ ±204 energy units of nonce-to-nonce variation.

    The above tracks well with our practical observations, which show energies around -14,200
    with a standard deviation of ~200 when we aren't spending significant compute time 
    on annealing, and better ranges when we do. You can run the tool in

    Problem Bounds:
    ------------------
    - Theoretical minimum: -M (all edges satisfied, unachievable for frustrated systems)
    - Practical SA solutions: Typically achieve ~35% of theoretical minimum (-14,709 vs -41,796)
      but ~80% of the empirical expected energy (-14,200 vs -14,709 from formula) unless 
      we work harder or search across random problems.
    - This formula provides a statistical expectation for real-world performance

    Args:
        nodes: List of node indices for the topology
        edges: List of edge tuples for the topology
        c: Empirical constant (default 0.75, calibrated from Advantage2 data)
        h_values: List of allowed h field values (default: [-1, 0, +1])
                 Use [0] for h=0 baseline (backward compatible)

    Returns:
        Expected ground state energy (negative value)

    Note on h_values impact:
        For h=0: GSE ≈ -c × √(avg_degree) × N
        For h≠0: h allows slightly deeper energy minima (h+J cooperation)
                 Formula predicts ~400-500 units better than observed SA
                 (due to local minima, limited sweeps, sample variance)

    Example:
        >>> # Advantage2 topology
        >>> nodes = list(range(4593))
        >>> edges = [(i, j) for i in range(4592) for j in range(i+1, 4593) if connected(i, j)]
        >>> expected_energy = expected_solution_energy(nodes, edges)
        >>> print(f"Expected GSE: {expected_energy:.1f}")
        Expected GSE: -14709.0
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]  # Default: ternary distribution

    N = len(nodes)
    M = len(edges)

    # Handle edge cases
    if N == 0 or M == 0:
        return 0.0

    # Calculate average node degree
    avg_degree = (2.0 * M) / N

    # Apply empirical scaling formula: GSE ≈ -c × √(avg_degree) × N
    j_contribution = -c * math.sqrt(avg_degree) * N

    # h field contribution (allows slightly smaller energies)
    #
    # Why h makes smaller gse:
    # ---------------------------
    # When h ≠ 0, nodes have local field biases (h_i wants spin s_i to align).
    # For edges (i,j) with coupling J_ij, three scenarios occur:
    #   1. h_i and J_ij cooperate (both want same spin alignment) → bonus energy reduction
    #   2. h_i and J_ij conflict (want opposite alignments) → frustration
    #   3. h_i = 0 (no local bias) → only J matters
    #
    # Since J ∈ {-1, +1} is 50/50 random and h is symmetric (zero mean):
    #   - ~25% of edges have h+J cooperation (both nodes' h and J agree)
    #   - SA can exploit these cooperative cases to achieve lower energies
    #   - Net effect: h ≠ 0 allows deeper energy minima than h = 0
    #
    # Empirical observations (Advantage2, 10 samples, num_sweeps=256):
    #   - h = 0:          SA achieves -14,286 (baseline)
    #   - h ∈ {-1, 0, +1}: SA achieves -14,759 (-473 better, ternary: 67% nonzero)
    #   - h ∈ {-1, +1}:    SA achieves -14,957 (-671 better, binary: 100% nonzero)
    #
    # Formula derivation:
    # ------------------
    # The h contribution scales as N/√(avg_degree) because:
    #   - N nodes each have local h bias
    #   - Higher degree → more J constraints → less freedom to satisfy h
    #   - Lower degree → fewer J constraints → more freedom to satisfy h
    #
    # The contribution is also scaled by c (SA efficiency factor) because:
    #   - Both J and h are optimization targets that SA tries to satisfy
    #   - SA's ability to exploit h+J cooperation is limited by the same
    #     algorithmic factors (temperature schedule, sweeps, local minima)
    #   - Using the same c ensures consistent scaling across problem difficulties
    if len(h_values) == 1 and h_values[0] == 0.0:
        h_contribution = 0.0
    else:
        # NOTE: This might not work well if
        # len(h) == 1 && h[0] = 1, for example...
        # We assume h is going to be some variant of [1, -1], [-1, 0, 1]
        # or equivalent.
        # Fraction of non-zero h values (ternary: 2/3, binary: 1.0)
        nonzero_fraction = sum(1 for v in h_values if v != 0) / len(h_values)

        # Empirical constant calibrated from experimental data:
        # - Binary h ∈ {-1, +1}: observed -671 units better than h=0
        # - Ternary h ∈ {-1, 0, +1}: observed -473 units better than h=0
        # - Ratio: -473/-671 ≈ 0.705 ≈ 2/3 (matches nonzero_fraction)
        #
        # This suggests linear scaling: h_contribution ∝ nonzero_fraction
        # (Even single node with h≠0 can exploit favorable J alignments)
        #
        # Solving for α:
        #   -671 ≈ -c × α × 1.0 × N/√(d)
        #   -671 ≈ -0.75 × α × 1075
        #   α ≈ 0.88
        # (this is roughly half the 25% h, J cooperation noted above)
        alpha = 0.88

        # h contribution: -c × α × nonzero_fraction × N/√(avg_degree)
        # Linear scaling with nonzero_fraction (not quadratic) because
        # even a single node with h≠0 helps, not just h+h pairs
        h_contribution = -c * alpha * nonzero_fraction * N / math.sqrt(avg_degree)

    return j_contribution + h_contribution


def adjust_energy_along_curve(
    current_energy: float,
    adjustment_rate: float,
    direction: str,
    nodes: List[int] = None,
    edges: List[Tuple[int, int]] = None,
    h_values: Optional[List[float]] = None,
    c_range: Tuple[float, float] = (0.717, 0.742)
) -> float:
    """Adjust energy along a curve that compresses adjustments near boundaries.

    Dynamically calculates min/knee/max energy using the GSE formula based on topology.
    This allows the function to work with any topology without hardcoded values.

    If nodes/edges not provided, falls back to Advantage2 calibrated values.

    The c_range represents SA efficiency at different computational efforts:
    - c_easy (0.717): 64 sweeps, quick mining
    - c_hard (0.742): 1024 sweeps, high-quality mining
    These values were empirically determined from Advantage2 calibration data.

    Args:
        current_energy: Current energy value
        adjustment_rate: Percentage to move (e.g., 0.05 for 5%)
        direction: 'harder' (more negative) or 'easier' (less negative)
        nodes: List of node IDs in topology (optional, for dynamic calculation)
        edges: List of edge tuples in topology (optional, for dynamic calculation)
        h_values: h field distribution (optional, default: [-1, 0, 1])
        c_range: (c_easy, c_hard) SA efficiency range (default: 0.717, 0.742)
                 Calibrated from Advantage2: 64 sweeps → c=0.717, 1024 sweeps → c=0.742

    Returns:
        New energy value after curve-based adjustment
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]

    # Calculate min/knee/max dynamically using GSE if topology provided
    if nodes is not None and edges is not None:
        c_easy, c_hard = c_range
        c_knee = (c_easy + c_hard) / 2

        # Use GSE formula to predict energy range
        min_energy = expected_solution_energy(nodes, edges, c=c_hard, h_values=h_values)
        knee_energy = expected_solution_energy(nodes, edges, c=c_knee, h_values=h_values)
        max_energy = expected_solution_energy(nodes, edges, c=c_easy, h_values=h_values)
    else:
        # Fall back to Advantage2 calibrated values (h ∈ {-1, 0, +1})
        # Based on experimental calibration (10 min per config, Metal GPU, 2025-01-27)
        # These match GSE predictions with c_range=(0.70, 0.85) for Advantage2
        min_energy = -15009.0   # Hardest difficulty (c=0.85, 1024 sweeps)
        knee_energy = -14810.0  # Mid-range difficulty (c=0.775, 512 sweeps)
        max_energy = -14550.0   # Easiest difficulty (c=0.70, 64 sweeps)
    
    # Convert energy to normalized position [0, 1] for observed range
    total_range = max_energy - min_energy
    
    # Handle out-of-range values with linear adjustment
    if current_energy < min_energy or current_energy > max_energy:
        linear_adjustment = total_range * adjustment_rate
        if direction == 'harder':
            return current_energy - linear_adjustment
        else:  # easier
            return current_energy + linear_adjustment
    
    # Normalize current position [0, 1]
    normalized_pos = (current_energy - min_energy) / total_range
    
    # Create curve using sqrt function
    # At position 0 (min_energy): curve_factor ≈ 0.1 (small adjustments)
    # At position 0.3 (knee): curve_factor ≈ 1.0 (full adjustments)  
    # At position 1 (max_energy): curve_factor ≈ 0.1 (small adjustments)
    
    knee_pos = (knee_energy - min_energy) / total_range  # ≈ 0.3
    
    if normalized_pos <= knee_pos:
        # Left side: increase from 0.1 to 1.0
        progress = normalized_pos / knee_pos
        curve_factor = 0.1 + 0.9 * math.sqrt(progress)
    else:
        # Right side: decrease from 1.0 to 0.1
        progress = (normalized_pos - knee_pos) / (1.0 - knee_pos)
        curve_factor = 1.0 - 0.9 * math.sqrt(progress)
    
    # Apply curved adjustment
    curved_adjustment = total_range * adjustment_rate * curve_factor
    
    if direction == 'harder':
        return current_energy - curved_adjustment
    else:  # easier
        return current_energy + curved_adjustment


class IsingModelValidator:
    """Validates Ising model solutions for correctness."""
    
    def __init__(self, h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int]):
        self.h = h
        self.J = J  
        self.nodes = nodes
        self.n = len(nodes)
        self.node_to_pos = {node_id: pos for pos, node_id in enumerate(nodes)}
        
    def validate_solution(self, spins: List[int], verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive validation of an Ising model solution.
        
        Args:
            spins: Spin configuration as list of {-1, +1} values
            verbose: Whether to print detailed analysis
            
        Returns:
            Dictionary with validation results
        """
        if verbose:
            print("🔍 Ising Model Solution Validation")
            print("=" * 40)
        
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "energy": 0.0,
            "energy_breakdown": {},
            "constraints": {},
            "statistics": {}
        }
        
        # 1. Basic Format Validation
        format_check = self._validate_format(spins)
        results.update(format_check)
        
        # 2. Energy Calculation Validation
        energy_check = self._validate_energy_calculation(spins)
        results.update(energy_check)
        
        # 3. Statistical Properties
        stats = self._analyze_statistics(spins)
        results["statistics"] = stats
        
        # 4. Coupling Satisfaction Analysis
        coupling_analysis = self._analyze_coupling_satisfaction(spins)
        results["constraints"] = coupling_analysis
        
        # 5. Overall Assessment
        overall = self._overall_assessment(results)
        results.update(overall)
        
        if verbose:
            self._print_validation_report(results)
            
        return results
    
    def _validate_format(self, spins: List[int]) -> Dict[str, Any]:
        """Validate basic solution format."""
        errors = []
        warnings = []
        
        # Check length
        if len(spins) != self.n:
            errors.append(f"Wrong solution length: {len(spins)} != {self.n}")
        
        # Check values are {-1, +1}
        unique_values = set(spins)
        if not unique_values.issubset({-1, 1}):
            invalid_values = unique_values - {-1, 1}
            errors.append(f"Invalid spin values: {invalid_values} (must be -1 or +1)")
        
        # Check for unusual patterns
        if len(spins) > 0:
            positive_count = sum(1 for s in spins if s == 1)
            negative_count = sum(1 for s in spins if s == -1)
            imbalance = abs(positive_count - negative_count) / len(spins)
            
            if imbalance > 0.8:
                warnings.append(f"Highly imbalanced solution: {positive_count}(+1) vs {negative_count}(-1)")
        
        return {"format_errors": errors, "format_warnings": warnings}
    
    def _validate_energy_calculation(self, spins: List[int]) -> Dict[str, Any]:
        """Validate energy calculation matches expected Ising formula."""
        
        # Calculate field energy: E_h = Σ h_i * s_i
        h_energy = 0.0
        for i in range(self.n):
            h_value = self.h.get(i, 0.0)
            h_energy += h_value * spins[i]
        
        # Calculate coupling energy: E_J = Σ J_ij * s_i * s_j
        j_energy = 0.0
        coupling_satisfactions = []
        
        for (node_i, node_j), val in self.J.items():
            pos_i = self.node_to_pos.get(int(node_i))
            pos_j = self.node_to_pos.get(int(node_j))
            
            if pos_i is not None and pos_j is not None:
                spin_i = spins[pos_i]
                spin_j = spins[pos_j]
                coupling_energy = val * spin_i * spin_j
                j_energy += coupling_energy
                
                # Track coupling satisfaction (negative energy = satisfied)
                coupling_satisfactions.append({
                    "edge": (node_i, node_j),
                    "J_value": val,
                    "spins": (spin_i, spin_j),
                    "energy": coupling_energy,
                    "satisfied": coupling_energy < 0
                })
        
        total_energy = h_energy + j_energy
        
        return {
            "energy": total_energy,
            "energy_breakdown": {
                "h_energy": h_energy,
                "j_energy": j_energy,
                "total": total_energy
            },
            "coupling_details": coupling_satisfactions
        }
    
    def _analyze_statistics(self, spins: List[int]) -> Dict[str, Any]:
        """Analyze statistical properties of the solution."""
        
        positive_spins = sum(1 for s in spins if s == 1)
        negative_spins = sum(1 for s in spins if s == -1)
        
        # Magnetization
        magnetization = sum(spins) / len(spins)
        
        # Local correlations (simple measure)
        correlations = []
        for (node_i, node_j), _ in self.J.items():
            pos_i = self.node_to_pos.get(int(node_i))
            pos_j = self.node_to_pos.get(int(node_j))
            if pos_i is not None and pos_j is not None:
                correlation = spins[pos_i] * spins[pos_j]
                correlations.append(correlation)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            "positive_spins": positive_spins,
            "negative_spins": negative_spins,
            "magnetization": magnetization,
            "avg_correlation": avg_correlation,
            "total_spins": len(spins)
        }
    
    def _analyze_coupling_satisfaction(self, spins: List[int]) -> Dict[str, Any]:
        """Analyze how well the solution satisfies coupling constraints."""
        
        satisfied_couplings = 0
        total_couplings = len(self.J)
        frustrated_couplings = []
        
        for (node_i, node_j), val in self.J.items():
            pos_i = self.node_to_pos.get(int(node_i))
            pos_j = self.node_to_pos.get(int(node_j))
            
            if pos_i is not None and pos_j is not None:
                spin_i = spins[pos_i]
                spin_j = spins[pos_j]
                coupling_energy = val * spin_i * spin_j
                
                if coupling_energy < 0:  # Satisfied (contributes negative energy)
                    satisfied_couplings += 1
                else:  # Frustrated (contributes positive energy)
                    frustrated_couplings.append({
                        "edge": (node_i, node_j),
                        "J_value": val,
                        "spins": (spin_i, spin_j),
                        "energy": coupling_energy
                    })
        
        satisfaction_rate = satisfied_couplings / total_couplings if total_couplings > 0 else 0
        
        return {
            "satisfied_couplings": satisfied_couplings,
            "total_couplings": total_couplings,
            "satisfaction_rate": satisfaction_rate,
            "frustrated_couplings": frustrated_couplings[:10],  # Show first 10
            "num_frustrated": len(frustrated_couplings)
        }
    
    def _overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide overall assessment of solution validity."""
        
        # Check for critical errors
        has_format_errors = len(results.get("format_errors", [])) > 0
        has_energy_issues = False
        
        # Energy sanity checks
        energy = results.get("energy", 0)
        satisfaction_rate = results.get("constraints", {}).get("satisfaction_rate", 0)
        
        # For max-cut problems (±1 couplings), expect ~50% satisfaction
        if satisfaction_rate > 0.95:  # >95% satisfaction is unrealistic
            has_energy_issues = True
            results.setdefault("warnings", []).append("Unrealistically high coupling satisfaction rate")
        
        # Check if energy is suspiciously good
        theoretical_min = -len(self.J)  # If all couplings could be satisfied
        if energy < theoretical_min * 0.9:  # Within 90% of theoretical optimum
            has_energy_issues = True
            results.setdefault("warnings", []).append("Energy suspiciously close to theoretical minimum")
        
        valid = not has_format_errors and not has_energy_issues
        
        return {
            "valid": valid,
            "assessment": "valid" if valid else "suspicious",
            "critical_issues": has_format_errors,
            "energy_concerns": has_energy_issues
        }
    
    def _print_validation_report(self, results: Dict[str, Any]):
        """Print detailed validation report."""
        
        print(f"\n📋 Validation Summary:")
        print(f"  Overall: {'✅ VALID' if results['valid'] else '❌ INVALID'}")
        
        # Format issues
        if results.get("format_errors"):
            print(f"\n❌ Format Errors:")
            for error in results["format_errors"]:
                print(f"  • {error}")
        
        if results.get("format_warnings"):
            print(f"\n⚠️ Format Warnings:")
            for warning in results["format_warnings"]:
                print(f"  • {warning}")
        
        # Energy breakdown
        energy_breakdown = results.get("energy_breakdown", {})
        print(f"\n⚡ Energy Breakdown:")
        print(f"  Field energy (h): {energy_breakdown.get('h_energy', 0):.1f}")
        print(f"  Coupling energy (J): {energy_breakdown.get('j_energy', 0):.1f}")
        print(f"  Total energy: {energy_breakdown.get('total', 0):.1f}")
        
        # Constraints
        constraints = results.get("constraints", {})
        satisfaction_rate = constraints.get("satisfaction_rate", 0)
        print(f"\n🔗 Coupling Analysis:")
        print(f"  Satisfied: {constraints.get('satisfied_couplings', 0)}/{constraints.get('total_couplings', 0)}")
        print(f"  Satisfaction rate: {satisfaction_rate:.1%}")
        print(f"  Frustrated couplings: {constraints.get('num_frustrated', 0)}")
        
        # Statistics
        stats = results.get("statistics", {})
        print(f"\n📊 Solution Statistics:")
        print(f"  Positive spins: {stats.get('positive_spins', 0)}")
        print(f"  Negative spins: {stats.get('negative_spins', 0)}")
        print(f"  Magnetization: {stats.get('magnetization', 0):.3f}")
        print(f"  Avg correlation: {stats.get('avg_correlation', 0):.3f}")
        
        # Assessment
        if results.get("warnings"):
            print(f"\n⚠️ Concerns:")
            for warning in results["warnings"]:
                print(f"  • {warning}")


def validate_sampler_solutions(sampler_name: str, result, h: Dict, J: Dict, nodes: List) -> Dict:
    """Validate all solutions from a sampler result."""
    
    print(f"\n🧪 Validating {sampler_name} Solutions")
    print("=" * 50)
    
    validator = IsingModelValidator(h, J, nodes)
    
    # Extract samples based on result type
    if hasattr(result, 'record'):  # Dimod SampleSet
        samples = [list(result.record.sample[i]) for i in range(len(result.record.sample))]
        energies = list(result.record.energy)
    else:  # Dictionary format
        samples = result.get('samples', [])
        energies = result.get('energies', [])
    
    print(f"Analyzing {len(samples)} solutions...")
    
    validation_results = []
    
    for i, (sample, energy) in enumerate(zip(samples, energies)):
        if i < 3:  # Validate first 3 solutions in detail
            print(f"\n--- Solution {i+1} (Energy: {energy:.1f}) ---")
            result = validator.validate_solution(sample, verbose=True)
        else:
            result = validator.validate_solution(sample, verbose=False)
        
        validation_results.append(result)
    
    # Summary analysis
    valid_count = sum(1 for r in validation_results if r["valid"])
    suspicious_count = len(validation_results) - valid_count
    
    avg_satisfaction = np.mean([r["constraints"]["satisfaction_rate"] for r in validation_results])
    
    print(f"\n📈 Overall Analysis:")
    print(f"  Valid solutions: {valid_count}/{len(validation_results)}")
    print(f"  Suspicious solutions: {suspicious_count}/{len(validation_results)}")
    print(f"  Average satisfaction rate: {avg_satisfaction:.1%}")
    
    if suspicious_count > len(validation_results) * 0.5:
        print(f"  🚨 ALGORITHM ISSUE: >50% of solutions are suspicious")
    elif suspicious_count > 0:
        print(f"  ⚠️ Some solutions may be unrealistic")
    else:
        print(f"  ✅ All solutions appear valid")
    
    return {
        "valid_count": valid_count,
        "suspicious_count": suspicious_count,
        "total_count": len(validation_results),
        "avg_satisfaction_rate": avg_satisfaction,
        "details": validation_results
    }
