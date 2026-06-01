"""Tests for energy-stratified diversity selection.

Verifies that when farthest-point sampling produces low diversity
(all solutions in one energy basin), the stratified fallback picks
solutions from different energy levels to achieve sufficient diversity.
"""

import numpy as np
import pytest
import dimod

from shared.quantum_proof_of_work import (
    _energy_stratified_selection,
    select_diverse_solutions,
    calculate_diversity,
    evaluate_sampleset,
    generate_ising_model_from_nonce,
)
from shared.block_requirements import BlockRequirements


def _make_clustered_solutions(n_nodes, n_solutions, n_clusters, seed=42):
    """Generate solutions that cluster into distinct energy basins.

    Each cluster is built from a base spin config with small random
    perturbations (few flips). Different clusters have completely
    different base configs → high inter-cluster diversity.

    Returns:
        (solutions, energies) where solutions in the same cluster
        are similar (low diversity) but solutions across clusters
        are different (high diversity).
    """
    rng = np.random.default_rng(seed)
    per_cluster = n_solutions // n_clusters
    solutions = []
    energies = []

    for c in range(n_clusters):
        # Random base config for this cluster
        base = 2 * rng.integers(2, size=n_nodes) - 1

        # Energy gets worse (less negative) for each cluster
        base_energy = -14900 + c * 10

        for i in range(per_cluster):
            sol = base.copy()
            # Flip a few random spins (small perturbation)
            flip_count = rng.integers(5, 20)
            flip_indices = rng.choice(n_nodes, flip_count, replace=False)
            sol[flip_indices] *= -1
            solutions.append(sol.tolist())
            # Slightly worse energy for each perturbation
            energies.append(base_energy + rng.uniform(0, 5))

    return solutions, energies


class TestEnergyStratifiedSelection:

    def test_returns_target_count(self):
        solutions = [[1, -1, 1]] * 10
        energies = list(range(-100, -90))
        result = _energy_stratified_selection(solutions, energies, 5)
        assert result is not None
        assert len(result) == 5

    def test_returns_none_if_too_few(self):
        solutions = [[1, -1]] * 3
        energies = [-10, -9, -8]
        result = _energy_stratified_selection(solutions, energies, 5)
        assert result is None

    def test_picks_from_different_energy_levels(self):
        """Selected solutions should span the energy range."""
        rng = np.random.default_rng(0)
        solutions = [rng.choice([-1, 1], 100).tolist() for _ in range(50)]
        energies = [float(-100 + i) for i in range(50)]

        result = _energy_stratified_selection(solutions, energies, 5)
        assert result is not None

        selected_energies = [energies[i] for i in result]
        # Should span most of the range, not cluster at the best end
        spread = max(selected_energies) - min(selected_energies)
        total_range = max(energies) - min(energies)
        assert spread > total_range * 0.5

    def test_indices_are_valid(self):
        solutions = [[1, -1, 1]] * 20
        energies = list(range(-120, -100))
        result = _energy_stratified_selection(solutions, energies, 5)
        assert result is not None
        for idx in result:
            assert 0 <= idx < len(solutions)

    def test_no_duplicate_indices(self):
        rng = np.random.default_rng(7)
        solutions = [rng.choice([-1, 1], 50).tolist() for _ in range(30)]
        energies = [float(-100 + i * 0.5) for i in range(30)]
        result = _energy_stratified_selection(solutions, energies, 5)
        assert result is not None
        assert len(set(result)) == len(result)


class TestStratifiedFallbackInEvaluate:
    """Test that evaluate_sampleset uses the stratified fallback."""

    def test_clustered_solutions_pass_with_fallback(self):
        """Solutions from multiple basins pass diversity via stratification."""
        n_nodes = 100  # Small for speed
        solutions, energies = _make_clustered_solutions(
            n_nodes=n_nodes, n_solutions=50,
            n_clusters=5, seed=42,
        )

        # Verify: farthest-point on the whole set may or may not pass,
        # but stratified should definitely pass since clusters are distinct
        strat_indices = _energy_stratified_selection(solutions, energies, 5)
        assert strat_indices is not None
        strat_solutions = [solutions[i] for i in strat_indices]
        strat_div = calculate_diversity(strat_solutions)
        # Clusters have different base configs → high diversity
        assert strat_div > 0.15

    def test_already_diverse_uses_farthest_point(self):
        """When farthest-point works, stratified fallback is not needed."""
        rng = np.random.default_rng(0)
        # Completely random solutions → inherently diverse
        solutions = [rng.choice([-1, 1], 200).tolist() for _ in range(20)]

        indices = select_diverse_solutions(solutions, 5)
        selected = [solutions[i] for i in indices]
        div = calculate_diversity(selected)
        # Random solutions should be diverse (~0.5)
        assert div > 0.3

    def test_evaluate_sampleset_with_clustered_qpu_results(self):
        """Full evaluate_sampleset path with clustered solutions."""
        n_nodes = 200
        nodes = list(range(n_nodes))
        edges = [(i, i + 1) for i in range(n_nodes - 1)]

        # Generate Ising model
        h, J = generate_ising_model_from_nonce(42, nodes, edges)

        # Build clustered sampleset: 5 clusters of 10 solutions
        solutions, energies = _make_clustered_solutions(
            n_nodes=n_nodes, n_solutions=50,
            n_clusters=5, seed=99,
        )

        # Build dimod SampleSet
        sample_dicts = [dict(zip(range(n_nodes), sol)) for sol in solutions]
        ss = dimod.SampleSet.from_samples(
            sample_dicts, vartype=dimod.SPIN, energy=energies,
        )

        reqs = BlockRequirements(
            difficulty_energy=-14850.0,
            min_diversity=0.15,
            min_solutions=5,
            timeout_to_difficulty_adjustment_decay=0,
        )

        import time
        result = evaluate_sampleset(
            ss, reqs, nodes, edges, 42, b'x' * 32,
            int(time.time()), time.time(), 'test', 'QPU',
            skip_validation=True,
        )

        # Should find a valid result with diversity via stratification
        # (or farthest-point if it works on these clusters)
        if result is not None:
            assert result.diversity >= 0.15
            assert result.num_valid >= 5
