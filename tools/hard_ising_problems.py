"""Hard Ising problems with known optimal ground state energies for QAOA benchmarking.

These problems are specifically chosen to challenge low-depth QAOA:
  - Large enough that random sampling can't cover the state space
    (16-20 qubits = 65K-1M states, vs ~4K final shots)
  - Dense connectivity creating many frustrated interactions
  - Spin glass disorder with competing couplings and fields

All optimal energies are verified by brute force (feasible for <= 20 nodes).
Fixed random seeds ensure reproducibility.

Each problem is a tuple: (h, J, optimal_energy, description)
  h: dict of linear biases {node: bias}
  J: dict of quadratic couplings {(node1, node2): coupling}
  optimal_energy: known ground state energy (brute-force verified)
  description: string description
"""

import itertools
import numpy as np


def _energy_of_config(spins, h, J, nodes):
    """Compute Ising energy for a spin configuration."""
    energy = 0.0
    node_to_spin = dict(zip(nodes, spins))
    for node, bias in h.items():
        energy += bias * node_to_spin[node]
    for (u, v), coupling in J.items():
        energy += coupling * node_to_spin[u] * node_to_spin[v]
    return energy


def _brute_force_optimal(nodes, h, J):
    """Find exact ground state energy by exhaustive search."""
    best = float('inf')
    for bits in itertools.product([-1, 1], repeat=len(nodes)):
        e = _energy_of_config(list(bits), h, J, nodes)
        if e < best:
            best = e
    return best


def _build_problems():
    """Build hard problems for QAOA benchmarking.

    Key design principle: problems must be large enough that the final
    sampling (typically 4096 shots) covers only a tiny fraction of the
    state space.  At 16 qubits, 4096 shots / 65536 states = 6%.
    At 18 qubits, 4096 / 262144 = 1.6%.  At 20 qubits, 4096 / 1M = 0.4%.
    This means QAOA actually has to find good solutions, not just
    stumble onto them.
    """
    problems = []

    # Problem 0: Dense spin glass, 16 nodes, ~65% connectivity
    # 2^16 = 65536 states.  4096 shots covers ~6%.
    rng16 = np.random.RandomState(31)
    nodes_16 = list(range(16))
    edges_16 = [(i, j) for i in range(16) for j in range(i + 1, 16)
                if rng16.random() < 0.65]
    h_16 = {i: float(rng16.uniform(-1.5, 1.5)) for i in nodes_16}
    J_16 = {e: float(rng16.choice([-1.0, 1.0])) for e in edges_16}
    opt_16 = _brute_force_optimal(nodes_16, h_16, J_16)
    problems.append((
        h_16, J_16, opt_16,
        f"16-spin dense spin glass ({len(edges_16)} edges, seed=31)"
    ))

    # Problem 1: Fully connected spin glass, 16 nodes
    # All 120 edges with random couplings.  Maximum frustration.
    rng16f = np.random.RandomState(88)
    edges_16f = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    h_16f = {i: float(rng16f.uniform(-1.0, 1.0)) for i in nodes_16}
    J_16f = {e: float(rng16f.choice([-1.0, 1.0])) for e in edges_16f}
    opt_16f = _brute_force_optimal(nodes_16, h_16f, J_16f)
    problems.append((
        h_16f, J_16f, opt_16f,
        "16-spin complete spin glass (120 edges, seed=88)"
    ))

    # Problem 2: Frustrated 4x4 grid with diagonals, 16 nodes
    # Grid + diagonals = triangular frustration everywhere.
    nodes_grid = list(range(16))
    edges_grid = []
    rng_grid = np.random.RandomState(44)
    for r in range(4):
        for c in range(4):
            i = r * 4 + c
            if c < 3:
                edges_grid.append((i, i + 1))
            if r < 3:
                edges_grid.append((i, i + 4))
            if r < 3 and c < 3:
                edges_grid.append((i, i + 5))  # diagonal down-right
            if r < 3 and c > 0:
                edges_grid.append((i, i + 3))  # diagonal down-left
    J_grid = {e: float(rng_grid.choice([-1.0, 1.0])) for e in edges_grid}
    h_grid = {i: float(rng_grid.uniform(-1.0, 1.0)) for i in nodes_grid}
    opt_grid = _brute_force_optimal(nodes_grid, h_grid, J_grid)
    problems.append((
        h_grid, J_grid, opt_grid,
        f"16-spin 4x4 grid+diagonals ({len(edges_grid)} edges, seed=44)"
    ))

    # Problem 3: Dense spin glass, 18 nodes, ~60% connectivity
    # 2^18 = 262144 states.  4096 shots covers ~1.6%.
    rng18 = np.random.RandomState(13)
    nodes_18 = list(range(18))
    edges_18 = [(i, j) for i in range(18) for j in range(i + 1, 18)
                if rng18.random() < 0.6]
    h_18 = {i: float(rng18.uniform(-1.5, 1.5)) for i in nodes_18}
    J_18 = {e: float(rng18.choice([-1.0, 1.0])) for e in edges_18}
    opt_18 = _brute_force_optimal(nodes_18, h_18, J_18)
    problems.append((
        h_18, J_18, opt_18,
        f"18-spin dense spin glass ({len(edges_18)} edges, seed=13)"
    ))

    # Problem 4: Competing fields spin glass, 18 nodes
    # Strong fields (up to +/-3.0) fighting random couplings.
    rng18c = np.random.RandomState(71)
    edges_18c = [(i, j) for i in range(18) for j in range(i + 1, 18)
                 if rng18c.random() < 0.55]
    h_18c = {i: float(rng18c.uniform(-3.0, 3.0)) for i in nodes_18}
    J_18c = {e: float(rng18c.choice([-1.0, 1.0])) for e in edges_18c}
    opt_18c = _brute_force_optimal(nodes_18, h_18c, J_18c)
    problems.append((
        h_18c, J_18c, opt_18c,
        f"18-spin strong-field spin glass ({len(edges_18c)} edges, seed=71)"
    ))

    # Problem 5: Dense spin glass, 20 nodes, ~55% connectivity
    # 2^20 = 1,048,576 states.  4096 shots covers ~0.4%.
    # This is the hardest problem — brute force takes a few seconds.
    rng20 = np.random.RandomState(7)
    nodes_20 = list(range(20))
    edges_20 = [(i, j) for i in range(20) for j in range(i + 1, 20)
                if rng20.random() < 0.55]
    h_20 = {i: float(rng20.uniform(-2.0, 2.0)) for i in nodes_20}
    J_20 = {e: float(rng20.choice([-1.0, 1.0])) for e in edges_20}
    opt_20 = _brute_force_optimal(nodes_20, h_20, J_20)
    problems.append((
        h_20, J_20, opt_20,
        f"20-spin dense spin glass ({len(edges_20)} edges, seed=7)"
    ))

    # Problem 6: Fully connected, 18 nodes
    # 153 edges, all random.  Extreme frustration density.
    rng18full = np.random.RandomState(23)
    edges_18full = [(i, j) for i in range(18) for j in range(i + 1, 18)]
    h_18full = {i: float(rng18full.uniform(-0.5, 0.5)) for i in nodes_18}
    J_18full = {e: float(rng18full.choice([-1.0, 1.0])) for e in edges_18full}
    opt_18full = _brute_force_optimal(nodes_18, h_18full, J_18full)
    problems.append((
        h_18full, J_18full, opt_18full,
        "18-spin complete spin glass (153 edges, seed=23)"
    ))

    return problems


HARD_ISING_PROBLEMS = _build_problems()