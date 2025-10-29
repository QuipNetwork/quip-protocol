"""Hard-coded Ising problems with known optimal ground state energies for testing."""

# Each problem is a tuple: (h, J, optimal_energy, description)
# h: dict of linear biases {node: bias}
# J: dict of quadratic couplings {(node1, node2): coupling}
# optimal_energy: known ground state energy
# description: string description

BASIC_ISING_PROBLEMS = [
    # Problem 0: 2 spins, ferromagnetic
    ({
        0: 0.0,
        1: 0.0
    }, {
        (0, 1): -1.0
    }, -1.0, "2-spin ferromagnetic chain"),

    # Problem 1: 3 spins, ferromagnetic chain
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0
    }, {
        (0, 1): -1.0,
        (1, 2): -1.0
    }, -2.0, "3-spin ferromagnetic chain"),

    # Problem 2: 4 spins, square lattice
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0
    }, {
        (0, 1): -1.0,
        (1, 2): -1.0,
        (2, 3): -1.0,
        (3, 0): -1.0
    }, -4.0, "4-spin square lattice"),

    # Problem 3: 4 spins, complete graph
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0
    }, {
        (0, 1): -1.0,
        (0, 2): -1.0,
        (0, 3): -1.0,
        (1, 2): -1.0,
        (1, 3): -1.0,
        (2, 3): -1.0
    }, -6.0, "4-spin complete graph"),

    # Problem 4: 5 spins, pentagon
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0
    }, {
        (0, 1): -1.0,
        (1, 2): -1.0,
        (2, 3): -1.0,
        (3, 4): -1.0,
        (4, 0): -1.0
    }, -5.0, "5-spin pentagon"),

    # Problem 5: 6 spins, hexagonal lattice
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0
    }, {
        (0, 1): -1.0,
        (1, 2): -1.0,
        (2, 3): -1.0,
        (3, 4): -1.0,
        (4, 5): -1.0,
        (5, 0): -1.0
    }, -6.0, "6-spin hexagon"),

    # Problem 6: 8 spins, cube (3D)
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0
    }, {
        (0, 1): -1.0,
        (0, 3): -1.0,
        (0, 4): -1.0,
        (1, 2): -1.0,
        (1, 5): -1.0,
        (2, 3): -1.0,
        (2, 6): -1.0,
        (3, 7): -1.0,
        (4, 5): -1.0,
        (4, 7): -1.0,
        (5, 6): -1.0,
        (6, 7): -1.0
    }, -12.0, "8-spin cube"),

    # Problem 7: 10 spins, more complex
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0
    }, {
        (0, 1): -1.0,
        (0, 4): -1.0,
        (0, 5): -1.0,
        (1, 2): -1.0,
        (1, 6): -1.0,
        (2, 3): -1.0,
        (2, 7): -1.0,
        (3, 8): -1.0,
        (4, 5): -1.0,
        (4, 9): -1.0,
        (5, 6): -1.0,
        (6, 7): -1.0,
        (7, 8): -1.0,
        (8, 9): -1.0
    }, -14.0, "10-spin complex lattice"),

    # Problem 8: 12 spins, even more complex
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: 0.0,
        11: 0.0
    }, {
        (0, 1): -1.0,
        (0, 3): -1.0,
        (0, 4): -1.0,
        (1, 2): -1.0,
        (1, 5): -1.0,
        (2, 3): -1.0,
        (2, 6): -1.0,
        (3, 7): -1.0,
        (4, 5): -1.0,
        (4, 8): -1.0,
        (5, 6): -1.0,
        (5, 9): -1.0,
        (6, 7): -1.0,
        (6, 10): -1.0,
        (7, 11): -1.0,
        (8, 9): -1.0,
        (9, 10): -1.0,
        (10, 11): -1.0
    }, -18.0, "12-spin extended lattice"),

    # Problem 9: 16 spins, larger grid
    ({
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0,
        10: 0.0,
        11: 0.0,
        12: 0.0,
        13: 0.0,
        14: 0.0,
        15: 0.0
    }, {
        (0, 1): -1.0,
        (0, 4): -1.0,
        (1, 2): -1.0,
        (1, 5): -1.0,
        (2, 3): -1.0,
        (2, 6): -1.0,
        (3, 7): -1.0,
        (4, 5): -1.0,
        (4, 8): -1.0,
        (5, 6): -1.0,
        (5, 9): -1.0,
        (6, 7): -1.0,
        (6, 10): -1.0,
        (7, 11): -1.0,
        (8, 9): -1.0,
        (8, 12): -1.0,
        (9, 10): -1.0,
        (9, 13): -1.0,
        (10, 11): -1.0,
        (10, 14): -1.0,
        (11, 15): -1.0,
        (12, 13): -1.0,
        (13, 14): -1.0,
        (14, 15): -1.0
    }, -24.0, "16-spin 4x4 grid"),

    # Problem 10: 32 spins, 4x8 grid
    ({
        **{i: 0.0 for i in range(32)}
    }, {
        **{(i, i+1): -1.0 for i in range(31) if (i+1) % 8 != 0},  # horizontal connections
        **{(i, i+8): -1.0 for i in range(24)}  # vertical connections
    }, -56.0, "32-spin 4x8 grid"),

    # Problem 11: 64 spins, 8x8 grid
    ({
        **{i: 0.0 for i in range(64)}
    }, {
        **{(i, i+1): -1.0 for i in range(63) if (i+1) % 8 != 0},  # horizontal connections
        **{(i, i+8): -1.0 for i in range(56)}  # vertical connections
    }, -112.0, "64-spin 8x8 grid"),

    # Problem 12: 128 spins, 16x8 grid
    ({
        **{i: 0.0 for i in range(128)}
    }, {
        **{(i, i+1): -1.0 for i in range(127) if (i+1) % 16 != 0},  # horizontal connections
        **{(i, i+16): -1.0 for i in range(112)}  # vertical connections
    }, -240.0, "128-spin 16x8 grid"),

    # Problem 13: 256 spins, 16x16 grid
    ({
        **{i: 0.0 for i in range(256)}
    }, {
        **{(i, i+1): -1.0 for i in range(255) if (i+1) % 16 != 0},  # horizontal connections
        **{(i, i+16): -1.0 for i in range(240)}  # vertical connections
    }, -480.0, "256-spin 16x16 grid"),

    # Problem 14: 64 spins, ferromagnetic complete graph (all spins same sign)
    ({
        **{i: 0.0 for i in range(64)}
    }, {
        **{(i, j): -1.0 for i in range(64) for j in range(i+1, 64)}
    }, -2016.0, "64-spin complete ferromagnetic graph"),

    # Problem 15: 128 spins, bipartite graph (two sets with negative between, positive within)
    ({
        **{i: 0.0 for i in range(128)}
    }, {
        **{(i, j): 1.0 for i in range(64) for j in range(64, 128)},  # positive between sets
        **{(i, j): -1.0 for i in range(64) for j in range(i+1, 64)},  # negative within set A
        **{(i, j): -1.0 for i in range(64, 128) for j in range(i+1, 128)}  # negative within set B
    }, -4096.0, "128-spin bipartite graph")
]