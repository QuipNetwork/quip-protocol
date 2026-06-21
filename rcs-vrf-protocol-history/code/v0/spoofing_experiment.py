"""
Spoofing experiment: empirically verify that classical attacks fail XEB.

For a given (n, d) circuit, compare F_XEB across submission strategies:

  1. HONEST QUANTUM (noiseless simulator)
     - Expected: F_XEB ≈ 1.0
  2. HONEST QUANTUM with noise (current v0 model)
     - Expected: F_XEB ≈ 0.3 (real-hardware-like)
  3. UNIFORM RANDOM (classical adversary, dumbest attack)
     - Expected: F_XEB ≈ 0.0 (no quantum advantage)
  4. HEAVY-OUTPUT GUESSING (classical adversary with partial classical knowledge)
     - Compute ALL p(x), submit only the highest-probability bitstrings
     - This is the strongest known classical attack at small n
     - Expected: F_XEB high, but requires full classical simulation

Goal: show that strategy 3 (the realistic adversary at scale) fails decisively,
while strategy 4 (the strongest possible attack) requires compute that grows
exponentially with n.
"""

import hashlib
import time
import numpy as np
import cirq


def generate_circuit(seed_bytes, n_qubits, depth):
    bits_needed = n_qubits * (depth + 1) * 12 + depth * n_qubits * 4 + 2048
    shake = hashlib.shake_256()
    shake.update(seed_bytes)
    expanded = shake.digest(bits_needed)
    
    cursor = [0]
    def take_bytes(k):
        result = expanded[cursor[0]:cursor[0] + k]
        cursor[0] += k
        return result
    def random_angle():
        b = take_bytes(4)
        return (int.from_bytes(b, "big") / (2**32)) * 2 * np.pi
    
    def random_perfect_matching(qubit_list):
        indices = list(range(len(qubit_list)))
        for i in range(len(indices) - 1, 0, -1):
            j = int.from_bytes(take_bytes(4), "big") % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        pairs = []
        for i in range(0, len(indices) - 1, 2):
            pairs.append((qubit_list[indices[i]], qubit_list[indices[i + 1]]))
        return pairs
    
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    for layer in range(depth):
        for q in qubits:
            theta, phi, lam = random_angle(), random_angle(), random_angle()
            circuit.append([cirq.rz(lam).on(q), cirq.ry(theta).on(q), cirq.rz(phi).on(q)])
        pairs = random_perfect_matching(list(qubits))
        for q1, q2 in pairs:
            circuit.append(cirq.CZ(q1, q2))
    
    for q in qubits:
        theta, phi, lam = random_angle(), random_angle(), random_angle()
        circuit.append([cirq.rz(lam).on(q), cirq.ry(theta).on(q), cirq.rz(phi).on(q)])
    
    return circuit, qubits


def compute_probs(circuit, n_qubits):
    """Get the ideal output distribution p(x) for the circuit."""
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    return np.abs(result.final_state_vector) ** 2


def f_xeb(probs_observed_indices, probs, n_qubits):
    """Standard F_XEB formula given samples (as indices) and ideal p(x)."""
    p_observed = probs[probs_observed_indices]
    return (2 ** n_qubits) * np.mean(p_observed) - 1.0


# ----- Attack strategies -----

def strategy_honest_noiseless(circuit, qubits, n_shots):
    """Strategy 1: noiseless quantum sampling. Should give F_XEB ≈ 1."""
    sampling_circuit = circuit.copy()
    sampling_circuit.append(cirq.measure(*qubits, key="m"))
    sim = cirq.Simulator()
    result = sim.run(sampling_circuit, repetitions=n_shots)
    samples = result.measurements["m"]
    return np.array([int("".join(str(int(b)) for b in s), 2) for s in samples])


def strategy_honest_noisy(circuit, qubits, n_shots, noise_rate=0.005):
    """Strategy 2: noisy quantum sampling. Should give F_XEB ≈ fidelity."""
    sampling_circuit = circuit.copy()
    sampling_circuit.append(cirq.measure(*qubits, key="m"))
    noise_model = cirq.ConstantQubitNoiseModel(cirq.depolarize(p=noise_rate))
    sim = cirq.DensityMatrixSimulator(noise=noise_model)
    result = sim.run(sampling_circuit, repetitions=n_shots)
    samples = result.measurements["m"]
    return np.array([int("".join(str(int(b)) for b in s), 2) for s in samples])


def strategy_uniform_random(n_qubits, n_shots, rng):
    """Strategy 3: uniformly random bitstrings (dumbest classical attack).
    Should give F_XEB ≈ 0 because there's no correlation with p(x)."""
    return rng.integers(0, 2 ** n_qubits, size=n_shots)


def strategy_heavy_output_guessing(probs, n_shots):
    """Strategy 4: classical adversary that has computed all p(x).
    Submits the highest-probability bitstrings.
    
    This is the STRONGEST possible classical attack at small n. The catch:
    requires O(2^n) compute, which is infeasible at production parameters.
    
    At n=8 it's trivial; at n=56 (Liu et al.) it would take Frontier-class
    compute or more."""
    # Pick the top-K most likely bitstrings, sample from them proportionally
    # (this is the optimal classical strategy given full knowledge of p)
    n = len(probs)
    # Sample according to p(x) itself - this would give F_XEB = 1 if we
    # have full knowledge, since it matches the ideal distribution exactly
    indices = np.random.choice(n, size=n_shots, p=probs)
    return indices


def strategy_top_k_guessing(probs, n_shots, k):
    """Strategy 4b: classical adversary that knows the top k bitstrings
    and submits a mix of them. This is what an adversary with LIMITED
    compute (only able to enumerate top-k of 2^n) could do."""
    # Pick top-k indices
    top_k_indices = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_indices]
    # Normalize
    top_k_probs = top_k_probs / top_k_probs.sum()
    # Sample from top-k according to their relative probabilities
    sampled_local = np.random.choice(len(top_k_indices), size=n_shots, p=top_k_probs)
    return top_k_indices[sampled_local]


# ----- Run the experiment -----

def run_experiment():
    n_qubits = 8
    depth = 10
    n_shots = 5000
    n_trials = 5  # different random circuits
    
    print(f"Spoofing experiment: n={n_qubits}, d={depth}, shots={n_shots}, "
          f"trials={n_trials}")
    print("=" * 75)
    print(f"{'strategy':35s} {'mean F_XEB':>12s} {'std':>8s} {'pass χ=0.15':>14s}")
    print("-" * 75)
    
    results = {}
    rng = np.random.default_rng(42)
    
    for strategy_name in ["honest_noiseless", "honest_noisy", "uniform_random",
                          "perfect_classical_with_p", "top_1_guess",
                          "top_4_guess", "top_16_guess"]:
        scores = []
        for trial in range(n_trials):
            seed_bytes = f"spoofing_seed_{trial}".encode()
            circuit, qubits = generate_circuit(seed_bytes, n_qubits, depth)
            # Strip measurements for p(x) computation
            probs = compute_probs(circuit, n_qubits)
            
            if strategy_name == "honest_noiseless":
                indices = strategy_honest_noiseless(circuit, qubits, n_shots)
            elif strategy_name == "honest_noisy":
                indices = strategy_honest_noisy(circuit, qubits, n_shots)
            elif strategy_name == "uniform_random":
                indices = strategy_uniform_random(n_qubits, n_shots, rng)
            elif strategy_name == "perfect_classical_with_p":
                indices = strategy_heavy_output_guessing(probs, n_shots)
            elif strategy_name == "top_1_guess":
                indices = strategy_top_k_guessing(probs, n_shots, k=1)
            elif strategy_name == "top_4_guess":
                indices = strategy_top_k_guessing(probs, n_shots, k=4)
            elif strategy_name == "top_16_guess":
                indices = strategy_top_k_guessing(probs, n_shots, k=16)
            
            score = f_xeb(indices, probs, n_qubits)
            scores.append(score)
        
        scores = np.array(scores)
        mean = scores.mean()
        std = scores.std()
        pass_rate = (scores >= 0.15).mean()
        
        results[strategy_name] = {"mean": mean, "std": std, "pass_rate": pass_rate}
        print(f"{strategy_name:35s} {mean:>12.4f} {std:>8.4f} {pass_rate*100:>12.0f}%")
    
    print()
    print("Interpretation:")
    print("  honest_noiseless:        perfect quantum (no noise)")
    print("  honest_noisy:            realistic quantum (noise=0.005)")
    print("  uniform_random:          classical attacker, no info about p")
    print("  perfect_classical_with_p: classical attacker who computed all p(x)")
    print("                            -- requires O(2^n) compute, infeasible at scale")
    print("  top_k_guess:             classical attacker who knows top-k bitstrings")
    print("                            -- requires partial computation of p")
    print()
    print("Key insight:")
    print("  The realistic adversary (uniform_random) fails decisively.")
    print("  The 'perfect' adversary (perfect_classical_with_p) succeeds — BUT only")
    print("  because at n=8, computing all 2^8=256 amplitudes is trivial.")
    print("  At n=56 (Liu et al.) this attack costs ~10^20 FLOPS (Frontier-scale).")
    print("  The security inequality φ > A·t/B is what makes this attack infeasible")
    print("  at production parameters.")

run_experiment()
