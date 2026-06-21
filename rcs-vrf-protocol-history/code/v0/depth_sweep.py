"""
Sweep depth d at n=8 to characterize F_XEB behavior.

For each depth:
  - Run NOISELESS: F_XEB should approach 1 as d grows (circuit becomes 2-design)
  - Run NOISY (rate=0.005): F_XEB should track circuit fidelity, decreasing with d

Multiple trials per (d, noise) with different seeds to estimate variance.
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
        # All-to-all: random perfect matching per layer
        pairs = random_perfect_matching(list(qubits))
        for q1, q2 in pairs:
            circuit.append(cirq.CZ(q1, q2))
    
    for q in qubits:
        theta, phi, lam = random_angle(), random_angle(), random_angle()
        circuit.append([cirq.rz(lam).on(q), cirq.ry(theta).on(q), cirq.rz(phi).on(q)])
    
    return circuit, qubits


def f_xeb(circuit, samples, n_qubits):
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    probs = np.abs(result.final_state_vector) ** 2
    
    sample_indices = np.array([
        int("".join(str(int(b)) for b in sample), 2)
        for sample in samples
    ])
    p_observed = probs[sample_indices]
    return (2 ** n_qubits) * np.mean(p_observed) - 1.0


def run_one(seed_int, n_qubits, depth, noise_rate, shots):
    seed_bytes = f"sweep_seed_{seed_int}_d{depth}".encode()
    circuit, qubits = generate_circuit(seed_bytes, n_qubits, depth)
    sampling_circuit = circuit.copy()
    sampling_circuit.append(cirq.measure(*qubits, key="m"))
    
    if noise_rate > 0:
        noise_model = cirq.ConstantQubitNoiseModel(cirq.depolarize(p=noise_rate))
        sim = cirq.DensityMatrixSimulator(noise=noise_model)
    else:
        sim = cirq.Simulator()
    
    result = sim.run(sampling_circuit, repetitions=shots)
    samples = result.measurements["m"]
    return f_xeb(circuit, samples, n_qubits)


def sweep():
    n_qubits = 8
    shots = 5000
    n_trials = 3
    depths = [2, 4, 6, 8, 10, 12, 14]
    noise_rates = [0.0, 0.005]
    
    print(f"Sweep: n={n_qubits}, shots/trial={shots}, trials per cell={n_trials}")
    print(f"Depths: {depths}")
    print(f"Noise rates: {noise_rates}")
    print()
    print(f"{'depth':>6} {'noiseless mean±std':>22} {'noisy(0.005) mean±std':>24} {'time(s)':>10}")
    print("-" * 70)
    
    results = {}
    for d in depths:
        row = {}
        for nr in noise_rates:
            t0 = time.time()
            vals = []
            for trial in range(n_trials):
                v = run_one(seed_int=trial, n_qubits=n_qubits, depth=d,
                            noise_rate=nr, shots=shots)
                vals.append(v)
            elapsed = time.time() - t0
            row[nr] = (np.mean(vals), np.std(vals), elapsed)
        results[d] = row
        
        nl_mean, nl_std, nl_t = row[0.0]
        ny_mean, ny_std, ny_t = row[0.005]
        print(f"{d:>6} {nl_mean:>10.4f} ± {nl_std:>6.4f}   {ny_mean:>10.4f} ± {ny_std:>6.4f}   {nl_t+ny_t:>8.1f}")
    
    print()
    print("Interpretation:")
    print("  Noiseless F_XEB should approach 1.0 as depth grows (circuit → 2-design)")
    print("  Noisy F_XEB should track circuit fidelity, decreasing with depth")
    print("  Below the 2-design threshold, F_XEB > 1 in both columns")

sweep()
