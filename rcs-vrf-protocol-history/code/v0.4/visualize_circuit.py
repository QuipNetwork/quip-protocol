"""
Visualize the v0 circuit.

Generates the same circuit our pipeline uses, then prints it as text and
optionally writes an SVG file you can open in a browser.

Usage:
    python visualize_circuit.py
    python visualize_circuit.py --public-input "round_1_block_height_5678"
    python visualize_circuit.py --n 8 --d 10
"""

import argparse
import hashlib
import numpy as np
import cirq


def vrf_seed_placeholder(public_input: bytes) -> bytes:
    """Same placeholder VRF as in v0."""
    outs = []
    for i in range(5):
        sk_i = hashlib.sha3_256(f"participant_{i}_secret_key".encode()).digest()
        outs.append(hashlib.sha3_256(sk_i + public_input).digest())
    return b"".join(outs)


def generate_circuit(seed: bytes, n_qubits: int, depth: int) -> cirq.Circuit:
    """Same circuit generation as v0's LEG 3."""
    bits_needed = n_qubits * (depth + 1) * 12 + depth * n_qubits * 4 + 2048
    shake = hashlib.shake_256()
    shake.update(seed)
    expanded = shake.digest(bits_needed)
    
    cursor = [0]
    def take_bytes(k):
        result = expanded[cursor[0]:cursor[0] + k]
        cursor[0] += k
        return result
    def random_angle():
        return (int.from_bytes(take_bytes(4), "big") / (2**32)) * 2 * np.pi
    def random_perfect_matching(qs):
        idx = list(range(len(qs)))
        for i in range(len(idx) - 1, 0, -1):
            j = int.from_bytes(take_bytes(4), "big") % (i + 1)
            idx[i], idx[j] = idx[j], idx[i]
        pairs = []
        for i in range(0, len(idx) - 1, 2):
            pairs.append((qs[idx[i]], qs[idx[i + 1]]))
        return pairs

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    for layer in range(depth):
        for q in qubits:
            t, p, l = random_angle(), random_angle(), random_angle()
            circuit.append([cirq.rz(l).on(q), cirq.ry(t).on(q), cirq.rz(p).on(q)])
        for q1, q2 in random_perfect_matching(list(qubits)):
            circuit.append(cirq.CZ(q1, q2))
    
    for q in qubits:
        t, p, l = random_angle(), random_angle(), random_angle()
        circuit.append([cirq.rz(l).on(q), cirq.ry(t).on(q), cirq.rz(p).on(q)])
    
    return circuit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-input", default="round_1_block_height_1234")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--svg", default="circuit.svg",
                        help="SVG output file (set to empty to skip)")
    parser.add_argument("--moments", type=int, default=0,
                        help="Print only first N moments (0 = print all)")
    args = parser.parse_args()
    
    seed = vrf_seed_placeholder(args.public_input.encode())
    circuit = generate_circuit(seed, args.n, args.d)
    
    print(f"Circuit for public_input='{args.public_input}', n={args.n}, d={args.d}")
    print(f"  Gates:    {sum(1 for _ in circuit.all_operations())}")
    print(f"  Moments:  {len(circuit)}")
    print(f"  Qubits:   {args.n}")
    print()
    print("=" * 70)
    print("TEXT DIAGRAM")
    print("=" * 70)
    
    if args.moments > 0 and args.moments < len(circuit):
        # Show only first N moments
        partial = cirq.Circuit(circuit[:args.moments])
        print(partial)
        print(f"\n... ({len(circuit) - args.moments} more moments omitted) ...")
    else:
        print(circuit)
    
    # SVG output
    if args.svg:
        try:
            from cirq.contrib.svg import circuit_to_svg
            svg_text = circuit_to_svg(circuit)
            with open(args.svg, "w", encoding="utf-8") as f:
                f.write(svg_text)
            print(f"\n✓ SVG written to {args.svg}")
            print(f"  Open it in a browser to see the circuit graphically.")
        except ImportError as e:
            print(f"\n[!] SVG output requires cirq.contrib.svg: {e}")
        except Exception as e:
            print(f"\n[!] SVG output failed: {e}")


if __name__ == "__main__":
    main()
