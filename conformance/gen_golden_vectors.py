"""Generate the cross-language golden vectors from the v0.2 shared/ reference.

These fixtures pin the two cross-language hazards: ChaCha8 draw order
(all h in node order, then all j in edge order, one next_u32 each) and
truncation-toward-zero milli conversion (int(energy*1000)).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.chacha8 import keystream_words  # noqa: E402
from shared.allowed_value_spec import AllowedValueSet  # noqa: E402
from shared.quantum_proof_of_work import (  # noqa: E402
    derive_nonce,
    generate_ising_model_from_nonce,
    energy_of_solution,
    calculate_diversity,
)

ALLOWED_H = AllowedValueSet((-1000, 0, 1000))
ALLOWED_J = AllowedValueSet((-1000, 1000))


def _int(energy: float) -> int:
    return int(energy * 1000)  # truncation toward zero — the wire convention


def build() -> dict:
    seed = bytes(range(32))
    nonce = derive_nonce(bytes(32), bytes([1] * 32), bytes([2] * 32))

    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    h, j = generate_ising_model_from_nonce(nonce, nodes, edges, ALLOWED_H, ALLOWED_J)
    h_milli = [int(round(h[n] * 1000)) for n in nodes]
    j_milli = [int(round(j[e] * 1000)) for e in edges]

    energy_cases = []
    for spins in ([1, -1, 1, -1], [1, 1, 1, 1], [-1, 1, -1, 1]):
        e = energy_of_solution(spins, h, j, nodes)
        energy_cases.append({
            "spins": spins,
            "h_milli": h_milli,
            "j_milli": j_milli,
            "edges": [list(e) for e in edges],
            "energy_milli": _int(e),
        })

    diversity_cases = [
        {"solutions": s, "diversity": calculate_diversity(s)}
        for s in ([[1, 1, -1], [-1, -1, 1]], [[1, 1, 1], [-1, 1, 1]])
    ]

    return {
        "version": 1,
        "chacha8": [{
            "seed_hex": seed.hex(),
            "n_words": 8,
            "words": [int(w) for w in keystream_words(seed, 8)],
        }],
        "derive_nonce": [{
            "last_proof_hex": bytes(32).hex(),
            "miner_hex": bytes([1] * 32).hex(),
            "salt_hex": bytes([2] * 32).hex(),
            "nonce_hex": nonce.hex(),
        }],
        "ising": [{
            "nonce_hex": nonce.hex(),
            "nodes": nodes,
            "edges": [list(e) for e in edges],
            "allowed_h_milli": [-1000, 0, 1000],
            "allowed_j_milli": [-1000, 1000],
            "h_milli": h_milli,
            "j_milli": j_milli,
        }],
        "energy": energy_cases,
        "diversity": diversity_cases,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    text = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    if args.stdout:
        sys.stdout.write(text)
    else:
        (ROOT / "conformance" / "golden_vectors.json").write_text(text)


if __name__ == "__main__":
    main()
