"""Regenerate the cross-language parity fixture ``python_parity.json``.

Run this whenever the PoW primitives change shape. The output goes to the
path given by ``--out``; copy it to ``quip-protocol-rs/crates/
quantum-validation/tests/fixtures/python_parity.json`` to feed the Rust
parity tests.

Schema for the two PoW sections:

    derive_nonce_cases: [
        {
          "name": str,
          "last_winning_hash_hex": 32-byte hex,
          "miner_hex":             32-byte hex,
          "salt_hex":              32-byte hex,
          "expected_nonce_hex":    32-byte hex
        }
    ]

    ising_model_cases: [
        {
          "name": str,
          "nonce_hex":    32-byte hex,
          "nodes":        [u32],
          "edges":        [[u32, u32]],
          "allowed_h":    {"Set": [milli i32], ...},
          "allowed_j":    {"Set": [milli i32], ...},
          "expected_h_milli": [i32],
          "expected_j_milli": [i32]
        }
    ]

Existing sections (``energy_cases``, ``expected_gse_cases``,
``diversity_cases``, ``hamming_cases``, ``topology_cases``,
``solution_validation_cases``) are passed through untouched if a previous
fixture is supplied via ``--merge``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from shared.allowed_value_spec import (
    AllowedValueContinuousRange,
    AllowedValueIntegerRange,
    AllowedValueSet,
    MILLI_SCALE,
    sample as _sample_spec,
)
from shared.chacha8 import ChaCha8Rng
from shared.quantum_proof_of_work import derive_nonce


def _spec_to_json(spec) -> Dict[str, Any]:
    """Render an AllowedValueSpec as a JSON-friendly tagged dict."""
    if isinstance(spec, AllowedValueSet):
        return {"Set": list(spec.values)}
    if isinstance(spec, AllowedValueIntegerRange):
        return {"IntegerRange": {"min": spec.min, "max": spec.max}}
    if isinstance(spec, AllowedValueContinuousRange):
        return {"ContinuousRange": {"min": spec.min, "max": spec.max}}
    raise TypeError(f"unknown spec variant: {type(spec).__name__}")


def _build_derive_nonce_cases() -> List[Dict[str, Any]]:
    """Hand-picked deterministic vectors covering last_winning_hash/miner/salt."""
    base_seed = bytes([0x01] * 32)
    alt_seed = bytes([0x42] * 32)
    alice = bytes([0xA1] * 32)
    bob = bytes([0xB0] * 32)
    salt_a = bytes([0x01] * 32)
    salt_b = bytes([0x02] * 32)

    inputs = [
        ("alice_seedA_saltA", base_seed, alice, salt_a),
        ("alice_seedA_saltB", base_seed, alice, salt_b),
        ("bob_seedA_saltA", base_seed, bob, salt_a),
        ("alice_seedB_saltA", alt_seed, alice, salt_a),
        ("alice_zeros", b"\x00" * 32, alice, b"\x00" * 32),
        ("alice_max", b"\xff" * 32, alice, b"\xff" * 32),
    ]
    cases: List[Dict[str, Any]] = []
    for name, last_winning_hash, miner, salt in inputs:
        nonce = derive_nonce(last_winning_hash, miner, salt)
        cases.append(
            {
                "name": name,
                "last_winning_hash_hex": last_winning_hash.hex(),
                "miner_hex": miner.hex(),
                "salt_hex": salt.hex(),
                "expected_nonce_hex": nonce.hex(),
            }
        )
    return cases


def _expected_ising(spec_h, spec_j, nonce_bytes, nodes, edges):
    """Run the same RNG path as generate_ising_model_from_nonce, returning
    the milli-precision arrays. Keeps the fixture independent of the
    float-precision boundary in the public API."""
    rng = ChaCha8Rng.from_seed(nonce_bytes)
    h_milli = [_sample_spec(spec_h, rng) for _ in nodes]
    j_milli = [_sample_spec(spec_j, rng) for _ in edges]
    return h_milli, j_milli


def _build_ising_model_cases() -> List[Dict[str, Any]]:
    """Cases covering Set / IntegerRange / ContinuousRange for h and j."""
    nodes_small = [0, 1, 2]
    edges_small = [(0, 1), (1, 2)]
    nodes_med = [0, 1, 2, 3, 4, 5]
    edges_med = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]

    cases_input = [
        (
            "ternary_h_binary_j_small",
            (1).to_bytes(32, "big"),
            nodes_small,
            edges_small,
            AllowedValueSet((-MILLI_SCALE, 0, MILLI_SCALE)),
            AllowedValueSet((-MILLI_SCALE, MILLI_SCALE)),
        ),
        (
            "ternary_h_binary_j_medium",
            (42).to_bytes(32, "big"),
            nodes_med,
            edges_med,
            AllowedValueSet((-MILLI_SCALE, 0, MILLI_SCALE)),
            AllowedValueSet((-MILLI_SCALE, MILLI_SCALE)),
        ),
        (
            "integer_range_h_set_j",
            bytes.fromhex("a1" * 32),
            nodes_med,
            edges_med,
            AllowedValueIntegerRange(min=-3, max=3),
            AllowedValueSet((-2 * MILLI_SCALE, 2 * MILLI_SCALE)),
        ),
        (
            "continuous_range_h_binary_j",
            bytes.fromhex("ff" * 32),
            nodes_small,
            edges_small,
            AllowedValueContinuousRange(min=-MILLI_SCALE, max=MILLI_SCALE),
            AllowedValueSet((-MILLI_SCALE, MILLI_SCALE)),
        ),
    ]

    cases: List[Dict[str, Any]] = []
    for name, nonce_bytes, nodes, edges, spec_h, spec_j in cases_input:
        h_milli, j_milli = _expected_ising(spec_h, spec_j, nonce_bytes, nodes, edges)
        cases.append(
            {
                "name": name,
                "nonce_hex": nonce_bytes.hex(),
                "nodes": nodes,
                "edges": [list(e) for e in edges],
                "allowed_h": _spec_to_json(spec_h),
                "allowed_j": _spec_to_json(spec_j),
                "expected_h_milli": h_milli,
                "expected_j_milli": j_milli,
            }
        )
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write python_parity.json",
    )
    ap.add_argument(
        "--merge",
        type=Path,
        default=None,
        help=(
            "Existing python_parity.json to merge — preserves sections this "
            "generator does not regenerate (energy_cases, diversity_cases, "
            "etc.). Without this flag those sections are dropped."
        ),
    )
    args = ap.parse_args()

    fixture: Dict[str, Any] = {}
    if args.merge is not None and args.merge.exists():
        fixture.update(json.loads(args.merge.read_text()))

    fixture["derive_nonce_cases"] = _build_derive_nonce_cases()
    fixture["ising_model_cases"] = _build_ising_model_cases()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {args.out} (derive={len(fixture['derive_nonce_cases'])} "
          f"cases, ising={len(fixture['ising_model_cases'])} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
