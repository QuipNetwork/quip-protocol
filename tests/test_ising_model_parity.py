"""Cross-language parity for ``generate_ising_model_from_nonce`` (post-MR-!20).

Consumes the ``ising_model_cases`` section of ``python_parity.json``. The
test is skipped if the fixture is missing or has no ising-model cases (the
section is only populated after running ``tests/generate_parity_vectors.py``).

Each case asserts that the milli-precision (h, j) arrays Python emits match
what the Rust ``quantum_validation::generate_ising_model`` would produce for
the same nonce / topology / allowed-value specs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from shared.allowed_value_spec import (
    AllowedValueContinuousRange,
    AllowedValueIntegerRange,
    AllowedValueSet,
    MILLI_SCALE,
)
from shared.quantum_proof_of_work import generate_ising_model_from_nonce


def _fixture_path() -> Path | None:
    env_override = os.environ.get("QUIP_RUST_FIXTURE_DIR")
    if env_override:
        override_path = Path(env_override) / "python_parity.json"
        if not override_path.exists():
            pytest.fail(
                f"QUIP_RUST_FIXTURE_DIR={env_override!r} does not contain "
                "python_parity.json"
            )
        return override_path

    sibling = (
        Path(__file__).parent.parent.parent
        / "quip-protocol-rs"
        / "crates"
        / "quantum-validation"
        / "tests"
        / "fixtures"
        / "python_parity.json"
    )
    return sibling if sibling.exists() else None


_FIXTURE_PATH = _fixture_path()
_SKIP_REASON = (
    "Rust parity fixture not found. Check out quip-protocol-rs alongside "
    "this repo or set QUIP_RUST_FIXTURE_DIR=/path/to/fixtures."
)


def _spec_from_json(spec_json) -> object:
    """Decode the tagged spec representation emitted by the generator."""
    if "Set" in spec_json:
        return AllowedValueSet(tuple(int(v) for v in spec_json["Set"]))
    if "IntegerRange" in spec_json:
        ir = spec_json["IntegerRange"]
        return AllowedValueIntegerRange(min=int(ir["min"]), max=int(ir["max"]))
    if "ContinuousRange" in spec_json:
        cr = spec_json["ContinuousRange"]
        return AllowedValueContinuousRange(min=int(cr["min"]), max=int(cr["max"]))
    raise ValueError(f"unknown spec variant: {spec_json!r}")


def _load_cases():
    if _FIXTURE_PATH is None:
        return [
            pytest.param(
                None,
                id="fixture-missing",
                marks=pytest.mark.skip(reason=_SKIP_REASON),
            )
        ]
    fixture = json.loads(_FIXTURE_PATH.read_text())
    cases = fixture.get("ising_model_cases", [])
    if not cases:
        return [
            pytest.param(
                None,
                id="ising_model_cases-missing",
                marks=pytest.mark.skip(
                    reason="fixture has no ising_model_cases yet; regenerate "
                    "with tests/generate_parity_vectors.py"
                ),
            )
        ]
    return [pytest.param(case, id=case["name"]) for case in cases]


@pytest.mark.parametrize("case", _load_cases())
def test_ising_model_matches_rust(case):
    nonce = bytes.fromhex(case["nonce_hex"])
    nodes = [int(n) for n in case["nodes"]]
    edges = [(int(u), int(v)) for u, v in case["edges"]]
    spec_h = _spec_from_json(case["allowed_h"])
    spec_j = _spec_from_json(case["allowed_j"])

    h_floats, j_floats = generate_ising_model_from_nonce(
        nonce, nodes, edges, allowed_h=spec_h, allowed_j=spec_j,
    )

    # The expected arrays are stored in milli-precision i32 to avoid
    # cross-language float comparison subtleties — convert Python's floats
    # back through MILLI_SCALE.
    actual_h_milli = [int(round(h_floats[n] * MILLI_SCALE)) for n in nodes]
    actual_j_milli = [int(round(j_floats[(u, v)] * MILLI_SCALE)) for (u, v) in edges]

    assert actual_h_milli == case["expected_h_milli"]
    assert actual_j_milli == case["expected_j_milli"]
