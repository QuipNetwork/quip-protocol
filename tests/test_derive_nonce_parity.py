"""Cross-language parity test for `shared.quantum_proof_of_work.derive_nonce`.

Loads the shared fixture maintained alongside the Rust pallet at
`crates/quantum-validation/tests/fixtures/python_parity.json` in the
`quip-protocol-rs` sibling checkout. Each `derive_nonce_cases` entry encodes
inputs (parent_hash_hex, miner string, block_number, salt_hex) and the
expected `u64` nonce computed by `quantum_validation::derive_nonce`.

The test is skipped if the fixture file cannot be located so that contributors
without the Rust repo checked out alongside this one can still run the suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.quantum_proof_of_work import derive_nonce


def _fixture_path() -> Path | None:
    """Search a few well-known locations for the parity fixture."""
    candidates = [
        # Sibling checkout (default dev layout).
        Path(__file__).parent.parent.parent
        / "quip-protocol-rs"
        / "crates"
        / "quantum-validation"
        / "tests"
        / "fixtures"
        / "python_parity.json",
        # Override via env var when the sibling is somewhere else.
    ]
    import os

    env_override = os.environ.get("QUIP_RUST_FIXTURE_DIR")
    if env_override:
        candidates.insert(0, Path(env_override) / "python_parity.json")

    for path in candidates:
        if path.exists():
            return path
    return None


def _load_cases():
    path = _fixture_path()
    if path is None:
        pytest.skip(
            "Rust parity fixture not found. Check out quip-protocol-rs alongside "
            "this repo or set QUIP_RUST_FIXTURE_DIR=/path/to/fixtures."
        )
    return json.loads(path.read_text())["derive_nonce_cases"]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
def test_derive_nonce_matches_rust(case):
    parent_hash = bytes.fromhex(case["parent_hash_hex"])
    salt = bytes.fromhex(case["salt_hex"])
    actual = derive_nonce(
        parent_hash=parent_hash,
        miner=case["miner"].encode(),
        block_number=case["block_number"],
        salt=salt,
    )
    assert actual == case["expected_nonce"]


def test_derive_nonce_rejects_oversize_block_number():
    with pytest.raises(ValueError, match="must be a u32"):
        derive_nonce(b"\x00" * 32, b"m", 2**32, b"\x00" * 32)


def test_derive_nonce_accepts_zero():
    # Smoke check that the function produces *some* deterministic output for
    # an all-zeros input — proves the helper at least runs end-to-end.
    out = derive_nonce(b"\x00" * 32, b"", 0, b"\x00" * 32)
    assert isinstance(out, int)
    assert 0 <= out < 2**64
