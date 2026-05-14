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
import os
from pathlib import Path

import pytest

from shared.quantum_proof_of_work import derive_nonce


def _fixture_path() -> Path | None:
    """Search a few well-known locations for the parity fixture.

    Set ``QUIP_RUST_FIXTURE_DIR`` to point at the directory containing
    ``python_parity.json``. If the env var is set but the file isn't there,
    we ``pytest.fail`` — silently ignoring an explicit override would let a
    misconfigured CI mask a real parity break.
    """
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


def _load_cases():
    path = _fixture_path()
    if path is None:
        # `allow_module_level=True` so this skips cleanly when called from
        # the parametrize decorator at collection time. Without it pytest
        # surfaces a collection error rather than a skip.
        pytest.skip(
            "Rust parity fixture not found. Check out quip-protocol-rs "
            "alongside this repo or set QUIP_RUST_FIXTURE_DIR=/path/to/fixtures.",
            allow_module_level=True,
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
