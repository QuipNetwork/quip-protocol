"""Cross-language test vectors for ChaCha8Rng and Ising model generation.

Tests verify that Python's ChaCha8Rng produces output identical to
Rust's rand_chacha::ChaCha8Rng v0.9.0. Shared test vectors live in
tests/chacha8_test_vectors.json — both Python and Rust should test
against the same file.

SPDX-License-Identifier: AGPL-3.0-or-later
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.allowed_value_spec import AllowedValueSet
from shared.chacha8 import (
    ChaCha8Rng,
    _pcg32,
    _seed_from_u64,
    _chacha_block,
    _CONSTANTS,
)
from shared.quantum_proof_of_work import (
    derive_nonce,
    generate_ising_model_from_nonce,
)

_VECTORS_PATH = Path(__file__).parent / 'chacha8_test_vectors.json'
_VECTORS = json.loads(_VECTORS_PATH.read_text())


# ---------------------------------------------------------------------------
# PCG32 seed expansion
# ---------------------------------------------------------------------------

class TestPCG32Expansion:
    """Verify the PCG32 expansion matches rand_core::SeedableRng."""

    @pytest.mark.parametrize(
        'vec',
        _VECTORS['seed_expansion'],
        ids=[f"seed={v['seed']}" for v in _VECTORS['seed_expansion']],
    )
    def test_seed_expansion_from_vectors(self, vec):
        key = _seed_from_u64(vec['seed'])
        assert key.hex() == vec['expected_key_hex']

    def test_pcg32_state_advance(self):
        """PCG32 advances state BEFORE computing output."""
        state, out = _pcg32(0)
        assert state != 0
        assert out != 0


# ---------------------------------------------------------------------------
# ChaCha8 block function
# ---------------------------------------------------------------------------

class TestChaCha8Block:
    """Verify the ChaCha8 block function internals."""

    def test_all_zero_key_nonce_produces_nonzero(self):
        state = list(_CONSTANTS) + [0] * 12
        output = _chacha_block(state)
        assert any(w != 0 for w in output)

    def test_different_counters_produce_different_blocks(self):
        state0 = list(_CONSTANTS) + [0] * 12
        state1 = list(_CONSTANTS) + [0] * 8 + [1, 0, 0, 0]
        assert _chacha_block(state0) != _chacha_block(state1)


# ---------------------------------------------------------------------------
# ChaCha8Rng next_u32() sequences (from JSON vectors)
# ---------------------------------------------------------------------------

class TestChaCha8Rng:
    """Verify next_u32() output sequences match Rust's rand_chacha."""

    @pytest.mark.parametrize(
        'vec',
        _VECTORS['rng_sequences'],
        ids=[f"seed={v['seed']}" for v in _VECTORS['rng_sequences']],
    )
    def test_rng_sequence_from_vectors(self, vec):
        rng = ChaCha8Rng.seed_from_u64(vec['seed'])
        actual = [rng.next_u32() for _ in range(len(vec['values']))]
        assert actual == vec['values']

    def test_deterministic(self):
        rng1 = ChaCha8Rng.seed_from_u64(999)
        rng2 = ChaCha8Rng.seed_from_u64(999)
        for _ in range(100):
            assert rng1.next_u32() == rng2.next_u32()

    def test_different_seeds_diverge(self):
        rng1 = ChaCha8Rng.seed_from_u64(0)
        rng2 = ChaCha8Rng.seed_from_u64(1)
        assert rng1.next_u32() != rng2.next_u32()

    def test_invalid_key_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            ChaCha8Rng(b'\x00' * 16)


# ---------------------------------------------------------------------------
# derive_nonce (post-MR-!20)
# ---------------------------------------------------------------------------
#
# The legacy JSON vectors in chacha8_test_vectors.json target the old
# truncated-u64 nonce shape; cross-language parity now goes through
# tests/test_derive_nonce_parity.py against python_parity.json. The smoke
# checks below pin the new shape (32-byte fixed inputs, 32-byte digest).


class TestDeriveNonce:
    """Smoke-tests for the fixed-width derive_nonce surface."""

    def test_returns_32_bytes(self):
        nonce = derive_nonce(b"\x00" * 32, b"\x00" * 32, b"\x00" * 32)
        assert isinstance(nonce, bytes)
        assert len(nonce) == 32

    def test_different_inputs_differ(self):
        base = derive_nonce(b"\x00" * 32, b"\x00" * 32, b"\x00" * 32)
        assert base != derive_nonce(b"\x01" * 32, b"\x00" * 32, b"\x00" * 32)
        assert base != derive_nonce(b"\x00" * 32, b"\x01" * 32, b"\x00" * 32)
        assert base != derive_nonce(b"\x00" * 32, b"\x00" * 32, b"\xff" * 32)

    def test_rejects_short_inputs(self):
        with pytest.raises(ValueError, match="last_proof_block_hash"):
            derive_nonce(b"\x00" * 16, b"\x00" * 32, b"\x00" * 32)
        with pytest.raises(ValueError, match="miner"):
            derive_nonce(b"\x00" * 32, b"m", b"\x00" * 32)
        with pytest.raises(ValueError, match="salt"):
            derive_nonce(b"\x00" * 32, b"\x00" * 32, b"")


# ---------------------------------------------------------------------------
# generate_ising_model_from_nonce (post-MR-!20)
# ---------------------------------------------------------------------------


class TestGenerateIsingModel:
    """Verify Ising model generation against the new AllowedValueSpec API."""

    def test_h_values_from_allowed_set(self):
        allowed = AllowedValueSet((-1000, 0, 1000))
        h, _ = generate_ising_model_from_nonce(
            99,
            [0, 1, 2, 3, 4],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            allowed_h=allowed,
        )
        assert all(v in (-1.0, 0.0, 1.0) for v in h.values())

    def test_j_values_from_allowed_set(self):
        _, J = generate_ising_model_from_nonce(
            99,
            [0, 1, 2, 3, 4],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        assert all(v in (-1.0, 1.0) for v in J.values())

    def test_deterministic(self):
        nodes, edges = [0, 1, 2], [(0, 1), (1, 2)]
        h1, J1 = generate_ising_model_from_nonce(42, nodes, edges)
        h2, J2 = generate_ising_model_from_nonce(42, nodes, edges)
        assert h1 == h2 and J1 == J2

    def test_accepts_bytes_and_int_nonce(self):
        nodes = [0, 1, 2]
        edges = [(0, 1)]
        from_int = generate_ising_model_from_nonce(42, nodes, edges)
        from_bytes = generate_ising_model_from_nonce(
            (42).to_bytes(32, "big"), nodes, edges,
        )
        assert from_int == from_bytes


# ---------------------------------------------------------------------------
# Edge cases and input validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Validate error handling and boundary conditions."""

    def test_empty_allowed_h_raises(self):
        with pytest.raises(ValueError):
            generate_ising_model_from_nonce(
                42, [0, 1], [(0, 1)], allowed_h=AllowedValueSet(()),
            )

    def test_empty_nodes_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            generate_ising_model_from_nonce(42, [], [])

    def test_empty_edges_returns_empty_j(self):
        h, J = generate_ising_model_from_nonce(42, [0, 1, 2], [])
        assert len(J) == 0
        assert len(h) == 3

    def test_seed_out_of_range_raises(self):
        with pytest.raises(ValueError, match="u64"):
            ChaCha8Rng.seed_from_u64(-1)
        with pytest.raises(ValueError, match="u64"):
            ChaCha8Rng.seed_from_u64(2**64)

    def test_counter_carry(self):
        """Verify 64-bit counter carry from state[12] to state[13]."""
        key = b'\x00' * 32
        rng = ChaCha8Rng(key, counter=0xFFFFFFFF)
        # Consume all 16 words from the first block
        for _ in range(16):
            rng.next_u32()
        # Next call triggers _refill_buffer with carry
        val = rng.next_u32()
        assert 0 <= val <= 0xFFFFFFFF
        # Counter should have wrapped: state[12]=0, state[13]=1
        assert rng._state[12] == 1
        assert rng._state[13] == 1
