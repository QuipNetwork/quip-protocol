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

from shared.chacha8 import (
    ChaCha8Rng,
    _pcg32,
    _seed_from_u64,
    _chacha_block,
    _CONSTANTS,
)
from shared.quantum_proof_of_work import derive_nonce, generate_ising_model

_VECTORS_PATH = Path(__file__).parent / 'chacha8_test_vectors.json'
_VECTORS = json.loads(_VECTORS_PATH.read_text())


# ---------------------------------------------------------------------------
# PCG32 seed expansion
# ---------------------------------------------------------------------------

class TestPCG32Expansion:
    """Verify the PCG32 expansion matches rand_core::SeedableRng."""

    def test_seed_zero_key_length(self):
        key = _seed_from_u64(0)
        assert len(key) == 32

    def test_seed_zero_key_not_all_zeros(self):
        key = _seed_from_u64(0)
        assert key != b'\x00' * 32

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

    def test_block_output_length(self):
        state = list(_CONSTANTS) + [0] * 12
        output = _chacha_block(state)
        assert len(output) == 16

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

    def test_many_values_no_crash(self):
        """Generate 1000 values (covers many block refills)."""
        rng = ChaCha8Rng.seed_from_u64(7)
        for _ in range(1000):
            val = rng.next_u32()
            assert 0 <= val <= 0xFFFFFFFF

    def test_key_constructor(self):
        key = b'\x01' * 32
        rng = ChaCha8Rng(key)
        assert 0 <= rng.next_u32() <= 0xFFFFFFFF

    def test_invalid_key_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            ChaCha8Rng(b'\x00' * 16)


# ---------------------------------------------------------------------------
# derive_nonce (from JSON vectors)
# ---------------------------------------------------------------------------

class TestDeriveNonce:
    """Verify nonce derivation matches Rust's derive_nonce."""

    @pytest.mark.parametrize(
        'vec',
        _VECTORS['derive_nonce'],
        ids=[
            f"{v['miner_id']}/blk{v['block_number']}"
            for v in _VECTORS['derive_nonce']
        ],
    )
    def test_derive_nonce_from_vectors(self, vec):
        nonce = derive_nonce(
            parent_hash=bytes.fromhex(vec['parent_hash_hex']),
            miner_id=vec['miner_id'],
            block_number=vec['block_number'],
            salt=bytes.fromhex(vec['salt_hex']),
        )
        assert nonce == vec['expected_nonce']

    def test_returns_u64_range(self):
        nonce = derive_nonce(b'\x00' * 32, 'miner', 0, b'\x00' * 32)
        assert 0 <= nonce < 2**64

    def test_different_inputs_differ(self):
        base = (b'\x00' * 32, 'miner', 0, b'\x00' * 32)
        n_base = derive_nonce(*base)
        assert n_base != derive_nonce(b'\x01' * 32, 'miner', 0, b'\x00' * 32)
        assert n_base != derive_nonce(b'\x00' * 32, 'other', 0, b'\x00' * 32)
        assert n_base != derive_nonce(b'\x00' * 32, 'miner', 1, b'\x00' * 32)
        assert n_base != derive_nonce(b'\x00' * 32, 'miner', 0, b'\xff' * 32)


# ---------------------------------------------------------------------------
# generate_ising_model (from JSON vectors)
# ---------------------------------------------------------------------------

class TestGenerateIsingModel:
    """Verify Ising model generation matches Rust's generate_ising_model."""

    @pytest.mark.parametrize(
        'vec',
        _VECTORS['generate_ising_model'],
        ids=[
            f"nonce={v['nonce']}/n{len(v['nodes'])}e{len(v['edges'])}"
            for v in _VECTORS['generate_ising_model']
        ],
    )
    def test_ising_model_from_vectors(self, vec):
        edges = [tuple(e) for e in vec['edges']]
        h, J = generate_ising_model(
            vec['nonce'], vec['nodes'], edges, vec['allowed_h_values']
        )
        expected_h = {int(k): v for k, v in vec['expected_h'].items()}
        expected_j = {
            tuple(int(x) for x in k.split(',')): v
            for k, v in vec['expected_j'].items()
        }
        assert h == expected_h
        assert J == expected_j

    def test_h_values_from_allowed_set(self):
        allowed = [-1.0, 0.0, 1.0]
        h, _ = generate_ising_model(99, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)], allowed)
        for v in h.values():
            assert v in allowed

    def test_j_values_are_pm1(self):
        _, J = generate_ising_model(99, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)])
        for v in J.values():
            assert v in (-1.0, 1.0)

    def test_deterministic(self):
        nodes, edges = [0, 1, 2], [(0, 1), (1, 2)]
        h1, J1 = generate_ising_model(42, nodes, edges)
        h2, J2 = generate_ising_model(42, nodes, edges)
        assert h1 == h2 and J1 == J2

    def test_h_generated_before_j(self):
        """Verify RNG consumption order: h first, then J."""
        nodes = [0, 1, 2, 3, 4]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        rng = ChaCha8Rng.seed_from_u64(42)
        expected_h_indices = [rng.next_u32() % 3 for _ in nodes]
        expected_j_bits = [rng.next_u32() & 1 for _ in edges]

        h, J = generate_ising_model(42, nodes, edges)

        allowed = [-1.0, 0.0, 1.0]
        for node_id, idx in zip(nodes, expected_h_indices):
            assert h[node_id] == allowed[idx]
        for (u, v), bit in zip(edges, expected_j_bits):
            assert J[(u, v)] == (-1.0 if bit == 0 else 1.0)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Old functions should still produce the same output."""

    def test_old_nonce_function_unchanged(self):
        from shared.quantum_proof_of_work import ising_nonce_from_block
        nonce = ising_nonce_from_block(
            prev_hash=b'\x00' * 32,
            miner_id='test',
            cur_index=0,
            salt=b'\x00' * 32,
        )
        assert isinstance(nonce, int)
        assert 0 <= nonce < 2**32

    def test_old_generate_function_unchanged(self):
        from shared.quantum_proof_of_work import generate_ising_model_from_nonce
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        h, J = generate_ising_model_from_nonce(42, nodes, edges)
        # Old function uses numpy PCG64, should still work
        assert len(h) == 3
        assert len(J) == 2
        for v in h.values():
            assert v in (-1.0, 0.0, 1.0)
        for v in J.values():
            assert v in (-1.0, 1.0)
