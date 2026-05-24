"""Tests for IsingModel dataclass, RandomIsingFeeder, and FixedIsingFeeder."""
from __future__ import annotations

import dataclasses
import time

import pytest

from shared.ising_feeder import FixedIsingFeeder, RandomIsingFeeder
from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
)

# Small graph for fast tests
_NODES = list(range(10))
_EDGES = [(i, i + 1) for i in range(9)]
_LAST_PROOF_BLOCK_HASH = b"testhash".ljust(32, b"\x00")
_MINER_BYTES = b"test-miner".ljust(32, b"\x00")


def _make_feeder(**kwargs):
    defaults = dict(
        last_proof_block_hash=_LAST_PROOF_BLOCK_HASH,
        miner_bytes=_MINER_BYTES,
        nodes=_NODES,
        edges=_EDGES,
        buffer_size=4,
        max_workers=1,
    )
    defaults.update(kwargs)
    return RandomIsingFeeder(**defaults)


_NONCE_BYTES_42 = (42).to_bytes(32, "big")


class TestIsingModel:
    def test_fields(self):
        model = IsingModel(
            h={0: 1.0}, J={(0, 1): -1.0},
            nonce=_NONCE_BYTES_42, salt=b"salt",
        )
        assert model.h == {0: 1.0}
        assert model.J == {(0, 1): -1.0}
        assert model.nonce == _NONCE_BYTES_42
        assert model.salt == b"salt"

    def test_model_immutable(self):
        model = IsingModel(
            h={0: 1.0}, J={(0, 1): -1.0},
            nonce=_NONCE_BYTES_42, salt=b"salt",
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            model.nonce = b"\x99" * 32


class TestRandomIsingFeeder:
    def test_pop_returns_ising_model(self):
        feeder = _make_feeder(seed=1)
        try:
            model = feeder.pop_blocking()
            assert isinstance(model, IsingModel)
            assert isinstance(model.h, dict)
            assert isinstance(model.J, dict)
            assert isinstance(model.nonce, bytes)
            assert len(model.nonce) == 32
            assert isinstance(model.salt, bytes)
            assert len(model.salt) == 32
        finally:
            feeder.stop()

    def test_deterministic_seed(self):
        feeder1 = _make_feeder(seed=42)
        feeder2 = _make_feeder(seed=42)
        try:
            m1 = [feeder1.pop_blocking() for _ in range(3)]
            m2 = [feeder2.pop_blocking() for _ in range(3)]
            for a, b in zip(m1, m2):
                assert a.nonce == b.nonce
                assert a.salt == b.salt
                assert a.h == b.h
                assert a.J == b.J
        finally:
            feeder1.stop()
            feeder2.stop()

    def test_different_seeds_differ(self):
        feeder1 = _make_feeder(seed=1)
        feeder2 = _make_feeder(seed=2)
        try:
            m1 = feeder1.pop_blocking()
            m2 = feeder2.pop_blocking()
            assert m1.nonce != m2.nonce
        finally:
            feeder1.stop()
            feeder2.stop()

    def test_pop_n(self):
        feeder = _make_feeder(seed=10, buffer_size=8)
        try:
            models = feeder.pop_n(4)
            assert len(models) >= 1
            nonces = {m.nonce for m in models}
            assert len(nonces) == len(models)
        finally:
            feeder.stop()

    def test_try_pop_returns_model_or_none(self):
        feeder = _make_feeder(seed=5)
        try:
            result = feeder.try_pop()
            assert (
                result is None
                or isinstance(result, IsingModel)
            )
        finally:
            feeder.stop()

    def test_buffer_stays_full(self):
        feeder = _make_feeder(seed=7, buffer_size=6)
        try:
            for _ in range(4):
                feeder.pop_blocking()
            # Give the background worker a moment to refill.
            time.sleep(0.1)
            model = feeder.try_pop()
            assert model is not None
        finally:
            feeder.stop()

    def test_stop_cleanup(self):
        feeder = _make_feeder(seed=3)
        feeder.stop()
        assert feeder._stopped
        assert len(feeder._futures) == 0

    def test_nonce_roundtrip(self):
        feeder = _make_feeder(seed=99)
        try:
            model = feeder.pop_blocking()
            h2, J2 = generate_ising_model_from_nonce(
                model.nonce, _NODES, _EDGES,
            )
            assert model.h == h2
            assert model.J == J2
        finally:
            feeder.stop()

def _make_model(seed: int = 0) -> IsingModel:
    """Build a small deterministic IsingModel for FixedIsingFeeder tests."""
    return IsingModel(
        h={0: float(seed), 1: -float(seed)},
        J={(0, 1): float(seed) * 0.5},
        nonce=(seed).to_bytes(32, "big"),
        salt=bytes([seed % 256] * 32),
    )


class TestFixedIsingFeeder:
    def test_single_model_cycles_forever(self):
        """One-element list → every pop returns the same model."""
        model = _make_model(seed=7)
        feeder = FixedIsingFeeder(models=[model])
        try:
            for _ in range(5):
                got = feeder.pop_blocking()
                assert got is model
        finally:
            feeder.stop()

    def test_multi_model_cycles_in_order(self):
        """Multi-element list → pops yield the input order, then repeat."""
        a, b, c = _make_model(1), _make_model(2), _make_model(3)
        feeder = FixedIsingFeeder(models=[a, b, c])
        try:
            sequence = [feeder.pop_blocking() for _ in range(7)]
            # First round: a, b, c. Second round restarts. Seventh = a.
            assert sequence == [a, b, c, a, b, c, a]
        finally:
            feeder.stop()

    def test_stop_is_idempotent(self):
        """``stop()`` is a no-op kept for API parity; safe to call repeatedly."""
        feeder = FixedIsingFeeder(models=[_make_model(seed=4)])
        feeder.stop()
        feeder.stop()  # second call must not raise
        assert feeder._stopped

    def test_rejects_empty_models(self):
        with pytest.raises(ValueError, match="at least one IsingModel"):
            FixedIsingFeeder(models=[])

    def test_pop_n_cycles(self):
        a, b = _make_model(10), _make_model(11)
        feeder = FixedIsingFeeder(models=[a, b])
        try:
            assert feeder.pop_n(5) == [a, b, a, b, a]
        finally:
            feeder.stop()

    def test_iteration_protocol(self):
        """``__iter__`` returns self; ``next()`` cycles indefinitely."""
        model = _make_model(seed=2)
        feeder = FixedIsingFeeder(models=[model])
        try:
            it = iter(feeder)
            assert it is feeder
            assert next(it) is model
            assert next(it) is model
        finally:
            feeder.stop()
