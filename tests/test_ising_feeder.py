"""Tests for IsingModel dataclass and IsingFeeder."""
from __future__ import annotations

import dataclasses
import time

import pytest

from shared.ising_feeder import IsingFeeder
from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
)

# Small graph for fast tests
_NODES = list(range(10))
_EDGES = [(i, i + 1) for i in range(9)]
_PREV_HASH = b"testhash"
_MINER_ID = "test-miner"
_CUR_INDEX = 0


def _make_feeder(**kwargs):
    defaults = dict(
        prev_hash=_PREV_HASH,
        miner_id=_MINER_ID,
        cur_index=_CUR_INDEX,
        nodes=_NODES,
        edges=_EDGES,
        buffer_size=4,
        max_workers=1,
    )
    defaults.update(kwargs)
    return IsingFeeder(**defaults)


class TestIsingModel:
    def test_fields(self):
        model = IsingModel(
            h={0: 1.0}, J={(0, 1): -1.0},
            nonce=42, salt=b"salt",
        )
        assert model.h == {0: 1.0}
        assert model.J == {(0, 1): -1.0}
        assert model.nonce == 42
        assert model.salt == b"salt"

    def test_model_immutable(self):
        model = IsingModel(
            h={0: 1.0}, J={(0, 1): -1.0},
            nonce=42, salt=b"salt",
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            model.nonce = 99


class TestIsingFeeder:
    def test_pop_returns_ising_model(self):
        feeder = _make_feeder(seed=1)
        try:
            model = feeder.pop_blocking()
            assert isinstance(model, IsingModel)
            assert isinstance(model.h, dict)
            assert isinstance(model.J, dict)
            assert isinstance(model.nonce, int)
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
            time.sleep(0.5)
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

    def test_update_block(self):
        feeder = _make_feeder(seed=11)
        try:
            m_before = feeder.pop_blocking()
            feeder.update_block(
                b"newhash", "new-miner", 1,
            )
            m_after = feeder.pop_blocking()
            assert m_before.nonce != m_after.nonce
        finally:
            feeder.stop()
