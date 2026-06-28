# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""build_feeder dispatches a feeder spec to the right IsingFeeder."""
from __future__ import annotations

import pytest

from shared.allowed_value_spec import AllowedValueSet
from shared.ising_feeder import FixedIsingFeeder, RandomIsingFeeder, build_feeder


def test_pow_spec_builds_random_feeder():
    f = build_feeder(
        ("pow", b"\x00" * 32, b"\x01" * 32), [0, 1], [(0, 1)],
        buffer_size=2, allowed_h=AllowedValueSet((0,)),
    )
    try:
        assert isinstance(f, RandomIsingFeeder)
        m = f.pop_blocking()
        assert hasattr(m, "nonce") and hasattr(m, "salt")
        # The allowed_h spec must reach the model: h=0 means every field is
        # zero. A regression where allowed_h is dropped reverts to the ternary
        # default (h in {-1,0,+1}), which the chain scores differently.
        assert all(v == 0 for v in m.h.values())
    finally:
        f.stop()


def test_pow_spec_without_allowed_h_raises():
    # Guards the Valid:0 regression: a None allowed_h on the PoW path would
    # silently build a ternary model the chain scores as the registered spec.
    with pytest.raises(ValueError, match="allowed_h"):
        build_feeder(("pow", b"\x00" * 32, b"\x01" * 32), [0, 1], [(0, 1)], buffer_size=2)


def test_unknown_spec_raises():
    with pytest.raises((ValueError, NotImplementedError)):
        build_feeder(("oneshot", []), [0, 1], [(0, 1)], buffer_size=2)


def test_mempool_spec_builds_fixed_feeder():
    import numpy as np

    from shared.ring_views import ProblemView

    nodes, edges = [0, 1, 2], [(0, 1), (1, 2)]
    pv = ProblemView(slots=1, n_nodes=3, n_edges=2)
    slot = pv.claim_free(timeout=1.0)
    pv.write(slot, np.array([0.1, -0.2, 0.3]), np.array([0.5, -0.5]))
    try:
        f = build_feeder(("mempool", pv.attach_args(), slot), nodes, edges, buffer_size=8)
        assert isinstance(f, FixedIsingFeeder)
        m = f.pop_blocking()
        assert m.h == pytest.approx({0: 0.1, 1: -0.2, 2: 0.3})
        assert m.J == pytest.approx({(0, 1): 0.5, (1, 2): -0.5})
        assert m.nonce == b"\x00" * 32 and m.salt == b"\x00" * 32
        f.stop()
    finally:
        pv.close_unlink()
