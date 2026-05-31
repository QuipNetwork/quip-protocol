# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""build_feeder dispatches a feeder spec to the right IsingFeeder."""
from __future__ import annotations

import pytest

from shared.ising_feeder import RandomIsingFeeder, build_feeder


def test_pow_spec_builds_random_feeder():
    f = build_feeder(("pow", b"\x00" * 32, b"\x01" * 32), [0, 1], [(0, 1)], buffer_size=2)
    try:
        assert isinstance(f, RandomIsingFeeder)
        m = f.pop_blocking()
        assert hasattr(m, "nonce") and hasattr(m, "salt")
    finally:
        f.stop()


def test_unknown_spec_raises():
    with pytest.raises((ValueError, NotImplementedError)):
        build_feeder(("mempool", 0), [0, 1], [(0, 1)], buffer_size=2)
    with pytest.raises((ValueError, NotImplementedError)):
        build_feeder(("oneshot", []), [0, 1], [(0, 1)], buffer_size=2)
