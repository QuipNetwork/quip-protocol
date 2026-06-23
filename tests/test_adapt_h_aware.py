# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 QUIP Protocol Contributors

"""adapt_parameters keys the difficulty band off the problem's allowed_h.

Zeroing the ternary field moves the achievable ground state ~+3% shallower, so
a fixed target energy is *harder* relative to the zero-field band than the
ternary band. adapt_parameters must therefore allocate more effort (reads /
sweeps) for that target when told ``allowed_h`` is zero-field — while leaving
ternary (default) callers byte-for-byte unchanged.
"""
from __future__ import annotations

from CPU.sa_miner import SimulatedAnnealingMiner
from shared.allowed_value_spec import AllowedValueSet
from shared.energy_utils import DEFAULT_H_VALUES

_M = SimulatedAnnealingMiner
_TERNARY = AllowedValueSet((-1000, 0, 1000))
_ZERO_FIELD = AllowedValueSet((0,))
# Energy inside the zero-field achievable band (observed CUDA-SA range).
_E = -14430.0


def test_h_values_from_allowed_variants():
    assert _M._h_values_from_allowed(None) == DEFAULT_H_VALUES
    assert _M._h_values_from_allowed(_ZERO_FIELD) == (0.0,)
    assert _M._h_values_from_allowed(_TERNARY) == (-1.0, 0.0, 1.0)
    # already-physical float sequence passes through
    assert _M._h_values_from_allowed([-1.0, 1.0]) == (-1.0, 1.0)


def test_none_matches_ternary_and_legacy_default():
    """allowed_h=None, explicit ternary, and the no-arg call all agree."""
    base = _M.adapt_parameters(_E, 0.2, 5)
    assert _M.adapt_parameters(_E, 0.2, 5, allowed_h=None) == base
    assert _M.adapt_parameters(_E, 0.2, 5, allowed_h=_TERNARY) == base


def test_zero_field_is_harder_than_ternary():
    """Same target → zero-field band rates it harder → more reads & sweeps."""
    ternary = _M.adapt_parameters(_E, 0.2, 5, allowed_h=_TERNARY)
    zero = _M.adapt_parameters(_E, 0.2, 5, allowed_h=_ZERO_FIELD)
    assert zero["num_reads"] > ternary["num_reads"]
    assert zero["num_sweeps"] > ternary["num_sweeps"]


def test_zero_field_difficulty_higher():
    d_tern = _M.energy_to_difficulty(_E, h_values=(-1.0, 0.0, 1.0))
    d_zero = _M.energy_to_difficulty(_E, h_values=(0.0,))
    assert d_zero > d_tern
    # default (no h_values) equals ternary
    assert _M.energy_to_difficulty(_E) == d_tern
