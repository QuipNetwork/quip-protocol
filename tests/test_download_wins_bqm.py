# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for the --dump-bqm serialization in download_and_validate_wins.

No network/chain access: these exercise the pure serialization step and its
round-trip back into the (h, J) dicts the energy functions consume.
"""
from __future__ import annotations

import json

from tools.download_and_validate_wins import _serialize_bqm


def test_serialize_bqm_is_json_safe_and_round_trips() -> None:
    """Serialized h/J survive json round-trip and rebuild the original dicts."""
    h = {0: -0.5, 1: 0.5, 2: 0.0}
    j = {(0, 1): 1.0, (1, 2): -1.0}

    serialized = _serialize_bqm(h, j)
    # Must survive a json encode/decode cycle (tuple keys would not).
    reloaded = json.loads(json.dumps(serialized))

    h_back = {node: bias for node, bias in reloaded["h"]}
    j_back = {(u, v): c for u, v, c in reloaded["j"]}

    assert h_back == h
    assert j_back == j


def test_serialize_bqm_handles_empty_couplings() -> None:
    """A model with biases but no couplings serializes to an empty j list."""
    serialized = _serialize_bqm({0: 1.0}, {})
    assert serialized["h"] == [[0, 1.0]]
    assert serialized["j"] == []
