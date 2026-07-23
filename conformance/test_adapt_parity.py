"""Regression guard: Python `energy_to_difficulty` still matches the committed
golden that the Rust port (`quip-miner-core/src/adapt.rs`) also consumes.

If the GSE model changes on either side, one of the two parity tests fails.
"""
import json
from pathlib import Path

from shared.energy_utils import energy_to_difficulty

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "conformance/golden_adapt.json").read_text()
)


def test_energy_to_difficulty_matches_golden():
    for c in GOLDEN["energy_to_difficulty"]:
        h = tuple(v / 1000.0 for v in c["allowed_h_milli"])
        got = energy_to_difficulty(
            c["target_milli"] / 1000.0,
            num_nodes=c["num_nodes"],
            num_edges=c["num_edges"],
            h_values=h,
        )
        assert abs(got - c["difficulty"]) < 1e-12
