import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from quip_proto import wire, scoring  # noqa: E402

GOLDEN = json.loads((ROOT / "conformance" / "golden_vectors.json").read_text())


def test_wire_roundtrip():
    assert wire.encode_i32_le([-1000, 0, 1000]) == bytes([0x18, 0xFC, 0xFF, 0xFF, 0, 0, 0, 0, 0xE8, 3, 0, 0])
    assert wire.decode_i32_le(wire.encode_i32_le([-1000, 0, 1000])) == [-1000, 0, 1000]
    assert wire.encode_spins([1, -1, 1]) == bytes([0x01, 0xFF, 0x01])
    assert wire.decode_spins(bytes([0x01, 0xFF, 0x01])) == [1, -1, 1]


def test_energy_matches_golden():
    for c in GOLDEN["energy"]:
        h = [v / 1000.0 for v in c["h_milli"]]
        j = [v / 1000.0 for v in c["j_milli"]]
        edges = [tuple(e) for e in c["edges"]]
        assert scoring.energy_milli(c["spins"], h, j, edges) == c["energy_milli"]


def test_truncation_matches_golden():
    # The golden `truncation` section (added by a Task-4 fix) pins truncation
    # toward zero cross-language; cases are chosen so int() != round(). Python's
    # int(e*1000) must match the committed (truncated) energy_milli, and the
    # matching Rust test (golden_scoring.rs) asserts the identical values.
    for c in GOLDEN["truncation"]:
        assert int(c["energy"] * 1000) == c["energy_milli"]


def test_diversity_matches_golden():
    for c in GOLDEN["diversity"]:
        assert abs(scoring.set_diversity(c["solutions"]) - c["diversity"]) < 1e-9


def test_energy_oob_edge_is_skipped_not_raising():
    # edge (0, 5) references node 5, out of range for a 2-spin problem; must be
    # skipped like a length-mismatched h/j entry, not raise IndexError.
    # E = (1*1) + (1*-1) = 0 -> 0 milli
    assert scoring.energy_milli([1, -1], [1.0, 1.0], [1.0], [(0, 5)]) == 0


def test_energy_truncates_toward_zero():
    # E = 0.0015 -> int(1.5) -> 1 (truncation, not round to 2); exercises the
    # truncation branch through scoring.energy_milli itself, mirroring Rust's
    # energy_truncates_toward_zero test (golden `energy` cases are all
    # integer-valued so this path is otherwise never hit via the function).
    assert scoring.energy_milli([1], [0.0015], [], []) == 1
