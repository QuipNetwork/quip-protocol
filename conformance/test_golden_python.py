import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "conformance" / "golden_vectors.json"


def test_golden_file_exists_and_is_stable():
    """Regenerating must reproduce the committed file byte-for-byte (drift guard)."""
    assert GOLDEN.exists(), "run: python conformance/gen_golden_vectors.py"
    committed = GOLDEN.read_text()
    regenerated = subprocess.run(
        [sys.executable, str(ROOT / "conformance" / "gen_golden_vectors.py"), "--stdout"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert regenerated == committed, "golden_vectors.json is stale; regenerate and commit"


def test_energy_cases_are_self_consistent():
    data = json.loads(GOLDEN.read_text())
    assert data["version"] == 1
    assert len(data["ising"]) >= 1
    assert len(data["energy"]) >= 3
    # at least one draw-order case: h length == len(nodes), j length == len(edges)
    case = data["ising"][0]
    assert len(case["h_milli"]) == len(case["nodes"])
    assert len(case["j_milli"]) == len(case["edges"])
    assert len(data["truncation"]) >= 4
    for c in data["truncation"]:
        assert int(c["energy"] * 1000) == c["energy_milli"]
    # at least one case where truncation != rounding (proves the hazard is exercised)
    assert any(int(c["energy"] * 1000) != round(c["energy"] * 1000) for c in data["truncation"])
