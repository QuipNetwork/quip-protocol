"""Tests for v0.4/rcs_vrf.py — the Liu many-circuits/one-shot-each structure."""
import sys
from pathlib import Path

# --- sys.path setup so we can import rcs_vrf as both 'base' (v0.3a) and 'v4' (v0.4) ---
_HERE = Path(__file__).resolve().parent
_V03A = _HERE.parent / "v0.3a"
for p in (str(_HERE), str(_V03A)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cirq

# Import v0.3a's rcs_vrf (the "base" with shared legs) and v0.4's rcs_vrf (this version)
# Because both files are literally named rcs_vrf.py, we use importlib for unambiguous loading.
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = _load("rcs_vrf_v03a", _V03A / "rcs_vrf.py")
v4 = _load("rcs_vrf_v04", _HERE / "rcs_vrf.py")

import leg5_qmin_liu as liu


def _params(M=20, model="phi_mixture"):
    p = dict(v4.PARAMS_V04)
    p.update({"num_circuits": M, "xeb_test_size": M, "sampling_model": model})
    return p


def test_circuits_are_distinct():
    # many DIFFERENT circuits, not one repeated
    seeds = v4.derive_circuit_seeds(b"seed", 10)
    assert len(set(seeds)) == 10                      # sub-seeds all distinct
    circuits = v4.leg3_generate_circuits(b"seed", _params(10), base.AuditLog())
    sigs = {tuple(str(op) for op in c.all_operations()) for c in circuits}
    assert len(sigs) == 10                            # circuits all distinct


def test_circuit_generation_is_deterministic():
    a = v4.derive_circuit_seeds(b"abc", 5)
    b = v4.derive_circuit_seeds(b"abc", 5)
    assert a == b                                     # reconstructible from master seed


def test_one_shot_per_circuit_shape():
    params = _params(15)
    circuits = v4.leg3_generate_circuits(b"s", params, base.AuditLog())
    samples = v4.leg4_sample_one_per_circuit(circuits, params, base.AuditLog(),
                                             master_seed=b"s")
    assert samples.shape == (15, params["n_qubits"])  # M rows, one shot each


def test_fxeb_noiseless_recovers_one():
    # noiseless quantum sampling (phi=1) over distinct circuits => F_XEB ~ 1
    params = _params(400)
    params["honest_fidelity"] = 1.0
    circuits = v4.leg3_generate_circuits(b"nl", params, base.AuditLog())
    samples = v4.leg4_sample_one_per_circuit(circuits, params, base.AuditLog(),
                                             master_seed=b"nl")
    f_xeb, _, _ = v4.leg5_xeb_score_multi(circuits, samples, b"nl", params,
                                          base.AuditLog())
    assert 0.8 < f_xeb < 1.2                          # concentrates near 1


def test_rounds_equal_num_circuits():
    out = v4.run_pipeline_liu()
    assert out["rounds"] == out["M"]                  # rounds is REAL (= M), not 1


def test_certified_matches_verified_module():
    # the certified honest-server number == Liu bound with Q_min = M
    M, n, eps_s = v4.PARAMS_V04["num_circuits"], v4.PARAMS_V04["n_qubits"], 2 ** -33
    out = v4.run_pipeline_liu()
    expected = liu.certified_min_entropy(q_min=M, n_qubits=n, eps_s=eps_s)
    assert abs(out["certified_H_honest_server"] - expected) < 1e-6


def test_fxeb_uses_distinct_circuit_per_term():
    # the Liu statistic p_{C_i}(x_i): mismatching circuits to samples should
    # destroy the signal (F_XEB drops toward 0), proving each term is circuit-specific
    params = _params(300)
    params["honest_fidelity"] = 1.0
    circuits = v4.leg3_generate_circuits(b"d", params, base.AuditLog())
    samples = v4.leg4_sample_one_per_circuit(circuits, params, base.AuditLog(),
                                             master_seed=b"d")
    f_good, _, _ = v4.leg5_xeb_score_multi(circuits, samples, b"d", params,
                                           base.AuditLog())
    # shuffle samples so each is scored against the WRONG circuit
    shuffled = samples[np.random.default_rng(0).permutation(len(samples))]
    f_bad, _, _ = v4.leg5_xeb_score_multi(circuits, shuffled, b"d", params,
                                          base.AuditLog())
    assert f_good > 0.6 and f_bad < 0.3               # signal needs correct pairing


def test_leg6_extraction_emits_bits_and_is_labeled_uncertified():
    out = v4.run_pipeline_liu(output_bits=128)
    if out["accepted"]:
        assert out["demo_output_hex"] is not None
        assert len(out["demo_output_hex"]) == 128 // 4      # 128 bits = 32 hex chars
        assert out["output_bits"] == 128
    # the honesty flag must be present and False regardless of acceptance
    assert out["output_is_certified_quantum"] is False
    assert out["output_bits"] <= out["lhl_max_extractable_bits"]


def test_leg6_refuses_to_exceed_lhl_bound():
    # requesting more bits than the LHL bound must NOT emit output
    out = v4.run_pipeline_liu(output_bits=10_000)
    assert out["demo_output_hex"] is None
    assert out["output_bits"] == 0
