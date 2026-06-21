"""
test_leg5_certified.py — unit tests for leg5_certified_entropy.

Run:  python3 -m pytest test_leg5_certified.py -v
      (or)  python3 test_leg5_certified.py   for a plain-stdlib runner.

Covers the parts that are easy to get subtly wrong:
  - GATE, DON'T SCALE (the core conceptual fix)
  - clamp-to-0 under the adversarial toy-scale framing
  - the assumption-flag / no-naked-quote discipline
  - numerical agreement with honest_server_bound.pdf (≈6967 vs ≈1659)
  - floor sits below the true PT entropy
  - the Liu min-tradeoff scaffold raises until filled
  - smoothing monotonicity, EAT-penalty effect, LHL behaviour
"""

import math
import pytest

import leg5_certified_entropy as L

BAND = dict(chi_low=0.15, chi_high=2.5)


# ----------------------------------------------------------------------------
# GATE, DON'T SCALE — the headline behavioural guarantee
# ----------------------------------------------------------------------------
def test_gate_not_scale_invariance():
    """Same #accepted rounds at different (in-band) F_XEB → identical certified
    entropy. F_XEB must NOT act as a multiplier."""
    low_band = L.certified_min_entropy([0.30] * 1000, 8, **BAND)
    high_band = L.certified_min_entropy([1.00] * 1000, 8, **BAND)
    assert low_band.bits == pytest.approx(high_band.bits)


def test_heuristic_does_scale_with_fxeb():
    """Contrast: the deprecated heuristic DOES (wrongly) vary with F_XEB —
    confirms the test above is detecting a real difference in behaviour."""
    h_low = L.estimate_min_entropy_heuristic(0.30, 8, 1000)
    h_high = L.estimate_min_entropy_heuristic(1.00, 8, 1000)
    assert h_high > h_low * 2  # 1.0 vs 0.3 → big difference


def test_gate_counts_only_in_band():
    """Rounds outside the band are discarded, not down-weighted."""
    scores = [0.30] * 600 + [0.01] * 200 + [3.0] * 200  # 600 in-band only
    res = L.certified_min_entropy(scores, 8, **BAND)
    assert res.m_accepted == 600
    assert res.breakdown["q_min"] == 600
    expected = 600 * 7 + math.log2(L.DEFAULT_SMOOTHING_EPS)
    assert res.bits == pytest.approx(max(0.0, expected))


# ----------------------------------------------------------------------------
# Clamp-to-0 under the adversarial toy-scale framing (correct, not a bug)
# ----------------------------------------------------------------------------
def test_adversarial_toy_scale_is_zero():
    res = L.certified_min_entropy([0.30] * 1000, 8, **BAND,
                                  assumption="adversarial_toy_scale")
    assert res.bits == 0.0
    assert res.breakdown["q_min"] == 0


def test_clamp_never_negative():
    """Few accepted rounds + tiny eps_s can push the raw value negative; the
    returned figure must clamp to 0."""
    res = L.certified_min_entropy([0.30] * 2, 8, **BAND, smoothing_eps=2.0 ** -60)
    assert res.bits >= 0.0
    assert res.breakdown["raw_before_clamp"] < 0.0  # confirms clamp engaged


# ----------------------------------------------------------------------------
# Assumption-flag discipline — no naked quoting
# ----------------------------------------------------------------------------
def test_quoted_carries_assumption():
    res = L.certified_min_entropy([0.30] * 1000, 8, **BAND)
    assert "honest_server" in res.quoted()
    assert "honest_server" in L.quote(res)


def test_empty_assumption_rejected():
    with pytest.raises(ValueError):
        L.CertifiedEntropy(bits=100.0, assumption="", n_qubits=8, m_accepted=10,
                           h_per_round=7.0, smoothing_eps=2.0 ** -33,
                           eat_penalty_bits=0.0, status="x", floor_slack_bits=0.0)


# ----------------------------------------------------------------------------
# Numerical agreement with the derivation
# ----------------------------------------------------------------------------
def test_canonical_certified_value():
    res = L.certified_min_entropy([0.30] * 1000, 8, **BAND)
    assert res.bits == pytest.approx(6967, abs=1)


def test_canonical_heuristic_value():
    assert L.estimate_min_entropy_heuristic(0.30, 8, 1000) == pytest.approx(1659, abs=1)


def test_ratio_about_four():
    cmp = L.compare_estimators([0.30] * 1000, 8, **BAND)
    assert cmp["ratio_certified_over_heuristic"] == pytest.approx(4.2, abs=0.1)


# ----------------------------------------------------------------------------
# Per-round entropy: floor sits strictly below the true PT value
# ----------------------------------------------------------------------------
def test_floor_below_true_entropy():
    for n in (4, 8, 12, 30):
        floor = L.min_tradeoff_function(n, mode="floor")
        true = L.true_pt_entropy(n)
        assert floor < true                       # conservative
        assert true - floor == pytest.approx(0.390, abs=0.01)  # ~0.39 slack/round


def test_true_pt_entropy_value():
    # n - (1-gamma)/ln2 ≈ n - 0.610
    assert L.true_pt_entropy(8) == pytest.approx(8 - 0.610, abs=0.005)


# ----------------------------------------------------------------------------
# The Liu min-tradeoff function is a scaffold until filled
# ----------------------------------------------------------------------------
def test_liu_mode_returns_exact_collision_entropy():
    # After reading the paper: per-round entropy is the EXACT collision entropy
    # n-1 (not a stub, not score-dependent). Both modes return n-1.
    assert L.min_tradeoff_function(8, f_xeb=1.0, mode="liu") == 7.0
    assert L.min_tradeoff_function(8, mode="floor") == 7.0


def test_unknown_floor_mode_raises():
    with pytest.raises(ValueError):
        L.min_tradeoff_function(8, mode="banana")


# ----------------------------------------------------------------------------
# Smoothing monotonicity and the EAT-penalty effect
# ----------------------------------------------------------------------------
def test_smaller_eps_fewer_bits():
    loose = L.certified_min_entropy([0.30] * 1000, 8, **BAND, smoothing_eps=2.0 ** -20)
    tight = L.certified_min_entropy([0.30] * 1000, 8, **BAND, smoothing_eps=2.0 ** -60)
    assert tight.bits < loose.bits


def test_eat_penalty_reduces_and_marks_verified():
    no_eta = L.certified_min_entropy([0.30] * 1000, 8, **BAND)
    with_eta = L.certified_min_entropy([0.30] * 1000, 8, **BAND, eat_penalty_eta=5.0)
    assert with_eta.bits < no_eta.bits
    assert with_eta.status == "verified"
    assert no_eta.status == "defensible_v1_unverified"
    # sqrt(1000)*5 ≈ 158 bits subtracted
    assert with_eta.eat_penalty_bits == pytest.approx(5.0 * math.sqrt(1000), rel=1e-6)


def test_eps_out_of_range_raises():
    with pytest.raises(ValueError):
        L.certified_min_entropy([0.30] * 10, 8, **BAND, smoothing_eps=1.5)


# ----------------------------------------------------------------------------
# Leftover Hash Lemma
# ----------------------------------------------------------------------------
def test_lhl_monotone_and_bounded():
    h = 6967.0
    loose = L.max_extractable_bits(h, 2.0 ** -30)
    tight = L.max_extractable_bits(h, 2.0 ** -60)
    assert tight < loose <= h
    assert L.max_extractable_bits(0.0, 2.0 ** -30) == 0
    assert L.max_extractable_bits(-5.0, 2.0 ** -30) == 0


def test_lhl_eps_validation():
    with pytest.raises(ValueError):
        L.max_extractable_bits(100.0, 0.0)
    with pytest.raises(ValueError):
        L.max_extractable_bits(100.0, 1.0)


# ----------------------------------------------------------------------------
# Audit block wiring
# ----------------------------------------------------------------------------
def test_audit_block_logs_both_and_extracts_from_certified():
    block = L.leg5_entropy_audit_block([0.30] * 1000, 8, epsilon_lhl=2.0 ** -50,
                                       **BAND)
    assert block["certified_assumption"] == "honest_server"
    assert block["heuristic_min_entropy_bits"] == pytest.approx(1659, abs=1)
    # extraction bound is derived from the certified number, not the heuristic
    expected_out = L.max_extractable_bits(block["certified_min_entropy_bits"], 2.0 ** -50)
    assert block["max_extractable_bits_at_epsilon"] == expected_out
    assert expected_out > block["heuristic_min_entropy_bits"]  # ~6867 > 1659


# ----------------------------------------------------------------------------
# plain-stdlib fallback runner
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
