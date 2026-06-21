"""
test_qmin_calibrated.py
=======================

Regression test for leg5_qmin_liu — the principled (not curve-fit) Q_min
implementation directly matching Liu et al.'s Zenodo reference code AND
their notebook parameters.

The pre-calibration version (leg5_liu_certified.compute_q_min) returned
Q_min ≈ 3860 with Liu's Table I parameters and c_eff = 1.0, and was xfail'd
with explicit "do not fudge" notice.

After reconciling with Zenodo entropy.py and Table2-bounds-on-extractable-
entropy.ipynb:
  - Replaced grid optimization over delta with closed-form Chernoff delta.
  - Threshold changed from 4*eps_s to eps_sou/2 (Liu's half-budget split).
  - Adversary model uses eff = 0.5 and A = 4 * FRONTIER_theoretical = 8e18,
    giving 4 exaFLOPS effective adversary compute.
  - Reproduces Liu's notebook exactly: Q_min = 1297, H = 71313.07,
    extractable = 71273.21.

Tests verify the calibration holds robustly across parameter perturbations.
The end-to-end check against Liu's experimental data is in a separate
script: verify_against_liu_data.py.
"""

import pytest

from leg5_qmin_liu import compute_q_min, PAPER


def _call_kwargs(**overrides):
    """Build the kwargs for compute_q_min from the PAPER dict.
    Use overrides to perturb individual parameters for testing."""
    kwargs = dict(
        epsilon_sou=PAPER['epsilon_sou'],
        chi=PAPER['chi'],
        m=PAPER['m'],
        M=PAPER['M'],
        t_tot=PAPER['M'] * PAPER['t_threshold_per_circuit'],
        eff=PAPER['eff'],
        A=PAPER['A'],
        B_val=PAPER['B_val_per_circuit'],
    )
    kwargs.update(overrides)
    return kwargs


def test_calibrated_qmin_reproduces_paper_exactly():
    """With Liu's exact notebook parameters, Q_min matches the paper exactly."""
    Q = compute_q_min(**_call_kwargs())
    target = PAPER['Q_min_paper']  # 1297
    assert Q == target, (
        f"Calibrated Q_min should match paper exactly (paper: {target}, "
        f"got: {Q})."
    )


def test_protocol_breaks_down_for_too_strong_adversary():
    """At sufficient adversary compute, Q_min collapses to 0 (no certification).

    This is the empirical 'protocol breakdown' point: the adversary has
    enough faking capacity that they can pass the audit even with zero
    real quantum rounds. The protocol returns Q_min = 0 meaning 'we can't
    certify any rounds were quantum.'

    Liu's Table 2 sweep shows this around 6 x Frontier (adversary 1.5x
    stronger than Liu's chosen model). Here we test with eff = 1.0
    (2x Liu's chosen eff) which should already be past the breakdown.
    """
    Q = compute_q_min(**_call_kwargs(eff=1.0))
    assert Q == 0, (
        f"Adversary 2x stronger than Liu's model should break the protocol "
        f"(expected Q_min = 0; got {Q})."
    )


def test_weaker_adversary_increases_qmin():
    """Halving Liu's adversary compute should significantly raise Q_min.

    eff = 0.25 corresponds to half of Liu's chosen eff = 0.5, i.e., the
    adversary runs at 25% sustained efficiency instead of 50%. With less
    effective compute, the adversary can fake fewer rounds, so the
    boundary Q (where they transition to 'can pass') moves UP.
    """
    Q = compute_q_min(**_call_kwargs(eff=0.25))
    assert Q > PAPER['Q_min_paper'] * 1.5, (
        f"Halving adversary efficiency should significantly raise Q_min "
        f"(got {Q}; expected > {PAPER['Q_min_paper'] * 1.5:.0f})."
    )


def test_qmin_decreases_with_stricter_soundness():
    """Stricter eps_sou yields smaller Q_min (more conservative claim).

    In Liu's convention, Q_min is the boundary where adversary pass prob
    crosses eps_sou/2. Smaller eps_sou requires Pr[pass] very small;
    since Pr[pass] is increasing in Q, the boundary moves DOWN.
    """
    Q_normal = compute_q_min(**_call_kwargs())
    Q_strict = compute_q_min(**_call_kwargs(epsilon_sou=1e-9))
    assert Q_strict < Q_normal, (
        f"Stricter soundness should yield smaller Q_min "
        f"(strict {Q_strict} should be less than normal {Q_normal})."
    )


def test_qmin_decreases_with_more_adversary_compute():
    """Higher eff (more adversary compute) yields smaller Q_min.

    Reasoning: stronger adversary can fake more rounds, so it can pass
    with fewer real quantum rounds. The Q boundary where they cross from
    'can't pass' to 'can pass' moves DOWN.

    Stronger adversary -> smaller Q_min -> less certified randomness.

    This matches Liu's Table 2 sweep (rate drops as a*Frontier increases)
    and design notes Section 4.8.
    """
    Q_weak   = compute_q_min(**_call_kwargs(eff=0.25))  # half of Liu's eff
    Q_strong = compute_q_min(**_call_kwargs(eff=1.0))   # double Liu's eff
    assert Q_strong < Q_weak, (
        f"Stronger adversary should yield smaller Q_min "
        f"(strong {Q_strong} should be less than weak {Q_weak})."
    )


if __name__ == "__main__":
    test_calibrated_qmin_reproduces_paper_exactly()
    test_protocol_breaks_down_for_too_strong_adversary()
    test_weaker_adversary_increases_qmin()
    test_qmin_decreases_with_stricter_soundness()
    test_qmin_decreases_with_more_adversary_compute()
    print("All Q_min calibration regression tests passed.")
