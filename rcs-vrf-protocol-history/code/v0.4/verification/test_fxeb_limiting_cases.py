"""
test_fxeb_limiting_cases.py
============================

Sanity-check test suite for the F_XEB formula and the certified-entropy chain,
covering known limiting cases where the answer is theoretically derivable.

These tests aren't about matching Liu's specific output — that's already covered
by test_qmin_calibrated.py and verify_against_liu_data.py. These tests verify
the formula and chain behave correctly across the *full range* of inputs.

Cases covered:
  1. Perfect quantum sampler (phi=1) -> F_XEB = 1
  2. Pure uniform noise sampler (phi=0) -> F_XEB = 0
  3. Mixed fidelity (phi=0.5) -> F_XEB = 0.5 (E-value to within statistical noise)
  4. Fixed bitstring NOT drawn from p_C -> F_XEB = 0 (not size-biased)
  5. Worst-case adversary (always picks argmax p_C) -> F_XEB >> 1 (motivates two-sided)
  6. Single shot edge case -> finite F_XEB, high variance
  7. F_XEB has correct expected value across many circuits (LLN sanity)

Run with:
    python test_fxeb_limiting_cases.py
or:
    python -m pytest test_fxeb_limiting_cases.py -v

The tests use small n (n=8, N=256) so we can directly compute p_C(x) for the
full distribution.
"""

from __future__ import annotations
import math
import numpy as np


# Use a fixed RNG seed for reproducibility — each test will set its own
# numpy random state explicitly
RNG_SEED = 20260614


def compute_fxeb(probabilities: np.ndarray, n: int) -> float:
    """The formula under test: F_XEB = (2^n / m) * sum(p_i) - 1."""
    N = 2 ** n
    m = len(probabilities)
    return (N / m) * probabilities.sum() - 1.0


def sample_haar_random_distribution(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a Porter-Thomas-distributed probability vector p over {0,...,N-1}.
    
    A Haar-random unitary on N states applied to |0> gives outcome probabilities
    {p_C(x)}_x that follow N*p ~ Exp(1) (Porter-Thomas). We can simulate this
    by drawing N exponentials and normalising so they sum to 1.
    """
    raw = rng.exponential(scale=1.0, size=N)
    return raw / raw.sum()


# ---------------------------------------------------------------
# CASE 1: Perfect quantum sampler (phi = 1) -> F_XEB ≈ 1
# ---------------------------------------------------------------

def test_perfect_quantum_sampler_gives_fxeb_one():
    """
    With phi=1 (ideal quantum), samples are drawn from p_C itself.
    The conditional E[p_C(X) | X ~ p_C] = 2/N (size-biased), so
    F_XEB = (N/m) * E[p_C(X)] - 1 = 2 - 1 = 1.
    """
    n = 8
    N = 2 ** n
    rng = np.random.default_rng(RNG_SEED)

    # Average over many random circuits to suppress per-circuit variance
    m_per_circuit = 10000
    num_circuits = 200
    all_observed_probs = []

    for _ in range(num_circuits):
        p_C = sample_haar_random_distribution(N, rng)
        samples = rng.choice(N, size=m_per_circuit, p=p_C)
        observed_probs = p_C[samples]
        all_observed_probs.append(observed_probs)

    all_probs = np.concatenate(all_observed_probs)
    fxeb = compute_fxeb(all_probs, n)

    print(f"  CASE 1 (phi=1): F_XEB = {fxeb:.4f}  (expected: ~1.0)")
    assert 0.95 < fxeb < 1.05, (
        f"Perfect quantum should give F_XEB near 1.0 (got {fxeb})"
    )


# ---------------------------------------------------------------
# CASE 2: Pure uniform noise (phi = 0) -> F_XEB ≈ 0
# ---------------------------------------------------------------

def test_pure_uniform_noise_gives_fxeb_zero():
    """
    With phi=0 (no quantum signal), samples are uniform over {0,...,N-1}.
    For any fixed circuit, E[p_C(X) | X ~ Uniform] = (1/N) * sum_x p_C(x) = 1/N
    (since p_C sums to 1). So F_XEB = (N/m) * (1/N) - 1 = 0.
    """
    n = 8
    N = 2 ** n
    rng = np.random.default_rng(RNG_SEED + 1)

    num_circuits = 200
    m_per_circuit = 10000
    all_observed_probs = []

    for _ in range(num_circuits):
        p_C = sample_haar_random_distribution(N, rng)
        # KEY: uniform sampling, NOT drawn from p_C
        samples = rng.integers(0, N, size=m_per_circuit)
        observed_probs = p_C[samples]
        all_observed_probs.append(observed_probs)

    all_probs = np.concatenate(all_observed_probs)
    fxeb = compute_fxeb(all_probs, n)

    print(f"  CASE 2 (phi=0): F_XEB = {fxeb:.4f}  (expected: ~0.0)")
    assert -0.05 < fxeb < 0.05, (
        f"Pure noise should give F_XEB near 0.0 (got {fxeb})"
    )


# ---------------------------------------------------------------
# CASE 3: Mixed fidelity (phi = 0.5) -> F_XEB ≈ 0.5
# ---------------------------------------------------------------

def test_mixed_fidelity_gives_fxeb_phi():
    """
    With phi=0.5, samples come from q_C = 0.5 * p_C + 0.5 * uniform.
    E[F_XEB] = phi exactly, so F_XEB should average to 0.5.
    """
    n = 8
    N = 2 ** n
    rng = np.random.default_rng(RNG_SEED + 2)
    phi = 0.5

    num_circuits = 200
    m_per_circuit = 10000
    all_observed_probs = []

    for _ in range(num_circuits):
        p_C = sample_haar_random_distribution(N, rng)
        # Mixture: with probability phi sample from p_C, else uniform
        choose_quantum = rng.random(m_per_circuit) < phi
        samples_q = rng.choice(N, size=m_per_circuit, p=p_C)
        samples_u = rng.integers(0, N, size=m_per_circuit)
        samples = np.where(choose_quantum, samples_q, samples_u)

        observed_probs = p_C[samples]
        all_observed_probs.append(observed_probs)

    all_probs = np.concatenate(all_observed_probs)
    fxeb = compute_fxeb(all_probs, n)

    print(f"  CASE 3 (phi=0.5): F_XEB = {fxeb:.4f}  (expected: ~0.5)")
    assert abs(fxeb - phi) < 0.05, (
        f"Mixed fidelity should give F_XEB near phi={phi} (got {fxeb})"
    )


# ---------------------------------------------------------------
# CASE 4: Fixed bitstring (NOT size-biased) -> F_XEB ≈ 0
# ---------------------------------------------------------------

def test_fixed_bitstring_not_drawn_from_pc_gives_fxeb_zero():
    """
    If we always pick bitstring 0 (independent of the circuit), then for each
    circuit we collect p_C(0), which has UNCONDITIONAL Porter-Thomas distribution
    (NOT size-biased). E[p_C(0)] = 1/N, so F_XEB = 0.

    This isolates the "factor of 2" effect: it comes from size-biased sampling,
    NOT from Porter-Thomas per se.
    """
    n = 8
    N = 2 ** n
    rng = np.random.default_rng(RNG_SEED + 3)

    num_circuits = 20000
    fixed_bitstring = 0
    observed_probs = []

    for _ in range(num_circuits):
        p_C = sample_haar_random_distribution(N, rng)
        observed_probs.append(p_C[fixed_bitstring])

    observed_probs = np.array(observed_probs)
    fxeb = compute_fxeb(observed_probs, n)

    print(f"  CASE 4 (fixed bitstring): F_XEB = {fxeb:.4f}  (expected: ~0.0)")
    assert -0.05 < fxeb < 0.05, (
        f"Fixed-bitstring (non-size-biased) should give F_XEB near 0 (got {fxeb})"
    )


# ---------------------------------------------------------------
# CASE 5: Worst-case adversary (always picks argmax p_C) -> F_XEB >> 1
# ---------------------------------------------------------------

def test_adversary_picking_argmax_violates_upper_bound():
    """
    A cherry-picking adversary always picks the bitstring with maximum p_C(x).
    For each Haar-random circuit, the max probability is roughly
    (log N + Euler) / N >> 2/N for the size-biased mean.

    This produces F_XEB >> 1, which would PASS the one-sided check
    (F_XEB >= chi=0.3) but FAIL our two-sided check (F_XEB > chi_high).
    
    This test motivates why v0.1 introduced the two-sided check.
    """
    n = 8
    N = 2 ** n
    rng = np.random.default_rng(RNG_SEED + 4)

    num_circuits = 500
    observed_probs = []

    for _ in range(num_circuits):
        p_C = sample_haar_random_distribution(N, rng)
        # Adversary cheats by always reporting argmax
        max_prob = p_C.max()
        observed_probs.append(max_prob)

    observed_probs = np.array(observed_probs)
    fxeb = compute_fxeb(observed_probs, n)

    # For Haar-random p over N states, E[max] ≈ (H_N / N) where H_N is the
    # N-th harmonic number, so N * E[max] ≈ ln(N) + gamma ≈ ln(256) + 0.577 ≈ 6.12
    # So F_XEB ≈ 5.12 expected
    expected_fxeb = math.log(N) + 0.577 - 1
    print(f"  CASE 5 (worst-case adversary): F_XEB = {fxeb:.4f}  (theoretical: ~{expected_fxeb:.2f})")
    assert fxeb > 2.0, (
        f"Worst-case adversary should give F_XEB way above 1 (got {fxeb})"
    )
    # Confirm that this DOES violate a sensible upper threshold
    # (Liu's lower threshold chi = 0.3; a reasonable upper might be 1.5 or 2.0)
    assert fxeb > 1.5, (
        f"This is the type of cheating that motivates the two-sided check"
    )


# ---------------------------------------------------------------
# CASE 6: Single-shot edge case -> finite F_XEB
# ---------------------------------------------------------------

def test_single_shot_per_circuit_does_not_break():
    """
    Liu's structure has 1 shot per circuit. Verify the formula handles m=1
    per circuit without numerical issues. F_XEB should still average to phi
    across many circuits.
    """
    n = 8
    N = 2 ** n
    rng = np.random.default_rng(RNG_SEED + 5)

    num_circuits = 10000  # Need many circuits since each has only 1 shot
    observed_probs = []

    for _ in range(num_circuits):
        p_C = sample_haar_random_distribution(N, rng)
        sample = rng.choice(N, p=p_C)
        observed_probs.append(p_C[sample])

    observed_probs = np.array(observed_probs)
    fxeb = compute_fxeb(observed_probs, n)

    print(f"  CASE 6 (m=1 per circuit, {num_circuits} circuits): F_XEB = {fxeb:.4f}  (expected: ~1.0)")
    assert 0.92 < fxeb < 1.08, (
        f"Single-shot Liu structure should give F_XEB near 1.0 (got {fxeb})"
    )


# ---------------------------------------------------------------
# CASE 7: LLN sanity — variance shrinks as we average over more circuits
# ---------------------------------------------------------------

def test_fxeb_variance_decreases_with_more_circuits():
    """
    By the law of large numbers, F_XEB computed over m circuits should have
    standard deviation O(1/sqrt(m)). Confirm by computing F_XEB at two
    different m values and checking the ratio of standard deviations is
    consistent with sqrt(m1/m2).
    """
    n = 8
    N = 2 ** n
    rng = np.random.default_rng(RNG_SEED + 6)

    def fxeb_realisation(m_circuits):
        observed_probs = []
        for _ in range(m_circuits):
            p_C = sample_haar_random_distribution(N, rng)
            sample = rng.choice(N, p=p_C)
            observed_probs.append(p_C[sample])
        return compute_fxeb(np.array(observed_probs), n)

    # Compute many F_XEB realisations at two different m values
    m_small = 100
    m_large = 1000
    num_trials = 50

    fxeb_small = np.array([fxeb_realisation(m_small) for _ in range(num_trials)])
    fxeb_large = np.array([fxeb_realisation(m_large) for _ in range(num_trials)])

    std_small = fxeb_small.std()
    std_large = fxeb_large.std()
    ratio = std_small / std_large
    expected_ratio = math.sqrt(m_large / m_small)  # = sqrt(10) ≈ 3.16

    print(f"  CASE 7 (LLN scaling): std at m={m_small}: {std_small:.4f}, "
          f"std at m={m_large}: {std_large:.4f}")
    print(f"           ratio: {ratio:.2f}  (expected ~{expected_ratio:.2f})")
    # Allow a 30% tolerance — this is a statistical test with finite trials
    assert 0.7 * expected_ratio < ratio < 1.4 * expected_ratio, (
        f"Variance ratio should match LLN scaling sqrt(m_large/m_small)"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("F_XEB sanity tests — limiting cases of the formula")
    print("=" * 60)
    print()

    tests = [
        test_perfect_quantum_sampler_gives_fxeb_one,
        test_pure_uniform_noise_gives_fxeb_zero,
        test_mixed_fidelity_gives_fxeb_phi,
        test_fixed_bitstring_not_drawn_from_pc_gives_fxeb_zero,
        test_adversary_picking_argmax_violates_upper_bound,
        test_single_shot_per_circuit_does_not_break,
        test_fxeb_variance_decreases_with_more_circuits,
    ]

    all_passed = True
    for t in tests:
        try:
            t()
            print()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            print()
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ All limiting-case sanity tests passed.")
    else:
        print("✗ Some tests failed — investigate before relying on the chain.")
    print("=" * 60)
