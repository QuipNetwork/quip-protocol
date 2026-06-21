"""
leg5_certified_entropy.py
=========================

LEG 5 — certified min-entropy, replacing the v0.3a heuristic.

This module implements the *honest-server* certified smooth-min-entropy bound
derived in `honest_server_bound.pdf`, alongside the original v0.3a heuristic so
the two can be logged side by side on real runs. The exact score-dependent
min-tradeoff function h(F_XEB) from Liu et al. is scaffolded as a stub to be
filled in after checking the paper.

Design rules baked in (see the derivation):
  1. GATE, DON'T SCALE.  A round whose F_XEB lands in the accepted band counts
     for its FULL per-round floor; F_XEB is never used as a multiplier.
  2. The certified number always travels with its `assumption` label, so it
     cannot be quoted naked. (See `CertifiedEntropy.quoted()` / `quote()`.)
  3. The EAT finite-size penalty is NOT invented. Left as a parameter; when
     absent the bound is flagged "defensible_v1_unverified", not "proven".
  4. At toy scale, the adversarial framing yields 0 — by design, not by bug.

References:
  Aaronson & Hung 2023, arXiv:2303.01625   (NB: corrected ID; v0.3a cited the
                                             wrong 2303.01514)
  Liu et al. 2025,        arXiv:2503.20498
  Dupuis, Fawzi, Renner,  Entropy Accumulation, CMP (2020)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Euler–Mascheroni constant
_GAMMA = 0.5772156649015329

# Default smoothing parameter ε_s. log2(2^-33) = -33 → at n=8,M=1000 gives ≈6967.
DEFAULT_SMOOTHING_EPS = 2.0 ** -33

# --- Real parameters from Liu et al. 2025 (Nature 640:343–348; arXiv:2503.20498) ---
# Confirmed from the published paper. Their headline figure — 71,313 certified bits
# — is against a RESTRICTED adversary (sustained 1.1e18 FLOP/s, ~4x Frontier) under
# explicitly stated additional assumptions; it is NOT an unconditional bound. These
# are provided for grounding; adopt with Mehdi rather than silently swapping the
# default, since they interact with the (deferred) adversary model.
PAPER_EPS_SOUNDNESS = 1.0e-6        # ε_sou  (soundness)
PAPER_EPS_SMOOTHING = 1.0e-6 / 4.0  # ε_s = ε_sou / 4
PAPER_P_FAIL = 1.0e-4               # target failure probability
# NB: the paper's *guaranteed smooth-min-entropy rate* in its adversarial regime is
# SMALL (figures discuss guaranteeing h ≈ 0.01 bits per circuit), i.e. far below the
# honest-server floor n-1. The two live in different regimes — see the 'liu' stub.


# ----------------------------------------------------------------------------
# Per-round entropy: the true value, the conservative floor, and the Liu stub
# ----------------------------------------------------------------------------
def true_pt_entropy(n_qubits: int) -> float:
    """
    Exact per-round von Neumann (= Shannon-of-outcome) entropy of one
    Porter–Thomas sample, in bits:   H_1 = n - (1-gamma)/ln 2  ≈  n - 0.610.

    Derived in honest_server_bound.pdf, Lemma 1. This is the *true* value under
    the honest, noiseless ensemble; we extract against a lower floor (below).
    """
    return n_qubits - (1.0 - _GAMMA) / math.log(2.0)


def min_tradeoff_function(n_qubits: int,
                          f_xeb: Optional[float] = None,
                          *,
                          mode: str = "floor") -> float:
    """
    Per-round entropy h used by the certified bound (bits per accepted round).

    mode="floor"  ->  h = n - 1 bits per round.   [CORRECTED JUSTIFICATION]
                      Reading Liu et al. (Nature 640:343; arXiv:2503.20498, SM
                      Eq. III.33) showed this is NOT a conservative von Neumann
                      floor (with "0.39 bits slack") as we first thought — it is
                      the EXACT 2-Renyi (collision) entropy of Porter-Thomas,
                      H_2 = -log2(2/N) = n-1, which is the correct single-round
                      quantity for i.i.d. smooth-min-entropy accumulation. The
                      value n-1 was right; the von Neumann justification was wrong.
                      See leg5_qmin_liu.collision_entropy_pt().

    mode="liu"    ->  also returns the exact collision entropy n-1. The observed
                      F_XEB does NOT modulate the per-round entropy (it is flat
                      n-1); the score's entire role is to set Q_min, the forced
                      quantum-round count, via the adversary model. So there is no
                      score->entropy curve to "fill in" — the real work is Q_min,
                      implemented in leg5_qmin_liu.compute_q_min().

    The two paths share a signature so swapping is a one-line change at call sites.
    """
    if mode == "floor" or mode == "liu":
        # Both return the EXACT collision entropy n-1 (see docstring). The score
        # does not enter here; it enters via Q_min (leg5_qmin_liu.compute_q_min).
        return float(n_qubits - 1)
    raise ValueError(f"unknown mode {mode!r}; use 'floor' or 'liu'")


# ----------------------------------------------------------------------------
# The certified bound (honest-server)
# ----------------------------------------------------------------------------
@dataclass
class CertifiedEntropy:
    """A certified smooth-min-entropy figure that carries its own caveat."""
    bits: float                       # clamped >= 0
    assumption: str                   # e.g. "honest_server"; never empty
    n_qubits: int
    m_accepted: int
    h_per_round: float
    smoothing_eps: float
    eat_penalty_bits: float           # sqrt(M)*eta actually subtracted (0 if folded)
    status: str                       # "verified" | "defensible_v1_unverified"
    floor_slack_bits: float           # (true - floor) * M, informational
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.assumption:
            raise ValueError("CertifiedEntropy.assumption must be non-empty — "
                             "the number may not be quoted without its caveat.")

    def quoted(self) -> str:
        """The only sanctioned way to render the number as text."""
        return f"{self.bits:.0f} bits (assuming {self.assumption})"


def quote(result: CertifiedEntropy) -> str:
    """Module-level convenience matching CertifiedEntropy.quoted()."""
    return result.quoted()


def certified_min_entropy(
    fxeb_scores: Sequence[float],
    n_qubits: int,
    *,
    chi_low: float,
    chi_high: float,
    smoothing_eps: float = DEFAULT_SMOOTHING_EPS,
    eat_penalty_eta: Optional[float] = None,
    floor_mode: str = "floor",
    assumption: str = "honest_server",
) -> CertifiedEntropy:
    """
    Honest-server certified smooth-min-entropy of the full record.

        H_min^{eps_s}(X^M | E)  >=  M * h  -  sqrt(M) * eta  +  log2(eps_s)

    Implements Recipe 1 of honest_server_bound.pdf.

    Args:
      fxeb_scores:     per-round observed F_XEB (one entry per attempted round).
      n_qubits:        n.
      chi_low/chi_high: the two-sided acceptance band (the GATE).
      smoothing_eps:   ε_s in (0,1).
      eat_penalty_eta: the GEAT second-order constant η. If None, the √M penalty
                       is folded into the floor slack and the result is flagged
                       "defensible_v1_unverified" (NOT proven). Supply η (a
                       literature lookup) to get a "verified" result.
      floor_mode:      "floor" (h=n-1, default) or "liu" (the stubbed curve).
      assumption:      "honest_server" for the real bound; anything else (e.g.
                       "adversarial_toy_scale") sets Q_min=0 → the bound is 0,
                       which is the *correct* answer at toy scale.

    Returns: CertifiedEntropy.
    """
    if not (0.0 < smoothing_eps < 1.0):
        raise ValueError(f"smoothing_eps must be in (0,1), got {smoothing_eps}")
    if n_qubits < 2:
        raise ValueError("n_qubits must be >= 2")

    # ---- 1. GATE, DON'T SCALE: count accepted rounds, ignore their scores ----
    m_attempted = len(fxeb_scores)
    m_accepted = sum(1 for s in fxeb_scores if chi_low <= s <= chi_high)

    # Q_min: under honest server, every accepted round counts. Adversarially at
    # toy scale, none do.
    if assumption == "honest_server":
        q_min = m_accepted
    else:
        q_min = 0  # adversarial / unknown framing → no round is un-spoofable here

    # ---- 2. per-round floor (or the scaffolded Liu curve) ----
    h = min_tradeoff_function(n_qubits, mode=floor_mode)

    # ---- 3. leading term ----
    leading = q_min * h

    # ---- 4. smoothing remainder ----
    smoothing = math.log2(smoothing_eps)  # negative

    # ---- 5. EAT penalty / verification gate ----
    true_h = true_pt_entropy(n_qubits)
    floor_slack = max(0.0, (true_h - h)) * q_min   # bits we already gave up
    if eat_penalty_eta is None:
        penalty = 0.0
        status = "defensible_v1_unverified"
    else:
        penalty = eat_penalty_eta * math.sqrt(max(q_min, 0))
        status = "verified"

    # ---- 6. assemble and clamp ----
    H = leading - penalty + smoothing
    bits = max(0.0, H)

    return CertifiedEntropy(
        bits=bits,
        assumption=assumption,
        n_qubits=n_qubits,
        m_accepted=m_accepted,
        h_per_round=h,
        smoothing_eps=smoothing_eps,
        eat_penalty_bits=penalty,
        status=status,
        floor_slack_bits=floor_slack,
        breakdown={
            "m_attempted": float(m_attempted),
            "m_accepted": float(m_accepted),
            "q_min": float(q_min),
            "h_per_round": h,
            "true_pt_entropy": true_h,
            "leading_term": leading,
            "smoothing": smoothing,
            "eat_penalty": penalty,
            "raw_before_clamp": H,
        },
    )


# ----------------------------------------------------------------------------
# Preserved v0.3a heuristic (DEPRECATED — kept for side-by-side comparison)
# ----------------------------------------------------------------------------
def estimate_min_entropy_heuristic(f_xeb: float, n_qubits: int,
                                   m_samples: int) -> float:
    """
    The original v0.3a heuristic:  m * F_XEB * (n - log2 n - log2 ln2).

    DEPRECATED. Kept only so runs can log heuristic-vs-certified side by side.
    Two known errors vs. the certified bound (see honest_server_bound.pdf):
      (i)  F_XEB used as a per-sample MULTIPLIER (wrong; it is a gate);
      (ii) per-sample constant (n - log2 n - log2 ln2 ≈ 5.53 at n=8) instead of
           the von Neumann floor (n-1 = 7).
    """
    if f_xeb <= 0 or n_qubits < 2:
        return 0.0
    h_haar = n_qubits - math.log2(n_qubits) - math.log2(math.log(2.0))
    return m_samples * f_xeb * h_haar


# ----------------------------------------------------------------------------
# Leftover Hash Lemma (preserved; unchanged semantics)
# ----------------------------------------------------------------------------
def max_extractable_bits(min_entropy: float, epsilon: float) -> int:
    """LHL:  ell <= H_min - 2 log2(1/eps).  Floored to int, >= 0."""
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must be in (0,1), got {epsilon}")
    if min_entropy <= 0:
        return 0
    return max(0, int(min_entropy - 2 * math.log2(1.0 / epsilon)))


# ----------------------------------------------------------------------------
# Side-by-side comparison + audit block for Leg 5
# ----------------------------------------------------------------------------
def compare_estimators(fxeb_scores: Sequence[float], n_qubits: int,
                       *, chi_low: float, chi_high: float,
                       representative_fxeb: Optional[float] = None,
                       **kwargs) -> Dict[str, object]:
    """
    Compute both estimators on the same data and return a comparison dict.

    The heuristic needs a single scalar F_XEB; we use the mean of accepted
    rounds (or `representative_fxeb` if given) so the comparison is apples-ish.
    """
    cert = certified_min_entropy(fxeb_scores, n_qubits,
                                 chi_low=chi_low, chi_high=chi_high, **kwargs)
    accepted = [s for s in fxeb_scores if chi_low <= s <= chi_high]
    f_rep = (representative_fxeb if representative_fxeb is not None
             else (sum(accepted) / len(accepted) if accepted else 0.0))
    heur = estimate_min_entropy_heuristic(f_rep, n_qubits, len(accepted))
    ratio = (cert.bits / heur) if heur > 0 else float("inf")
    return {
        "certified_bits": cert.bits,
        "certified_quoted": cert.quoted(),
        "certified_status": cert.status,
        "heuristic_bits": heur,
        "heuristic_representative_fxeb": f_rep,
        "ratio_certified_over_heuristic": ratio,
        "certified": cert,
    }


def leg5_entropy_audit_block(fxeb_scores: Sequence[float], n_qubits: int,
                             epsilon_lhl: float, *,
                             chi_low: float, chi_high: float,
                             **kwargs) -> Dict[str, object]:
    """
    Drop-in producer of the Leg-5 audit dict, logging BOTH estimators and the
    LHL bound derived from the *certified* number (the heuristic is recorded for
    comparison only and must not drive extraction).
    """
    cmp = compare_estimators(fxeb_scores, n_qubits,
                             chi_low=chi_low, chi_high=chi_high, **kwargs)
    cert: CertifiedEntropy = cmp["certified"]  # type: ignore
    max_out = max_extractable_bits(cert.bits, epsilon_lhl)
    return {
        # certified (authoritative)
        "certified_min_entropy_bits": cert.bits,
        "certified_assumption": cert.assumption,
        "certified_status": cert.status,
        "min_entropy_method": (
            "honest-server certified bound: M*(n-1) + log2(eps_s) "
            "[honest_server_bound.pdf]; AH arXiv:2303.01625, Liu 2503.20498"
        ),
        "max_extractable_bits_at_epsilon": int(max_out),
        # heuristic (comparison only — DO NOT extract against this)
        "heuristic_min_entropy_bits": cmp["heuristic_bits"],
        "heuristic_note": "DEPRECATED v0.3a heuristic, logged for comparison only",
        "ratio_certified_over_heuristic": cmp["ratio_certified_over_heuristic"],
        # breakdown for auditors
        "certified_breakdown": cert.breakdown,
    }


if __name__ == "__main__":
    # Demo at the canonical toy parameters: n=8, M=1000 accepted, F_XEB≈0.3
    scores = [0.30] * 1000
    cmp = compare_estimators(scores, 8, chi_low=0.15, chi_high=2.5)
    print("n=8, M=1000, accepted F_XEB≈0.30")
    print(f"  heuristic : {cmp['heuristic_bits']:.0f} bits")
    print(f"  certified : {cmp['certified_quoted']}  [{cmp['certified_status']}]")
    print(f"  ratio     : {cmp['ratio_certified_over_heuristic']:.2f}x")
    adv = certified_min_entropy(scores, 8, chi_low=0.15, chi_high=2.5,
                                assumption="adversarial_toy_scale")
    print(f"  adversarial toy-scale framing: {adv.quoted()}  (0 by design)")
