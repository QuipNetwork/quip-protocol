"""
RCS + VRF Subnet — v0.4 Implementation (Liu structure)
======================================================

v0.3a-rev → v0.4 change: switch the protocol structure from Aaronson-Hung
(ONE circuit, MANY shots) to Liu et al. (MANY distinct circuits, ONE shot each).

Why this matters:
  * F_XEB is now the average of p_{C_i}(x_i) over DISTINCT circuits C_i (the Liu
    statistic), not p_C(x_i) over repeated shots of one circuit C (the AH statistic).
  * Each fake sample now forces a fresh full-circuit simulation for an attacker
    (no amortization across shots) — this is the anti-amortization property.
  * "Rounds" — the unit the certified bound counts — is now REAL (= number of
    circuits), not a single-circuit placeholder stuck at protocol_rounds=1.

This module reuses v0.3a's building blocks (Leg 1 VRF, the single-circuit Leg 3
generator, the Leg 4 noise model, Legs 6-7, AuditLog) and adds the multi-circuit
path on top. It does NOT mutate the v0.3a functions, so the existing 52 pipeline
tests stay valid.

Toy scale only (n=8): every circuit is fully simulated, so M stays small.
"""

from __future__ import annotations
import hashlib
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- sys.path setup for cross-version imports (v0.4 reuses some leg implementations
# from v0.3a, which lives in a sibling directory in the audit-trail repo layout) ---
_HERE = Path(__file__).resolve().parent
_V03A = _HERE.parent / "v0.3a"
if str(_V03A) not in sys.path:
    sys.path.insert(0, str(_V03A))

import numpy as np
import cirq

import rcs_vrf as base                # v0.3a's rcs_vrf (imported via sys.path above)
import leg5_qmin_liu as liu           # calibrated certified-bound module (v0.4.1)


# ============================================================
# v0.4 parameters (Liu structure)
# ============================================================
PARAMS_V04 = dict(base.PROTOCOL_PARAMS)
PARAMS_V04.update({
    "num_circuits": 300,       # M = number of DISTINCT circuits (each one shot)
    "xeb_test_size": 300,      # m = random audit subset (m <= M). Larger m needed:
                               # F_XEB std ~ 1/sqrt(m) per single shot, so small m
                               # gives big fluctuations (the paper's m-optimization
                               # tradeoff). At m=30 F_XEB can swing +/-0.24 and even
                               # go negative on an honest run; m>=200 stabilises it.
    "shots_per_circuit": 1,    # ONE shot per circuit (the whole point)
    "sampling_model": "phi_mixture",  # "phi_mixture" (fast, paper's depolarising
                               # model Eq III.10) | "density_matrix" (realistic, slow)
    "honest_fidelity": 0.30,   # phi for the fast model (matches Liu's H2-1 fidelity)
    # protocol_rounds is now DERIVED (= num_circuits), not a fixed placeholder.
})


# ============================================================
# LEG 3 (v0.4) — generate M DISTINCT circuits from one master seed
# ============================================================
def derive_circuit_seeds(master_seed: bytes, num_circuits: int) -> List[bytes]:
    """
    Deterministically derive M distinct, domain-separated sub-seeds from the
    master seed. Reconstructible by anyone holding the master seed (so the
    verifier and any auditor can regenerate the exact circuits).
    """
    seeds = []
    for i in range(num_circuits):
        shake = hashlib.shake_256()
        shake.update(master_seed + b"circuit-index" + i.to_bytes(4, "big"))
        seeds.append(shake.digest(32))
    return seeds


def leg3_generate_circuits(master_seed: bytes, params: Dict[str, Any],
                           audit: base.AuditLog) -> List[cirq.Circuit]:
    """
    Generate M = params['num_circuits'] DISTINCT circuits, each from its own
    sub-seed (calls the v0.3a single-circuit generator under the hood).
    """
    M = params["num_circuits"]
    sub_seeds = derive_circuit_seeds(master_seed, M)
    circuits = []
    for s in sub_seeds:
        # silent per-circuit audit (avoid M verbose entries): use a throwaway log
        circuits.append(base.leg3_generate_circuit(s, params, base.AuditLog()))
    audit.append("LEG_3_CIRCUITS_MULTI", {
        "num_circuits": M,
        "structure": "Liu: many distinct circuits, one shot each",
        "master_seed_hex": master_seed.hex()[:64] + "...",
        "sub_seed_derivation": "SHAKE256(master || 'circuit-index' || i)",
        "n_qubits": params["n_qubits"], "depth": params["depth"],
    })
    return circuits


# ============================================================
# LEG 4 (v0.4) — ONE shot from each distinct circuit
# ============================================================
def leg4_sample_one_per_circuit(circuits: List[cirq.Circuit],
                                params: Dict[str, Any],
                                audit: base.AuditLog,
                                master_seed: bytes = b"") -> np.ndarray:
    """
    Take exactly ONE measurement from each circuit. Returns array of shape
    (M, n_qubits) — one bitstring per circuit (NOT many shots of one circuit).

    Two sampling models:
      "density_matrix" — realistic depolarising-noise simulation (slow: a full
            density-matrix sim per circuit).
      "phi_mixture"    — the paper's depolarising model (Eq III.10): draw from the
            ideal (Porter-Thomas) distribution with prob phi, else uniform. Fast
            (state-vector sim per circuit), used for toy demos at large M.
    """
    n = params["n_qubits"]
    model = params.get("sampling_model", "density_matrix")
    samples = []
    start = time.time()

    if model == "phi_mixture":
        phi = params.get("honest_fidelity", 0.30)
        rng = np.random.default_rng(
            int.from_bytes(hashlib.shake_256(master_seed + b"sampling").digest(8), "big")
        )
        sim = cirq.Simulator()
        for c in circuits:
            cm = cirq.Circuit([op for op in c.all_operations()
                               if not isinstance(op.gate, cirq.MeasurementGate)])
            probs = np.abs(sim.simulate(cm).final_state_vector) ** 2
            x = rng.choice(2 ** n, p=probs) if rng.random() < phi else rng.integers(2 ** n)
            bits = [int(b) for b in format(int(x), f"0{n}b")]
            samples.append(bits)
    else:  # density_matrix (realistic, slow)
        noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(p=params["noise_rate"]))
        sim = cirq.DensityMatrixSimulator(noise=noise)
        for c in circuits:
            result = sim.run(c, repetitions=1)
            samples.append(result.measurements["m"][0])

    elapsed = time.time() - start
    samples = np.array(samples)
    audit.append("LEG_4_SAMPLING_MULTI", {
        "num_circuits": len(circuits),
        "shots_per_circuit": 1,
        "sampling_model": model,
        "samples_shape": list(samples.shape),
        "wall_time_seconds": elapsed,
        "first_5_samples": samples[:5].tolist(),
    })
    return samples


# ============================================================
# LEG 5 (v0.4) — F_XEB over DISTINCT circuits (the Liu statistic)
# ============================================================
def leg5_xeb_score_multi(circuits: List[cirq.Circuit], samples: np.ndarray,
                         master_seed: bytes, params: Dict[str, Any],
                         audit: base.AuditLog) -> Tuple[float, bool, float]:
    """
    F_XEB = (2^n / m) * sum_{i in V} p_{C_i}(x_i) - 1   over a random audit subset
    V of size m, drawn from the M (circuit, one-shot) pairs. Each term uses a
    DIFFERENT circuit C_i evaluated at its own single returned string x_i.

    Returns (F_XEB, accepted, certified_min_entropy_bits[honest-server]).
    """
    n = params["n_qubits"]
    M = len(circuits)
    m = min(params.get("xeb_test_size", M), M)
    chi_low, chi_high = params["chi_low"], params["chi_high"]

    # --- audit subset chosen via the seed, AFTER samples are in (post-commit) ---
    # Unpredictable to the server: it can't know which circuits will be graded.
    rng = np.random.default_rng(
        int.from_bytes(hashlib.shake_256(master_seed + b"xeb-subset").digest(8), "big")
    )
    V = rng.choice(M, size=m, replace=False)

    # --- compute p_{C_i}(x_i) for each audited circuit (full state-vector sim) ---
    sim = cirq.Simulator()
    p_observed = []
    for i in V:
        c_nomeas = cirq.Circuit(
            [op for op in circuits[i].all_operations()
             if not isinstance(op.gate, cirq.MeasurementGate)]
        )
        probs = np.abs(sim.simulate(c_nomeas).final_state_vector) ** 2
        idx = int("".join(str(b) for b in samples[i]), 2)
        p_observed.append(probs[idx])
    p_observed = np.array(p_observed)

    f_xeb = (2 ** n) * np.mean(p_observed) - 1.0
    accepted = (f_xeb >= chi_low) and (f_xeb <= chi_high)

    # --- certified entropy: honest server => Q_min = M (every round quantum) ---
    # Uses the VERIFIED leg5_qmin_liu (calibrated as of v0.4.1) bound. Adversarially at toy scale this
    # would be ~0; the honest-server number (Q_min=M) is what we quote, labelled.
    eps_s = params.get("smoothing_eps", 2 ** -33)
    H_honest = liu.certified_min_entropy(q_min=M, n=n, epsilon_sou=eps_s)

    audit.append("LEG_5_XEB_MULTI", {
        "structure": "Liu: F_XEB averaged over DISTINCT circuits, one shot each",
        "f_xeb": float(f_xeb),
        "M_circuits": M, "m_audited": m,
        "audit_subset_selection": "seed-derived, post-commit (unpredictable to server)",
        "chi_low": chi_low, "chi_high": chi_high,
        "accepted": bool(accepted),
        "mean_p_observed": float(np.mean(p_observed)),
        "rounds_counted": M,   # REAL now: one round per distinct circuit
        "certified_min_entropy_bits_honest_server": float(H_honest),
        "certified_assumption": "honest server (Q_min = M); adversarial toy-scale ~ 0",
        "certified_method": "Liu Thm 1/3: Q_min*(n-1) - log2(1/eps_s) [leg5_qmin_liu]",
        "smoothing_eps": float(eps_s),
    })
    return f_xeb, accepted, float(H_honest)


# ============================================================
# Runner — wire the Liu-structure path end to end
# ============================================================
def run_pipeline_liu(round_number: int = 1, output_path: str = "audit_log_v0_4.json",
                     output_bits: int = 256) -> Dict[str, Any]:
    audit = base.AuditLog()
    params = PARAMS_V04

    # LEG 1 — VRF seed (reuse v0.3a)
    public_input = f"round-{round_number}".encode()
    seed = base.leg1_vrf_seed(public_input, audit)

    # LEG 3 — many distinct circuits
    circuits = leg3_generate_circuits(seed, params, audit)
    # LEG 4 — one shot each
    samples = leg4_sample_one_per_circuit(circuits, params, audit, master_seed=seed)
    # LEG 5 — F_XEB over distinct circuits + certified (honest-server) entropy
    f_xeb, accepted, H = leg5_xeb_score_multi(circuits, samples, seed, params, audit)

    # ---- LEG 6 — extraction (wired onto the v0.4 multi-circuit samples) ----
    # HONESTY: we extract against the HONEST-SERVER entropy H (Q_min=M). The output
    # is uniform *conditioned on* that assumption and on a SIMULATED source. It is a
    # demonstration of the extraction plumbing — NOT certified-quantum randomness.
    extracted_hex = None
    lhl_bound = base.max_extractable_bits(H, params["epsilon"])
    if accepted and output_bits <= lhl_bound:
        drand_seed = base.fetch_drand_pulse()   # independent extractor seed (fallback if offline)
        out = base.leg6_toeplitz_extract(samples, drand_seed, output_bits, audit,
                                         max_extractable_bits_bound=lhl_bound)
        extracted_hex = out.hex()

    # LEG 7 — anchor the run
    audit.save(output_path)

    return {
        "f_xeb": f_xeb, "accepted": accepted,
        "M": params["num_circuits"], "m": params["xeb_test_size"],
        "rounds": params["num_circuits"],
        "certified_H_honest_server": H,
        "lhl_max_extractable_bits": lhl_bound,
        "output_bits": output_bits if extracted_hex else 0,
        "demo_output_hex": extracted_hex,
        # explicit honesty flags so callers can't misread the output:
        "output_is_certified_quantum": False,
        "output_caveat": (
            "Demonstration only: uniform bits CONDITIONED on the honest-server "
            "assumption (Q_min=M) and a SIMULATED source. NOT certified-quantum "
            "randomness — that needs a real QPU (Leg 4), enforced timing (Leg 5), "
            "production scale (so Q_min>0 adversarially), and a verifiable drand seed."
        ),
        "audit_log": output_path,
    }


if __name__ == "__main__":
    out = run_pipeline_liu()
    print("v0.4 (Liu structure) — many circuits, one shot each")
    print(f"  M circuits / m audited : {out['M']} / {out['m']}")
    print(f"  F_XEB                   : {out['f_xeb']:.4f}  (accepted: {out['accepted']})")
    print(f"  rounds counted (= M)    : {out['rounds']}   <- REAL now, not a placeholder")
    print(f"  certified H_min (honest): {out['certified_H_honest_server']:.1f} bits "
          f"[Q_min=M; adversarial toy-scale ~0]")
    print(f"  LHL extractable bound   : {out['lhl_max_extractable_bits']} bits")
    if out["demo_output_hex"]:
        print(f"  --- LEG 6 extraction (DEMO ONLY) ---")
        print(f"  output ({out['output_bits']} bits): {out['demo_output_hex']}")
        print(f"  [!] {out['output_caveat']}")
    print(f"  audit log saved: {out['audit_log']}")
