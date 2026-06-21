"""
RCS + VRF Subnet — v0.2 Implementation
=======================================

Minimal end-to-end prototype of the seven-leg pipeline:
    LEG 1 — VRF Seed Generation (real cryptographic VRF-like construction)
    LEG 2 — Parameter Precommitment (hard-coded in PROTOCOL_PARAMS)
    LEG 3 — Deterministic Circuit Specification (SHAKE256 → cirq circuit)
    LEG 4 — Quantum Sampling (cirq noisy simulator standing in for real QPUs)
    LEG 5 — Classical XEB Verification (two-sided check: chi_low ≤ F_XEB ≤ chi_high)
    LEG 6 — Randomness Extraction (Toeplitz extractor + drand pulse)
    LEG 7 — Hash-Chain Anchoring (single JSON audit log)

v0.1 → v0.2 changes:
    LEG 1 now uses a real cryptographic VRF construction (Ed25519 signatures +
    SHA3-256 hash, rather than the deterministic-hash placeholder).
    
    Each of 5 simulated participants has a real Ed25519 keypair. Each produces
    (beta_i, pi_i) by signing the public input. All proofs are verified before
    aggregation. The beta_i values are concatenated, then SHA3-256 (= H) is
    applied to compress to a 32-byte seed (the architecture's H + SHAKE256
    pattern), which LEG 3 then expands via SHAKE256.
    
    Honest framing: this is a "VRF-like" construction (Sign + Hash), not full
    RFC 9381 ECVRF. The differences don't matter at v0.2 scope but would for
    the eventual spec. Production target is full ECVRF (RFC 9381) — Mehdi will
    own that decision.
    
    Other v0.2 limits (unchanged from v0.1):
    - Participants are simulated locally, not real subnet participants
    - Keys are regenerated each pipeline run (no persistent registration)
    - Public input format is simplified (not full epoch state)
    - No real participant identity, slashing, or governance

Goal: produce a small number of certified-random bits through the full pipeline,
demonstrate the data flow, and surface the engineering decisions that matter.

This is NOT production code. It's a learning artifact.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import cirq
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


# ============================================================
# Protocol parameters (LEG 2 — hard-coded for v0)
# ============================================================
PROTOCOL_PARAMS = {
    "n_qubits": 8,           # qubit count
    "depth": 10,             # circuit depth (entangler layers)
                             # With all-to-all connectivity (random perfect
                             # matching per layer, à la Liu et al.'s edge-
                             # coloured graphs), d≥8 brings noiseless F_XEB
                             # to ~1.0 (verified via depth sweep). d=10 matches
                             # Liu et al.'s choice and gives noisy F_XEB ≈ 0.29
                             # at noise rate 0.005 — comfortably above any
                             # reasonable χ threshold.
    "shots_per_circuit": 1000,
    "num_circuits": 100,
    "chi_low": 0.15,         # Lower XEB acceptance threshold
                             # Rejects circuit-blind attacks (uniform random,
                             # fixed bitstring, replay of unrelated samples).
                             # Set below the per-circuit variance band of
                             # honest miners (F_XEB ≈ 0.29 ± 0.03) with margin.
    "chi_high": 2.5,         # Upper XEB acceptance threshold (v0.1 addition)
                             # Rejects heavy-output attacks where attacker
                             # submits only top-k high-probability bitstrings.
                             # Set above honest noiseless F_XEB (≤1.5 empirically
                             # at n=8 d=10 from depth sweep) with margin, and
                             # below top-k attack F_XEB (≥3.18 from spoofing
                             # experiment at n=8 d=10 with k=16).
    "noise_rate": 0.005,     # depolarizing noise per operation
    "epsilon": 2 ** -32,     # soundness target
    "extractor_output_bits": 64,  # final output length
}


# ============================================================
# LEG 7 — Audit log (passed through all legs)
# ============================================================
class AuditLog:
    """A hash chain of artifacts produced by each leg."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def append(self, leg: str, data: Dict[str, Any]):
        prev_hash = self.entries[-1]["hash"] if self.entries else "0" * 64
        payload = json.dumps({"leg": leg, "data": data, "prev": prev_hash},
                             sort_keys=True, default=str)
        entry_hash = hashlib.sha3_256(payload.encode()).hexdigest()
        self.entries.append({
            "leg": leg,
            "data": data,
            "prev_hash": prev_hash,
            "hash": entry_hash,
            "timestamp": time.time(),
        })

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)


# ============================================================
# LEG 1 — VRF Seed Generation (v0.2: real cryptographic construction)
# ============================================================
# Participant keypairs are generated once per Python session (module-level cache)
# In production, these would be registered on-chain and persistent across rounds.
_PARTICIPANT_CACHE = None


def _setup_participants(n_participants: int = 5) -> List[Dict[str, Any]]:
    """Generate n participants with real Ed25519 keypairs.
    
    For v0.2: keys are fresh each Python process. Production target:
    persistent participants registered via the Bittensor subnet's identity
    system (TBD with Mehdi).
    """
    participants = []
    for i in range(n_participants):
        sk = Ed25519PrivateKey.generate()
        pk = sk.public_key()
        pk_bytes = pk.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        participants.append({
            "index": i,
            "sk": sk,           # Used for proving (stays in memory)
            "pk_bytes": pk_bytes,  # Published for verification
        })
    return participants


def get_participants() -> List[Dict[str, Any]]:
    """Lazy-initialize participants on first call."""
    global _PARTICIPANT_CACHE
    if _PARTICIPANT_CACHE is None:
        _PARTICIPANT_CACHE = _setup_participants()
    return _PARTICIPANT_CACHE


def leg1_vrf_seed(public_input: bytes, audit: AuditLog) -> bytes:
    """
    Real cryptographic VRF (v0.2): each participant signs the public input
    with their Ed25519 secret key, producing a (beta, pi) pair.
    
    For each participant i:
        pi_i  = Ed25519_sign(sk_i, public_input)
        beta_i = SHA3-256(pi_i)
    
    All proofs are verified before aggregation. The architecture's H + SHAKE256
    pattern is implemented:
        seed = H(beta_1 || beta_2 || ... || beta_n)   <- this leg's output
    where H = SHA3-256, producing a 32-byte seed. LEG 3 then expands this
    32-byte seed via SHAKE256 into the bytes needed for circuit generation.
    
    Honest note: this is a "VRF-like" construction (Sign + Hash). Full RFC 9381
    ECVRF has a more sophisticated proof structure with additional uniqueness
    properties. For v0.2, the simpler construction demonstrates the cryptographic
    integration; production target is full ECVRF.
    
    Returns: 32-byte seed (= H(concatenated beta_i values))
    """
    participants = get_participants()
    
    # Each participant produces (beta_i, pi_i)
    contributions = []
    for p in participants:
        pi_i = p["sk"].sign(public_input)  # Ed25519 signature, 64 bytes
        beta_i = hashlib.sha3_256(pi_i).digest()  # 32 bytes
        contributions.append({
            "index": p["index"],
            "pk": p["pk_bytes"],
            "beta": beta_i,
            "pi": pi_i,
        })
    
    # Verify each contribution independently (the verifier's job)
    # This is where real cryptographic verification happens — not a stub.
    verification_results = []
    for c in contributions:
        pk = Ed25519PublicKey.from_public_bytes(c["pk"])
        try:
            pk.verify(c["pi"], public_input)
            sig_valid = True
        except InvalidSignature:
            sig_valid = False
        
        beta_recomputed = hashlib.sha3_256(c["pi"]).digest()
        beta_matches = (beta_recomputed == c["beta"])
        
        verification_results.append({
            "index": c["index"],
            "signature_valid": sig_valid,
            "beta_matches_hash_of_pi": beta_matches,
            "accepted": sig_valid and beta_matches,
        })
    
    # All contributions must verify or we abort
    all_valid = all(v["accepted"] for v in verification_results)
    if not all_valid:
        raise ValueError(
            f"VRF verification failed for one or more participants: "
            f"{verification_results}"
        )
    
    # Aggregate: concatenate beta_i values, then apply H
    concatenated = b"".join(c["beta"] for c in contributions)
    
    # H = SHA3-256: compresses to 32-byte seed (architecture's H step)
    seed = hashlib.sha3_256(concatenated).digest()
    
    audit.append("LEG_1_VRF", {
        "public_input": public_input.hex(),
        "n_participants": len(participants),
        "vrf_construction": "Ed25519-Sign + SHA3-256 (VRF-like, v0.2)",
        "contributions": [
            {
                "index": c["index"],
                "pk": c["pk"].hex(),
                "beta": c["beta"].hex(),
                "pi": c["pi"].hex(),
            }
            for c in contributions
        ],
        "verification_results": verification_results,
        "concatenated_length_bytes": len(concatenated),
        "aggregation": "H(beta_1 || ... || beta_n) where H = SHA3-256",
        "seed_after_H": seed.hex(),
        "seed_length_bytes": len(seed),
    })
    return seed


# ============================================================
# LEG 3 — Deterministic Circuit Specification
# ============================================================
def leg3_generate_circuit(seed: bytes, params: Dict[str, Any],
                          audit: AuditLog) -> cirq.Circuit:
    """
    Expand seed via SHAKE256 into a deterministic random circuit.
    Recipe inspired by Liu et al.: alternating layers of random SU(2)
    single-qubit gates and entangling layers using random perfect matchings
    (the simplest version of their edge-coloured random graph scheme).
    """
    n = params["n_qubits"]
    d = params["depth"]
    
    # Estimate bits needed:
    # - n*(d+1) single-qubit gates * 3 angles * 4 bytes each
    # - d layers * (n-1) Fisher-Yates shuffle bytes per layer
    # Generous overshoot to be safe.
    bits_needed = n * (d + 1) * 12 + d * n * 4 + 2048
    
    # SHAKE256 in XOF mode
    shake = hashlib.shake_256()
    shake.update(seed)
    expanded = shake.digest(bits_needed)
    
    cursor = [0]
    def take_bytes(k: int) -> bytes:
        result = expanded[cursor[0]:cursor[0] + k]
        cursor[0] += k
        return result
    
    def random_angle() -> float:
        b = take_bytes(4)
        return (int.from_bytes(b, "big") / (2**32)) * 2 * np.pi
    
    def random_perfect_matching(qubit_list):
        """Random pairing of all qubits (assumes n even)."""
        indices = list(range(len(qubit_list)))
        # Fisher-Yates shuffle using random bytes from SHAKE stream
        for i in range(len(indices) - 1, 0, -1):
            j = int.from_bytes(take_bytes(4), "big") % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        # Pair consecutive elements after shuffle
        pairs = []
        for i in range(0, len(indices) - 1, 2):
            pairs.append((qubit_list[indices[i]], qubit_list[indices[i + 1]]))
        return pairs
    
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()
    
    for layer in range(d):
        # Single-qubit layer: random SU(2) on each qubit
        for q in qubits:
            theta = random_angle()
            phi = random_angle()
            lam = random_angle()
            # Decompose general SU(2) as Rz(phi) Ry(theta) Rz(lam)
            circuit.append([cirq.rz(lam).on(q), cirq.ry(theta).on(q), cirq.rz(phi).on(q)])
        
        # Entangling layer: random perfect matching (all-to-all)
        # This is the v0 simplification of Liu et al.'s edge-coloured random
        # graph scheme. Each layer gets a fresh random pairing, giving
        # effective all-to-all connectivity over multiple layers.
        pairs = random_perfect_matching(list(qubits))
        for q1, q2 in pairs:
            circuit.append(cirq.CZ(q1, q2))
    
    # Final single-qubit layer
    for q in qubits:
        theta = random_angle()
        phi = random_angle()
        lam = random_angle()
        circuit.append([cirq.rz(lam).on(q), cirq.ry(theta).on(q), cirq.rz(phi).on(q)])
    
    # Measurements
    circuit.append(cirq.measure(*qubits, key="m"))
    
    audit.append("LEG_3_CIRCUIT", {
        "n_qubits": n,
        "depth": d,
        "connectivity": "all-to-all (random perfect matching per layer)",
        "seed_hex": seed.hex()[:64] + "...",  # truncated for log readability
        "bits_consumed": cursor[0],
        "circuit_moment_count": len(circuit),
        "circuit_gate_count": sum(1 for _ in circuit.all_operations()),
    })
    
    return circuit


# ============================================================
# LEG 4 — Quantum Sampling (simulator stand-in)
# ============================================================
def leg4_sample(circuit: cirq.Circuit, params: Dict[str, Any],
                audit: AuditLog) -> np.ndarray:
    """
    Run the circuit on a noisy simulator. Returns array of shape
    (shots, n_qubits) with measurement outcomes.
    """
    # Add depolarizing noise to mimic real hardware
    noise_model = cirq.ConstantQubitNoiseModel(
        cirq.depolarize(p=params["noise_rate"])
    )
    
    noisy_simulator = cirq.DensityMatrixSimulator(noise=noise_model)
    
    start = time.time()
    result = noisy_simulator.run(circuit, repetitions=params["shots_per_circuit"])
    elapsed = time.time() - start
    
    samples = result.measurements["m"]  # shape (shots, n_qubits)
    
    audit.append("LEG_4_SAMPLING", {
        "shots": int(samples.shape[0]),
        "n_qubits": int(samples.shape[1]),
        "wall_time_seconds": elapsed,
        "first_5_samples": samples[:5].tolist(),
    })
    
    return samples


# ============================================================
# LEG 5 — XEB Verification
# ============================================================
def leg5_xeb_score(circuit: cirq.Circuit, samples: np.ndarray,
                   params: Dict[str, Any], audit: AuditLog) -> Tuple[float, bool]:
    """
    Compute F_XEB by full state-vector simulation of the (noiseless) circuit.
    F_XEB = (2^n / m) * sum_i p(x_i) - 1
    where p(x_i) is the ideal probability of measuring bitstring x_i.
    
    v0.1: two-sided check. F_XEB must satisfy chi_low ≤ F_XEB ≤ chi_high.
        chi_low rejects circuit-blind attacks (uniform random, etc.) where
            samples are uncorrelated with the ideal distribution.
        chi_high rejects heavy-output attacks where samples are CHERRY-PICKED
            from high-probability bitstrings, inflating F_XEB above what
            honest quantum sampling can produce.
    
    Returns: (F_XEB, accepted)
    """
    n = params["n_qubits"]
    chi_low = params["chi_low"]
    chi_high = params["chi_high"]
    
    # Strip measurements for state-vector simulation
    circuit_no_measure = cirq.Circuit(
        [op for op in circuit.all_operations() 
         if not isinstance(op.gate, cirq.MeasurementGate)]
    )
    
    # Full state-vector simulation
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit_no_measure)
    state_vector = result.final_state_vector  # shape (2^n,)
    
    # P(x) for each bitstring
    probs = np.abs(state_vector) ** 2
    
    # For each sample, look up p(x) and accumulate
    sample_indices = np.array([
        int("".join(str(b) for b in sample), 2)
        for sample in samples
    ])
    p_observed = probs[sample_indices]
    
    f_xeb = (2 ** n) * np.mean(p_observed) - 1.0
    
    # Two-sided acceptance check
    lower_check = f_xeb >= chi_low
    upper_check = f_xeb <= chi_high
    accepted = lower_check and upper_check
    
    # Classify rejection reason if rejected
    if not accepted:
        if not lower_check:
            rejection_reason = "F_XEB below chi_low (circuit-blind attack suspected)"
        elif not upper_check:
            rejection_reason = "F_XEB above chi_high (heavy-output attack suspected)"
        else:
            rejection_reason = None
    else:
        rejection_reason = None
    
    audit.append("LEG_5_XEB", {
        "f_xeb": float(f_xeb),
        "chi_low": chi_low,
        "chi_high": chi_high,
        "lower_check_passed": bool(lower_check),
        "upper_check_passed": bool(upper_check),
        "accepted": bool(accepted),
        "rejection_reason": rejection_reason,
        "mean_p_observed": float(np.mean(p_observed)),
        "n_samples_scored": int(len(samples)),
    })
    
    return f_xeb, accepted


# ============================================================
# Helper — fetch a drand pulse for the extractor seed
# ============================================================
def fetch_drand_pulse() -> bytes:
    """
    Fetch a recent pulse from drand (League of Entropy).
    Returns the 512-bit pulse as bytes.
    Falls back to local randomness if drand is unreachable.
    """
    try:
        # drand mainnet API
        r = requests.get(
            "https://api.drand.sh/8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce/public/latest",
            timeout=5
        )
        r.raise_for_status()
        pulse = r.json()
        randomness_hex = pulse["randomness"]
        return bytes.fromhex(randomness_hex)
    except Exception as e:
        print(f"[!] drand fetch failed ({e}); using local fallback")
        return os.urandom(32)


# ============================================================
# LEG 6 — Randomness Extraction (Toeplitz)
# ============================================================
def leg6_toeplitz_extract(samples: np.ndarray, drand_seed: bytes,
                          output_bits: int, audit: AuditLog) -> bytes:
    """
    Apply a Toeplitz extractor: output = T * input (mod 2)
    where T is an output_bits x input_bits Toeplitz matrix
    constructed from the drand seed.
    """
    # Flatten samples to a single bit vector
    input_bits_array = samples.flatten().astype(np.uint8)
    input_length = len(input_bits_array)
    
    # Expand drand seed to enough bytes for the Toeplitz matrix's
    # defining vector (length = output_bits + input_length - 1)
    needed_seed_bytes = (output_bits + input_length - 1 + 7) // 8
    shake = hashlib.shake_256()
    shake.update(drand_seed)
    seed_bytes = shake.digest(needed_seed_bytes)
    
    # Convert seed bytes to a bit array
    seed_bits = np.unpackbits(np.frombuffer(seed_bytes, dtype=np.uint8))
    toeplitz_defining = seed_bits[:output_bits + input_length - 1]
    
    # Build Toeplitz matrix T (output_bits x input_length)
    # T[i, j] = toeplitz_defining[i - j + (input_length - 1)]
    T = np.zeros((output_bits, input_length), dtype=np.uint8)
    for i in range(output_bits):
        for j in range(input_length):
            T[i, j] = toeplitz_defining[i - j + (input_length - 1)]
    
    # Matrix-vector multiplication mod 2
    output_bits_array = (T @ input_bits_array) % 2
    
    # Pack into bytes
    pad = (-len(output_bits_array)) % 8
    if pad:
        output_bits_array = np.concatenate([output_bits_array, np.zeros(pad, dtype=np.uint8)])
    output_bytes = np.packbits(output_bits_array).tobytes()
    
    audit.append("LEG_6_EXTRACTION", {
        "input_bits": int(input_length),
        "output_bits": int(output_bits),
        "drand_seed_hex": drand_seed.hex(),
        "extractor": "Toeplitz",
        "output_hex": output_bytes.hex(),
    })
    
    return output_bytes


# ============================================================
# Main pipeline orchestration
# ============================================================
def run_pipeline(round_number: int = 1, output_path: str = "audit_log.json"):
    print("=" * 60)
    print(f"RCS + VRF Subnet — v0.2 Pipeline — Round {round_number}")
    print("=" * 60)
    
    audit = AuditLog()
    audit.append("PROTOCOL_PARAMS", PROTOCOL_PARAMS)
    
    # LEG 1: VRF seed (v0.2: real Ed25519-based VRF)
    print("\n[LEG 1] Generating VRF seed via real cryptographic VRF...")
    print(f"        5 participants with Ed25519 keypairs, signing public input")
    public_input = f"round_{round_number}_block_height_1234".encode()
    seed = leg1_vrf_seed(public_input, audit)
    print(f"        All 5 proofs verified independently")
    print(f"        Aggregated via H = SHA3-256, then expansion in LEG 3 via SHAKE256")
    print(f"        Seed (32 bytes after H): {seed.hex()[:32]}...")
    
    # LEG 2 already happened (parameters precommitted in PROTOCOL_PARAMS)
    print(f"\n[LEG 2] Parameters precommitted: n={PROTOCOL_PARAMS['n_qubits']}, "
          f"d={PROTOCOL_PARAMS['depth']}, "
          f"χ_low={PROTOCOL_PARAMS['chi_low']}, χ_high={PROTOCOL_PARAMS['chi_high']}")
    
    # LEG 3: Generate circuit
    print(f"\n[LEG 3] Generating circuit from seed via SHAKE256...")
    circuit = leg3_generate_circuit(seed, PROTOCOL_PARAMS, audit)
    print(f"        Circuit: {len(list(circuit.all_operations()))} operations, "
          f"{len(circuit)} moments")
    
    # LEG 4: Sample on noisy simulator
    print(f"\n[LEG 4] Running circuit on noisy simulator "
          f"({PROTOCOL_PARAMS['shots_per_circuit']} shots)...")
    samples = leg4_sample(circuit, PROTOCOL_PARAMS, audit)
    print(f"        Samples shape: {samples.shape}")
    
    # LEG 5: XEB verification (v0.1: two-sided check)
    print(f"\n[LEG 5] Computing F_XEB...")
    f_xeb, accepted = leg5_xeb_score(circuit, samples, PROTOCOL_PARAMS, audit)
    chi_low = PROTOCOL_PARAMS['chi_low']
    chi_high = PROTOCOL_PARAMS['chi_high']
    print(f"        F_XEB = {f_xeb:.4f}, acceptance window = [{chi_low}, {chi_high}]")
    if accepted:
        print(f"        ACCEPTED ✓")
    else:
        # Find rejection reason from audit log
        leg5_entry = audit.entries[-1]
        reason = leg5_entry["data"].get("rejection_reason", "unknown")
        print(f"        REJECTED ✗  ({reason})")
    
    if not accepted:
        print("\n[!] Round rejected at LEG 5. Pipeline terminating.")
        audit.save(output_path)
        return None
    
    # LEG 6: Extract randomness
    print(f"\n[LEG 6] Fetching drand pulse for extractor seed...")
    drand_seed = fetch_drand_pulse()
    print(f"        drand seed: {drand_seed.hex()[:32]}...")
    print(f"        Running Toeplitz extractor...")
    output = leg6_toeplitz_extract(samples, drand_seed,
                                    PROTOCOL_PARAMS["extractor_output_bits"],
                                    audit)
    print(f"        Output ({len(output) * 8} bits): {output.hex()}")
    
    # LEG 7: Audit chain already being built throughout
    print(f"\n[LEG 7] Audit log entries: {len(audit.entries)}")
    
    audit.save(output_path)
    print(f"\n✓ Pipeline complete. Audit log saved to {output_path}")
    print(f"✓ Certified-random output: {output.hex()}")
    
    return output


if __name__ == "__main__":
    run_pipeline(round_number=1, output_path="audit_log.json")
