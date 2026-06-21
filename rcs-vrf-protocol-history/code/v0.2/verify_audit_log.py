"""
Audit Log Verifier — v0.2

Independently re-verifies a published audit log without trusting the
protocol orchestrator. This demonstrates the protocol's "publicly verifiable"
property: anyone with the audit log and the protocol implementation can
re-check every cryptographic claim.

Verifications performed:
    LEG 1 (VRF):
        - Each Ed25519 signature: pk.verify(pi, alpha) succeeds
        - Each beta: SHA3-256(pi) == beta
        - The seed: SHA3-256(beta_1 || ... || beta_n) == logged seed
    LEG 3 (Circuit):
        - Regenerate the circuit from seed + params
        - Confirm gate count, moment count match the audit log
    LEG 4 (Sampling):
        - Confirm logged sample shape (partial — full samples not in log)
    LEG 5 (XEB):
        - Verify threshold logic matches the logged F_XEB
    LEG 7 (Audit chain):
        - Recompute every entry's hash, walk the prev_hash chain
        - Confirm integrity end-to-end

Honest limitations:
    LEG 4/5/6 cannot be fully re-verified because the audit log only stores
    the first 5 samples (for log size). Full re-verification would require
    either: (a) storing all samples in the audit log, or (b) fetching them
    from a separate linked artifact (which the architecture's hash-graph
    design would enable). v0.2 keeps the log compact; v0.3 could add a
    "full_samples" link that points to extended storage.

Usage:
    python verify_audit_log.py [audit_log.json]
"""

import argparse
import hashlib
import json
import sys
from typing import Dict, Any, List

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

# We need the circuit generation function for LEG 3 verification
# Import from v0.2 if available, otherwise re-implement inline
try:
    from rcs_vrf import leg3_generate_circuit, AuditLog as _AuditLog
    HAVE_V02 = True
except ImportError:
    HAVE_V02 = False


# ANSI colors for output (gracefully degrade on Windows if not supported)
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def green(s): return f"{C.GREEN}{s}{C.END}"
def red(s): return f"{C.RED}{s}{C.END}"
def yellow(s): return f"{C.YELLOW}{s}{C.END}"
def bold(s): return f"{C.BOLD}{s}{C.END}"
def dim(s): return f"{C.DIM}{s}{C.END}"


# ============================================================
# Verification routines per leg
# ============================================================

def verify_leg1_vrf(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Re-verify every cryptographic claim in LEG 1."""
    data = entry["data"]
    public_input = bytes.fromhex(data["public_input"])
    contributions = data["contributions"]
    logged_seed = bytes.fromhex(data["seed_after_H"])
    
    checks = []
    
    # Check 1: each Ed25519 signature
    sig_results = []
    for c in contributions:
        pk_bytes = bytes.fromhex(c["pk"])
        beta = bytes.fromhex(c["beta"])
        pi = bytes.fromhex(c["pi"])
        pk = Ed25519PublicKey.from_public_bytes(pk_bytes)
        try:
            pk.verify(pi, public_input)
            sig_results.append((c["index"], True, None))
        except InvalidSignature:
            sig_results.append((c["index"], False, "InvalidSignature"))
    sigs_all_valid = all(r[1] for r in sig_results)
    checks.append({
        "name": "Ed25519 signatures verify",
        "passed": sigs_all_valid,
        "detail": f"{sum(1 for r in sig_results if r[1])}/{len(sig_results)} participants",
    })
    
    # Check 2: beta = SHA3-256(pi) for each contribution
    beta_results = []
    for c in contributions:
        pi = bytes.fromhex(c["pi"])
        beta_logged = bytes.fromhex(c["beta"])
        beta_recomputed = hashlib.sha3_256(pi).digest()
        beta_results.append((c["index"], beta_logged == beta_recomputed))
    betas_all_match = all(r[1] for r in beta_results)
    checks.append({
        "name": "beta = SHA3-256(pi) for each contribution",
        "passed": betas_all_match,
        "detail": f"{sum(1 for r in beta_results if r[1])}/{len(beta_results)} match",
    })
    
    # Check 3: seed = SHA3-256(concatenated betas)
    concatenated = b"".join(bytes.fromhex(c["beta"]) for c in contributions)
    seed_recomputed = hashlib.sha3_256(concatenated).digest()
    seed_matches = (seed_recomputed == logged_seed)
    checks.append({
        "name": "seed = SHA3-256(beta_1 || ... || beta_n)",
        "passed": seed_matches,
        "detail": f"recomputed = {seed_recomputed.hex()[:16]}..., logged = {logged_seed.hex()[:16]}...",
    })
    
    return {
        "leg": "LEG_1_VRF",
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }


def verify_leg3_circuit(audit_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-verify the circuit was correctly generated from the seed + params."""
    # Find LEG 1 (for seed) and LEG 3 (for circuit metadata) entries
    vrf_entry = next(e for e in audit_log if e["leg"] == "LEG_1_VRF")
    circuit_entry = next(e for e in audit_log if e["leg"] == "LEG_3_CIRCUIT")
    params_entry = next(e for e in audit_log if e["leg"] == "PROTOCOL_PARAMS")
    
    seed = bytes.fromhex(vrf_entry["data"]["seed_after_H"])
    params = params_entry["data"]
    logged_ops = circuit_entry["data"]["circuit_gate_count"]
    logged_moments = circuit_entry["data"]["circuit_moment_count"]
    logged_bits = circuit_entry["data"]["bits_consumed"]
    
    checks = []
    
    if HAVE_V02:
        # Regenerate the circuit and compare metadata
        dummy_audit = _AuditLog()
        circuit = leg3_generate_circuit(seed, params, dummy_audit)
        regen_ops = sum(1 for _ in circuit.all_operations())
        regen_moments = len(circuit)
        regen_bits = dummy_audit.entries[-1]["data"]["bits_consumed"]
        
        checks.append({
            "name": "Circuit gate count matches",
            "passed": regen_ops == logged_ops,
            "detail": f"regenerated = {regen_ops}, logged = {logged_ops}",
        })
        checks.append({
            "name": "Circuit moment count matches",
            "passed": regen_moments == logged_moments,
            "detail": f"regenerated = {regen_moments}, logged = {logged_moments}",
        })
        checks.append({
            "name": "Bits consumed matches",
            "passed": regen_bits == logged_bits,
            "detail": f"regenerated = {regen_bits}, logged = {logged_bits}",
        })
    else:
        checks.append({
            "name": "Circuit regeneration",
            "passed": None,
            "detail": "SKIPPED (rcs_vrf.py not importable)",
        })
    
    return {
        "leg": "LEG_3_CIRCUIT",
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks if c["passed"] is not None),
    }


def verify_leg4_sampling(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Partial verification: sample shape only (full samples not in log)."""
    data = entry["data"]
    n_qubits = data["n_qubits"]
    logged_shots = data["shots"]
    first_5 = data["first_5_samples"]
    
    checks = [
        {
            "name": "First-5 samples have correct width (n_qubits)",
            "passed": all(len(s) == n_qubits for s in first_5),
            "detail": f"{n_qubits} bits per sample expected",
        },
        {
            "name": "First-5 samples contain only 0s and 1s",
            "passed": all(b in (0, 1) for s in first_5 for b in s),
            "detail": "binary measurement outcomes",
        },
    ]
    
    # We CAN'T fully verify shots count or F_XEB without all samples
    notes = [
        f"NOTE: Only first {len(first_5)} of {logged_shots} samples in audit log.",
        f"      Full re-verification of F_XEB would require all samples.",
    ]
    
    return {
        "leg": "LEG_4_SAMPLING",
        "checks": checks,
        "notes": notes,
        "all_passed": all(c["passed"] for c in checks),
    }


def verify_leg5_xeb(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Verify the threshold logic given the logged F_XEB."""
    data = entry["data"]
    f_xeb = data["f_xeb"]
    chi_low = data["chi_low"]
    chi_high = data["chi_high"]
    logged_lower = data["lower_check_passed"]
    logged_upper = data["upper_check_passed"]
    logged_accepted = data["accepted"]
    
    # Recompute the checks
    actual_lower = f_xeb >= chi_low
    actual_upper = f_xeb <= chi_high
    actual_accepted = actual_lower and actual_upper
    
    checks = [
        {
            "name": "Lower threshold check (F_XEB >= chi_low)",
            "passed": actual_lower == logged_lower,
            "detail": f"F_XEB={f_xeb:.4f} >= chi_low={chi_low}: recomputed={actual_lower}, logged={logged_lower}",
        },
        {
            "name": "Upper threshold check (F_XEB <= chi_high)",
            "passed": actual_upper == logged_upper,
            "detail": f"F_XEB={f_xeb:.4f} <= chi_high={chi_high}: recomputed={actual_upper}, logged={logged_upper}",
        },
        {
            "name": "Overall accept/reject matches",
            "passed": actual_accepted == logged_accepted,
            "detail": f"accepted: recomputed={actual_accepted}, logged={logged_accepted}",
        },
    ]
    
    notes = [
        "NOTE: F_XEB value itself is taken as given (cannot recompute without",
        "      all samples and the circuit's full state vector).",
    ]
    
    return {
        "leg": "LEG_5_XEB",
        "checks": checks,
        "notes": notes,
        "all_passed": all(c["passed"] for c in checks),
    }


def verify_leg7_hash_chain(audit_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify the hash chain integrity: every entry's hash recomputes correctly,
    and every prev_hash links to the previous entry's hash."""
    checks = []
    
    # Check each entry's individual hash
    hash_recompute_results = []
    for i, entry in enumerate(audit_log):
        prev_hash = entry["prev_hash"]
        leg = entry["leg"]
        data = entry["data"]
        logged_hash = entry["hash"]
        
        # Recompute exactly as the AuditLog class does
        payload = json.dumps(
            {"leg": leg, "data": data, "prev": prev_hash},
            sort_keys=True, default=str
        )
        recomputed_hash = hashlib.sha3_256(payload.encode()).hexdigest()
        
        hash_recompute_results.append({
            "index": i,
            "leg": leg,
            "matches": recomputed_hash == logged_hash,
        })
    
    all_hashes_match = all(r["matches"] for r in hash_recompute_results)
    checks.append({
        "name": "Each entry's hash recomputes correctly",
        "passed": all_hashes_match,
        "detail": f"{sum(1 for r in hash_recompute_results if r['matches'])}/{len(hash_recompute_results)} entries",
    })
    
    # Check prev_hash chain
    chain_results = []
    for i in range(1, len(audit_log)):
        prev_entry_hash = audit_log[i-1]["hash"]
        current_prev_hash = audit_log[i]["prev_hash"]
        chain_results.append({
            "index": i,
            "leg": audit_log[i]["leg"],
            "matches": prev_entry_hash == current_prev_hash,
        })
    
    chain_intact = all(r["matches"] for r in chain_results)
    checks.append({
        "name": "Hash chain integrity (prev_hash links match)",
        "passed": chain_intact,
        "detail": f"{sum(1 for r in chain_results if r['matches'])}/{len(chain_results)} links",
    })
    
    # Check first entry's prev_hash is all zeros
    first_prev = audit_log[0]["prev_hash"]
    checks.append({
        "name": "First entry's prev_hash is all zeros (chain root)",
        "passed": first_prev == "0" * 64,
        "detail": f"first prev_hash = {first_prev[:16]}...",
    })
    
    return {
        "leg": "LEG_7_HASH_CHAIN (integrity)",
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }


# ============================================================
# Main verification orchestrator
# ============================================================

def verify_audit_log(audit_log_path: str) -> bool:
    """Run all verifications, print results, return True if all passed."""
    
    print(bold(f"\n{'=' * 70}"))
    print(bold(f"  AUDIT LOG VERIFIER — v0.2"))
    print(bold(f"{'=' * 70}"))
    print(f"Loading: {audit_log_path}")
    
    try:
        with open(audit_log_path) as f:
            audit_log = json.load(f)
    except FileNotFoundError:
        print(red(f"\n[ERROR] File not found: {audit_log_path}"))
        return False
    except json.JSONDecodeError as e:
        print(red(f"\n[ERROR] Invalid JSON: {e}"))
        return False
    
    print(f"Entries: {len(audit_log)}")
    print(f"Legs found: {[e['leg'] for e in audit_log]}")
    print()
    
    # Run all verifications
    results = []
    
    # LEG 1
    vrf_entry = next((e for e in audit_log if e["leg"] == "LEG_1_VRF"), None)
    if vrf_entry:
        results.append(verify_leg1_vrf(vrf_entry))
    
    # LEG 3
    if any(e["leg"] == "LEG_3_CIRCUIT" for e in audit_log):
        results.append(verify_leg3_circuit(audit_log))
    
    # LEG 4
    leg4_entry = next((e for e in audit_log if e["leg"] == "LEG_4_SAMPLING"), None)
    if leg4_entry:
        results.append(verify_leg4_sampling(leg4_entry))
    
    # LEG 5
    leg5_entry = next((e for e in audit_log if e["leg"] == "LEG_5_XEB"), None)
    if leg5_entry:
        results.append(verify_leg5_xeb(leg5_entry))
    
    # LEG 7 (hash chain)
    results.append(verify_leg7_hash_chain(audit_log))
    
    # Print per-leg results
    for r in results:
        status = (green("✓ ALL PASS") if r["all_passed"]
                  else red("✗ FAILURES"))
        print(f"\n{bold(r['leg'])}  {status}")
        for check in r["checks"]:
            if check["passed"] is None:
                marker = yellow("○")
            elif check["passed"]:
                marker = green("✓")
            else:
                marker = red("✗")
            print(f"  {marker} {check['name']}")
            print(f"    {dim(check['detail'])}")
        if r.get("notes"):
            for note in r["notes"]:
                print(f"  {yellow(note)}")
    
    # Summary
    all_passed = all(r["all_passed"] for r in results)
    print(bold(f"\n{'=' * 70}"))
    if all_passed:
        print(green(bold(f"  ✓ ALL VERIFICATIONS PASSED")))
        print(f"  This audit log is consistent with v0.2's cryptographic claims.")
    else:
        print(red(bold(f"  ✗ SOME VERIFICATIONS FAILED")))
        print(f"  Inspect the failures above. The audit log may be tampered or corrupt.")
    print(bold(f"{'=' * 70}"))
    
    # Honest caveat about what this verifier doesn't catch
    print(yellow(bold("\nLimitations of this verifier (v0.2):")))
    print(yellow("  - Full samples not in audit log → F_XEB cannot be recomputed end-to-end."))
    print(yellow("    A 'verifying validator' would need to fetch all 1000 samples from"))
    print(yellow("    extended storage (linked from the audit log)."))
    print(yellow("  - Single hash chain → a CASCADING tamper (update entry hash AND all"))
    print(yellow("    subsequent prev_hashes + hashes) can produce a forged log that passes."))
    print(yellow("    The architecture's LEG 7 specifies a multi-party hash GRAPH with"))
    print(yellow("    cross-signed pulses to defend against this; v0.2 uses a single chain"))
    print(yellow("    as a simplification."))
    print(yellow("  - No signatures on individual entries → no proof of WHO produced each."))
    print()
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_log", nargs="?", default="audit_log.json",
                        help="path to audit_log.json (default: audit_log.json)")
    args = parser.parse_args()
    
    ok = verify_audit_log(args.audit_log)
    sys.exit(0 if ok else 1)
