"""
Test suite for RCS + VRF Subnet v0.3a and the audit log verifier.

This is the proto-conformance-suite for the eventual spec. Every test
encodes a specific cryptographic or protocol property that v0.3a must
satisfy. Each test is small (< 20 lines) and deterministic where possible.

Run with:
    pytest test_v0_3a.py
    pytest test_v0_3a.py -v          # verbose
    pytest test_v0_3a.py -k vrf      # only VRF tests
    pytest test_v0_3a.py --tb=short  # shorter tracebacks on failure

Test organization mirrors the leg structure of the protocol:
    TestVRFCryptography      → LEG 1
    TestCircuitGeneration    → LEG 3
    TestXEBThresholds        → LEG 5 (acceptance window)
    TestMinEntropyEstimator  → LEG 5 (v0.3a new)
    TestLHLBound             → LEG 6 (v0.3a new)
    TestExtractor            → LEG 6
    TestAuditChain           → LEG 7
    TestVerifier             → audit log verification
    TestEndToEnd             → full pipeline integration

v0.2 → v0.3a test additions:
    - TestMinEntropyEstimator: estimator properties (monotonic in F_XEB,
      proportional to m, scales with n, zero for F_XEB ≤ 0)
    - TestLHLBound: LHL formula correctness; output_bits exceeding bound
      raises ValueError; relationship to epsilon
    - LEG_5 audit entry now contains min-entropy fields
    - LEG_6 audit entry now records LHL bound enforcement
"""

import hashlib
import json
import math
import os
import tempfile
import pytest
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

import rcs_vrf as v03a
from rcs_vrf import (
    leg1_vrf_seed, leg3_generate_circuit, leg6_toeplitz_extract,
    _setup_participants, AuditLog, PROTOCOL_PARAMS, run_pipeline,
    estimate_min_entropy, max_extractable_bits,
)
from verify_audit_log import verify_audit_log


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def fresh_audit():
    """Provide a fresh AuditLog for each test."""
    return AuditLog()


@pytest.fixture
def reset_participants():
    """Reset the module-level participant cache between tests."""
    v03a._PARTICIPANT_CACHE = None
    yield
    v03a._PARTICIPANT_CACHE = None


@pytest.fixture
def sample_seed_and_audit(reset_participants, fresh_audit):
    """Run LEG 1 and return (seed, audit). Cached per test."""
    public_input = b"test_round_public_input"
    seed = leg1_vrf_seed(public_input, fresh_audit)
    return seed, fresh_audit


# ============================================================
# LEG 1 — VRF cryptography
# ============================================================

class TestVRFCryptography:
    """Real Ed25519-based VRF (v0.2 construction)."""

    def test_ed25519_signature_verifies(self):
        """A correctly-signed message verifies with the right public key."""
        sk = Ed25519PrivateKey.generate()
        pk = sk.public_key()
        message = b"hello world"
        signature = sk.sign(message)
        pk.verify(signature, message)  # raises InvalidSignature if bad

    def test_ed25519_tampered_input_rejected(self):
        """Signature over original message must NOT verify against a different message."""
        sk = Ed25519PrivateKey.generate()
        pk = sk.public_key()
        signature = sk.sign(b"original message")
        with pytest.raises(InvalidSignature):
            pk.verify(signature, b"different message")

    def test_ed25519_is_deterministic(self):
        """Ed25519 (per RFC 8032) is deterministic: same key + message → same signature."""
        sk = Ed25519PrivateKey.generate()
        message = b"deterministic input"
        sig1 = sk.sign(message)
        sig2 = sk.sign(message)
        assert sig1 == sig2

    def test_beta_equals_sha3_of_pi(self, sample_seed_and_audit):
        """For each contribution, beta must equal SHA3-256(pi)."""
        _, audit = sample_seed_and_audit
        vrf_entry = next(e for e in audit.entries if e["leg"] == "LEG_1_VRF")
        for c in vrf_entry["data"]["contributions"]:
            pi = bytes.fromhex(c["pi"])
            beta = bytes.fromhex(c["beta"])
            assert beta == hashlib.sha3_256(pi).digest()

    def test_seed_equals_sha3_of_concatenated_betas(self, sample_seed_and_audit):
        """The architecture's H step: seed = SHA3-256(beta_1 || ... || beta_n)."""
        seed, audit = sample_seed_and_audit
        vrf_entry = next(e for e in audit.entries if e["leg"] == "LEG_1_VRF")
        concatenated = b"".join(
            bytes.fromhex(c["beta"]) for c in vrf_entry["data"]["contributions"]
        )
        assert seed == hashlib.sha3_256(concatenated).digest()

    def test_seed_is_32_bytes(self, sample_seed_and_audit):
        """H = SHA3-256 produces a 32-byte (256-bit) seed."""
        seed, _ = sample_seed_and_audit
        assert len(seed) == 32

    def test_vrf_deterministic_same_keys_same_input(self, reset_participants, fresh_audit):
        """Same participant keys + same public input → same seed.
        
        (Resets cache, generates keys, runs twice with same input → same beta_i.)
        """
        audit1 = AuditLog()
        seed1 = leg1_vrf_seed(b"same input", audit1)
        # Don't reset — use same cached participants
        audit2 = AuditLog()
        seed2 = leg1_vrf_seed(b"same input", audit2)
        assert seed1 == seed2

    def test_vrf_different_inputs_different_seeds(self, reset_participants, fresh_audit):
        """Same participants + different public inputs → different seeds."""
        audit1 = AuditLog()
        seed1 = leg1_vrf_seed(b"input alpha", audit1)
        audit2 = AuditLog()
        seed2 = leg1_vrf_seed(b"input beta", audit2)
        assert seed1 != seed2

    def test_five_participants_generated(self, reset_participants):
        """Default v0.2 setup has 5 simulated participants."""
        participants = _setup_participants()
        assert len(participants) == 5

    def test_participant_keys_unique(self, reset_participants):
        """Each participant must have a distinct public key (no collisions)."""
        participants = _setup_participants()
        pks = [p["pk_bytes"] for p in participants]
        assert len(set(pks)) == len(pks)


# ============================================================
# LEG 3 — Circuit Generation
# ============================================================

class TestCircuitGeneration:
    """Deterministic circuit from seed via SHAKE256."""

    def test_same_seed_produces_same_circuit(self, fresh_audit):
        """Reproducibility: same seed + params → bit-identical circuit."""
        seed = b"X" * 32
        c1 = leg3_generate_circuit(seed, PROTOCOL_PARAMS, AuditLog())
        c2 = leg3_generate_circuit(seed, PROTOCOL_PARAMS, AuditLog())
        # Compare gate-by-gate string representations
        assert str(c1) == str(c2)

    def test_different_seeds_produce_different_circuits(self, fresh_audit):
        """Sensitivity: different seeds → different circuits."""
        c1 = leg3_generate_circuit(b"A" * 32, PROTOCOL_PARAMS, AuditLog())
        c2 = leg3_generate_circuit(b"B" * 32, PROTOCOL_PARAMS, AuditLog())
        assert str(c1) != str(c2)

    def test_circuit_gate_count_at_n8_d10(self, fresh_audit):
        """Specific expected count at default params (regression guard)."""
        c = leg3_generate_circuit(b"X" * 32, PROTOCOL_PARAMS, AuditLog())
        gate_count = sum(1 for _ in c.all_operations())
        # 8 qubits, depth 10:
        # - 11 single-qubit sublayers × 8 qubits × 3 gates = 264
        # - 10 entangling sublayers × 4 CZ each = 40
        # - 1 measurement gate
        # Total = 305
        assert gate_count == 305

    def test_circuit_moment_count_at_n8_d10(self, fresh_audit):
        """Specific moment count at default params."""
        c = leg3_generate_circuit(b"X" * 32, PROTOCOL_PARAMS, AuditLog())
        assert len(c) == 44

    def test_seed_size_independence(self, fresh_audit):
        """LEG 3 should accept seed bytes of any length (SHAKE256 absorbs arbitrary input)."""
        # 32-byte seed (current v0.2 convention)
        c1 = leg3_generate_circuit(b"x" * 32, PROTOCOL_PARAMS, AuditLog())
        # 16-byte seed (smaller)
        c2 = leg3_generate_circuit(b"x" * 16, PROTOCOL_PARAMS, AuditLog())
        # Both should produce valid circuits, different from each other
        assert sum(1 for _ in c1.all_operations()) == sum(1 for _ in c2.all_operations())
        assert str(c1) != str(c2)


# ============================================================
# LEG 5 — XEB threshold logic
# ============================================================

class TestXEBThresholds:
    """Two-sided F_XEB acceptance window (v0.1+ behavior)."""

    def _audit_entry_for_fxeb(self, f_xeb_value, params=None):
        """Helper: simulate the threshold logic for a given F_XEB."""
        p = dict(params or PROTOCOL_PARAMS)
        chi_low = p["chi_low"]
        chi_high = p["chi_high"]
        lower = f_xeb_value >= chi_low
        upper = f_xeb_value <= chi_high
        return lower and upper, lower, upper

    def test_below_chi_low_rejected(self):
        """F_XEB below the lower bound → rejected (circuit-blind attack)."""
        accepted, lower, upper = self._audit_entry_for_fxeb(0.05)
        assert not accepted
        assert not lower
        assert upper

    def test_above_chi_high_rejected(self):
        """F_XEB above the upper bound → rejected (heavy-output attack)."""
        accepted, lower, upper = self._audit_entry_for_fxeb(3.5)
        assert not accepted
        assert lower
        assert not upper

    def test_in_window_accepted(self):
        """F_XEB inside [chi_low, chi_high] → accepted (honest miner)."""
        accepted, lower, upper = self._audit_entry_for_fxeb(0.3)
        assert accepted
        assert lower
        assert upper

    def test_exactly_at_chi_low_accepted(self):
        """Boundary: F_XEB == chi_low should be accepted (>= comparison)."""
        accepted, _, _ = self._audit_entry_for_fxeb(PROTOCOL_PARAMS["chi_low"])
        assert accepted

    def test_exactly_at_chi_high_accepted(self):
        """Boundary: F_XEB == chi_high should be accepted (<= comparison)."""
        accepted, _, _ = self._audit_entry_for_fxeb(PROTOCOL_PARAMS["chi_high"])
        assert accepted

    def test_chi_low_below_chi_high(self):
        """Sanity: chi_low must be strictly less than chi_high in any sane params."""
        assert PROTOCOL_PARAMS["chi_low"] < PROTOCOL_PARAMS["chi_high"]


# ============================================================
# LEG 5 — Min-entropy estimator (v0.3a)
# ============================================================

class TestMinEntropyEstimator:
    """Properties of estimate_min_entropy(f_xeb, n, m)."""

    def test_zero_at_fxeb_zero(self):
        """No fidelity → no entropy claim (conservative)."""
        assert estimate_min_entropy(0.0, 8, 1000) == 0.0

    def test_zero_at_negative_fxeb(self):
        """Defensive: negative F_XEB is degenerate; estimator returns 0."""
        assert estimate_min_entropy(-0.5, 8, 1000) == 0.0

    def test_positive_at_positive_fxeb(self):
        """Positive F_XEB → positive min-entropy estimate."""
        assert estimate_min_entropy(0.3, 8, 1000) > 0

    def test_monotonic_in_fxeb(self):
        """Higher F_XEB → more estimated min-entropy (with n and m fixed)."""
        h_low = estimate_min_entropy(0.1, 8, 1000)
        h_mid = estimate_min_entropy(0.3, 8, 1000)
        h_high = estimate_min_entropy(0.9, 8, 1000)
        assert h_low < h_mid < h_high

    def test_proportional_to_m(self):
        """Doubling samples should double the estimate (independent samples)."""
        h_1k = estimate_min_entropy(0.3, 8, 1000)
        h_2k = estimate_min_entropy(0.3, 8, 2000)
        assert math.isclose(h_2k, 2 * h_1k, rel_tol=1e-9)

    def test_monotonic_in_n(self):
        """More qubits → more entropy per sample → more total entropy."""
        h_n4 = estimate_min_entropy(0.3, 4, 1000)
        h_n8 = estimate_min_entropy(0.3, 8, 1000)
        h_n16 = estimate_min_entropy(0.3, 16, 1000)
        assert h_n4 < h_n8 < h_n16

    def test_degenerate_n_returns_zero(self):
        """n < 2 is degenerate; return 0 rather than NaN/error."""
        assert estimate_min_entropy(0.5, 1, 1000) == 0.0
        assert estimate_min_entropy(0.5, 0, 1000) == 0.0

    def test_at_default_params_matches_run(self):
        """Sanity: at n=8, F_XEB=0.3, m=1000 we expect about 1660 bits."""
        h = estimate_min_entropy(0.3, 8, 1000)
        # h_haar = 8 - log2(8) - log2(ln 2) ≈ 5.529
        # h = 1000 * 0.3 * 5.529 ≈ 1659
        assert 1500 < h < 1800


# ============================================================
# LEG 6 — Leftover Hash Lemma bound (v0.3a)
# ============================================================

class TestLHLBound:
    """max_extractable_bits and the LHL enforcement in LEG 6."""

    def test_lhl_formula_correctness(self):
        """k = H_∞ - 2·log_2(1/ε); check the arithmetic."""
        # H_∞ = 1000 bits, ε = 2^-32 → k = 1000 - 2*32 = 936
        assert max_extractable_bits(1000.0, 2 ** -32) == 936

    def test_lhl_zero_min_entropy_returns_zero(self):
        """Can't extract from nothing."""
        assert max_extractable_bits(0.0, 2 ** -32) == 0

    def test_lhl_negative_result_clamps_to_zero(self):
        """When H_∞ < 2·log(1/ε), result would be negative — clamp to 0."""
        # H_∞ = 10, ε = 2^-32 → 10 - 64 = -54 → clamp to 0
        assert max_extractable_bits(10.0, 2 ** -32) == 0

    def test_lhl_smaller_epsilon_gives_fewer_bits(self):
        """Tighter ε → smaller LHL bound (more entropy "spent" on soundness)."""
        h = 1000.0
        k_loose = max_extractable_bits(h, 2 ** -8)
        k_tight = max_extractable_bits(h, 2 ** -64)
        assert k_loose > k_tight

    def test_lhl_invalid_epsilon_raises(self):
        """ε must be in (0, 1)."""
        with pytest.raises(ValueError):
            max_extractable_bits(1000.0, 0.0)
        with pytest.raises(ValueError):
            max_extractable_bits(1000.0, 1.0)
        with pytest.raises(ValueError):
            max_extractable_bits(1000.0, -0.1)

    def test_extraction_within_bound_succeeds(self, fresh_audit):
        """Requesting fewer bits than the bound → extraction proceeds."""
        samples = np.random.randint(0, 2, size=(100, 8), dtype=np.uint8)
        # Plenty of headroom
        out = leg6_toeplitz_extract(samples, b"x" * 32, 64, fresh_audit,
                                     max_extractable_bits_bound=1000)
        assert len(out) == 8

    def test_extraction_exceeding_bound_raises(self, fresh_audit):
        """Requesting more bits than the bound → ValueError; refuses to extract."""
        samples = np.random.randint(0, 2, size=(100, 8), dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds the Leftover Hash Lemma bound"):
            leg6_toeplitz_extract(samples, b"x" * 32, 2000, fresh_audit,
                                   max_extractable_bits_bound=1000)

    def test_extraction_at_exactly_bound_succeeds(self, fresh_audit):
        """Boundary: requested == bound → extraction proceeds (≤ comparison)."""
        samples = np.random.randint(0, 2, size=(100, 8), dtype=np.uint8)
        out = leg6_toeplitz_extract(samples, b"x" * 32, 64, fresh_audit,
                                     max_extractable_bits_bound=64)
        assert len(out) == 8

    def test_extraction_without_bound_still_works(self, fresh_audit):
        """Backwards compat: no bound supplied → no enforcement, extraction proceeds."""
        samples = np.random.randint(0, 2, size=(100, 8), dtype=np.uint8)
        out = leg6_toeplitz_extract(samples, b"x" * 32, 64, fresh_audit)
        assert len(out) == 8


class TestExtractor:
    """Toeplitz randomness extractor."""

    def test_output_length_matches_request(self, fresh_audit):
        """Extract N bits → output is N bits packed into ceil(N/8) bytes."""
        samples = np.random.randint(0, 2, size=(100, 8), dtype=np.uint8)
        drand_seed = b"x" * 32
        output = leg6_toeplitz_extract(samples, drand_seed, 64, fresh_audit)
        assert len(output) == 8  # 64 bits = 8 bytes

    def test_deterministic_given_inputs(self):
        """Same (samples, drand_seed, output_bits) → same output bits."""
        samples = np.random.RandomState(42).randint(0, 2, size=(100, 8), dtype=np.uint8)
        drand_seed = b"deterministic seed"
        out1 = leg6_toeplitz_extract(samples, drand_seed, 64, AuditLog())
        out2 = leg6_toeplitz_extract(samples, drand_seed, 64, AuditLog())
        assert out1 == out2

    def test_different_drand_seed_different_output(self):
        """Same samples + different drand seeds → different output (the seed matters)."""
        samples = np.random.RandomState(42).randint(0, 2, size=(100, 8), dtype=np.uint8)
        out1 = leg6_toeplitz_extract(samples, b"seed-A", 64, AuditLog())
        out2 = leg6_toeplitz_extract(samples, b"seed-B", 64, AuditLog())
        assert out1 != out2


# ============================================================
# LEG 7 — Audit chain integrity
# ============================================================

class TestAuditChain:
    """Hash-chained audit log."""

    def test_first_entry_has_zero_prev_hash(self, fresh_audit):
        """The chain root: first entry's prev_hash is all zeros."""
        fresh_audit.append("test_leg", {"x": 1})
        assert fresh_audit.entries[0]["prev_hash"] == "0" * 64

    def test_chain_links_correctly(self, fresh_audit):
        """Each entry's prev_hash equals the previous entry's hash."""
        fresh_audit.append("leg_a", {"a": 1})
        fresh_audit.append("leg_b", {"b": 2})
        fresh_audit.append("leg_c", {"c": 3})
        for i in range(1, len(fresh_audit.entries)):
            assert fresh_audit.entries[i]["prev_hash"] == fresh_audit.entries[i-1]["hash"]

    def test_entry_hash_recomputes(self, fresh_audit):
        """Each entry's hash equals SHA3-256 of its serialized payload."""
        fresh_audit.append("leg_x", {"value": 42})
        entry = fresh_audit.entries[0]
        payload = json.dumps(
            {"leg": entry["leg"], "data": entry["data"], "prev": entry["prev_hash"]},
            sort_keys=True, default=str
        )
        expected_hash = hashlib.sha3_256(payload.encode()).hexdigest()
        assert entry["hash"] == expected_hash

    def test_save_and_load_roundtrip(self, fresh_audit, tmp_path):
        """Saving and loading the audit log preserves all entries."""
        fresh_audit.append("leg_a", {"a": 1})
        fresh_audit.append("leg_b", {"b": "string"})
        path = tmp_path / "audit.json"
        fresh_audit.save(str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["data"]["a"] == 1
        assert loaded[1]["data"]["b"] == "string"


# ============================================================
# Verifier — negative cases
# ============================================================

class TestVerifier:
    """The audit log verifier catches tampering at multiple layers."""

    @pytest.fixture
    def clean_audit_log_path(self, tmp_path, reset_participants):
        """Run the full pipeline and write the audit log to a temp file."""
        path = tmp_path / "audit_log.json"
        run_pipeline(round_number=1, output_path=str(path))
        return str(path)

    def test_clean_log_passes(self, clean_audit_log_path):
        """A clean audit log produced by v0.3a must pass all verifier checks."""
        assert verify_audit_log(clean_audit_log_path)

    def test_tampered_beta_fails(self, clean_audit_log_path, tmp_path):
        """Modifying a VRF beta breaks the beta-from-pi check AND the seed check."""
        with open(clean_audit_log_path) as f:
            log = json.load(f)
        # Zero out a beta value
        log[1]["data"]["contributions"][0]["beta"] = "0" * 64
        tampered_path = tmp_path / "tampered_beta.json"
        with open(tampered_path, "w") as f:
            json.dump(log, f)
        # The verifier should detect this
        assert not verify_audit_log(str(tampered_path))

    def test_tampered_fxeb_fails(self, clean_audit_log_path, tmp_path):
        """Modifying F_XEB without updating the entry hash breaks chain integrity."""
        with open(clean_audit_log_path) as f:
            log = json.load(f)
        # Pretend F_XEB was much higher (a heavy-output attack)
        log[4]["data"]["f_xeb"] = 5.0
        tampered_path = tmp_path / "tampered_fxeb.json"
        with open(tampered_path, "w") as f:
            json.dump(log, f)
        assert not verify_audit_log(str(tampered_path))

    def test_cascading_tamper_is_not_caught_v0_3a(self, clean_audit_log_path, tmp_path):
        """
        DELIBERATE NEGATIVE TEST: documents an architectural limit of v0.3a.
        
        A cascading tamper (update entry hash AND all subsequent prev_hashes
        and hashes) produces an internally-consistent forged log that the
        verifier does NOT catch. This is why the architecture's LEG 7
        specifies a multi-party hash GRAPH with cross-signed pulses, not a
        single hash chain.
        
        This test passes (no assertion fails) iff the limit still holds.
        When a future version closes this gap (e.g., v0.3c adds per-entry
        signatures), this test must be updated to assert the opposite.
        """
        with open(clean_audit_log_path) as f:
            log = json.load(f)
        
        # Tamper at entry 4 (LEG_5_XEB)
        log[4]["data"]["f_xeb"] = 5.0
        log[4]["data"]["upper_check_passed"] = False
        log[4]["data"]["accepted"] = False
        
        # Recompute every hash from entry 4 onward, cascading prev_hash updates
        def recompute(entry):
            payload = json.dumps(
                {"leg": entry["leg"], "data": entry["data"], "prev": entry["prev_hash"]},
                sort_keys=True, default=str
            )
            return hashlib.sha3_256(payload.encode()).hexdigest()
        
        log[4]["hash"] = recompute(log[4])
        for i in range(5, len(log)):
            log[i]["prev_hash"] = log[i-1]["hash"]
            log[i]["hash"] = recompute(log[i])
        
        tampered_path = tmp_path / "cascade.json"
        with open(tampered_path, "w") as f:
            json.dump(log, f)
        
        # Verifier does NOT catch this — known limitation of single-chain v0.3a.
        # If this assertion ever fails (verifier rejects the cascading tamper),
        # it means a future version has closed the gap — update this test.
        assert verify_audit_log(str(tampered_path)), (
            "Cascading tamper was caught — the v0.3a limit has been closed; "
            "update this test accordingly"
        )


# ============================================================
# End-to-end integration
# ============================================================

class TestEndToEnd:
    """Full pipeline integration."""

    def test_pipeline_produces_six_audit_entries(self, tmp_path, reset_participants):
        """A successful pipeline run logs 6 entries: params + 5 legs."""
        path = tmp_path / "audit.json"
        run_pipeline(round_number=1, output_path=str(path))
        with open(path) as f:
            log = json.load(f)
        # PROTOCOL_PARAMS + LEG_1_VRF + LEG_3_CIRCUIT + LEG_4_SAMPLING +
        # LEG_5_XEB + LEG_6_EXTRACTION = 6 entries
        # (LEG 2 is just the params dict; LEG 7 is implicit in the chain itself)
        assert len(log) == 6

    def test_pipeline_audit_log_passes_verifier(self, tmp_path, reset_participants):
        """End-to-end consistency: pipeline output verifies cleanly."""
        path = tmp_path / "audit.json"
        run_pipeline(round_number=1, output_path=str(path))
        assert verify_audit_log(str(path))

    def test_pipeline_output_has_correct_size(self, tmp_path, reset_participants):
        """The extracted output is exactly extractor_output_bits / 8 bytes."""
        path = tmp_path / "audit.json"
        output = run_pipeline(round_number=1, output_path=str(path))
        if output is not None:  # If round was accepted
            assert len(output) == PROTOCOL_PARAMS["extractor_output_bits"] // 8
