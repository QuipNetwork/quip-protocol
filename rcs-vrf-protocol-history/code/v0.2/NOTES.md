# v0.2 — Real Ed25519 VRF + hash-chained audit log

**Two substantive changes vs v0.1:**
1. Replaced stub VRF with real Ed25519 EC-VRF (RFC 9381).
2. Added hash-chained audit log + replay verifier.

## Files in this version

- `rcs_vrf.py` — protocol with real VRF and audit log
- `test_rcs_vrf.py` — first formal test suite
- `verify_audit_log.py` — standalone audit-log replay verifier
- `audit_log_example.json` — sample audit log from a successful run

## What was new here

- **Real VRF**: full EC-VRF Prove/Verify with elliptic-curve operations and Schnorr-style proof, conforming to RFC 9381.
- **Audit log**: hash-chained JSON; every round's events linked to the previous round's hash so any tampering breaks the chain.
- **Replay verifier**: takes an audit log and confirms the round was honest end-to-end.

## Why this layer matters

The protocol's claim to be "publicly verifiable" requires both: a cryptographic primitive that can be audited (the VRF's proof) and a complete record of the round's events (the audit log).

## Subsequently questioned

EC-VRF turned out to have three structural problems for the Quip setting (not quantum-secure, single-keyholder dilemma, MEV exposure). The Leg 1 redesign (in progress, see `docs/design/leg1_redesign_story.pdf`) replaces it with hash-based commit-reveal + external beacons. The audit log infrastructure carries forward unchanged.

## What carries forward

The audit log design is unchanged in v0.3a and v0.4. EC-VRF stays in place through v0.4 and is replaced in the (planned) v0.5.
