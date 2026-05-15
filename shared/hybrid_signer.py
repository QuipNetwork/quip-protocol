"""Hybrid sr25519 + ML-DSA-44 signer for quip-protocol-rs Phase 7.

Matches the Rust suite `quip_crypto_primitives::substrate::sr25519_mldsa44`
byte-for-byte where it matters (key derivation, signature composition,
account-id derivation, message-prep prefix).

Contract preserved across the v0.1 -> v0.2 refactor:
  - 32-byte master seed -> deterministic keypair via HKDF-SHA256
      classical_seed = HKDF-SHA256(salt="hybrid-sig", ikm=seed, info="classical", L=32)
      pq_seed        = HKDF-SHA256(salt="hybrid-sig", ikm=seed, info="pq", L=32)
      sr25519 pair   = sr25519::from_seed(classical_seed)         (sp_core::sr25519::Pair)
      ml_dsa pair    = ML_DSA_44.key_derive(pq_seed)              (FIPS 204 keygen)
  - HybridPublic    = sr25519_pk(32) || ml_dsa_pk(1312)           1344 bytes total
  - HybridSignature = sr25519_sig(64) || ml_dsa_sig(2420)         2484 bytes total
  - Message prep    M' = 0x01 || b"hybrid-sr25519-mldsa44-v1\\x00" || 0x00 || msg
                    (ctx is empty for runtime extrinsic signing — see
                     quip/primitives/crypto/src/substrate/signature.rs line 448)
  - AccountId       = blake2_256(b"quip-account-v1" || HybridPublic)
                    (quip-transaction-crypto crate ACCOUNT_ID_DOMAIN constant)

The chain's `Verify` impl only checks signature validity, not byte-equality
with its own signing path. So Python uses hedged (randomized) signing for
both components by default — both still verify under the same pubkey.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from dilithium_py.ml_dsa import ML_DSA_44
from scalecodec.utils.ss58 import ss58_encode
from substrateinterface import Keypair, KeypairType

from shared.signer import Signer, SignatureKind


# ----------------------------------------------------------------------
# Constants — must match `quip-crypto-primitives` exactly. Pinned by parity
# tests (see test_hybrid_signer_parity.py).
# ----------------------------------------------------------------------

HYBRID_LABEL = b"hybrid-sr25519-mldsa44-v1\0"  # 26 bytes
HYBRID_VERSION = 0x01
HKDF_SALT = b"hybrid-sig"
HKDF_CLASSICAL_INFO = b"classical"
HKDF_PQ_INFO = b"pq"
MASTER_SEED_LEN = 32
ACCOUNT_ID_DOMAIN = b"quip-account-v1"

SR_PK_LEN = 32
SR_SIG_LEN = 64
ML_PK_LEN = 1312
ML_SK_LEN = 2560
ML_SIG_LEN = 2420

HYBRID_PK_LEN = SR_PK_LEN + ML_PK_LEN          # 1344
HYBRID_SIG_LEN = SR_SIG_LEN + ML_SIG_LEN       # 2484


# ----------------------------------------------------------------------
# HKDF-SHA256 (RFC 5869). hashlib doesn't ship one in stdlib so we write
# the minimal expand-only form we need. Verified byte-for-byte against the
# Rust `hkdf` crate via parity fixtures.
# ----------------------------------------------------------------------


def _hkdf_extract_expand(*, salt: bytes, ikm: bytes, info: bytes, length: int) -> bytes:
    """Single-block (length<=32) HKDF-SHA256 extract + expand."""
    if length > 32:
        # Single-block expand only emits SHA-256 length (32 bytes); a caller
        # asking for more would silently get a truncated value. Raise so the
        # contract is loud — multi-block expand can be added if needed.
        raise ValueError(
            f"single-block HKDF only supports length<=32, got {length}"
        )
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    t = hmac.new(prk, info + bytes([1]), hashlib.sha256).digest()
    return t[:length]


def derive_component_seeds(master_seed: bytes) -> tuple[bytes, bytes]:
    """Expand a 32-byte master seed into (classical_seed, pq_seed).

    Mirrors `quip-crypto-primitives::seed::derive_component_seeds`.
    """
    if len(master_seed) != MASTER_SEED_LEN:
        raise ValueError(
            f"master_seed must be {MASTER_SEED_LEN} bytes, got {len(master_seed)}"
        )
    classical = _hkdf_extract_expand(
        salt=HKDF_SALT, ikm=master_seed, info=HKDF_CLASSICAL_INFO, length=MASTER_SEED_LEN
    )
    pq = _hkdf_extract_expand(
        salt=HKDF_SALT, ikm=master_seed, info=HKDF_PQ_INFO, length=MASTER_SEED_LEN
    )
    return classical, pq


def prepare_message(msg: bytes, *, ctx: bytes = b"") -> bytes:
    """Domain-prepared message: M' = version || label || len(ctx) || ctx || msg.

    Mirrors `quip-crypto-primitives::domain::prepare_message`.
    """
    if len(ctx) > 255:
        raise ValueError(f"ctx must be at most 255 bytes, got {len(ctx)}")
    return (
        bytes([HYBRID_VERSION])
        + HYBRID_LABEL
        + bytes([len(ctx)])
        + ctx
        + msg
    )


def derive_account_id(hybrid_public: bytes) -> bytes:
    """AccountId32 derivation: blake2_256(domain || pubkey).

    Mirrors `quip-transaction-crypto::account_id_from_public`.
    """
    if len(hybrid_public) != HYBRID_PK_LEN:
        raise ValueError(
            f"hybrid_public must be {HYBRID_PK_LEN} bytes, got {len(hybrid_public)}"
        )
    h = hashlib.blake2b(ACCOUNT_ID_DOMAIN + hybrid_public, digest_size=32)
    return h.digest()


# ----------------------------------------------------------------------
# HybridSigner
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _HybridKeypair:
    """Resolved hybrid keypair held by `HybridSigner`."""

    sr25519: Keypair       # substrate-interface Keypair (handles sr25519 sign/verify)
    ml_dsa_pk: bytes       # 1312-byte ML-DSA-44 public key
    ml_dsa_sk: bytes       # 2560-byte ML-DSA-44 secret key (FIPS 204 packed form)


class HybridSigner(Signer):
    """sr25519 + ML-DSA-44 composite signer.

    Construct via `from_master_seed(seed)` (deterministic, recommended) or
    `generate()` (randomized; useful in tests).
    """

    def __init__(self, master_seed: bytes, _keypair: _HybridKeypair) -> None:
        # Private __init__: callers should use the factory classmethods so
        # the seed validation lives in one place.
        self._master_seed = master_seed
        self._kp = _keypair

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_master_seed(cls, master_seed: bytes) -> "HybridSigner":
        """Deterministic keypair from a 32-byte master seed.

        The classical sub-key is derived via `sp_core::sr25519::Pair::from_seed`
        (substrate-interface's `Keypair.create_from_seed`), and the PQ sub-key
        via FIPS 204 `ML_DSA.KeyGen(seed)`. Same recipe the Rust suite uses
        in `fixed::from_seed_slice`.
        """
        if len(master_seed) != MASTER_SEED_LEN:
            raise ValueError(
                f"master_seed must be {MASTER_SEED_LEN} bytes, got {len(master_seed)}"
            )
        classical_seed, pq_seed = derive_component_seeds(master_seed)
        sr25519_pair = Keypair.create_from_seed(
            seed_hex=classical_seed.hex(),
            crypto_type=KeypairType.SR25519,
            # ss58_format here is for display only; AccountId is derived from
            # the hybrid pubkey, not the sr25519-only address.
            ss58_format=42,
        )
        ml_pk, ml_sk = ML_DSA_44.key_derive(pq_seed)
        return cls(
            master_seed=master_seed,
            _keypair=_HybridKeypair(sr25519=sr25519_pair, ml_dsa_pk=ml_pk, ml_dsa_sk=ml_sk),
        )

    @classmethod
    def generate(cls) -> "HybridSigner":
        """Fresh random master seed; useful for tests + dev keygen."""
        return cls.from_master_seed(os.urandom(MASTER_SEED_LEN))

    # ------------------------------------------------------------------
    # Signer protocol
    # ------------------------------------------------------------------

    def public_bytes(self) -> bytes:
        """Composite hybrid public key: sr25519_pk(32) || ml_dsa_pk(1312)."""
        return bytes(self._kp.sr25519.public_key) + self._kp.ml_dsa_pk

    def account_id_bytes(self) -> bytes:
        """32-byte chain AccountId32 derived from the composite pubkey."""
        return derive_account_id(self.public_bytes())

    def ss58_address(self) -> str:
        """SS58 address of the *chain* AccountId (blake2_256-derived).

        Not the same as the sr25519 sub-key's SS58 address — the chain
        identifies accounts by `blake2_256(domain || hybrid_pubkey)`.
        """
        return ss58_encode(self.account_id_bytes(), ss58_format=42)

    def sign(self, payload: bytes) -> bytes:
        """Composite signature: sr25519_sig(64) || ml_dsa_sig(2420) over M'.

        M' = `prepare_message(payload)` — i.e., domain-bound with the hybrid
        suite version + label. Both component signs are hedged (randomized);
        both still verify under the same pubkey because verification is
        signature-correctness-only, not byte-equality with the chain's path.
        """
        msg_prime = prepare_message(payload)
        sr_sig = self._kp.sr25519.sign(data=msg_prime)
        if isinstance(sr_sig, str):
            sr_sig = bytes.fromhex(sr_sig[2:] if sr_sig.startswith("0x") else sr_sig)
        sr_sig = bytes(sr_sig)
        ml_sig = ML_DSA_44.sign(self._kp.ml_dsa_sk, msg_prime)
        if len(sr_sig) != SR_SIG_LEN:
            raise RuntimeError(
                f"sr25519 signature length {len(sr_sig)} != expected {SR_SIG_LEN}; "
                f"substrate-interface signing returned an unexpected shape"
            )
        if len(ml_sig) != ML_SIG_LEN:
            raise RuntimeError(
                f"ML-DSA-44 signature length {len(ml_sig)} != expected {ML_SIG_LEN}; "
                f"dilithium-py returned an unexpected shape"
            )
        return sr_sig + ml_sig

    def signature_kind(self) -> SignatureKind:
        return "Hybrid"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def master_seed(self) -> bytes:
        """The 32-byte master seed this signer was constructed from.

        Exposed so `keystore_hybrid.py` can persist it. Treat as secret —
        anyone with the seed can derive both sub-keys.
        """
        return self._master_seed

    @property
    def sr25519_public_bytes(self) -> bytes:
        return bytes(self._kp.sr25519.public_key)

    @property
    def ml_dsa_public_bytes(self) -> bytes:
        return self._kp.ml_dsa_pk

    def verify(self, payload: bytes, signature: bytes) -> bool:
        """Re-verify a signature with this signer's keypair.

        Mirrors the chain's `Verify::verify` flow without the AccountId
        cross-check: just confirm both component signatures are valid.
        """
        if len(signature) != HYBRID_SIG_LEN:
            return False
        sr_sig = signature[:SR_SIG_LEN]
        ml_sig = signature[SR_SIG_LEN:]
        msg_prime = prepare_message(payload)
        # Both component verifiers can raise on malformed inputs (wrong-shape
        # signature bytes, internal library asserts). Catch the documented
        # exception types and treat as verification failure so the function's
        # `-> bool` contract holds; let unexpected types propagate.
        try:
            sr_ok = self._kp.sr25519.verify(data=msg_prime, signature=sr_sig)
        except (ValueError, TypeError):
            sr_ok = False
        try:
            ml_ok = ML_DSA_44.verify(self._kp.ml_dsa_pk, msg_prime, ml_sig)
        except (ValueError, TypeError):
            ml_ok = False
        return sr_ok and ml_ok


__all__ = [
    "ACCOUNT_ID_DOMAIN",
    "HKDF_SALT",
    "HYBRID_LABEL",
    "HYBRID_PK_LEN",
    "HYBRID_SIG_LEN",
    "HYBRID_VERSION",
    "HybridSigner",
    "MASTER_SEED_LEN",
    "derive_account_id",
    "derive_component_seeds",
    "prepare_message",
]
