"""Abstract signer interface plus the sr25519 implementation.

This module ships `Sr25519Signer`, which targets the historical
`MultiSignature` extrinsic signing path. The hybrid sr25519 + ML-DSA-44
implementation (`HybridSigner`) lives in `shared.hybrid_signer` and is
selected when the chain advertises the `HybridTxSignature` extension.
The abstract `Signer` interface is what the substrate client, controller,
and submitter consume — callers don't need to know which scheme is in use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from substrateinterface import Keypair, KeypairType


SignatureKind = Literal["Sr25519", "Ed25519", "Ecdsa", "Hybrid"]


def _normalize_sr25519_sig(raw: object) -> bytes:
    """Coerce substrate-interface's sign() return (bytes or hex str) to raw bytes."""
    if isinstance(raw, str):
        raw = bytes.fromhex(raw[2:] if raw.startswith("0x") else raw)
    return bytes(raw)


class Signer(ABC):
    """Sign extrinsic payloads for the substrate chain.

    Implementations are *not* required to expose the raw secret material. The
    `sign` method takes whatever payload bytes the caller has assembled (the
    full SCALE-encoded extrinsic body after the `SignedPayload` wrapping) and
    returns the signature bytes that go into the `MultiSignature` enum.
    """

    @abstractmethod
    def public_bytes(self) -> bytes:
        """The full public key, raw bytes (e.g. 32B for sr25519)."""

    @abstractmethod
    def account_id_bytes(self) -> bytes:
        """The 32-byte `AccountId32` derived from the public key.

        For sr25519/ed25519 this is the public key itself. For hybrid schemes
        this is a `blake2_256` digest of the composite public key — see
        Phase 7 plan for details.
        """

    @abstractmethod
    def ss58_address(self) -> str:
        """Human-readable SS58 address (network prefix 42 = generic substrate)."""

    @abstractmethod
    def sign(self, payload: bytes) -> bytes:
        """Sign an already-prepared payload. No hashing or wrapping here."""

    @abstractmethod
    def signature_kind(self) -> SignatureKind:
        """Identifies the `MultiSignature` variant the caller should emit."""


class Sr25519Signer(Signer):
    """sr25519 signer backed by `substrate-interface`'s `Keypair`.

    Constructed via the named constructors rather than `__init__` directly so
    the call sites read clearly:

        Sr25519Signer.from_uri("//Alice")
        Sr25519Signer.from_seed(bytes_32)
        Sr25519Signer.from_keypair(existing_keypair)
    """

    def __init__(self, keypair: Keypair) -> None:
        if keypair.crypto_type != KeypairType.SR25519:
            raise ValueError(
                f"Sr25519Signer requires SR25519 keypair, got crypto_type={keypair.crypto_type}"
            )
        self._keypair = keypair

    @classmethod
    def from_uri(cls, uri: str, ss58_format: int = 42) -> "Sr25519Signer":
        keypair = Keypair.create_from_uri(uri, crypto_type=KeypairType.SR25519, ss58_format=ss58_format)
        return cls(keypair)

    @classmethod
    def from_seed(cls, seed: bytes, ss58_format: int = 42) -> "Sr25519Signer":
        if len(seed) != 32:
            raise ValueError(f"sr25519 seed must be 32 bytes, got {len(seed)}")
        keypair = Keypair.create_from_seed(
            seed_hex=seed.hex(),
            crypto_type=KeypairType.SR25519,
            ss58_format=ss58_format,
        )
        return cls(keypair)

    @classmethod
    def from_keypair(cls, keypair: Keypair) -> "Sr25519Signer":
        return cls(keypair)

    def public_bytes(self) -> bytes:
        return bytes(self._keypair.public_key)

    def account_id_bytes(self) -> bytes:
        # For sr25519, AccountId32 is the public key itself.
        return bytes(self._keypair.public_key)

    def ss58_address(self) -> str:
        return self._keypair.ss58_address

    def sign(self, payload: bytes) -> bytes:
        # substrate-interface's sign() accepts bytes via `data=` and returns
        # raw 64-byte sr25519 signature.
        return _normalize_sr25519_sig(self._keypair.sign(data=payload))

    def signature_kind(self) -> SignatureKind:
        return "Sr25519"

    @property
    def keypair(self) -> Keypair:
        """Underlying `substrate-interface` Keypair.

        Exposed so the submitter can pass it directly to `compose_call` /
        `create_signed_extrinsic` until the substrate client wraps that path
        behind the abstract `Signer` interface.
        """
        return self._keypair
