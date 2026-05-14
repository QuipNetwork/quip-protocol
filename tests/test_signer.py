"""Unit tests for `shared.signer.Sr25519Signer`.

These don't require a live chain — they only exercise the keypair
construction, type guards, and the signature-kind contract that the
substrate client branches on.
"""
from __future__ import annotations

import pytest

from substrateinterface import Keypair, KeypairType

from shared.signer import Sr25519Signer


# //Alice's SS58 address on the generic substrate prefix (42) is a
# well-known constant — pin it here so any unintended change to the
# constructor path surfaces immediately.
ALICE_SS58 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def test_from_uri_alice_well_known_address():
    signer = Sr25519Signer.from_uri("//Alice")
    assert signer.ss58_address() == ALICE_SS58
    # AccountId for sr25519 is the raw 32-byte public key.
    assert len(signer.account_id_bytes()) == 32
    assert signer.account_id_bytes() == signer.public_bytes()


def test_signature_kind_is_literal_sr25519():
    # `SubstrateClient.submit_extrinsic` branches on the exact string
    # "Sr25519" — a typo or case change there or here would silently send
    # everything down the NotImplementedError path.
    signer = Sr25519Signer.from_uri("//Alice")
    assert signer.signature_kind() == "Sr25519"


def test_from_seed_rejects_wrong_length():
    with pytest.raises(ValueError, match="32 bytes"):
        Sr25519Signer.from_seed(b"\x00" * 31)
    with pytest.raises(ValueError, match="32 bytes"):
        Sr25519Signer.from_seed(b"\x00" * 33)


def test_from_seed_accepts_32_bytes():
    signer = Sr25519Signer.from_seed(b"\x42" * 32)
    assert len(signer.public_bytes()) == 32


def test_init_rejects_non_sr25519_keypair():
    # Ed25519 keypairs are structurally similar but the chain's
    # MultiSignature wire format demands sr25519; we want a loud rejection
    # rather than a sign-and-be-rejected path. (substrate-interface won't
    # accept `//Alice` for ed25519 — derivation paths are sr25519-only —
    # so construct from a seed instead.)
    ed25519_kp = Keypair.create_from_seed(
        seed_hex=("00" * 32),
        crypto_type=KeypairType.ED25519,
        ss58_format=42,
    )
    with pytest.raises(ValueError, match="SR25519"):
        Sr25519Signer(ed25519_kp)


def test_sign_returns_64_byte_signature():
    signer = Sr25519Signer.from_uri("//Alice")
    sig = signer.sign(b"hello quip")
    assert isinstance(sig, bytes)
    assert len(sig) == 64


def test_from_keypair_accepts_sr25519():
    kp = Keypair.create_from_uri(
        "//Bob", crypto_type=KeypairType.SR25519, ss58_format=42
    )
    signer = Sr25519Signer.from_keypair(kp)
    assert signer.signature_kind() == "Sr25519"
    assert signer.keypair is kp
