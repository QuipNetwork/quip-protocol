"""Unit tests for `shared.hybrid_signer` + `shared.keystore_hybrid`.

Covers:
  - byte-level invariants pinned to the Rust suite (HKDF inputs, message-prep
    prefix, account-id domain string, component sizes)
  - HybridSigner round-trip (sign + self-verify)
  - Determinism across HybridSigner re-instantiation from the same seed
  - Keystore generate/load/tamper-detection
  - Tampering rejection — flipped pubkey hex in the keystore JSON fails to load

End-to-end chain submission (signing an actual extrinsic, chain accepts it)
lands once the SubstrateClient hybrid path is in place — see Phase 7 plan.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from shared.hybrid_signer import (
    ACCOUNT_ID_DOMAIN,
    HKDF_SALT,
    HYBRID_LABEL,
    HYBRID_PK_LEN,
    HYBRID_SIG_LEN,
    HYBRID_VERSION,
    HybridSigner,
    MASTER_SEED_LEN,
    derive_account_id,
    derive_component_seeds,
    prepare_message,
)
from shared.keystore_hybrid import (
    KEYSTORE_FILE_MODE,
    KEYSTORE_VERSION,
    generate,
    load,
)


# ----------------------------------------------------------------------
# Constants — pinned. Changing any of these re-keys every chain account or
# silently weakens the hash inputs. The Rust side tests `account_id_domain_is_pinned`
# similarly; this is the Python-side mirror.
# ----------------------------------------------------------------------


def test_account_id_domain_pinned():
    assert ACCOUNT_ID_DOMAIN == b"quip-account-v1"


def test_hkdf_salt_pinned():
    assert HKDF_SALT == b"hybrid-sig"


def test_hybrid_label_pinned():
    # The trailing NUL is part of the domain string, not a C convention —
    # changing its presence breaks signature parity with the Rust suite.
    assert HYBRID_LABEL == b"hybrid-sr25519-mldsa44-v1\x00"
    assert len(HYBRID_LABEL) == 26


def test_hybrid_version_pinned():
    assert HYBRID_VERSION == 0x01


def test_master_seed_len():
    assert MASTER_SEED_LEN == 32


# ----------------------------------------------------------------------
# derive_component_seeds
# ----------------------------------------------------------------------


def test_derive_component_seeds_lengths():
    seed = bytes(range(32))
    classical, pq = derive_component_seeds(seed)
    assert len(classical) == 32
    assert len(pq) == 32


def test_derive_component_seeds_are_distinct():
    """Classical and PQ seeds must be derived with different `info` strings
    so the same master seed produces unrelated sub-keys. A bug that used
    the same info would re-key both components from one HKDF block."""
    seed = bytes(range(32))
    classical, pq = derive_component_seeds(seed)
    assert classical != pq


def test_derive_component_seeds_deterministic():
    seed = bytes(range(32))
    a = derive_component_seeds(seed)
    b = derive_component_seeds(seed)
    assert a == b


def test_derive_component_seeds_rejects_wrong_length():
    with pytest.raises(ValueError):
        derive_component_seeds(b"\x00" * 16)
    with pytest.raises(ValueError):
        derive_component_seeds(b"\x00" * 64)


# ----------------------------------------------------------------------
# prepare_message
# ----------------------------------------------------------------------


def test_prepare_message_layout():
    """M' = 0x01 || label || 0x00 (empty ctx len) || msg."""
    msg = b"hello"
    out = prepare_message(msg)
    assert out[0] == 0x01
    assert out[1:1 + len(HYBRID_LABEL)] == HYBRID_LABEL
    assert out[1 + len(HYBRID_LABEL)] == 0x00  # empty ctx length
    assert out[2 + len(HYBRID_LABEL):] == msg


def test_prepare_message_with_ctx():
    msg = b"hello"
    ctx = b"some-ctx"
    out = prepare_message(msg, ctx=ctx)
    # version + label + ctx_len + ctx + msg
    expected = (
        bytes([HYBRID_VERSION])
        + HYBRID_LABEL
        + bytes([len(ctx)])
        + ctx
        + msg
    )
    assert out == expected


def test_prepare_message_rejects_long_ctx():
    with pytest.raises(ValueError):
        prepare_message(b"msg", ctx=b"x" * 256)


# ----------------------------------------------------------------------
# derive_account_id
# ----------------------------------------------------------------------


def test_derive_account_id_lengths():
    pubkey = b"\x00" * HYBRID_PK_LEN
    aid = derive_account_id(pubkey)
    assert len(aid) == 32


def test_derive_account_id_rejects_wrong_pubkey_len():
    with pytest.raises(ValueError):
        derive_account_id(b"\x00" * 32)
    with pytest.raises(ValueError):
        derive_account_id(b"\x00" * (HYBRID_PK_LEN + 1))


def test_derive_account_id_deterministic():
    pubkey = bytes(range(256)) * 6  # 1536 bytes, take prefix
    pubkey = pubkey[:HYBRID_PK_LEN]
    a = derive_account_id(pubkey)
    b = derive_account_id(pubkey)
    assert a == b


# ----------------------------------------------------------------------
# HybridSigner end-to-end
# ----------------------------------------------------------------------


def test_signer_public_bytes_length():
    s = HybridSigner.from_master_seed(bytes(range(32)))
    assert len(s.public_bytes()) == HYBRID_PK_LEN


def test_signer_account_id_length():
    s = HybridSigner.from_master_seed(bytes(range(32)))
    assert len(s.account_id_bytes()) == 32


def test_signer_signature_length():
    s = HybridSigner.from_master_seed(bytes(range(32)))
    sig = s.sign(b"hello chain")
    assert len(sig) == HYBRID_SIG_LEN


def test_signer_verify_round_trip():
    s = HybridSigner.from_master_seed(bytes(range(32)))
    sig = s.sign(b"hello chain")
    assert s.verify(b"hello chain", sig)


def test_signer_verify_rejects_wrong_message():
    s = HybridSigner.from_master_seed(bytes(range(32)))
    sig = s.sign(b"hello chain")
    assert not s.verify(b"tampered", sig)


def test_signer_verify_rejects_truncated_signature():
    s = HybridSigner.from_master_seed(bytes(range(32)))
    sig = s.sign(b"hello chain")
    # Drop the last byte → length mismatch → fast reject
    assert not s.verify(b"hello chain", sig[:-1])


def test_signer_verify_rejects_tampered_signature():
    s = HybridSigner.from_master_seed(bytes(range(32)))
    sig = bytearray(s.sign(b"hello chain"))
    # Flip a byte in the middle of the ml_dsa section
    sig[1000] ^= 0xFF
    assert not s.verify(b"hello chain", bytes(sig))


def test_signer_keypair_deterministic_from_seed():
    seed = bytes(range(32))
    a = HybridSigner.from_master_seed(seed)
    b = HybridSigner.from_master_seed(seed)
    assert a.public_bytes() == b.public_bytes()
    assert a.account_id_bytes() == b.account_id_bytes()
    assert a.sr25519_public_bytes == b.sr25519_public_bytes
    assert a.ml_dsa_public_bytes == b.ml_dsa_public_bytes


def test_signer_different_seeds_give_different_keys():
    a = HybridSigner.from_master_seed(b"\x00" * 32)
    b = HybridSigner.from_master_seed(b"\x01" * 32)
    assert a.public_bytes() != b.public_bytes()
    assert a.account_id_bytes() != b.account_id_bytes()


def test_signer_kind_is_hybrid():
    s = HybridSigner.from_master_seed(bytes(range(32)))
    assert s.signature_kind() == "Hybrid"


def test_signer_rejects_wrong_seed_length():
    with pytest.raises(ValueError):
        HybridSigner.from_master_seed(b"\x00" * 16)


def test_signer_generate_produces_unique_keys():
    """Sanity: HybridSigner.generate() uses os.urandom — two calls must yield
    different keys."""
    a = HybridSigner.generate()
    b = HybridSigner.generate()
    assert a.master_seed != b.master_seed
    assert a.account_id_bytes() != b.account_id_bytes()


# ----------------------------------------------------------------------
# Hybrid keystore
# ----------------------------------------------------------------------


def test_keystore_generate_writes_file_with_600_perms(tmp_path: Path):
    path = tmp_path / "hybrid.json"
    keystore = generate(path)
    assert path.exists()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == KEYSTORE_FILE_MODE
    assert keystore.signer.ss58_address().startswith("5")


def test_keystore_generate_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "hybrid.json"
    generate(path)
    with pytest.raises(FileExistsError):
        generate(path)


def test_keystore_generate_overwrite(tmp_path: Path):
    path = tmp_path / "hybrid.json"
    a = generate(path)
    b = generate(path, overwrite=True)
    assert a.signer.account_id_bytes() != b.signer.account_id_bytes()


def test_keystore_round_trip(tmp_path: Path):
    path = tmp_path / "hybrid.json"
    written = generate(path)
    loaded = load(path)
    assert loaded.signer.account_id_bytes() == written.signer.account_id_bytes()
    assert loaded.signer.public_bytes() == written.signer.public_bytes()
    # Signatures from both signers verify under the same pubkey.
    sig = written.signer.sign(b"hello")
    assert loaded.signer.verify(b"hello", sig)


def test_keystore_load_rejects_wrong_scheme(tmp_path: Path):
    path = tmp_path / "hybrid.json"
    path.write_text(json.dumps({
        "version": KEYSTORE_VERSION,
        "scheme": "sr25519",
        "encrypted": False,
    }))
    os.chmod(path, KEYSTORE_FILE_MODE)
    with pytest.raises(ValueError, match="expected scheme=hybrid"):
        load(path)


def test_keystore_load_rejects_encrypted_marker(tmp_path: Path):
    path = tmp_path / "hybrid.json"
    path.write_text(json.dumps({
        "version": KEYSTORE_VERSION,
        "scheme": "hybrid",
        "encrypted": True,
    }))
    os.chmod(path, KEYSTORE_FILE_MODE)
    with pytest.raises(ValueError, match="encrypted"):
        load(path)


def test_keystore_load_detects_tampered_pubkey(tmp_path: Path):
    """A tampered keystore that ships a mismatched cached pubkey is caught
    at load time, not later when the chain rejects extrinsics for a
    different AccountId. Defense-in-depth against on-disk modification."""
    path = tmp_path / "hybrid.json"
    generate(path)
    raw = json.loads(path.read_text())
    # Flip a byte in the cached sr25519 pubkey hex.
    pubkey_hex = raw["sr25519_public_hex"][2:]
    tampered = list(pubkey_hex)
    tampered[0] = "f" if tampered[0] != "f" else "e"
    raw["sr25519_public_hex"] = "0x" + "".join(tampered)
    path.write_text(json.dumps(raw))
    os.chmod(path, KEYSTORE_FILE_MODE)

    with pytest.raises(ValueError, match="sr25519_public_hex does not match"):
        load(path)
