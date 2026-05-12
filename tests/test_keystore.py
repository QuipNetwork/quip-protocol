"""Unit tests for `shared.keystore`."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from shared.keystore import (
    KEYSTORE_FILE_MODE,
    generate,
    load,
    load_or_generate,
)


def test_generate_writes_file_with_600_perms(tmp_path: Path):
    path = tmp_path / "signing.json"
    keystore = generate(path)
    assert path.exists()
    assert keystore.signer.ss58_address().startswith("5")

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == KEYSTORE_FILE_MODE, f"expected 0o{KEYSTORE_FILE_MODE:o}, got 0o{mode:o}"


def test_generate_refuses_to_overwrite(tmp_path: Path):
    path = tmp_path / "signing.json"
    generate(path)
    with pytest.raises(FileExistsError):
        generate(path)


def test_generate_overwrite_flag_replaces(tmp_path: Path):
    path = tmp_path / "signing.json"
    first = generate(path)
    second = generate(path, overwrite=True)
    assert first.signer.ss58_address() != second.signer.ss58_address()


def test_load_round_trips(tmp_path: Path):
    path = tmp_path / "signing.json"
    written = generate(path)
    loaded = load(path)
    assert loaded.signer.ss58_address() == written.signer.ss58_address()
    assert loaded.signer.account_id_bytes() == written.signer.account_id_bytes()
    # Signing the same payload must produce identical output (sr25519 is
    # actually randomized, so test the verify path instead via public bytes).
    assert loaded.signer.public_bytes() == written.signer.public_bytes()


def test_load_or_generate_creates_when_missing(tmp_path: Path):
    path = tmp_path / "nested" / "signing.json"
    keystore = load_or_generate(path)
    assert path.exists()
    assert keystore.path == path


def test_load_or_generate_reads_existing(tmp_path: Path):
    path = tmp_path / "signing.json"
    a = generate(path)
    b = load_or_generate(path)
    assert a.signer.ss58_address() == b.signer.ss58_address()


def test_load_rejects_unsupported_version(tmp_path: Path):
    path = tmp_path / "signing.json"
    path.write_text(json.dumps({"version": 99, "scheme": "sr25519"}))
    os.chmod(path, KEYSTORE_FILE_MODE)
    with pytest.raises(ValueError, match="version 99"):
        load(path)


def test_load_rejects_encrypted_keystore(tmp_path: Path):
    path = tmp_path / "signing.json"
    path.write_text(json.dumps({
        "version": 1,
        "scheme": "sr25519",
        "encrypted": True,
    }))
    os.chmod(path, KEYSTORE_FILE_MODE)
    with pytest.raises(ValueError, match="marked encrypted"):
        load(path)


def test_load_rejects_non_sr25519_scheme(tmp_path: Path):
    path = tmp_path / "signing.json"
    path.write_text(json.dumps({
        "version": 1,
        "scheme": "hybrid",
        "encrypted": False,
        "seed_hex": "00" * 32,
    }))
    os.chmod(path, KEYSTORE_FILE_MODE)
    with pytest.raises(ValueError, match="not supported in Phase 2"):
        load(path)
