"""sr25519 + ML-DSA-44 hybrid keystore for quip-miner Phase 7.

Stores the 32-byte master seed only — both component keypairs are
re-derived deterministically on load via `HybridSigner.from_master_seed`.
Same on-disk shape and `0o600` permissions as `shared.keystore`, with a
`scheme: "hybrid"` discriminator so `Keystore.load(path)` can dispatch.

File layout:

```json
{
    "version": 1,
    "scheme": "hybrid",
    "encrypted": false,
    "master_seed_hex": "0x<32-byte hex>",
    "ss58": "5...",
    "account_id_hex": "0x<32-byte hex>",
    "sr25519_public_hex": "0x<32-byte hex>",
    "ml_dsa_public_hex": "0x<1312-byte hex>"
}
```

`sr25519_public_hex` and `ml_dsa_public_hex` are cached for fast load — we
re-derive on every load anyway and assert they match, so a tampered
keystore that pre-computes a wrong pubkey is caught immediately.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from shared.hybrid_signer import HybridSigner
from shared.logging_config import get_logger
from shared.signer import strip_0x


logger = get_logger("keystore_hybrid")


KEYSTORE_VERSION = 1
KEYSTORE_FILE_MODE = 0o600


@dataclass(frozen=True)
class HybridKeystoreFile:
    path: Path
    signer: HybridSigner


def generate(path: Path, *, overwrite: bool = False) -> HybridKeystoreFile:
    """Create a fresh hybrid keystore at `path`."""
    path = Path(path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"keystore already exists at {path}; pass overwrite=True to replace"
        )
    signer = HybridSigner.generate()
    _write(path, signer)
    logger.info(
        "generated hybrid keystore: path=%s ss58=%s", path, signer.ss58_address()
    )
    return HybridKeystoreFile(path=path, signer=signer)


def load(path: Path) -> HybridKeystoreFile:
    """Open an existing hybrid keystore and return its signer.

    Re-derives both component keypairs from the master seed and asserts the
    cached `sr25519_public_hex` / `ml_dsa_public_hex` fields match — a
    tampered keystore that ships a mismatched pubkey is caught here, not
    later when the chain rejects extrinsics.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"keystore not found: {path}")
    _check_mode(path)
    raw = json.loads(path.read_text())
    if raw.get("version") != KEYSTORE_VERSION:
        raise ValueError(
            f"keystore version {raw.get('version')} not supported; expected "
            f"{KEYSTORE_VERSION}"
        )
    if raw.get("scheme") != "hybrid":
        raise ValueError(
            f"expected scheme=hybrid, got {raw.get('scheme')!r}; use "
            "shared.keystore.load for sr25519 keystores"
        )
    if raw.get("encrypted"):
        raise ValueError(
            "passphrase-encrypted hybrid keystores not supported yet; the "
            "encrypted-on-disk format is a follow-on once the dev flow stabilises"
        )
    master_seed = bytes.fromhex(strip_0x(raw["master_seed_hex"]))
    signer = HybridSigner.from_master_seed(master_seed)

    # Tamper check: cached pubkey hex must match what we derive from the seed.
    expected_sr = signer.sr25519_public_bytes.hex()
    expected_ml = signer.ml_dsa_public_bytes.hex()
    raw_sr = raw.get("sr25519_public_hex")
    raw_ml = raw.get("ml_dsa_public_hex")
    # Require both cached pubkey fields present and non-empty. An attacker
    # editing the JSON to delete/blank them used to silently disable the
    # integrity check; now a tampered file fails to load instead.
    if not isinstance(raw_sr, str) or not raw_sr.strip():
        raise ValueError(
            "keystore tampered: sr25519_public_hex field is missing or empty"
        )
    if not isinstance(raw_ml, str) or not raw_ml.strip():
        raise ValueError(
            "keystore tampered: ml_dsa_public_hex field is missing or empty"
        )
    if strip_0x(raw_sr) != expected_sr:
        raise ValueError(
            "keystore tampered: sr25519_public_hex does not match the key "
            "derived from master_seed_hex"
        )
    if strip_0x(raw_ml) != expected_ml:
        raise ValueError(
            "keystore tampered: ml_dsa_public_hex does not match the key "
            "derived from master_seed_hex"
        )
    return HybridKeystoreFile(path=path, signer=signer)


def load_or_generate(path: Path) -> HybridKeystoreFile:
    """Bootstrap helper: open if present, else generate."""
    path = Path(path).expanduser()
    if path.exists():
        return load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return generate(path)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _write(path: Path, signer: HybridSigner) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": KEYSTORE_VERSION,
        "scheme": "hybrid",
        "encrypted": False,
        "master_seed_hex": "0x" + signer.master_seed.hex(),
        "ss58": signer.ss58_address(),
        "account_id_hex": "0x" + signer.account_id_bytes().hex(),
        "sr25519_public_hex": "0x" + signer.sr25519_public_bytes.hex(),
        "ml_dsa_public_hex": "0x" + signer.ml_dsa_public_bytes.hex(),
    }
    # Open the tmp file with O_EXCL and the final mode so the master seed is
    # never on disk at 0o644: a crash between `write_text` and `chmod` used
    # to leave a world-readable seed for any concurrent reader.
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    fd = os.open(
        tmp,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        KEYSTORE_FILE_MODE,
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
    except BaseException:
        # Don't leave a half-written temp file behind on error.
        tmp.unlink(missing_ok=True)
        raise
    os.chmod(tmp, KEYSTORE_FILE_MODE)  # belt-and-braces: ignore umask
    tmp.replace(path)


def _check_mode(path: Path) -> None:
    st = path.stat()
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning(
            "hybrid keystore %s has group/world-readable permissions (%o); "
            "tighten to %o",
            path,
            stat.S_IMODE(st.st_mode),
            KEYSTORE_FILE_MODE,
        )


__all__ = [
    "HybridKeystoreFile",
    "KEYSTORE_FILE_MODE",
    "KEYSTORE_VERSION",
    "generate",
    "load",
    "load_or_generate",
]
