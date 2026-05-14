"""sr25519 keypair persistence for `quip-miner`.

Stores the keypair as a JSON file with strict file mode (0o600). For Phase 2
the seed is written in plaintext — quip-miner is dev-targeted at this stage
and the alternative (passphrase prompt on every miner start) would block
unattended workflows. The on-disk format reserves an `encrypted` field for a
future migration to passphrase-protected storage; readers refuse files where
that field is `true` because they can't decrypt yet.

Format:

```json
{
    "version": 1,
    "scheme": "sr25519",
    "encrypted": false,
    "seed_hex": "0x<32-byte hex>",
    "ss58": "5...",
    "account_id_hex": "0x<32-byte hex>"
}
```
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from shared.logging_config import get_logger
from shared.signer import Sr25519Signer


logger = get_logger("keystore")


KEYSTORE_VERSION = 1
KEYSTORE_FILE_MODE = 0o600


@dataclass(frozen=True)
class KeystoreFile:
    """In-memory view of an unlocked keystore."""

    path: Path
    signer: Sr25519Signer


def generate(path: Path, *, overwrite: bool = False) -> KeystoreFile:
    """Create a new sr25519 keypair and write it to `path`.

    Raises `FileExistsError` if the file is present and `overwrite=False`.
    """
    path = Path(path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"keystore already exists at {path}; pass overwrite=True to replace"
        )
    seed = os.urandom(32)
    signer = Sr25519Signer.from_seed(seed)
    _write(path, signer, seed)
    logger.info(
        "generated keystore: path=%s ss58=%s",
        path,
        signer.ss58_address(),
    )
    return KeystoreFile(path=path, signer=signer)


def load(path: Path) -> KeystoreFile:
    """Open an existing keystore and return its signer."""
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
    if raw.get("scheme") != "sr25519":
        raise ValueError(
            f"keystore scheme {raw.get('scheme')!r} not supported in Phase 2; "
            "hybrid scheme lands in Phase 7"
        )
    if raw.get("encrypted"):
        raise ValueError(
            "keystore is marked encrypted; passphrase-protected keystores are "
            "not supported yet (planned for Phase 7 alongside HybridSigner)"
        )
    seed_hex = raw["seed_hex"]
    if seed_hex.startswith("0x"):
        seed_hex = seed_hex[2:]
    seed = bytes.fromhex(seed_hex)
    signer = Sr25519Signer.from_seed(seed)
    return KeystoreFile(path=path, signer=signer)


def load_or_generate(path: Path) -> KeystoreFile:
    """Convenience helper for bootstrap: open if present, else generate."""
    path = Path(path).expanduser()
    if path.exists():
        return load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return generate(path)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _write(path: Path, signer: Sr25519Signer, seed: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": KEYSTORE_VERSION,
        "scheme": "sr25519",
        "encrypted": False,
        "seed_hex": "0x" + seed.hex(),
        "ss58": signer.ss58_address(),
        "account_id_hex": "0x" + signer.account_id_bytes().hex(),
    }
    # Write to a sibling tempfile and rename so a crash mid-write can't leave
    # a half-populated keystore on disk.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(tmp, KEYSTORE_FILE_MODE)
    tmp.replace(path)


def _check_mode(path: Path) -> None:
    """Warn when the keystore has wider-than-owner permissions."""
    st = path.stat()
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning(
            "keystore %s has group/world-readable permissions (%o); "
            "tighten to %o",
            path,
            stat.S_IMODE(st.st_mode),
            KEYSTORE_FILE_MODE,
        )
