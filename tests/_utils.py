"""Shared test-only helpers.

Importable from any test file as ``from _utils import ...`` because
pytest auto-adds the ``tests/`` directory to ``sys.path`` via the
sibling ``conftest.py``.
"""

import json
import os


def hash_for(i: int) -> bytes:
    """Deterministic 32-byte block hash for a test index (big-endian)."""
    return i.to_bytes(32, "big")


def index_of(h: bytes) -> int:
    """Recover the synthetic index from a hash produced by ``hash_for``."""
    return int.from_bytes(h, "big")


def write_json(path, data) -> None:
    """Write ``data`` as JSON to ``path``, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
