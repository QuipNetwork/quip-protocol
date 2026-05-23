"""Unit tests for `faucet_bot.py` helpers.

These tests don't need a live chain and stay in the default pytest run.
The integration tests in `tests/test_substrate_faucet.py` still require
the docker compose chain and are skipped without it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import faucet_bot  # noqa: E402


# ----------------------------------------------------------------------
# _normalize_dest
# ----------------------------------------------------------------------


def test_normalize_dest_canonicalizes_hex_case():
    """Mixed-case hex must collapse to lowercase so the rate-limit key is
    stable regardless of how the caller spelled the address."""
    a = faucet_bot._normalize_dest("0x" + "AB" * 32)
    b = faucet_bot._normalize_dest("0x" + "ab" * 32)
    assert a == b == "0x" + "ab" * 32


def test_normalize_dest_accepts_0X_prefix():
    out = faucet_bot._normalize_dest("0X" + "01" * 32)
    assert out == "0x" + "01" * 32


def test_normalize_dest_rejects_short_hex():
    """A 32-byte AccountId32 is 64 hex chars; anything else is bogus."""
    assert faucet_bot._normalize_dest("0x" + "ab" * 16) is None


def test_normalize_dest_rejects_non_hex_chars():
    assert faucet_bot._normalize_dest("0x" + "zz" * 32) is None


def test_normalize_dest_rejects_garbage():
    """Pre-fix the helper passed garbage through unchanged, which gave
    each malformed string its own rate-limit slot — trivial throttle
    bypass."""
    assert faucet_bot._normalize_dest("not-an-address") is None
    assert faucet_bot._normalize_dest("") is None


def test_normalize_dest_ss58_matches_hex():
    """SS58 and 0x-hex of the same AccountId must canonicalize to the
    same rate-limit key. Without this a caller alternates representations
    to defeat the per-destination throttle."""
    from scalecodec.utils.ss58 import ss58_encode

    raw = bytes(range(32))
    ss58_form = ss58_encode(raw, ss58_format=42)
    via_ss58 = faucet_bot._normalize_dest(ss58_form)
    via_hex = faucet_bot._normalize_dest("0x" + raw.hex())
    assert via_ss58 == via_hex


# ----------------------------------------------------------------------
# _HYBRID_TERMINAL_FAILURES
# ----------------------------------------------------------------------


def test_hybrid_terminal_failures_covers_all_dead_states():
    """All five terminal `TransactionStatus` variants must be enumerated;
    a missing one re-introduces the silent-hang bug (subscription stays
    open after the chain has dropped/usurped/retracted/finality-timed-out
    the extrinsic)."""
    expected = {"dropped", "invalid", "usurped", "retracted", "finalitytimeout"}
    assert faucet_bot._HYBRID_TERMINAL_FAILURES == expected


def test_hybrid_terminal_failures_matches_shared():
    """Pin the faucet's inline copy against the canonical version in
    `substrate.client`. Drift between the two would re-introduce
    exactly the bug class MR !82 + !83 cleaned up."""
    from substrate.client import _HYBRID_TERMINAL_FAILURES as shared_set
    assert faucet_bot._HYBRID_TERMINAL_FAILURES == shared_set


# ----------------------------------------------------------------------
# _chain_uses_hybrid_signature
# ----------------------------------------------------------------------


def _fake_metadata(types_path_lists, version_key="V14"):
    """Build a mock SubstrateInterface metadata response.

    `types_path_lists` is a list of `path` lists — one per fake type
    entry. The hybrid signature is detected when any path contains
    `"HybridTxSignature"`.
    """
    md = MagicMock()
    md.value = (
        b"meta-magic",
        {
            version_key: {
                "types": {
                    "types": [{"type": {"path": p}} for p in types_path_lists]
                }
            }
        },
    )
    return md


def test_chain_uses_hybrid_signature_detects_v14_hybrid():
    iface = MagicMock()
    iface.get_metadata.return_value = _fake_metadata(
        [["quip_transaction_crypto", "HybridTxSignature"]]
    )
    assert faucet_bot._chain_uses_hybrid_signature(iface) is True


def test_chain_uses_hybrid_signature_detects_v15_hybrid():
    """A chain on V15 metadata must still be detected. The pre-fix
    version hardcoded `["V14"]` and silently fell back to sr25519 on V15
    hybrid chains — every transfer would then fail at submit time."""
    iface = MagicMock()
    iface.get_metadata.return_value = _fake_metadata(
        [["quip_transaction_crypto", "HybridTxSignature"]],
        version_key="V15",
    )
    assert faucet_bot._chain_uses_hybrid_signature(iface) is True


def test_chain_uses_hybrid_signature_returns_false_on_vanilla_chain():
    iface = MagicMock()
    iface.get_metadata.return_value = _fake_metadata(
        [["sp_runtime", "MultiSignature"], ["other", "type"]]
    )
    assert faucet_bot._chain_uses_hybrid_signature(iface) is False


def test_chain_uses_hybrid_signature_raises_on_unrecognized_shape():
    """If we can't find any `types.types` under any version key, fail
    loud rather than silently default to sr25519 on what might be a
    hybrid chain. Defaulting soft here produces opaque downstream
    submission failures."""
    iface = MagicMock()
    md = MagicMock()
    md.value = (b"meta-magic", {"Unknown": {"no": "types"}})
    iface.get_metadata.return_value = md
    with pytest.raises(RuntimeError, match="could not locate"):
        faucet_bot._chain_uses_hybrid_signature(iface)


# ----------------------------------------------------------------------
# _encode_compact_u32 / _encode_compact_u128 mode boundaries
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, b"\x00"),
        (63, b"\xfc"),                                          # last single-byte
        (64, (64 << 2 | 0b01).to_bytes(2, "little")),           # first 2-byte
        (16_383, (16_383 << 2 | 0b01).to_bytes(2, "little")),   # last 2-byte
        (16_384, (16_384 << 2 | 0b10).to_bytes(4, "little")),   # first 4-byte
    ],
)
def test_encode_compact_u32_boundaries(value, expected):
    assert faucet_bot._encode_compact_u32(value) == expected


def test_encode_compact_u32_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        faucet_bot._encode_compact_u32(-1)


def test_encode_compact_u128_big_int_layout():
    """2**32 needs 5 bytes; mode byte = ((5-4) << 2) | 0b11 = 0x07."""
    value = 2 ** 32
    encoded = faucet_bot._encode_compact_u128(value)
    assert encoded[0] == 0x07
    assert int.from_bytes(encoded[1:], "little") == value
    assert len(encoded) == 1 + 5


def test_encode_compact_u128_overflow_message_says_67_byte():
    """The SCALE big-int mode caps at 67 raw bytes. The pre-fix message
    said "64-byte" which mismatched the real limit and confused diagnosis."""
    huge = 2 ** (67 * 8 + 1)
    with pytest.raises(OverflowError, match="67-byte"):
        faucet_bot._encode_compact_u128(huge)


# ----------------------------------------------------------------------
# Hybrid crypto parity with shared.hybrid_signer
# ----------------------------------------------------------------------


def test_hybrid_signer_parity_with_shared():
    """The inline `_HybridSigner` in faucet_bot must produce the same
    public bytes, account ID, and SS58 address as the canonical version
    in `shared.hybrid_signer.HybridSigner` for the same seed. This pins
    the standalone-deploy duplication against silent drift."""
    from shared.hybrid_signer import HybridSigner

    seed = bytes(range(32))
    inline = faucet_bot._HybridSigner(seed)
    canonical = HybridSigner.from_master_seed(seed)

    assert inline.public_bytes() == canonical.public_bytes()
    assert inline.account_id_bytes() == canonical.account_id_bytes()
    assert inline.ss58_address() == canonical.ss58_address()


def test_hybrid_signer_signature_round_trip():
    """The faucet's inline signer signs in a verifiable way (both
    components verify under the same composite pubkey)."""
    from shared.hybrid_signer import HybridSigner

    seed = bytes(range(1, 33))
    inline = faucet_bot._HybridSigner(seed)
    canonical = HybridSigner.from_master_seed(seed)

    sig = inline.sign(b"hello faucet")
    # ML-DSA is randomized so we can't byte-compare signatures; but the
    # canonical verifier must accept the inline signature under the
    # shared composite pubkey.
    assert canonical.verify(b"hello faucet", sig)


# ----------------------------------------------------------------------
# DEV_HYBRID_SEEDS parity with shared.miner_bootstrap
# ----------------------------------------------------------------------


def test_dev_hybrid_seeds_match_shared():
    """The faucet's duplicated `DEV_HYBRID_SEEDS` must stay byte-identical
    to `shared.miner_bootstrap.DEV_HYBRID_SEEDS` — otherwise //Alice maps
    to two different accounts depending on which entry point you went
    through."""
    from shared.miner_bootstrap import DEV_HYBRID_SEEDS as shared_seeds
    assert faucet_bot.DEV_HYBRID_SEEDS == shared_seeds


# ----------------------------------------------------------------------
# Standalone-deploy invariant
# ----------------------------------------------------------------------


def test_faucet_bot_imports_nothing_from_shared():
    """The MR's stated invariant is "no imports from `shared/`". An
    accidental `from shared.X import ...` would silently break the
    standalone deploy contract; this AST scan catches it at test time."""
    import ast

    src = (Path(__file__).parent.parent / "faucet_bot.py").read_text()
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "shared" or mod.startswith("shared."):
                bad.append(f"from {mod} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "shared" or alias.name.startswith("shared."):
                    bad.append(f"import {alias.name}")
    assert not bad, f"faucet_bot.py imports from shared/: {bad}"
