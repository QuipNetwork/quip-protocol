"""Unit tests for `faucet_bot._normalize_dest`.

The rate-limit dict is keyed by `_normalize_dest(dest)`. If the function
fails to canonicalize equivalent SS58 / hex representations of the same
account, a caller can alternate forms to bypass the per-destination
throttle and drain the funding key. These tests pin the canonicalization.
"""
from __future__ import annotations

import sys
from pathlib import Path

# faucet_bot.py is a standalone script at the repo root, not a package
# member. Add the repo root to sys.path so we can import it under test.
sys.path.insert(0, str(Path(__file__).parent.parent))
import faucet_bot  # noqa: E402


# //Alice's well-known SS58 + matching public-key hex on prefix 42.
ALICE_SS58 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
ALICE_HEX = "0xd43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d"


def test_hex_is_lowercased():
    assert faucet_bot._normalize_dest("0xAB" + "cd" * 31) == "0xab" + "cd" * 31


def test_hex_uppercase_prefix_is_lowercased():
    assert faucet_bot._normalize_dest("0XAB" + "cd" * 31) == "0xab" + "cd" * 31


def test_ss58_and_hex_canonicalize_to_same_key():
    # The whole point: rate-limit slot must be the same for both
    # representations of one account.
    assert faucet_bot._normalize_dest(ALICE_SS58) == faucet_bot._normalize_dest(ALICE_HEX)
    assert faucet_bot._normalize_dest(ALICE_SS58) == ALICE_HEX


def test_malformed_dest_is_rejected():
    """Garbage input now returns None so `_handle_faucet` can reject it
    with HTTP 400 before keying the rate-limit table on it. The original
    pass-through behaviour let a caller alternate garbage strings to
    bypass the per-destination throttle entirely (each distinct
    malformed string was its own slot)."""
    assert faucet_bot._normalize_dest("not-an-address") is None
    assert faucet_bot._normalize_dest("") is None
    assert faucet_bot._normalize_dest("0xtoo-short") is None
