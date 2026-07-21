"""Unit tests for `_fetch_extrinsic_dispatch_error`.

The receipt-success path on the hybrid signing flow trusts whatever this
helper returns: `None` → success, any string → non-success. The function
must therefore (a) normalise extrinsic_hash across bytes/str shapes,
(b) accept `phase` in several variant forms, and (c) refuse to silently
return None when it cannot locate the target extrinsic in the block.
"""
from __future__ import annotations

from substrate.client import _fetch_extrinsic_dispatch_error


# ----------------------------------------------------------------------
# Stand-in `SubstrateInterface` — just the two methods we need.
# ----------------------------------------------------------------------


class _FakeMetadata:
    """metadata.get_module_error stand-in: pallet 9 error 1 is the
    SolverNotRegistered case observed live."""

    def get_module_error(self, module_index, error_index):
        from types import SimpleNamespace

        if module_index == 9 and error_index == 1:
            return SimpleNamespace(name="SolverNotRegistered")
        raise ValueError(f"unknown error {module_index}/{error_index}")


class _FakeIface:
    def __init__(self, block, events):
        self._block = block
        self._events = events
        self.metadata = _FakeMetadata()

    def get_block(self, **kwargs):  # noqa: ARG002
        return self._block

    def get_events(self, **kwargs):  # noqa: ARG002
        return self._events


def _block_with_extrinsic(ext_hash_value) -> dict:
    """One-extrinsic block where the extrinsic_hash field matches the
    given value (the helper must canonicalise across shapes)."""
    return {"extrinsics": [{"extrinsic_hash": ext_hash_value}]}


# ----------------------------------------------------------------------
# Hash normalisation
# ----------------------------------------------------------------------


def test_dispatch_error_matches_bytes_hash():
    """A target hash given as a hex string must match an extrinsic
    whose `extrinsic_hash` came back as raw bytes."""
    raw = bytes.fromhex("aa" * 32)
    block = _block_with_extrinsic(raw)
    iface = _FakeIface(block, events=[])
    assert (
        _fetch_extrinsic_dispatch_error(
            iface, block_hash="0x" + "cc" * 32, ext_hash="0x" + "aa" * 32
        )
        is None
    )


def test_dispatch_error_matches_uppercase_hex_hash():
    block = _block_with_extrinsic("0x" + "AB" * 32)
    iface = _FakeIface(block, events=[])
    assert (
        _fetch_extrinsic_dispatch_error(
            iface, block_hash="0x" + "cc" * 32, ext_hash="ab" * 32
        )
        is None
    )


# ----------------------------------------------------------------------
# Phase shapes
# ----------------------------------------------------------------------


def test_dispatch_error_apply_extrinsic_int_phase():
    """`phase: {"ApplyExtrinsic": 0}` — the legacy form."""
    ext_hash = "0x" + "aa" * 32
    block = _block_with_extrinsic(ext_hash)
    events = [
        {
            "phase": {"ApplyExtrinsic": 0},
            "event": {
                "module_id": "System",
                "event_id": "ExtrinsicFailed",
                "attributes": {"dispatch_error": "BadOrigin"},
            },
        }
    ]
    iface = _FakeIface(block, events)
    err = _fetch_extrinsic_dispatch_error(
        iface, block_hash="0x" + "cc" * 32, ext_hash=ext_hash
    )
    assert err is not None
    assert "ExtrinsicFailed" in err


def test_dispatch_error_apply_extrinsic_dict_phase():
    """`phase: {"ApplyExtrinsic": {"extrinsic_idx": 0}}` — typed variant."""
    ext_hash = "0x" + "aa" * 32
    block = _block_with_extrinsic(ext_hash)
    events = [
        {
            "phase": {"ApplyExtrinsic": {"extrinsic_idx": 0}},
            "event": {
                "module_id": "System",
                "event_id": "ExtrinsicFailed",
                "attributes": {},
            },
        }
    ]
    iface = _FakeIface(block, events)
    err = _fetch_extrinsic_dispatch_error(
        iface, block_hash="0x" + "cc" * 32, ext_hash=ext_hash
    )
    assert err is not None
    assert "ExtrinsicFailed" in err


def test_dispatch_error_bare_string_phase_with_sibling_extrinsic_idx():
    """`phase: 'ApplyExtrinsic'` + top-level `extrinsic_idx` — the shape
    the current substrate-interface event decoder actually produces.

    Found live by T9: every event's phase parsed to None, so a real
    in-block ExtrinsicFailed (SolverNotRegistered) was never attributed
    to the extrinsic and the receipt reported success — a false OK on
    EVERY failed hybrid extrinsic. The pow path was shielded by its
    chain-state verify; the mempool submit/claim classification was not.
    """
    ext_hash = "0x" + "aa" * 32
    block = {"extrinsics": [
        {"extrinsic_hash": "0x" + "bb" * 32},  # timestamp inherent
        {"extrinsic_hash": ext_hash},
    ]}
    events = [
        {
            "phase": "ApplyExtrinsic",
            "extrinsic_idx": 0,
            "event": {
                "module_id": "System",
                "event_id": "ExtrinsicSuccess",
                "attributes": {},
            },
        },
        {
            "phase": "ApplyExtrinsic",
            "extrinsic_idx": 1,
            "event": {
                "module_id": "System",
                "event_id": "ExtrinsicFailed",
                "attributes": {
                    "dispatch_error": {"Module": {"index": 9, "error": "0x01000000"}}
                },
            },
        },
    ]
    iface = _FakeIface(block, events)
    err = _fetch_extrinsic_dispatch_error(
        iface, block_hash="0x" + "cc" * 32, ext_hash=ext_hash
    )
    assert err is not None
    # The module error must be decoded to its NAME: the submit/claim
    # classifiers match error-name substrings (SolverNotRegistered is
    # mempool-fatal; OrderNotOpen is merely stale) — raw indexes would
    # misclassify every stale receipt as mempool-fatal.
    assert "SolverNotRegistered" in err


def test_dispatch_error_sibling_idx_success_short_circuits_none():
    """Same decoder shape, success case: ExtrinsicSuccess at OUR index
    (not the inherent's) returns None."""
    ext_hash = "0x" + "aa" * 32
    block = {"extrinsics": [
        {"extrinsic_hash": "0x" + "bb" * 32},
        {"extrinsic_hash": ext_hash},
    ]}
    events = [
        {
            "phase": "ApplyExtrinsic",
            "extrinsic_idx": 1,
            "event": {
                "module_id": "System",
                "event_id": "ExtrinsicSuccess",
                "attributes": {},
            },
        },
    ]
    iface = _FakeIface(block, events)
    assert (
        _fetch_extrinsic_dispatch_error(
            iface, block_hash="0x" + "cc" * 32, ext_hash=ext_hash
        )
        is None
    )


def test_dispatch_error_sibling_idx_other_extrinsic_failure_not_attributed():
    """A sibling extrinsic's failure must not be pinned on ours."""
    ext_hash = "0x" + "aa" * 32
    block = {"extrinsics": [
        {"extrinsic_hash": "0x" + "bb" * 32},
        {"extrinsic_hash": ext_hash},
    ]}
    events = [
        {
            "phase": "ApplyExtrinsic",
            "extrinsic_idx": 0,
            "event": {
                "module_id": "System",
                "event_id": "ExtrinsicFailed",
                "attributes": {"dispatch_error": "BadOrigin"},
            },
        },
        {
            "phase": "ApplyExtrinsic",
            "extrinsic_idx": 1,
            "event": {
                "module_id": "System",
                "event_id": "ExtrinsicSuccess",
                "attributes": {},
            },
        },
    ]
    iface = _FakeIface(block, events)
    assert (
        _fetch_extrinsic_dispatch_error(
            iface, block_hash="0x" + "cc" * 32, ext_hash=ext_hash
        )
        is None
    )


def test_dispatch_error_pallet_name_keys():
    """Newer substrate-interface uses `pallet_name` / `event_name`."""
    ext_hash = "0x" + "aa" * 32
    block = _block_with_extrinsic(ext_hash)
    events = [
        {
            "phase": {"ApplyExtrinsic": 0},
            "event": {
                "pallet_name": "System",
                "event_name": "ExtrinsicFailed",
                "fields": {},
            },
        }
    ]
    iface = _FakeIface(block, events)
    err = _fetch_extrinsic_dispatch_error(
        iface, block_hash="0x" + "cc" * 32, ext_hash=ext_hash
    )
    assert err is not None
    assert "ExtrinsicFailed" in err


# ----------------------------------------------------------------------
# Success / not-found semantics
# ----------------------------------------------------------------------


def test_dispatch_error_extrinsic_success_returns_none():
    ext_hash = "0x" + "aa" * 32
    block = _block_with_extrinsic(ext_hash)
    events = [
        {
            "phase": {"ApplyExtrinsic": 0},
            "event": {"module_id": "System", "event_id": "ExtrinsicSuccess"},
        }
    ]
    iface = _FakeIface(block, events)
    assert (
        _fetch_extrinsic_dispatch_error(
            iface, block_hash="0x" + "cc" * 32, ext_hash=ext_hash
        )
        is None
    )


def test_dispatch_error_extrinsic_not_in_block_returns_unclassified():
    """Used to return None silently — that masked acceptance failures.
    Must now return a non-empty string so the receipt is treated as
    non-success and the work_key is NOT closed."""
    block = {"extrinsics": [{"extrinsic_hash": "0x" + "bb" * 32}]}
    iface = _FakeIface(block, events=[])
    err = _fetch_extrinsic_dispatch_error(
        iface, block_hash="0x" + "cc" * 32, ext_hash="0x" + "aa" * 32
    )
    assert err is not None
    assert "not found" in err


def test_dispatch_error_get_block_none_returns_unclassified():
    """If the chain returns no block for the hash (extreme reorg / RPC
    blip), don't claim success — return a diagnostic string."""
    iface = _FakeIface(block=None, events=[])
    err = _fetch_extrinsic_dispatch_error(
        iface, block_hash="0x" + "cc" * 32, ext_hash="0x" + "aa" * 32
    )
    assert err is not None
    assert "get_block returned None" in err


# ----------------------------------------------------------------------
# Decode-error tolerance — see SHALLOW IFACE POLICY in substrate_client.py
# ----------------------------------------------------------------------


class _DecodingFailureIface:
    """Simulates substrate-interface raising NotImplementedError on an
    unknown SCALE type even when `ignore_decoding_errors=True` was
    passed — happens in some versions when the failure is in outer-frame
    decoding rather than inner field decoding."""

    def get_block(self, **kwargs):  # noqa: ARG002
        raise NotImplementedError("No decoding class found for 'DigestItem'")

    def get_events(self, **kwargs):  # noqa: ARG002
        return []


def test_dispatch_error_survives_get_block_decode_crash():
    """A DigestItem-shaped NotImplementedError from get_block must not
    crash into the controller — it surfaces as a non-None unclassified
    string so the receipt is treated as non-success and the work_key
    stays open for retry."""
    iface = _DecodingFailureIface()
    err = _fetch_extrinsic_dispatch_error(
        iface, block_hash="0x" + "cc" * 32, ext_hash="0x" + "aa" * 32
    )
    assert err is not None
    assert "decode failure" in err
    assert "DigestItem" in err
