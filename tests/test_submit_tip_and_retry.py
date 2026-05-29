"""Tests for the submission primitives added in Task 5:

  - configurable extrinsic tip threaded through both signing paths,
  - pre-submit liveness probe + reconnect (`ensure_connected`),
  - resilient `submit_with_retry` with an injected sleeper.

All hermetic — no live chain. The substrate iface and the build/pool
clients are faked so the wire-format and control-flow assertions run
without a websocket.
"""
from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

from shared.allowed_value_spec import AllowedValueSet
from shared.miner_types import MiningResult
from substrate.client import (
    _build_hybrid_signed_extrinsic,
    _encode_compact_u128,
)
from substrate.submitter import (
    SubmitRetryAction,
    submit_proof,
    submit_with_retry,
)
from substrate.types import (
    ExtrinsicReceipt,
    SubstrateDifficulty,
    SubstrateMiningContext,
)


_BIN_SPEC = AllowedValueSet((-1000, 1000))
_TER_SPEC = AllowedValueSet((-1000, 0, 1000))


def _make_context(**overrides) -> SubstrateMiningContext:
    defaults = dict(
        last_proof_block_hash=b"\x11" * 32,
        topology_hash=b"\x22" * 32,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        difficulty=SubstrateDifficulty(
            min_solutions=5,
            max_energy_milli=-4_100_000,
            min_diversity_milli=150,
        ),
        miner_account_bytes=b"\x33" * 32,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
        block_hash=b"\x77" * 32,
        block_number=1,
    )
    defaults.update(overrides)
    return SubstrateMiningContext(**defaults)


def _make_result(**overrides) -> MiningResult:
    defaults = dict(
        miner_id="test-miner",
        miner_type="CPU",
        nonce=bytes.fromhex("aa" * 32),
        salt=b"\xab" * 32,
        timestamp=1_700_000_000,
        prev_timestamp=1_700_000_000 - 6,
        solutions=[[0, 1, 0, 1]],
        energy=-4250.5,
        diversity=0.4,
        num_valid=5,
        mining_time=4500,
        node_list=[],
        edge_list=[],
    )
    defaults.update(overrides)
    return MiningResult(**defaults)


# ----------------------------------------------------------------------
# Fake substrate iface for the hybrid wire-format tip test.
# ----------------------------------------------------------------------


class _FakeCallData:
    def __init__(self, raw: bytes) -> None:
        self.data = raw


class _FakeCall:
    def __init__(self, raw: bytes) -> None:
        self.data = _FakeCallData(raw)


class _FakeHybridIface:
    """Minimal iface stand-in for `_build_hybrid_signed_extrinsic`.

    Returns deterministic chain state so the only thing that varies
    between two builds is the tip.
    """

    def __init__(self) -> None:
        self._call_bytes = bytes([0x09, 0x00, 0xde, 0xad, 0xbe, 0xef])

    def compose_call(self, *, call_module, call_function, call_params):
        return _FakeCall(self._call_bytes)

    def get_account_nonce(self, *, account_address):  # noqa: ARG002
        return 7

    def get_block_hash(self, *, block_id):  # noqa: ARG002
        return "0x" + ("01" * 32)

    def rpc_request(self, method, params):  # noqa: ARG002
        if method == "state_getRuntimeVersion":
            return {"result": {"specVersion": 100, "transactionVersion": 2}}
        raise AssertionError(f"unexpected rpc {method}")


class _FakeHybridSigner:
    """Signer with deterministic, length-correct hybrid material."""

    def signature_kind(self) -> str:
        return "Hybrid"

    def account_id_bytes(self) -> bytes:
        return b"\x44" * 32

    def public_bytes(self) -> bytes:
        return b"\x00" * 1344

    def sign(self, message: bytes) -> bytes:  # noqa: ARG002
        return b"\x00" * 2484


def test_hybrid_tip_zero_reproduces_pre_tip_bytes():
    """tip=0 must produce the exact bytes the hardcoded `tip=0` produced."""
    iface = _FakeHybridIface()
    signer = _FakeHybridSigner()
    ext_zero, _ = _build_hybrid_signed_extrinsic(
        iface=iface,
        signer=signer,
        call_module="QuantumPow",
        call_function="submit_proof",
        call_params={},
        tip=0,
    )
    ext_default, _ = _build_hybrid_signed_extrinsic(
        iface=iface,
        signer=signer,
        call_module="QuantumPow",
        call_function="submit_proof",
        call_params={},
    )
    assert ext_zero == ext_default


def test_hybrid_nonzero_tip_changes_bytes_and_embeds_compact_tip():
    """A non-zero tip changes the extrinsic and the tip compact-u128 bytes
    appear in the body's `extra` section (after the compact nonce)."""
    iface = _FakeHybridIface()
    signer = _FakeHybridSigner()
    tip = 1_000
    ext_zero, _ = _build_hybrid_signed_extrinsic(
        iface=iface, signer=signer, call_module="QuantumPow",
        call_function="submit_proof", call_params={}, tip=0,
    )
    ext_tip, _ = _build_hybrid_signed_extrinsic(
        iface=iface, signer=signer, call_module="QuantumPow",
        call_function="submit_proof", call_params={}, tip=tip,
    )
    assert ext_tip != ext_zero
    # The tip is SCALE compact-u128 encoded; those bytes must be present.
    assert _encode_compact_u128(tip) in ext_tip


# ----------------------------------------------------------------------
# Sr25519 path: tip kwarg reaches create_signed_extrinsic.
# ----------------------------------------------------------------------


async def test_sr25519_build_passes_tip_to_create_signed_extrinsic():
    from substrate.client import SubstrateClient

    client = SubstrateClient(url="ws://unused:9944")

    # Fake the iface so no network is touched. create_signed_extrinsic
    # returns an object whose .data.to_hex() yields a sentinel.
    fake_iface = MagicMock()
    fake_iface.compose_call.return_value = MagicMock(name="call")
    ext = MagicMock()
    ext.data.to_hex.return_value = "0xdeadbeef"
    fake_iface.create_signed_extrinsic.return_value = ext
    client._iface = fake_iface
    # Run synchronously in-thread; bypass the executor lock setup.
    client._loop = None

    signer = MagicMock()
    signer.signature_kind.return_value = "Sr25519"
    signer.keypair = MagicMock(name="keypair")

    out = await client.build_signed_extrinsic(
        "QuantumPow", "submit_proof", {"proof": {}}, signer, tip=2_500,
    )
    assert out == "0xdeadbeef"
    _, kwargs = fake_iface.create_signed_extrinsic.call_args
    assert kwargs["tip"] == 2_500


async def test_sr25519_build_default_tip_is_zero():
    from substrate.client import SubstrateClient

    client = SubstrateClient(url="ws://unused:9944")
    fake_iface = MagicMock()
    fake_iface.compose_call.return_value = MagicMock(name="call")
    ext = MagicMock()
    ext.data.to_hex.return_value = "0x00"
    fake_iface.create_signed_extrinsic.return_value = ext
    client._iface = fake_iface
    client._loop = None

    signer = MagicMock()
    signer.signature_kind.return_value = "Sr25519"
    signer.keypair = MagicMock(name="keypair")

    await client.build_signed_extrinsic(
        "QuantumPow", "submit_proof", {"proof": {}}, signer,
    )
    _, kwargs = fake_iface.create_signed_extrinsic.call_args
    assert kwargs["tip"] == 0


def test_build_signed_extrinsic_rejects_negative_tip():
    import asyncio

    from substrate.client import SubstrateClient

    client = SubstrateClient(url="ws://unused:9944")
    signer = MagicMock()
    signer.signature_kind.return_value = "Sr25519"
    with pytest.raises(ValueError, match="non-negative"):
        asyncio.run(
            client.build_signed_extrinsic(
                "QuantumPow", "submit_proof", {}, signer, tip=-1,
            )
        )


# ----------------------------------------------------------------------
# Pre-submit liveness: healthy connection does NOT reconnect; a failing
# probe DOES reconnect before submit.
# ----------------------------------------------------------------------


class _FakeBuildClient:
    """Stands in for SubstrateClient on the submit path."""

    def __init__(self, *, ensure_raises: bool = False) -> None:
        self.ensure_calls = 0
        self.reconnects = 0
        self.build_calls: List[int] = []
        self._ensure_raises = ensure_raises

    async def ensure_connected(self) -> bool:
        self.ensure_calls += 1
        if self._ensure_raises:
            # Mimic the real method: probe fails -> reconnect, return False.
            self.reconnects += 1
            return False
        return True

    async def build_signed_extrinsic(
        self, call_module, call_function, call_params, signer, *, tip=0,
    ):  # noqa: ARG002
        self.build_calls.append(tip)
        return "0xabc"


class _FakePoolClient:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.submit_calls: List[str] = []
        self.receipt = ExtrinsicReceipt(extrinsic_hash="0xhash")

    async def ensure_connected(self) -> bool:
        self.ensure_calls += 1
        return True

    async def submit_signed_extrinsic(self, extrinsic_hex, wait_for="inblock"):  # noqa: ARG002
        self.submit_calls.append(extrinsic_hex)
        return self.receipt


async def test_submit_proof_probes_both_clients_before_submit():
    build = _FakeBuildClient()
    pool = _FakePoolClient()
    receipt = await submit_proof(
        build, pool, MagicMock(), _make_result(), _make_context(), tip=42,
    )
    assert receipt is pool.receipt
    # Liveness probe ran on both clients exactly once.
    assert build.ensure_calls == 1
    assert pool.ensure_calls == 1
    # No reconnect on a healthy probe.
    assert build.reconnects == 0
    # Tip threaded through to the build call.
    assert build.build_calls == [42]


async def test_submit_proof_ensure_live_false_skips_probe():
    build = _FakeBuildClient()
    pool = _FakePoolClient()
    await submit_proof(
        build, pool, MagicMock(), _make_result(), _make_context(),
        ensure_live=False,
    )
    assert build.ensure_calls == 0
    assert pool.ensure_calls == 0


async def test_ensure_connected_reconnects_on_dead_socket(monkeypatch):
    """A failing health probe must call reconnect() before returning."""
    from substrate.client import SubstrateClient

    c = SubstrateClient(url="ws://unused:9944")
    c._iface = MagicMock()
    c._loop = None

    # Make the raw probe raise a connection-class error.
    async def _raw_run(fn):  # noqa: ARG001
        raise ConnectionError("socket dead")

    reconnected = {"n": 0}

    async def _reconnect():
        reconnected["n"] += 1

    monkeypatch.setattr(c, "_raw_run", _raw_run)
    monkeypatch.setattr(c, "reconnect", _reconnect)

    result = await c.ensure_connected()
    assert result is False
    assert reconnected["n"] == 1


async def test_ensure_connected_healthy_does_not_reconnect(monkeypatch):
    from substrate.client import SubstrateClient

    c = SubstrateClient(url="ws://unused:9944")
    c._iface = MagicMock()
    c._loop = None

    async def _raw_run(fn):  # noqa: ARG001
        return {"peers": 1, "isSyncing": False, "shouldHavePeers": True}

    reconnected = {"n": 0}

    async def _reconnect():
        reconnected["n"] += 1

    monkeypatch.setattr(c, "_raw_run", _raw_run)
    monkeypatch.setattr(c, "reconnect", _reconnect)

    result = await c.ensure_connected()
    assert result is True
    assert reconnected["n"] == 0


# ----------------------------------------------------------------------
# submit_with_retry: classification + bounded retry + injected sleeper.
# ----------------------------------------------------------------------


class _ScriptedBuildClient:
    async def ensure_connected(self) -> bool:
        return True

    async def build_signed_extrinsic(
        self, call_module, call_function, call_params, signer, *, tip=0,
    ):  # noqa: ARG002
        return "0xabc"


class _ScriptedPoolClient:
    """Pool client whose submit returns a scripted sequence of
    receipts and/or raises scripted exceptions.

    Each element of `script` is either an ExtrinsicReceipt (returned) or
    an Exception instance (raised).
    """

    def __init__(self, script: List[Any]) -> None:
        self._script = list(script)
        self.attempts = 0

    async def ensure_connected(self) -> bool:
        return True

    async def submit_signed_extrinsic(self, extrinsic_hex, wait_for="inblock"):  # noqa: ARG002
        item = self._script[self.attempts]
        self.attempts += 1
        if isinstance(item, Exception):
            raise item
        return item


def _err_receipt(name: str) -> ExtrinsicReceipt:
    return ExtrinsicReceipt(
        extrinsic_hash="0xhash",
        block_hash="0xblock",
        error=f"Module(error={name}, index=9)",
    )


def _ok_receipt() -> ExtrinsicReceipt:
    return ExtrinsicReceipt(extrinsic_hash="0xhash", block_hash="0xblock")


class _RecordingSleeper:
    def __init__(self) -> None:
        self.delays: List[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


async def _run(pool_script, **kwargs):
    sleeper = _RecordingSleeper()
    result = await submit_with_retry(
        _ScriptedBuildClient(),
        _ScriptedPoolClient(pool_script),
        MagicMock(),
        _make_result(),
        _make_context(),
        sleeper=sleeper,
        **kwargs,
    )
    return result, sleeper


async def test_retry_success_first_try():
    result, sleeper = await _run([_ok_receipt()])
    assert result.action is SubmitRetryAction.SUCCESS
    assert result.attempts == 1
    assert sleeper.delays == []


async def test_retry_on_insufficient_energy_then_success():
    result, sleeper = await _run(
        [_err_receipt("InsufficientEnergy"), _ok_receipt()],
        max_retries=3,
        retry_backoff_ms=10,
    )
    assert result.action is SubmitRetryAction.SUCCESS
    assert result.attempts == 2
    # One backoff between the two attempts.
    assert sleeper.delays == [0.01]


async def test_retry_on_proof_limit_reached_then_success():
    result, _ = await _run(
        [_err_receipt("ProofLimitReached"), _ok_receipt()],
        retry_backoff_ms=0,
    )
    assert result.action is SubmitRetryAction.SUCCESS
    assert result.attempts == 2


async def test_retry_on_transient_exception_then_success():
    result, _ = await _run(
        [BrokenPipeError("idle ws dropped"), _ok_receipt()],
        retry_backoff_ms=0,
    )
    assert result.action is SubmitRetryAction.SUCCESS
    assert result.attempts == 2


async def test_stops_on_invalid_nonce_round_stale():
    result, sleeper = await _run([_err_receipt("InvalidNonce")])
    assert result.action is SubmitRetryAction.STOP_ROUND_STALE
    assert result.attempts == 1
    # No retry on a round-stale stop.
    assert sleeper.delays == []


async def test_stops_on_insufficient_solutions_fatal():
    result, _ = await _run([_err_receipt("InsufficientSolutions")])
    assert result.action is SubmitRetryAction.STOP_FATAL
    assert result.attempts == 1


async def test_stops_on_insufficient_diversity_fatal():
    result, _ = await _run([_err_receipt("InsufficientDiversity")])
    assert result.action is SubmitRetryAction.STOP_FATAL
    assert result.attempts == 1


async def test_topology_not_registered_is_round_stale():
    result, _ = await _run([_err_receipt("TopologyNotRegistered")])
    assert result.action is SubmitRetryAction.STOP_ROUND_STALE


async def test_invalid_topology_is_round_stale():
    result, _ = await _run([_err_receipt("InvalidTopology")])
    assert result.action is SubmitRetryAction.STOP_ROUND_STALE


async def test_unknown_error_is_fatal_not_retried():
    # Guards against a future infinite-retry regression: an unrecognized
    # pallet error must STOP_FATAL, never loop. Asserting attempts==1 pins
    # that the loop did not retry it.
    result, sleeper = await _run(
        [_err_receipt("SomeBrandNewPalletError")], max_retries=5
    )
    assert result.action is SubmitRetryAction.STOP_FATAL
    assert result.attempts == 1
    assert sleeper.delays == []


async def test_retry_on_validator_swapped_then_success():
    # ValidatorSwapped means the pool hot-swapped validators mid-submit.
    # Because submit_proof re-signs from scratch each attempt (fresh nonce
    # against the new validator), submit_with_retry treats it as a
    # retryable transient — it must NOT propagate, and the next attempt
    # succeeds.
    from substrate.validator_handle import ValidatorSwapped

    result, _ = await _run(
        [ValidatorSwapped("pool swapped mid-submit"), _ok_receipt()],
        retry_backoff_ms=0,
    )
    assert result.action is SubmitRetryAction.SUCCESS
    assert result.attempts == 2


async def test_validator_swapped_does_not_escape_when_persistent():
    # A persistent swap exhausts retries and returns RETRY — never raises
    # ValidatorSwapped out of submit_with_retry.
    from substrate.validator_handle import ValidatorSwapped

    result, _ = await _run(
        [ValidatorSwapped("swap"), ValidatorSwapped("swap")],
        max_retries=1,
        retry_backoff_ms=0,
    )
    assert result.action is SubmitRetryAction.RETRY
    assert result.attempts == 2


async def test_respects_max_retries_then_returns_retry():
    # Always InsufficientEnergy; max_retries=2 -> 3 attempts total, then
    # gives up with action=RETRY (caller decides to re-mine).
    script = [_err_receipt("InsufficientEnergy")] * 3
    result, sleeper = await _run(script, max_retries=2, retry_backoff_ms=5)
    assert result.action is SubmitRetryAction.RETRY
    assert result.attempts == 3
    # Two backoffs between three attempts.
    assert sleeper.delays == [0.005, 0.01]


async def test_never_raises_on_single_transient_failure():
    # A single transient failure with max_retries=0 must return RETRY,
    # not raise.
    result, _ = await _run([BrokenPipeError("boom")], max_retries=0)
    assert result.action is SubmitRetryAction.RETRY
    assert result.attempts == 1


async def test_rejects_negative_max_retries():
    with pytest.raises(ValueError, match="non-negative"):
        await submit_with_retry(
            _ScriptedBuildClient(),
            _ScriptedPoolClient([_ok_receipt()]),
            MagicMock(),
            _make_result(),
            _make_context(),
            max_retries=-1,
        )
