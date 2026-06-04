"""Unit tests for Phase 6 (live-miner verification) production code.

These mirror the bugs the new live integration tests caught — but as
fast unit tests so a regression fails in CI without needing the docker
chain. Live integration tests still live in
`test_substrate_miner_controller.py::test_controller_long_haul_multi_block`
and `test_telemetry_live_miner.py`; this file pins the unit-level
contracts those tests exercise end-to-end.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock


from substrate.miner_bootstrap import (
    BootstrapConfig,
    DEFAULT_SEED_DIFFICULTY,
    _maybe_seed_chain,
)
from substrate.client import SubstrateClient


# ----------------------------------------------------------------------
# SubstrateClient._call_lock — proves the lock actually serialises
# ----------------------------------------------------------------------


async def test_call_lock_serialises_concurrent_run():
    """Three concurrent `_run` calls execute one-at-a-time under the lock.

    Without serialisation, substrate-interface 1.8.1 corrupts mid-decode
    (this MR's whole motivation). Pinned via wallclock: with the lock
    three 50ms calls take ~150ms; without it they'd run in parallel
    and finish in ~50ms.
    """
    client = SubstrateClient(url="ws://nowhere")
    client._call_lock = asyncio.Lock()
    client._loop = asyncio.get_running_loop()

    def _slow_fn() -> int:
        time.sleep(0.05)
        return 0

    start = time.monotonic()
    await asyncio.gather(
        client._run(_slow_fn),
        client._run(_slow_fn),
        client._run(_slow_fn),
    )
    elapsed = time.monotonic() - start
    # Serial → ~0.15s. Parallel → ~0.05s. 0.12s margin accommodates
    # async/executor scheduler jitter on a busy CI runner.
    assert elapsed >= 0.12, (
        f"three serial 50ms _run calls took only {elapsed:.3f}s — "
        "lock is not being honoured"
    )


# ----------------------------------------------------------------------
# force_reseed_difficulty — pin the conditional that the long-haul
# integration test relies on
# ----------------------------------------------------------------------


async def test_force_reseed_difficulty_overrides_idempotency(monkeypatch):
    """`force_reseed_difficulty=True` overwrites an existing on-chain
    difficulty via sudo, even though the idempotent path would normally
    skip the call. Production CLI keeps the flag off; long-haul tests
    flip it so the runtime's between-proof tightening can be reset."""
    fake_client = MagicMock(spec=SubstrateClient)
    fake_client.query_difficulty = AsyncMock(return_value=DEFAULT_SEED_DIFFICULTY)
    fake_client.get_mining_snapshot = AsyncMock(return_value=MagicMock())

    # Capture sudo calls instead of actually submitting to a chain.
    sudo_calls: list = []

    async def fake_sudo_call(client, signer, module, function, params):
        sudo_calls.append((module, function))

    monkeypatch.setattr(
        "substrate.miner_bootstrap._sudo_call", fake_sudo_call
    )
    # `_assert_dev_chain` calls into substrate-interface; replace with a no-op.
    monkeypatch.setattr(
        "substrate.miner_bootstrap._assert_dev_chain", AsyncMock()
    )
    # `_resolve_dev_signer` is fine but builds a real signer; stub it
    # so the test doesn't depend on DEV_HYBRID_SEEDS layout.
    monkeypatch.setattr(
        "substrate.miner_bootstrap._resolve_dev_signer", lambda uri: MagicMock()
    )

    config = BootstrapConfig(
        validators=("ws://nowhere",),
        signer_key_path="/tmp/never-read",  # noqa: S108
        seed_chain=True,
        force_reseed_difficulty=True,
    )
    await _maybe_seed_chain(fake_client, config)

    # set_difficulty must have been called despite query_difficulty
    # returning a non-None value (the idempotency check is overridden).
    set_difficulty_calls = [c for c in sudo_calls if c == ("QuantumPow", "set_difficulty")]
    assert len(set_difficulty_calls) == 1


async def test_force_reseed_difficulty_default_is_idempotent(monkeypatch):
    """`force_reseed_difficulty=False` (the default) does NOT call sudo
    set_difficulty when the chain already has one. Production callers
    rely on this; flipping the default would mass-overwrite difficulty
    on every CLI invocation."""
    fake_client = MagicMock(spec=SubstrateClient)
    fake_client.query_difficulty = AsyncMock(return_value=DEFAULT_SEED_DIFFICULTY)
    fake_client.get_mining_snapshot = AsyncMock(return_value=MagicMock())

    sudo_calls: list = []

    async def fake_sudo_call(client, signer, module, function, params):
        sudo_calls.append((module, function))

    monkeypatch.setattr("substrate.miner_bootstrap._sudo_call", fake_sudo_call)
    monkeypatch.setattr("substrate.miner_bootstrap._assert_dev_chain", AsyncMock())
    monkeypatch.setattr(
        "substrate.miner_bootstrap._resolve_dev_signer", lambda uri: MagicMock()
    )

    config = BootstrapConfig(
        validators=("ws://nowhere",),
        signer_key_path="/tmp/never-read",  # noqa: S108
        seed_chain=True,
        # force_reseed_difficulty defaults False
    )
    await _maybe_seed_chain(fake_client, config)

    set_difficulty_calls = [c for c in sudo_calls if c == ("QuantumPow", "set_difficulty")]
    assert set_difficulty_calls == []
