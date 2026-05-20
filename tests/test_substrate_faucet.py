"""Integration tests for the standalone `faucet_bot.py`.

Requires the docker chain at `ws://localhost:9944`. Brings the faucet up on
an ephemeral port, posts a funding request from a fresh test key, and
asserts the destination balance reflects the transfer.
"""
from __future__ import annotations

import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiohttp
import pytest

# faucet_bot.py is a standalone script at repo root, not a package member.
# Add the repo root to sys.path so we can import it under test.
sys.path.insert(0, str(Path(__file__).parent.parent))
import faucet_bot  # noqa: E402

from shared.signer import Sr25519Signer  # noqa: E402
from shared.substrate_client import SubstrateClient  # noqa: E402
from substrateinterface import SubstrateInterface  # noqa: E402


DEFAULT_URL = os.environ.get("QUIP_SUBSTRATE_URL", "ws://localhost:9944")


def _chain_reachable(url: str) -> bool:
    bare = url.split("://", 1)[1]
    host, _, port_str = bare.partition(":")
    port = int(port_str) if port_str else 9944
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, socket.timeout):
        return False


def _chain_requires_hybrid_signer(url: str) -> bool:
    """Reuse faucet_bot's own detector against a live chain.

    Used only to pick the right test signer type for the balance assertion
    (a fresh hybrid signer on hybrid chains, vs `Sr25519Signer` on vanilla
    chains). Delegates to `faucet_bot._chain_uses_hybrid_signature` so the
    test and the bot stay in lockstep — when the bot's metadata-version
    fallback ships, this helper inherits it for free.
    """
    if not _chain_reachable(url):
        return False
    iface = SubstrateInterface(url=url)
    try:
        return faucet_bot._chain_uses_hybrid_signature(iface)
    except Exception:
        return False
    finally:
        try:
            iface.close()
        except Exception:
            pass


pytestmark = [
    pytest.mark.skipif(
        not _chain_reachable(DEFAULT_URL),
        reason=f"substrate chain not reachable at {DEFAULT_URL}",
    ),
]


def _fresh_test_account_id_bytes() -> bytes:
    """Mint a fresh AccountId for the destination of a faucet transfer.

    On hybrid chains the AccountId is derived from the *composite* pubkey
    via `blake2_256(domain || pubkey)` — `Sr25519Signer.account_id_bytes`
    won't match because the chain expects the hybrid derivation. We just
    need any unused 32-byte AccountId for the balance check; producing one
    from a random hybrid pubkey works on both chain types.
    """
    if _chain_requires_hybrid_signer(DEFAULT_URL):
        signer = faucet_bot._HybridSigner(os.urandom(faucet_bot.MASTER_SEED_LEN))
        return signer.account_id_bytes()
    return Sr25519Signer.from_seed(os.urandom(32)).account_id_bytes()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@asynccontextmanager
async def _running_faucet(port: int) -> AsyncIterator["faucet_bot.SubstrateFaucet"]:
    bot = faucet_bot.SubstrateFaucet(
        faucet_bot.FaucetConfig(
            node_url=DEFAULT_URL,
            faucet_key_uri="//Alice",
            listen_host="127.0.0.1",
            listen_port=port,
            rate_limit_seconds=0.0,  # disabled for tests
        )
    )
    await bot.start()
    try:
        yield bot
    finally:
        await bot.stop()


async def test_faucet_funds_fresh_account():
    port = _free_port()
    # Fresh per-test destination so the balance check is unambiguous. The
    # chain-mode-appropriate derivation is picked inside the helper so this
    # test exercises both signing paths.
    dest_bytes = _fresh_test_account_id_bytes()
    dest_hex = "0x" + dest_bytes.hex()
    amount = 1_000_000_000_000  # 1 UNIT

    async with _running_faucet(port):
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"http://127.0.0.1:{port}/request",
                json={"dest": dest_hex, "amount": amount},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                payload = await resp.json()
                assert resp.status == 200, f"faucet returned {resp.status}: {payload}"
                assert payload["amount"] == amount

    client = SubstrateClient(url=DEFAULT_URL)
    await client.connect()
    try:
        balance = await client.query_balance(dest_bytes)
        assert balance == amount, f"expected balance {amount}, got {balance}"
    finally:
        await client.close()


async def test_faucet_rejects_bad_request():
    port = _free_port()
    config = faucet_bot.FaucetConfig(
        node_url=DEFAULT_URL,
        listen_host="127.0.0.1",
        listen_port=port,
        rate_limit_seconds=0.0,
    )
    bot = faucet_bot.SubstrateFaucet(config)
    await bot.start()
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"http://127.0.0.1:{port}/request",
                data="not-json",
                headers={"Content-Type": "application/json"},
            ) as resp:
                assert resp.status == 400

            async with http.post(
                f"http://127.0.0.1:{port}/request",
                json={"amount": 100},  # missing dest
            ) as resp:
                assert resp.status == 400

            async with http.post(
                f"http://127.0.0.1:{port}/request",
                json={"dest": "0x" + "00" * 32, "amount": -1},
            ) as resp:
                assert resp.status == 400
    finally:
        await bot.stop()


async def test_faucet_rate_limits():
    port = _free_port()
    config = faucet_bot.FaucetConfig(
        node_url=DEFAULT_URL,
        listen_host="127.0.0.1",
        listen_port=port,
        rate_limit_seconds=60.0,
    )
    bot = faucet_bot.SubstrateFaucet(config)
    await bot.start()
    try:
        dest_hex = "0x" + os.urandom(32).hex()
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"http://127.0.0.1:{port}/request",
                json={"dest": dest_hex, "amount": 1_000_000_000_000},
            ) as resp:
                assert resp.status == 200

            async with http.post(
                f"http://127.0.0.1:{port}/request",
                json={"dest": dest_hex, "amount": 1_000_000_000_000},
            ) as resp:
                assert resp.status == 429
                payload = await resp.json()
                assert "retry_after_seconds" in payload
    finally:
        await bot.stop()


async def test_faucet_health_endpoint():
    port = _free_port()
    async with _running_faucet(port):
        async with aiohttp.ClientSession() as http:
            async with http.get(f"http://127.0.0.1:{port}/health") as resp:
                assert resp.status == 200
                assert (await resp.json()) == {"status": "ok"}
