#!/usr/bin/env python3
"""Standalone dev-only faucet for substrate chains.

Listens on an HTTP endpoint and submits `Balances.transfer_keep_alive` from a
single funded account (typically `//Alice` on a dev chain) to whichever
destination is requested. Rate-limits per destination so a misbehaving caller
can't drain the funding key.

Designed to be deployable independently of the rest of `quip-protocol`:
copy this single file plus `substrate-interface` and `aiohttp` and it runs.
No imports from `shared/` and no other quip modules.

**Never run against a production chain.** The startup check refuses to bind
unless the connected chain's `System.chain` matches a known dev name. Pass
`--allow-any-chain` or set `QUIP_FAUCET_ALLOW_ANY_CHAIN=1` to override —
only do that in deliberately controlled environments.

Usage:
    python faucet_bot.py --node-url ws://localhost:9944 \\
        --faucet-key //Alice --listen 127.0.0.1 --port 8087

Funding request:
    POST /faucet  {"dest": "0x<32-byte hex>", "amount": <plancks>}
    Response 200: {"extrinsic_hash": "0x...", "block_hash": "0x...",
                   "amount": N, "dest": "..."}
    Response 429: rate limited (includes retry_after_seconds)
    Response 4xx/5xx: error detail in the JSON body
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

from aiohttp import web
from substrateinterface import Keypair, KeypairType, SubstrateInterface


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dev chain names matched by prefix. `--chain=local3` reports
# "Local Testnet (3 Validators)" so a prefix match keeps the list short.
DEV_CHAIN_PREFIXES = (
    "Development",
    "Local Testnet",
    "quip-local",
)

# One funding per minute per destination address. Keeps a misbehaving caller
# from draining the funding key, but doesn't get in the way of legitimate
# bootstrap workflows that run a couple of times back-to-back.
DEFAULT_RATE_LIMIT_SECONDS = 60.0

# Default funding amount: 1000 UNIT assuming 12-decimal chains (matches
# standard Substrate dev presets). Override per request.
DEFAULT_AMOUNT_PLANCKS = 1_000_000_000_000_000


logger = logging.getLogger("faucet_bot")


@dataclass
class FaucetConfig:
    node_url: str
    faucet_key_uri: str = "//Alice"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8087
    rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS
    allow_any_chain: bool = False


# ---------------------------------------------------------------------------
# Faucet service
# ---------------------------------------------------------------------------


class SubstrateFaucet:
    """HTTP service that signs `Balances.transfer_keep_alive` extrinsics."""

    def __init__(self, config: FaucetConfig) -> None:
        self.config = config
        self._iface: Optional[SubstrateInterface] = None
        self._keypair = Keypair.create_from_uri(
            config.faucet_key_uri, crypto_type=KeypairType.SR25519
        )
        self._last_funded: Dict[str, float] = {}
        self._rate_limit_lock = asyncio.Lock()
        # `SubstrateInterface` keeps a single ws connection and isn't safe
        # against concurrent calls. The faucet serializes every chain
        # interaction (compose / sign / submit) behind this lock. For a dev
        # faucet the loss of parallelism is fine, and it prevents the
        # Broken-pipe and torn-JSON failure modes that show up under load.
        self._chain_lock = asyncio.Lock()
        self._runner: Optional[web.AppRunner] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._iface = await self._run(
            lambda: SubstrateInterface(url=self.config.node_url)
        )
        await self._verify_dev_chain()

        app = web.Application()
        app.router.add_post("/faucet", self._handle_faucet)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(
            self._runner,
            host=self.config.listen_host,
            port=self.config.listen_port,
        )
        await site.start()
        logger.info(
            "faucet listening: http://%s:%d funder=%s",
            self.config.listen_host,
            self.config.listen_port,
            self._keypair.ss58_address,
        )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._iface is not None:
            iface = self._iface
            self._iface = None
            def _close() -> None:
                try:
                    iface.close()
                except AttributeError:
                    pass  # older substrate-interface versions
            await self._run(_close)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_faucet(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except ValueError:
            return web.json_response({"error": "invalid json"}, status=400)

        dest = body.get("dest")
        if not isinstance(dest, str) or not dest:
            return web.json_response(
                {"error": "missing or invalid 'dest' (ss58 or 0x-hex AccountId)"},
                status=400,
            )

        amount = body.get("amount", DEFAULT_AMOUNT_PLANCKS)
        if not isinstance(amount, int) or amount <= 0:
            return web.json_response(
                {"error": "missing or invalid 'amount' (positive integer plancks)"},
                status=400,
            )

        normalized_dest = _normalize_dest(dest)

        async with self._rate_limit_lock:
            now = time.monotonic()
            last = self._last_funded.get(normalized_dest, 0.0)
            wait = self.config.rate_limit_seconds - (now - last)
            if wait > 0:
                return web.json_response(
                    {
                        "error": "rate limited",
                        "retry_after_seconds": round(wait, 1),
                    },
                    status=429,
                )
            self._last_funded[normalized_dest] = now

        logger.info(
            "funding %s with %d plancks from %s",
            normalized_dest,
            amount,
            self._keypair.ss58_address,
        )
        try:
            async with self._chain_lock:
                receipt = await self._run(
                    lambda: self._submit_transfer(dest=dest, amount=amount)
                )
        except Exception as exc:  # noqa: BLE001 — surface any RPC error to caller
            logger.exception(
                "faucet transfer failed: dest=%s amount=%d", dest, amount
            )
            return web.json_response({"error": str(exc)}, status=502)

        if not getattr(receipt, "is_success", True):
            return web.json_response(
                {
                    "error": str(getattr(receipt, "error_message", "unknown")),
                    "extrinsic_hash": str(getattr(receipt, "extrinsic_hash", "")),
                },
                status=502,
            )

        return web.json_response(
            {
                "extrinsic_hash": str(getattr(receipt, "extrinsic_hash", "")),
                "block_hash": str(getattr(receipt, "block_hash", "") or "") or None,
                "amount": amount,
                "dest": dest,
            }
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _submit_transfer(self, *, dest: str, amount: int):
        # Idle websocket connections to the substrate node go stale after
        # ~minutes and the first send fails with BrokenPipeError. Retry once
        # by tearing down the interface and reconnecting before giving up.
        for attempt in (1, 2):
            try:
                call = self._iface.compose_call(
                    call_module="Balances",
                    call_function="transfer_keep_alive",
                    call_params={"dest": dest, "value": amount},
                )
                extrinsic = self._iface.create_signed_extrinsic(
                    call=call, keypair=self._keypair
                )
                return self._iface.submit_extrinsic(
                    extrinsic, wait_for_inclusion=True
                )
            except (BrokenPipeError, ConnectionError) as exc:
                if attempt == 2:
                    raise
                logger.warning(
                    "substrate websocket dead (%s); reconnecting and retrying",
                    type(exc).__name__,
                )
                self._reconnect_iface()
        raise RuntimeError("unreachable: retry loop exhausted")

    def _reconnect_iface(self) -> None:
        old = self._iface
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001 — best-effort tear-down
                pass
        self._iface = SubstrateInterface(url=self.config.node_url)

    async def _verify_dev_chain(self) -> None:
        if self.config.allow_any_chain or os.environ.get(
            "QUIP_FAUCET_ALLOW_ANY_CHAIN"
        ) == "1":
            logger.warning(
                "faucet running against non-dev chain because allow_any_chain=true; "
                "this is unsafe outside controlled environments"
            )
            return

        chain_name = await self._run(lambda: self._iface.chain)
        if not any(chain_name.startswith(p) for p in DEV_CHAIN_PREFIXES):
            raise RuntimeError(
                f"refusing to run faucet against chain {chain_name!r}; pass "
                "--allow-any-chain only if you really mean to fund accounts "
                "on a non-dev chain"
            )
        logger.info("faucet verified dev chain: %s", chain_name)

    async def _run(self, fn):
        loop = self._loop or asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_dest(dest: str) -> str:
    """Canonicalize dest as 0x-prefixed lowercase hex for rate-limit keys."""
    if dest.startswith("0x") or dest.startswith("0X"):
        return "0x" + dest[2:].lower()
    return dest


def _is_port_free(host: str, port: int) -> bool:
    try:
        with socket.socket() as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faucet_bot",
        description="Standalone dev faucet for substrate chains.",
    )
    parser.add_argument(
        "--node-url",
        required=True,
        help="Substrate node WebSocket URL (e.g. ws://localhost:9944)",
    )
    parser.add_argument(
        "--faucet-key",
        default="//Alice",
        help="Substrate URI for the funded sender account (default: //Alice)",
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8087,
        help="Bind port (default: 8087)",
    )
    parser.add_argument(
        "--rate-limit-seconds",
        type=float,
        default=DEFAULT_RATE_LIMIT_SECONDS,
        help="Seconds between requests per destination address",
    )
    parser.add_argument(
        "--allow-any-chain",
        action="store_true",
        help="Allow running against non-dev chains. UNSAFE.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log verbosity (default: INFO)",
    )
    return parser


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


async def _run(config: FaucetConfig) -> None:
    faucet = SubstrateFaucet(config)
    await faucet.start()
    try:
        # Block forever; aiohttp's TCPSite has already bound the port and
        # the runner serves requests in the background.
        await asyncio.Event().wait()
    finally:
        await faucet.stop()


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.log_level)

    if not _is_port_free(args.listen, args.port):
        logger.error(
            "port %s:%d already in use; pick a different --port",
            args.listen,
            args.port,
        )
        return 2

    config = FaucetConfig(
        node_url=args.node_url,
        faucet_key_uri=args.faucet_key,
        listen_host=args.listen,
        listen_port=args.port,
        rate_limit_seconds=args.rate_limit_seconds,
        allow_any_chain=args.allow_any_chain,
    )
    try:
        asyncio.run(_run(config))
    except KeyboardInterrupt:
        logger.info("faucet stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
