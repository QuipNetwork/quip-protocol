"""Telemetry sibling process entry point.

`telemetry_main` is the child entry: it runs an asyncio loop hosting
aiohttp.web with the telemetry routes. The /api/v1/stats handler reads
the stats snapshot file the controller writes (no live IPC); other
handlers query the chain directly via a SubstrateClient with URL
failover.

Process isolation rationale: telemetry serves HTTP requests of
arbitrary cost (operator dashboards, ad-hoc queries). Co-locating
that on the controller's main loop risks starving the event drain
during request spikes. The original silent-subscription bug taught us
to be paranoid about shared event loops; telemetry isolation is the
follow-through.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import multiprocessing.synchronize
import signal
from pathlib import Path
from typing import Optional

from aiohttp import web

from shared.stats_snapshot import read_snapshot
from substrate.url_failover import SubstrateUrlFailover

logger = logging.getLogger(__name__)


def telemetry_main(
    listen_host: str,
    listen_port: int,
    stats_snapshot_path: str,
    validator_urls: list[str],
    shutdown_event: mp.synchronize.Event,
) -> None:
    """Child process entry point for telemetry.

    Args:
        listen_host: aiohttp bind address.
        listen_port: aiohttp listen port.
        stats_snapshot_path: file path the controller writes via
            StatsSnapshotWriter. The /api/v1/stats handler reads this.
        validator_urls: URLs to try for direct chain queries. Telemetry
            uses its own client; no pool/event-manager involvement.
        shutdown_event: Set by the parent (or by SIGTERM handler) to
            request graceful exit.
    """
    # SIGTERM → set shutdown_event so the asyncio loop notices.
    def _sigterm(signum, frame):
        logger.info("telemetry process received SIGTERM")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)

    asyncio.run(_run(
        listen_host=listen_host,
        listen_port=listen_port,
        stats_snapshot_path=Path(stats_snapshot_path),
        validator_urls=validator_urls,
        shutdown_event=shutdown_event,
    ))


async def _run(
    listen_host: str,
    listen_port: int,
    stats_snapshot_path: Path,
    validator_urls: list[str],
    shutdown_event: mp.synchronize.Event,
) -> None:
    failover = SubstrateUrlFailover(validator_urls, initial_backoff_s=1.0, max_backoff_s=60.0)

    app = web.Application()
    app["stats_snapshot_path"] = stats_snapshot_path
    app["failover"] = failover

    app.router.add_get("/api/v1/stats", _handle_stats)
    app.router.add_get("/api/v1/status", _handle_status)
    app.router.add_get("/health", _handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, listen_host, listen_port)
    await site.start()

    logger.info("telemetry process listening on %s:%d", listen_host, listen_port)

    try:
        # Wait for shutdown signal (polled because mp.Event isn't asyncio-native).
        while not shutdown_event.is_set():
            await asyncio.sleep(0.1)
    finally:
        logger.info("telemetry process shutting down")
        await runner.cleanup()


async def _handle_stats(request: web.Request) -> web.Response:
    """Return the latest stats snapshot the controller has written."""
    path: Path = request.app["stats_snapshot_path"]
    snapshot = read_snapshot(path)
    if snapshot is None:
        return web.json_response(
            {
                "success": False,
                "error": "stats snapshot not yet available",
                "code": "STATS_UNAVAILABLE",
            },
            status=503,
        )
    return web.json_response({"success": True, "data": snapshot})


async def _handle_status(request: web.Request) -> web.Response:
    """Basic status from the chain itself. Uses telemetry's own client."""
    # For Plan 4 we keep this stub — the existing /api/v1/status handler
    # logic (get_head, get_block_number, query_miner) ports over from
    # shared/telemetry_api.py. Wiring it through the failover-aware
    # SubstrateClient is a mechanical move.
    return web.json_response({"success": True, "data": {"telemetry_process": "alive"}})


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})
