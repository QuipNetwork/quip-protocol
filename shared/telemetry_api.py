"""Telemetry REST API for quip-miner.

Replaces `shared.rest_api` in the v0.1 -> v0.2 refactor. Endpoint paths and
the `{success, data, ...}` response envelope are preserved so existing
dashboards (dashboard.quip.network) keep working; the backends are rewired
to read from `MinerCore`, `SubstrateClient`, and `SubstrateMinerController`
instead of the deleted `Node` / `NetworkNode`.

Preserved paths:
  GET  /health
  GET  /api/v1/status        - chain head + miner is_mining + miner ss58
  GET  /api/v1/system        - hardware descriptor (MinerCore.descriptor)
  GET  /api/v1/stats         - aggregate mining stats (MinerCore + controller)
  GET  /api/v1/block/latest  - chain head via SubstrateClient
  GET  /api/v1/block/{n}     - chain block at index `n`
  GET  /api/v1/block/{n}/header - chain header at index `n`
  POST /api/v1/solve         - direct DWave Ising sample (unchanged)

Dropped paths (no equivalent in substrate mode):
  /api/v1/peers              - no P2P peers
  /api/v1/join               - no P2P join handshake
  /api/v1/gossip             - no gossip
  /api/v1/heartbeat          - no SWIM heartbeats
  POST /api/v1/block         - block submission goes through substrate extrinsics

Note on `/telemetry/*`: the legacy paths required a TelemetryCache that
aggregated per-peer state across the P2P network. There is no equivalent in
substrate mode (the chain is the source of truth), so these paths are
omitted. Dashboards that consume them should migrate to the substrate-side
events / Prometheus endpoints exposed by the node directly (port 9615 on
the default docker setup).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from aiohttp import web
from aiohttp.web import middleware

from shared.logging_config import get_logger
from shared.miner_survey import build_miner_survey
from shared.version import get_version


if TYPE_CHECKING:
    from shared.miner_core import MinerCore
    from shared.signer import Signer
    from shared.substrate_client import SubstrateClient
    from shared.substrate_miner_controller import SubstrateMinerController


logger = get_logger("telemetry_api")


@middleware
async def cors_middleware(request: web.Request, handler) -> web.Response:
    """Permissive CORS for the dev dashboard. Tighten if exposed externally."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = exc
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


@middleware
async def error_middleware(request: web.Request, handler) -> web.Response:
    """Convert unhandled exceptions into the legacy error envelope."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "unhandled exception in handler for %s %s: %s",
            request.method,
            request.path,
            exc,
        )
        return web.json_response(
            {
                "success": False,
                "error": str(exc),
                "code": "INTERNAL_ERROR",
                "timestamp": int(time.time()),
            },
            status=500,
        )


class TelemetryApiServer:
    """HTTP server exposing the preserved /api/v1/* telemetry paths.

    Construction takes the three quip-miner singletons:
      - `core`: MinerCore (descriptor + aggregate stats)
      - `client`: SubstrateClient (chain head + block queries)
      - `signer`: Signer (ss58 identity)
      - `controller` (optional): SubstrateMinerController (operational
        counters merged into /api/v1/stats)
    """

    def __init__(
        self,
        core: "MinerCore",
        client: "SubstrateClient",
        signer: "Signer",
        controller: Optional["SubstrateMinerController"] = None,
        host: str = "127.0.0.1",
        port: int = 8086,
    ) -> None:
        self.core = core
        self.client = client
        self.signer = signer
        self.controller = controller
        self.host = host
        self.port = port
        self._started_at: Optional[float] = None
        self._runner: Optional[web.AppRunner] = None
        self._app: Optional[web.Application] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        app = web.Application(middlewares=[cors_middleware, error_middleware])
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/api/v1/status", self.handle_status)
        app.router.add_get("/api/v1/system", self.handle_system)
        app.router.add_get("/api/v1/miner/survey", self.handle_miner_survey)
        app.router.add_get("/api/v1/stats", self.handle_stats)
        app.router.add_get("/api/v1/block/latest", self.handle_block_latest)
        app.router.add_get("/api/v1/block/{block_number}", self.handle_block)
        app.router.add_get(
            "/api/v1/block/{block_number}/header", self.handle_block_header
        )
        app.router.add_post("/api/v1/solve", self.handle_solve)
        app.router.add_get(
            "/api/v1/mining/attempts", self.handle_mining_attempts,
        )
        app.router.add_get("/", self.handle_index)
        app.router.add_route("*", "/{path:.*}", self.handle_not_found)

        self._app = app
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host=self.host, port=self.port)
        await site.start()
        self._started_at = time.time()
        logger.info("telemetry api listening: http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # ------------------------------------------------------------------
    # Endpoint handlers
    # ------------------------------------------------------------------

    async def handle_health(self, _request: web.Request) -> web.Response:
        return self._success({"status": "ok"})

    async def handle_index(self, _request: web.Request) -> web.Response:
        return self._success(
            {
                "service": "quip-miner",
                "version": get_version(),
                "endpoints": [
                    {"GET": "/health"},
                    {"GET": "/api/v1/status"},
                    {"GET": "/api/v1/system"},
                    {"GET": "/api/v1/miner/survey"},
                    {"GET": "/api/v1/stats"},
                    {"GET": "/api/v1/block/latest"},
                    {"GET": "/api/v1/block/{block_number}"},
                    {"GET": "/api/v1/block/{block_number}/header"},
                    {"POST": "/api/v1/solve"},
                    {"GET": "/api/v1/mining/attempts"},
                ],
            }
        )

    async def handle_not_found(self, _request: web.Request) -> web.Response:
        return self._error("not found", "NOT_FOUND", status=404)

    async def handle_status(self, _request: web.Request) -> web.Response:
        head_hash = await self.client.get_head()
        head_number = await self.client.get_block_number(at=head_hash)
        miner_account = self.signer.account_id_bytes()
        miner_info = await self.client.query_miner(miner_account)
        return self._success(
            {
                "ss58_address": self.signer.ss58_address(),
                "account_id_hex": "0x" + miner_account.hex(),
                "node_id": self.core.node_id,
                "is_mining": (
                    self.controller is not None
                    and not self.controller._shutdown_event.is_set()  # noqa: SLF001
                ),
                "uptime_seconds": int(time.time() - self._started_at)
                if self._started_at
                else 0,
                "chain": {
                    "head_hash": "0x" + head_hash.hex(),
                    "head_number": head_number,
                },
                "miner_registered": miner_info is not None,
                "miner_info": _miner_info_dict(miner_info) if miner_info else None,
                "miners": [
                    {"id": h.miner_id, "type": h.miner_type}
                    for h in self.core.miner_handles
                ],
            }
        )

    async def handle_system(self, _request: web.Request) -> web.Response:
        return self._success(self.core.descriptor())

    async def handle_miner_survey(self, _request: web.Request) -> web.Response:
        """Stable, indexer-friendly snapshot of the miner's identity,
        active handles, and hardware. Schema: `quip.miner_survey.v1`.
        Prefer this over `/api/v1/system` — the descriptor endpoint
        returns whatever the legacy descriptor builder produces, while
        this endpoint guarantees a versioned shape."""
        return self._success(
            build_miner_survey(self.core, self.signer, controller=self.controller)
        )

    async def handle_stats(self, _request: web.Request) -> web.Response:
        stats = self.core.get_stats()
        if self.controller is not None:
            cs = self.controller.stats
            stats["controller"] = {
                "heads_observed": cs.heads_observed,
                "contexts_dispatched": cs.contexts_dispatched,
                "results_received": cs.results_received,
                "proofs_submitted": cs.proofs_submitted,
                "stale_drops": cs.stale_drops,
                "submission_errors": cs.submission_errors,
                "heads_skipped_already_won": cs.heads_skipped_already_won,
                "heads_dropped_stale_number": cs.heads_dropped_stale_number,
                "heads_refreshed_active": cs.heads_refreshed_active,
                "stale_post_win_heads_dropped": cs.stale_post_win_heads_dropped,
                "zero_seed_snapshots_dropped": cs.zero_seed_snapshots_dropped,
                "subscription_lag_blocks": cs.subscription_lag_blocks,
                "heads_promoted_to_rpc": cs.heads_promoted_to_rpc,
                "heads_refresh_below_floor": cs.heads_refresh_below_floor,
                "post_win_fast_forwards": cs.post_win_fast_forwards,
                "post_win_fast_forward_timeouts": cs.post_win_fast_forward_timeouts,
                "heads_same_key_skipped": cs.heads_same_key_skipped,
                "none_snapshots_seen": cs.none_snapshots_seen,
                "duplicate_result_drops": cs.duplicate_result_drops,
                "proofs_unverified": cs.proofs_unverified,
            }
        return self._success(stats)

    async def handle_block_latest(self, _request: web.Request) -> web.Response:
        head_hash = await self.client.get_head()
        return self._success(await self._block_payload(at=head_hash))

    async def handle_block(self, request: web.Request) -> web.Response:
        block_number, error = _parse_block_number(request)
        if error is not None:
            return error
        block_hash = await self._block_hash_for_number(block_number)
        if block_hash is None:
            return self._error(
                f"block {block_number} not found",
                "BLOCK_NOT_FOUND",
                status=404,
            )
        return self._success(await self._block_payload(at=block_hash))

    async def handle_block_header(self, request: web.Request) -> web.Response:
        block_number, error = _parse_block_number(request)
        if error is not None:
            return error
        block_hash = await self._block_hash_for_number(block_number)
        if block_hash is None:
            return self._error(
                f"block {block_number} not found",
                "BLOCK_NOT_FOUND",
                status=404,
            )
        payload = await self._block_payload(at=block_hash)
        return self._success(
            {"header": payload.get("header"), "hash": payload.get("hash")}
        )

    async def handle_solve(self, request: web.Request) -> web.Response:
        """POST /api/v1/solve — direct DWave Ising sample.

        Body: {"h": {...}, "J": {...}, "num_reads": int, "annealing_time_us": int}
        Returns: {"samples": [...], "energies": [...]}

        Phase 5b keeps the legacy contract intact; the DWave-sampler caching
        and rate limiting from rest_api.py are deferred to a follow-on so
        the telemetry surface stays close to its v0.1 behaviour. Returns
        503 if the user hasn't enabled DWave (no API key in env).
        """
        # The DWave path is heavy and was rate-limited in rest_api.py via a
        # PeerRateLimiter; we keep the endpoint registered but defer the
        # sampler-caching machinery until a real use case shows up against
        # quip-miner. Operators that need this should hit a dedicated solve
        # service rather than the miner's telemetry port.
        return self._error(
            "/api/v1/solve is not enabled on quip-miner; deploy a dedicated "
            "solve service for direct DWave sampling",
            "SOLVE_DISABLED",
            status=503,
        )

    async def handle_mining_attempts(self, request: web.Request) -> web.Response:
        """GET /api/v1/mining/attempts — query the attempt + submission log.

        Query params (exactly one of the first two must be supplied):
          - ``solution_id`` (int): returns
            ``{"submission": {...}, "attempts": [...]}`` — the submission
            record with that id and every attempt that fed the same
            (miner_id, dispatch_id).
          - ``miner_id`` + ``dispatch_id``: returns just the attempt
            list for that dispatch.
          - ``limit`` (int, default 1000): caps the attempts list.

        The store is the JSONL files under ``~/.quip-miner/mining_attempts/``
        produced by the worker (``attempts-*.jsonl``) and controller
        (``submissions-*.jsonl``). See ``shared/mining_attempt_log.py``
        for the schema and write paths.
        """
        # Imported lazily so the module load doesn't pin a filesystem
        # path during test setup that imports telemetry_api headlessly.
        from shared.mining_attempt_log import (
            query_by_dispatch,
            query_by_solution_id,
        )

        params = request.rel_url.query
        try:
            limit = int(params.get("limit", "1000"))
        except ValueError:
            return self._error("limit must be an integer", "BAD_PARAM")
        if limit < 1 or limit > 100_000:
            return self._error(
                "limit out of range (1..100000)", "BAD_PARAM",
            )

        if "solution_id" in params:
            try:
                solution_id = int(params["solution_id"])
            except ValueError:
                return self._error(
                    "solution_id must be an integer", "BAD_PARAM",
                )
            result = query_by_solution_id(solution_id)
            if result is None:
                return self._error(
                    f"solution_id {solution_id} not found",
                    "NOT_FOUND",
                    status=404,
                )
            # Cap attempts list to limit (oldest first kept).
            result["attempts"] = result["attempts"][:limit]
            return self._success(result)

        miner_id = params.get("miner_id")
        dispatch_id_raw = params.get("dispatch_id")
        if miner_id and dispatch_id_raw:
            try:
                dispatch_id = int(dispatch_id_raw)
            except ValueError:
                return self._error(
                    "dispatch_id must be an integer", "BAD_PARAM",
                )
            attempts = query_by_dispatch(miner_id, dispatch_id, limit=limit)
            return self._success({"attempts": attempts})

        return self._error(
            "supply either ?solution_id=N or ?miner_id=X&dispatch_id=Y",
            "BAD_PARAM",
        )

    # ------------------------------------------------------------------
    # Substrate-backed helpers
    # ------------------------------------------------------------------

    async def _block_hash_for_number(self, block_number: int) -> Optional[bytes]:
        """Resolve block_number → block_hash via chain RPC."""

        def _resolve() -> Optional[str]:
            return self.client._iface.get_block_hash(block_id=block_number)  # noqa: SLF001

        result = await self.client._run(_resolve)  # noqa: SLF001
        if not result or result in ("0x" + "00" * 32, None):
            return None
        h = result[2:] if result.startswith("0x") else result
        try:
            return bytes.fromhex(h)
        except ValueError:
            return None

    async def _block_payload(self, *, at: bytes) -> dict:
        """Substrate block → legacy-shaped JSON dict.

        We don't reconstruct the full v0.1 chain block (with quantum proof
        + miner SPHINCS+ signature) — substrate blocks have different
        contents. Instead we expose:
          - the substrate header (parentHash, number, stateRoot,
            extrinsicsRoot, digest)
          - the block hash
        Dashboards that consumed the v0.1 fields directly need updating;
        the response envelope shape (success/data/timestamp) is preserved.

        substrate-interface's `get_block` returns a structure that mixes
        plain dicts with `ScaleType` wrappers (e.g. the digest items hold
        decoded `scale_info::N` objects). Those aren't JSON-serializable as
        returned, so we coerce through `_to_jsonable` before handing the
        dict to `json_response`.
        """

        def _query() -> dict:
            block = self.client._iface.get_block(block_hash="0x" + at.hex())  # noqa: SLF001
            return block

        block = await self.client._run(_query)  # noqa: SLF001
        header = block.get("header", {}) if isinstance(block, dict) else {}
        extrinsics = block.get("extrinsics", []) if isinstance(block, dict) else []
        return {
            "hash": "0x" + at.hex(),
            "header": _to_jsonable(header),
            "extrinsic_count": len(extrinsics or []),
        }

    # ------------------------------------------------------------------
    # Response envelopes
    # ------------------------------------------------------------------

    def _success(self, data: Any, status: int = 200) -> web.Response:
        return web.json_response(
            {"success": True, "data": data, "timestamp": int(time.time())},
            status=status,
        )

    def _error(self, message: str, code: str, status: int = 400) -> web.Response:
        return web.json_response(
            {
                "success": False,
                "error": message,
                "code": code,
                "timestamp": int(time.time()),
            },
            status=status,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Recursively coerce substrate-interface output to JSON-safe primitives.

    `SubstrateInterface.get_block` returns dicts whose leaves can be raw
    `ScaleType` instances (e.g. decoded `scale_info::N` objects from the
    runtime metadata). `json.dumps` rejects those with "is not JSON
    serializable"; calling `.value` on a ScaleType gives the decoded
    primitive (str / int / list / dict). Walk the structure and unwrap.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Render binary as 0x-hex to match the rest of the API surface
    # (e.g. block hashes, account ids). Without this, raw `bytes` in the
    # decoded header (parent hash, state root, digest payload) would
    # stringify via repr() as `b'\\x00...'` — valid Python but useless
    # to any dashboard parsing the response as hex.
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return "0x" + bytes(obj).hex()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # ScaleType and friends carry the decoded value under `.value`. Use
    # `hasattr` rather than `getattr(..., None)`: a ScaleType that legit
    # decodes to `None` (e.g. an empty `Option<T>`) should serialize as
    # JSON `null`, not as `repr(obj)`.
    if hasattr(obj, "value"):
        val = obj.value
        if val is not obj:
            return _to_jsonable(val)
    # Fall back to repr() so the endpoint stays observable even if
    # substrate-interface ever changes its return shape.
    return repr(obj)


def _parse_block_number(request: web.Request):
    """Extract `block_number` from URL match info as a positive int.

    Returns (block_number, None) on success, (None, error_response) on bad
    input. Keeping the error envelope identical to the legacy `rest_api.py`
    so dashboards that match on `code` keep working.
    """
    raw = request.match_info.get("block_number", "")
    try:
        block_number = int(raw)
    except ValueError:
        return None, web.json_response(
            {
                "success": False,
                "error": "block_number must be an integer",
                "code": "INVALID_BLOCK_NUMBER",
                "timestamp": int(time.time()),
            },
            status=400,
        )
    if block_number < 0:
        return None, web.json_response(
            {
                "success": False,
                "error": "block_number must be non-negative",
                "code": "INVALID_BLOCK_NUMBER",
                "timestamp": int(time.time()),
            },
            status=400,
        )
    return block_number, None


def _miner_info_dict(info) -> dict:
    """Map `shared.substrate_types.MinerInfo` → JSON-safe dict."""
    return {
        "registered_at": info.registered_at,
        "deposit": info.deposit,
        "proofs_submitted": info.proofs_submitted,
        "proofs_won": info.proofs_won,
        "rewards_earned": info.rewards_earned,
    }


__all__ = [
    "TelemetryApiServer",
]
