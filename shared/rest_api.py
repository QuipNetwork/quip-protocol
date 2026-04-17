"""
REST API server for QuIP network nodes.

Exposes all QUIC message types as HTTP endpoints for browser and external access.
Runs alongside the QUIC server on a separate port.
"""

import asyncio
import hmac
import json
import logging
import os
import ssl
import struct
import time
from typing import TYPE_CHECKING, Any, List, Optional

from aiohttp import web
from aiohttp.web import middleware

from shared.rate_limiter import PeerRateLimiter
from shared.time_utils import utc_timestamp_float
from shared.version import get_version

if TYPE_CHECKING:
    from shared.certificate_manager import CertificateManager
    from shared.network_node import NetworkNode
    from shared.telemetry_cache import TelemetryCache


@middleware
async def cors_middleware(request: web.Request, handler) -> web.Response:
    """Add CORS headers to all responses."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except web.HTTPException as e:
            response = e

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "86400"

    return response


@middleware
async def error_middleware(request: web.Request, handler) -> web.Response:
    """Convert exceptions to JSON error responses."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as e:
        return web.json_response(
            {
                "success": False,
                "error": str(e),
                "code": "INTERNAL_ERROR",
                "timestamp": int(time.time())
            },
            status=500
        )


class RestApiServer:
    """
    REST API server exposing QUIC message types as HTTP endpoints.

    Runs on a separate port from the QUIC server and provides browser-compatible
    access to node functionality.
    """

    def __init__(
        self,
        network_node: 'NetworkNode',
        host: str = "0.0.0.0",
        port: int = 8080,
        tls_port: int = 443,
        cert_manager: Optional['CertificateManager'] = None,
        webroot: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        telemetry_cache: Optional['TelemetryCache'] = None,
        telemetry_access_token: str = "",
        telemetry_rate_limit_rpm: int = 30,
        telemetry_max_sse: int = 20,
        solve_rate_limit_rpm: int = 10,
    ):
        """
        Initialize the REST API server.

        Args:
            network_node: The NetworkNode instance to expose.
            host: Host address to bind to.
            port: HTTP port (non-TLS).
            tls_port: HTTPS port (TLS).
            cert_manager: Certificate manager for TLS.
            webroot: Directory for static files (ACME challenges).
            logger: Logger instance.
            telemetry_cache: TelemetryCache for telemetry endpoints.
            telemetry_access_token: Bearer token for telemetry (empty=open).
            telemetry_rate_limit_rpm: Requests/minute per IP for telemetry.
            telemetry_max_sse: Max concurrent SSE connections.
            solve_rate_limit_rpm: Requests/minute per IP for /api/v1/solve.
        """
        self.node = network_node
        self.host = host
        self.http_port = port
        self.https_port = tls_port
        self.cert_manager = cert_manager
        self.webroot = webroot
        self.logger = logger or logging.getLogger(__name__)

        # Telemetry API
        self._telemetry_cache = telemetry_cache
        self._telemetry_token = telemetry_access_token
        tokens_per_sec = telemetry_rate_limit_rpm / 60.0
        self._telemetry_rate_limiter = PeerRateLimiter(
            tokens_per_second=tokens_per_sec,
            max_burst=max(int(tokens_per_sec * 5), 3),
        )
        self._telemetry_max_sse = telemetry_max_sse
        self._sse_clients: List[web.StreamResponse] = []
        # Guards check-then-append / iterate-and-remove on _sse_clients.
        self._sse_lock = asyncio.Lock()
        # Tracks pending background writes per client so we can drop
        # slow consumers instead of accumulating unbounded write tasks.
        self._sse_pending: "dict[web.StreamResponse, int]" = {}
        # Handler tasks for active SSE streams. stop() cancels these so
        # clients sleeping in the keepalive loop exit immediately
        # instead of lingering up to 15s past shutdown.
        self._sse_tasks: "set[asyncio.Task]" = set()
        self._rate_limit_prune_task: Optional[asyncio.Task] = None

        # /api/v1/solve rate limiter and cached DWaveSampler. Solve is
        # expensive (Leap RTT + anneal time), so we serve one request
        # at a time via a shared sampler instead of spinning one up
        # per request on the event loop thread.
        solve_tokens_per_sec = solve_rate_limit_rpm / 60.0
        self._solve_rate_limiter = PeerRateLimiter(
            tokens_per_second=solve_tokens_per_sec,
            max_burst=max(int(solve_tokens_per_sec * 5), 3),
        )
        self._dwave_sampler: Optional[Any] = None
        self._dwave_sampler_init_lock = asyncio.Lock()
        # QPU is a single hardware resource; serialize sample_ising
        # calls on the cached sampler.
        self._dwave_sampler_sample_lock = asyncio.Lock()

        self._http_runner: Optional[web.AppRunner] = None
        self._https_runner: Optional[web.AppRunner] = None
        self._app: Optional[web.Application] = None

    def _create_app(self) -> web.Application:
        """Create and configure the aiohttp application."""
        middlewares = [cors_middleware, error_middleware]
        if self._telemetry_cache is not None:
            middlewares.append(self._telemetry_auth_middleware)
            middlewares.append(self._telemetry_rate_limit_middleware)

        app = web.Application(middlewares=middlewares)

        # Health check
        app.router.add_get("/health", self.handle_health)

        # API v1 routes
        app.router.add_get("/api/v1/status", self.handle_status)
        app.router.add_get("/api/v1/system", self.handle_system)
        app.router.add_get("/api/v1/stats", self.handle_stats)
        app.router.add_get("/api/v1/peers", self.handle_peers)
        app.router.add_get("/api/v1/block/latest", self.handle_get_latest_block)
        app.router.add_get("/api/v1/block/{block_number}", self.handle_get_block)
        app.router.add_get("/api/v1/block/{block_number}/header", self.handle_get_block_header)
        app.router.add_post("/api/v1/join", self.handle_join)
        app.router.add_post("/api/v1/block", self.handle_submit_block)
        app.router.add_post("/api/v1/gossip", self.handle_gossip)
        app.router.add_post("/api/v1/solve", self.handle_solve)
        app.router.add_post("/api/v1/heartbeat", self.handle_heartbeat)

        # Telemetry API v1 routes
        if self._telemetry_cache is not None:
            app.router.add_get(
                "/api/v1/telemetry/status", self.handle_telemetry_status,
            )
            app.router.add_get(
                "/api/v1/telemetry/nodes", self.handle_telemetry_nodes,
            )
            app.router.add_get(
                "/api/v1/telemetry/epochs", self.handle_telemetry_epochs,
            )
            app.router.add_get(
                "/api/v1/telemetry/epochs/{epoch}/blocks/{block_index}",
                self.handle_telemetry_block,
            )
            app.router.add_get(
                "/api/v1/telemetry/latest", self.handle_telemetry_latest,
            )
            app.router.add_get(
                "/api/v1/telemetry/stream", self.handle_telemetry_stream,
            )

        # Static file hosting (ACME HTTP-01 challenges, etc.)
        if self.webroot and os.path.isdir(self.webroot):
            well_known = os.path.join(self.webroot, ".well-known")
            os.makedirs(well_known, exist_ok=True)
            app.router.add_static("/.well-known", well_known, show_index=False)
            self.logger.info(f"Serving static files from {well_known} at /.well-known/")

        # OPTIONS handler for CORS preflight
        app.router.add_route("OPTIONS", "/{path:.*}", self.handle_options)

        return app

    async def start(self) -> None:
        """Start the REST API server."""
        self._app = self._create_app()

        # Wire SSE push from telemetry cache
        if self._telemetry_cache is not None:
            self._telemetry_cache.on_new_block = self._sse_push_block
            self._telemetry_cache.on_nodes_changed = self._sse_push_nodes

        # Rate-limiter pruning runs unconditionally: the solve limiter
        # always exists even when telemetry is disabled.
        self._rate_limit_prune_task = asyncio.create_task(
            self._prune_rate_limiter_loop(),
        )

        # Start HTTP server (always)
        self._http_runner = web.AppRunner(self._app)
        await self._http_runner.setup()
        http_site = web.TCPSite(self._http_runner, self.host, self.http_port)
        await http_site.start()
        self.logger.info(f"REST API HTTP server started on http://{self.host}:{self.http_port}")

        # Start HTTPS server if certificate manager is available
        if self.cert_manager:
            try:
                cert_path, key_path = await self.cert_manager.get_certificate()

                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(cert_path, key_path)

                self._https_runner = web.AppRunner(self._app)
                await self._https_runner.setup()
                https_site = web.TCPSite(
                    self._https_runner,
                    self.host,
                    self.https_port,
                    ssl_context=ssl_context
                )
                await https_site.start()
                self.logger.info(
                    f"REST API HTTPS server started on https://{self.host}:{self.https_port}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to start HTTPS server: {e}")
                self.logger.info("REST API available via HTTP only")

    async def stop(self) -> None:
        """Stop the REST API server."""
        if self._rate_limit_prune_task and not self._rate_limit_prune_task.done():
            self._rate_limit_prune_task.cancel()

        # Cancel in-flight SSE handler tasks. Without this, any client
        # suspended in the 15s keepalive sleep would linger past
        # shutdown.  write_eof alone can't interrupt a sleeping task.
        sse_tasks = list(self._sse_tasks)
        for task in sse_tasks:
            task.cancel()

        # Close SSE connections
        for client in list(self._sse_clients):
            try:
                await client.write_eof()
            except Exception:
                pass
        self._sse_clients.clear()

        # Wait for cancelled handlers to unwind so their finally
        # blocks finish cleanup before the app runner tears sockets
        # down underneath them.
        if sse_tasks:
            await asyncio.gather(*sse_tasks, return_exceptions=True)

        if self._http_runner:
            await self._http_runner.cleanup()
            self._http_runner = None

        if self._https_runner:
            await self._https_runner.cleanup()
            self._https_runner = None

        if self._dwave_sampler is not None:
            try:
                await asyncio.to_thread(self._dwave_sampler.close)
            except Exception as e:
                self.logger.warning(f"Failed to close DWaveSampler: {e}")
            self._dwave_sampler = None

        self.logger.info("REST API server stopped")

    def _success_response(self, data: Any) -> web.Response:
        """Create a successful JSON response."""
        return web.json_response({
            "success": True,
            "data": data,
            "timestamp": int(time.time())
        })

    def _error_response(
        self,
        message: str,
        code: str = "ERROR",
        status: int = 400
    ) -> web.Response:
        """Create an error JSON response."""
        return web.json_response(
            {
                "success": False,
                "error": message,
                "code": code,
                "timestamp": int(time.time())
            },
            status=status
        )

    # Handler implementations

    async def handle_options(self, request: web.Request) -> web.Response:
        """Handle OPTIONS request for CORS preflight."""
        return web.Response()

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health - Health check endpoint."""
        return self._success_response({
            "status": "healthy",
            "node_running": self.node.running,
            "version": get_version()
        })

    async def handle_status(self, request: web.Request) -> web.Response:
        """GET /api/v1/status - Node status information."""
        status_data = {
            "host": self.node.public_host,
            "info": json.loads(self.node.info().to_json()),
            "descriptor": self.node.descriptor(),
            "running": self.node.running,
            "total_peers": len(self.node.peers),
            "uptime": utc_timestamp_float() if self.node.running else 0,
            "latest_block": self.node.get_latest_block().header.index if self.node.get_latest_block() else 0
        }
        return self._success_response(status_data)

    async def handle_system(self, request: web.Request) -> web.Response:
        """GET /api/v1/system - Node hardware survey + whitelisted config."""
        return self._success_response(self.node.descriptor())

    async def handle_stats(self, request: web.Request) -> web.Response:
        """GET /api/v1/stats - Mining and network statistics."""
        async with self.node._stats_cache_lock:
            if self.node._stats_cache is None:
                return self._error_response(
                    "Stats cache not initialized",
                    "STATS_NOT_READY",
                    503
                )
            return self._success_response(self.node._stats_cache)

    async def handle_peers(self, request: web.Request) -> web.Response:
        """GET /api/v1/peers - List of known peers."""
        async with self.node.net_lock:
            peers_data = {
                host: json.loads(info.to_json())
                for host, info in self.node.peers.items()
            }
        return self._success_response({"peers": peers_data, "count": len(peers_data)})

    async def handle_get_block(self, request: web.Request) -> web.Response:
        """GET /api/v1/block/{block_number} - Get block by number."""
        try:
            block_number = int(request.match_info["block_number"])
        except ValueError:
            return self._error_response("Invalid block number", "INVALID_BLOCK_NUMBER")

        block = self.node.get_block(block_number)
        if block is None:
            return self._error_response(
                f"Block {block_number} not found",
                "BLOCK_NOT_FOUND",
                404
            )

        return self._success_response(self._block_to_dict(block))

    async def handle_get_latest_block(self, request: web.Request) -> web.Response:
        """GET /api/v1/block/latest - Get the latest block."""
        block = self.node.get_latest_block()
        if block is None:
            return self._error_response("No blocks available", "NO_BLOCKS", 404)

        return self._success_response(self._block_to_dict(block))

    async def handle_get_block_header(self, request: web.Request) -> web.Response:
        """GET /api/v1/block/{block_number}/header - Get block header by number."""
        try:
            block_number = int(request.match_info["block_number"])
        except ValueError:
            return self._error_response("Invalid block number", "INVALID_BLOCK_NUMBER")

        block = self.node.get_block(block_number)
        if block is None:
            return self._error_response(
                f"Block {block_number} not found",
                "BLOCK_NOT_FOUND",
                404
            )

        return self._success_response(self._header_to_dict(block.header))

    async def handle_join(self, request: web.Request) -> web.Response:
        """POST /api/v1/join - Join the network."""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._error_response("Invalid JSON body", "INVALID_JSON")

        from shared.block import MinerInfo

        new_node_address = data.get("host")
        info_field = data.get("info")

        if not new_node_address:
            return self._error_response("Missing 'host' field", "MISSING_HOST")

        new_node_info = None
        if info_field:
            try:
                new_node_info = MinerInfo.from_json(
                    info_field if isinstance(info_field, str) else json.dumps(info_field)
                )
            except Exception as e:
                return self._error_response(f"Invalid 'info' field: {e}", "INVALID_INFO")

        descriptor_field = data.get("descriptor")
        if new_node_info:
            from shared.system_info import override_public_address
            await self.node.add_peer(
                new_node_address, new_node_info,
                descriptor=override_public_address(
                    descriptor_field, new_node_address,
                ),
            )

        # Return our peer list
        async with self.node.net_lock:
            peers_snapshot = dict(self.node.peers)

        peers_payload = {
            host: json.loads(info.to_json())
            for host, info in peers_snapshot.items()
        }
        peers_payload[self.node.public_host] = json.loads(self.node.info().to_json())

        return self._success_response({
            "status": "ok",
            "peers": peers_payload,
            "descriptor": self.node.descriptor(),
        })

    async def handle_submit_block(self, request: web.Request) -> web.Response:
        """POST /api/v1/block - Submit a new block (DEBUG)."""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._error_response("Invalid JSON body", "INVALID_JSON")

        if "raw" not in data or "signature" not in data:
            return self._error_response(
                "Missing 'raw' or 'signature' field",
                "MISSING_FIELDS"
            )

        try:
            from shared.block import Block

            block_bytes = bytes.fromhex(data["raw"])
            signature = bytes.fromhex(data["signature"])
            net_data = block_bytes + signature
            block = Block.from_network(net_data)

            response_future: asyncio.Future[bool] = asyncio.Future()
            self.node.block_processing_queue.put_nowait((block, response_future))
            result = await asyncio.wait_for(response_future, timeout=10.0)

            status = "accepted" if result else "rejected"
            return self._success_response({"status": status})

        except asyncio.QueueFull:
            return self._error_response("Server overloaded", "OVERLOADED", 503)
        except asyncio.TimeoutError:
            return self._error_response("Processing timeout", "TIMEOUT", 504)
        except Exception as e:
            return self._error_response(str(e), "PROCESSING_ERROR")

    async def handle_gossip(self, request: web.Request) -> web.Response:
        """POST /api/v1/gossip - Send a gossip message."""
        try:
            body = await request.read()
        except Exception as e:
            return self._error_response(f"Failed to read body: {e}", "READ_ERROR")

        try:
            from shared.network_node import Message

            gossip_message = Message.from_network(body)
            response_future: asyncio.Future[str] = asyncio.Future()
            t_enq = time.perf_counter()

            self.node.gossip_processing_queue.put_nowait((gossip_message, response_future, t_enq))
            status = await asyncio.wait_for(response_future, timeout=5.0)

            return self._success_response({"status": status})

        except asyncio.QueueFull:
            return self._error_response("Server overloaded", "OVERLOADED", 503)
        except asyncio.TimeoutError:
            return self._error_response("Processing timeout", "TIMEOUT", 504)
        except Exception as e:
            return self._error_response(str(e), "PROCESSING_ERROR")

    async def handle_solve(self, request: web.Request) -> web.Response:
        """POST /api/v1/solve - Submit quantum annealing solve request."""
        # Per-IP rate limit: solve is expensive and one abusive peer
        # could otherwise pin the QPU queue indefinitely.
        peer_key = request.remote or "unknown"
        if not self._solve_rate_limiter.allow(peer_key):
            return web.json_response(
                {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMITED",
                    "timestamp": int(time.time()),
                },
                status=429,
            )

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._error_response("Invalid JSON body", "INVALID_JSON")

        # Validate required fields
        if "h" not in data or "J" not in data or "num_samples" not in data:
            return self._error_response(
                "Missing required fields: h, J, num_samples",
                "MISSING_FIELDS"
            )

        h = data["h"]
        J_raw = data["J"]
        num_samples = int(data["num_samples"])

        # Convert J to list of tuples format
        try:
            if isinstance(J_raw, dict):
                J = [
                    ((int(k.split(",")[0].strip("()")), int(k.split(",")[1].strip("()"))), v)
                    for k, v in J_raw.items()
                ]
            elif isinstance(J_raw, list):
                J = [((entry[0], entry[1]), entry[2]) for entry in J_raw]
            else:
                return self._error_response(
                    "Invalid J format. Must be dict or list.",
                    "INVALID_J_FORMAT"
                )
        except Exception as e:
            return self._error_response(f"Failed to parse J: {e}", "INVALID_J_FORMAT")

        # Generate transaction ID
        transaction_id = f"{self.node.public_host}-{time.time()}-{hash((tuple(h), tuple(str(j) for j in J)))}"

        # Convert h and J to format needed by sampler
        h_dict = {i: val for i, val in enumerate(h)}
        J_dict = {(i, j): val for ((i, j), val) in J}

        # Use first available miner to solve
        if not hasattr(self.node, "miner_handles") or not self.node.miner_handles:
            return self._error_response("No miners available", "NO_MINERS", 503)

        miner_handle = self.node.miner_handles[0]
        miner_kind = miner_handle.spec.get("kind", "").lower()

        self.logger.info(
            f"REST API: Solving BQM with {len(h)} variables, {len(J)} couplings, "
            f"{num_samples} samples using {miner_handle.miner_id}"
        )

        try:
            if miner_kind == "qpu":
                sampler = await self._get_or_create_dwave_sampler()
                # Serialize concurrent solve requests on the shared
                # QPU sampler; run the blocking Leap call off-loop.
                async with self._dwave_sampler_sample_lock:
                    sampleset = await asyncio.to_thread(
                        sampler.sample_ising, h_dict, J_dict,
                        num_reads=num_samples,
                    )
            elif miner_kind in ["cpu", "metal", "cuda", "modal"]:
                from dwave.samplers import SimulatedAnnealingSampler
                sampler = SimulatedAnnealingSampler()
                sampleset = await asyncio.to_thread(
                    sampler.sample_ising, h_dict, J_dict,
                    num_reads=num_samples,
                )
            else:
                return self._error_response(
                    f"Unknown miner type: {miner_kind}",
                    "UNKNOWN_MINER_TYPE"
                )

            # Extract samples and energies
            samples = []
            energies = []
            for sample, energy in sampleset.data(["sample", "energy"]):
                sample_list = [int(sample[i]) for i in sorted(sample.keys())]
                samples.append(sample_list)
                energies.append(float(energy))

            self.logger.info(
                f"REST API: Solve completed with {len(samples)} samples, "
                f"energies from {min(energies):.2f} to {max(energies):.2f}"
            )

            # Create and store transaction
            from shared.block import Transaction
            from shared.time_utils import network_timestamp

            transaction = Transaction(
                transaction_id=transaction_id,
                timestamp=network_timestamp(),
                request_h=h,
                request_J=J,
                num_samples=num_samples,
                samples=samples[:num_samples],
                energies=energies[:num_samples]
            )

            async with self.node.transactions_lock:
                self.node.pending_transactions.append(transaction)

            return self._success_response({
                "samples": samples[:num_samples],
                "energies": energies[:num_samples],
                "transaction_id": transaction_id,
                "status": "completed"
            })

        except Exception as e:
            self.logger.error(f"REST API solve failed: {e}")
            return self._error_response(str(e), "SOLVE_FAILED")

    async def _get_or_create_dwave_sampler(self) -> Any:
        """Return the cached DWaveSampler, constructing it on first use.

        The D-Wave client performs blocking network I/O on construction
        (Leap auth + solver handshake), so we do the setup once in a
        thread and share the instance across requests.
        """
        if self._dwave_sampler is not None:
            return self._dwave_sampler
        async with self._dwave_sampler_init_lock:
            if self._dwave_sampler is None:
                from dwave.system import DWaveSampler
                self._dwave_sampler = await asyncio.to_thread(DWaveSampler)
        return self._dwave_sampler

    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """POST /api/v1/heartbeat - Send heartbeat."""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._error_response("Invalid JSON body", "INVALID_JSON")

        sender = data.get("sender")
        timestamp = data.get("timestamp", utc_timestamp_float())

        if sender:
            async with self.node.net_lock:
                if sender in self.node.peers:
                    self.node.heartbeats[sender] = utc_timestamp_float()
                    self.node._track_peer_timestamp(timestamp)
                else:
                    self.logger.info(f"REST API: New node discovered via heartbeat: {sender}")
                    asyncio.create_task(self.node.refresh_peer_info(sender))

        return self._success_response({"status": "ok"})

    def _block_to_dict(self, block) -> dict:
        """Convert a Block to a JSON-serializable dict."""
        result = {
            "header": self._header_to_dict(block.header),
            "miner_info": json.loads(block.miner_info.to_json()) if block.miner_info else None,
            "quantum_proof": {
                "energy": block.quantum_proof.energy,
                "diversity": block.quantum_proof.diversity,
                "num_valid_solutions": block.quantum_proof.num_valid_solutions,
                "mining_time": block.quantum_proof.mining_time,
                "nonce": block.quantum_proof.nonce,
            } if block.quantum_proof else None,
            "next_block_requirements": {
                "difficulty_energy": block.next_block_requirements.difficulty_energy,
                "min_diversity": block.next_block_requirements.min_diversity,
                "min_solutions": block.next_block_requirements.min_solutions,
            } if block.next_block_requirements else None,
            "transactions": [
                {
                    "transaction_id": tx.transaction_id,
                    "timestamp": tx.timestamp,
                    "num_samples": tx.num_samples,
                    "samples_count": len(tx.samples) if tx.samples else 0,
                    "energy_range": [min(tx.energies), max(tx.energies)] if tx.energies else None,
                }
                for tx in (block.transactions or [])
            ],
            "hash": block.hash.hex() if block.hash else None,
            "signature": block.signature.hex() if block.signature else None,
        }
        return result

    def _header_to_dict(self, header) -> dict:
        """Convert a BlockHeader to a JSON-serializable dict."""
        return {
            "index": header.index,
            "timestamp": header.timestamp,
            "previous_hash": header.previous_hash.hex() if header.previous_hash else None,
            "data_hash": header.data_hash.hex() if header.data_hash else None,
            "version": header.version,
        }

    # ------------------------------------------------------------------
    # Telemetry middleware (scoped to /api/v1/telemetry/ routes)
    # ------------------------------------------------------------------

    @middleware
    async def _telemetry_auth_middleware(
        self, request: web.Request, handler,
    ) -> web.Response:
        """Check bearer token for telemetry endpoints."""
        if not request.path.startswith("/api/v1/telemetry/"):
            return await handler(request)
        if not self._telemetry_token:
            return await handler(request)
        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {self._telemetry_token}"
        # Constant-time comparison — a plain `==` leaks token length and
        # prefix bytes via response timing.
        if hmac.compare_digest(auth, expected):
            return await handler(request)
        return self._error_response("Unauthorized", "UNAUTHORIZED", 401)

    @middleware
    async def _telemetry_rate_limit_middleware(
        self, request: web.Request, handler,
    ) -> web.Response:
        """Per-IP token-bucket rate limit for telemetry endpoints."""
        if not request.path.startswith("/api/v1/telemetry/"):
            return await handler(request)
        key = request.remote or "unknown"
        if not self._telemetry_rate_limiter.allow(key):
            return web.json_response(
                {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMITED",
                    "timestamp": int(time.time()),
                },
                status=429,
            )
        return await handler(request)

    async def _prune_rate_limiter_loop(self) -> None:
        """Periodically prune stale rate-limiter buckets."""
        try:
            while True:
                await asyncio.sleep(300)
                if self._telemetry_cache is not None:
                    self._telemetry_rate_limiter.prune()
                self._solve_rate_limiter.prune()
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Telemetry ETag helpers
    # ------------------------------------------------------------------

    def _check_etag(
        self, request: web.Request, etag: str,
    ) -> Optional[web.Response]:
        """Return 304 response if client ETag matches, else None."""
        if not etag:
            return None
        if_none_match = request.headers.get("If-None-Match", "")
        if if_none_match == f'"{etag}"':
            return web.Response(status=304)
        return None

    def _etag_response(self, data: Any, etag: str) -> web.Response:
        """Success response with ETag header."""
        resp = web.json_response({
            "success": True,
            "data": data,
            "timestamp": int(time.time()),
        })
        if etag:
            resp.headers["ETag"] = f'"{etag}"'
        return resp

    # ------------------------------------------------------------------
    # Telemetry REST handlers
    # ------------------------------------------------------------------

    async def handle_telemetry_status(
        self, request: web.Request,
    ) -> web.Response:
        """GET /api/v1/telemetry/status"""
        cache = self._telemetry_cache
        if cache is None:
            return self._error_response(
                "Telemetry not enabled", "TELEMETRY_DISABLED", 503,
            )
        etag = cache.status_etag()
        not_modified = self._check_etag(request, etag)
        if not_modified:
            return not_modified
        return self._etag_response(cache.get_status(), etag)

    async def handle_telemetry_nodes(
        self, request: web.Request,
    ) -> web.Response:
        """GET /api/v1/telemetry/nodes"""
        cache = self._telemetry_cache
        if cache is None:
            return self._error_response(
                "Telemetry not enabled", "TELEMETRY_DISABLED", 503,
            )
        data = cache.get_nodes()
        if data is None:
            return self._error_response(
                "No node data available", "NO_DATA", 404,
            )
        etag = cache.nodes_etag()
        not_modified = self._check_etag(request, etag)
        if not_modified:
            return not_modified
        return self._etag_response(data, etag)

    async def handle_telemetry_epochs(
        self, request: web.Request,
    ) -> web.Response:
        """GET /api/v1/telemetry/epochs"""
        cache = self._telemetry_cache
        if cache is None:
            return self._error_response(
                "Telemetry not enabled", "TELEMETRY_DISABLED", 503,
            )
        return self._success_response({"epochs": cache.get_epochs()})

    async def handle_telemetry_block(
        self, request: web.Request,
    ) -> web.Response:
        """GET /api/v1/telemetry/epochs/{epoch}/blocks/{block_index}"""
        cache = self._telemetry_cache
        if cache is None:
            return self._error_response(
                "Telemetry not enabled", "TELEMETRY_DISABLED", 503,
            )
        epoch = request.match_info["epoch"]
        try:
            block_index = int(request.match_info["block_index"])
        except ValueError:
            return self._error_response(
                "Invalid block index", "INVALID_BLOCK_INDEX",
            )
        data = cache.get_block(epoch, block_index)
        if data is None:
            return self._error_response(
                f"Block {epoch}/{block_index} not found",
                "BLOCK_NOT_FOUND",
                404,
            )
        etag = cache.block_etag(data)
        not_modified = self._check_etag(request, etag)
        if not_modified:
            return not_modified
        return self._etag_response(data, etag)

    async def handle_telemetry_latest(
        self, request: web.Request,
    ) -> web.Response:
        """GET /api/v1/telemetry/latest"""
        cache = self._telemetry_cache
        if cache is None:
            return self._error_response(
                "Telemetry not enabled", "TELEMETRY_DISABLED", 503,
            )
        data = cache.get_latest()
        if data is None:
            return self._error_response(
                "No blocks available", "NO_BLOCKS", 404,
            )
        etag = cache.block_etag(data.get("block", {}))
        not_modified = self._check_etag(request, etag)
        if not_modified:
            return not_modified
        return self._etag_response(data, etag)

    # ------------------------------------------------------------------
    # SSE (Server-Sent Events)
    # ------------------------------------------------------------------

    async def handle_telemetry_stream(
        self, request: web.Request,
    ) -> web.StreamResponse:
        """GET /api/v1/telemetry/stream — SSE endpoint."""
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"

        # Hold the lock through the limit check AND prepare+append so
        # (a) concurrent connects can't all pass the check before any
        # append, and (b) broadcasts never see an unprepared response.
        # prepare() only sends headers, so the critical section is
        # short. _sse_broadcast does not acquire this lock.
        async with self._sse_lock:
            if len(self._sse_clients) >= self._telemetry_max_sse:
                return self._error_response(
                    "Too many SSE connections", "SSE_LIMIT", 429,
                )
            await resp.prepare(request)
            self._sse_clients.append(resp)
            self._sse_pending[resp] = 0

        # Register this handler so stop() can cancel it mid-sleep.
        task = asyncio.current_task()
        if task is not None:
            self._sse_tasks.add(task)
        try:
            # Keep connection alive until client disconnects
            while True:
                await asyncio.sleep(15)
                # Send keepalive comment
                try:
                    await resp.write(b":keepalive\n\n")
                except (ConnectionResetError, ConnectionAbortedError):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            async with self._sse_lock:
                if resp in self._sse_clients:
                    self._sse_clients.remove(resp)
                self._sse_pending.pop(resp, None)
            if task is not None:
                self._sse_tasks.discard(task)
        return resp

    def _sse_push_block(
        self, epoch: str, block_index: int, data: dict,
    ) -> None:
        """Push a new-block event to all SSE clients."""
        payload = json.dumps({
            "epoch": epoch, "block_index": block_index, "block": data,
        })
        msg = f"event: block\ndata: {payload}\n\n".encode("utf-8")
        self._sse_broadcast(msg)

    def _sse_push_nodes(self, data: dict) -> None:
        """Push a nodes-changed event to all SSE clients."""
        payload = json.dumps(data)
        msg = f"event: nodes\ndata: {payload}\n\n".encode("utf-8")
        self._sse_broadcast(msg)

    # Max outstanding writes allowed per SSE client before we drop it.
    # A slow client that builds up a backlog is ejected rather than
    # allowing the queue of pending write tasks to grow unboundedly.
    _SSE_MAX_PENDING_WRITES = 32

    def _sse_broadcast(self, msg: bytes) -> None:
        """Write an SSE message to all connected clients."""
        for client in list(self._sse_clients):
            pending = self._sse_pending.get(client, 0)
            if pending >= self._SSE_MAX_PENDING_WRITES:
                self.logger.warning(
                    "Dropping slow SSE client: %d pending writes",
                    pending,
                )
                self._sse_drop_client(client)
                continue
            self._sse_pending[client] = pending + 1
            task = asyncio.create_task(self._sse_write_one(client, msg))
            # Hold a strong reference so the task isn't GC'd mid-write.
            task.add_done_callback(lambda _t, c=client: self._sse_after_write(c))

    async def _sse_write_one(
        self, client: web.StreamResponse, msg: bytes,
    ) -> None:
        try:
            await client.write(msg)
        except (ConnectionResetError, ConnectionAbortedError, RuntimeError):
            self._sse_drop_client(client)

    def _sse_after_write(self, client: web.StreamResponse) -> None:
        pending = self._sse_pending.get(client)
        if pending is not None and pending > 0:
            self._sse_pending[client] = pending - 1

    def _sse_drop_client(self, client: web.StreamResponse) -> None:
        if client in self._sse_clients:
            self._sse_clients.remove(client)
        self._sse_pending.pop(client, None)
