"""
REST API server for QuIP network nodes.

Exposes all QUIC message types as HTTP endpoints for browser and external access.
Runs alongside the QUIC server on a separate port.
"""

import asyncio
import json
import logging
import ssl
import struct
import time
from typing import TYPE_CHECKING, Any, Optional

from aiohttp import web
from aiohttp.web import middleware

from shared.time_utils import utc_timestamp_float

if TYPE_CHECKING:
    from shared.network_node import NetworkNode
    from shared.certificate_manager import CertificateManager


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
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the REST API server.

        Args:
            network_node: The NetworkNode instance to expose
            host: Host address to bind to
            port: HTTP port (non-TLS)
            tls_port: HTTPS port (TLS)
            cert_manager: Certificate manager for TLS
            logger: Logger instance
        """
        self.node = network_node
        self.host = host
        self.http_port = port
        self.https_port = tls_port
        self.cert_manager = cert_manager
        self.logger = logger or logging.getLogger(__name__)

        self._http_runner: Optional[web.AppRunner] = None
        self._https_runner: Optional[web.AppRunner] = None
        self._app: Optional[web.Application] = None

    def _create_app(self) -> web.Application:
        """Create and configure the aiohttp application."""
        app = web.Application(middlewares=[cors_middleware, error_middleware])

        # Health check
        app.router.add_get("/health", self.handle_health)

        # API v1 routes
        app.router.add_get("/api/v1/status", self.handle_status)
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

        # OPTIONS handler for CORS preflight
        app.router.add_route("OPTIONS", "/{path:.*}", self.handle_options)

        return app

    async def start(self) -> None:
        """Start the REST API server."""
        self._app = self._create_app()

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
        if self._http_runner:
            await self._http_runner.cleanup()
            self._http_runner = None

        if self._https_runner:
            await self._https_runner.cleanup()
            self._https_runner = None

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
            "version": self.node.info().version if self.node.running else "unknown"
        })

    async def handle_status(self, request: web.Request) -> web.Response:
        """GET /api/v1/status - Node status information."""
        status_data = {
            "host": self.node.public_host,
            "info": json.loads(self.node.info().to_json()),
            "running": self.node.running,
            "total_peers": len(self.node.peers),
            "uptime": utc_timestamp_float() if self.node.running else 0,
            "latest_block": self.node.get_latest_block().header.index if self.node.get_latest_block() else 0
        }
        return self._success_response(status_data)

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

        if new_node_info:
            await self.node.add_peer(new_node_address, new_node_info)

        # Return our peer list
        async with self.node.net_lock:
            peers_snapshot = dict(self.node.peers)

        peers_payload = {
            host: json.loads(info.to_json())
            for host, info in peers_snapshot.items()
        }
        peers_payload[self.node.public_host] = json.loads(self.node.info().to_json())

        return self._success_response({"status": "ok", "peers": peers_payload})

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
            # Create appropriate sampler based on miner type
            if miner_kind == "qpu":
                from dwave.system import DWaveSampler
                sampler = DWaveSampler()
            elif miner_kind in ["cpu", "metal", "cuda", "modal"]:
                from dwave.samplers import SimulatedAnnealingSampler
                sampler = SimulatedAnnealingSampler()
            else:
                return self._error_response(
                    f"Unknown miner type: {miner_kind}",
                    "UNKNOWN_MINER_TYPE"
                )

            # Sample the Ising problem
            sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=num_samples)

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
            from shared.time_utils import utc_timestamp

            transaction = Transaction(
                transaction_id=transaction_id,
                timestamp=utc_timestamp(),
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
        return {
            "header": self._header_to_dict(block.header),
            "transactions": [
                {
                    "transaction_id": tx.transaction_id,
                    "timestamp": tx.timestamp,
                    "num_samples": tx.num_samples,
                    "samples_count": len(tx.samples) if tx.samples else 0,
                    "energy_range": [min(tx.energies), max(tx.energies)] if tx.energies else None
                }
                for tx in (block.transactions or [])
            ],
            "signature_hex": block.signature.hex() if block.signature else None
        }

    def _header_to_dict(self, header) -> dict:
        """Convert a BlockHeader to a JSON-serializable dict."""
        return {
            "index": header.index,
            "timestamp": header.timestamp,
            "prev_hash_hex": header.prev_hash.hex() if header.prev_hash else None,
            "pow_hash_hex": header.pow_hash.hex() if header.pow_hash else None,
            "merkle_root_hex": header.merkle_root.hex() if header.merkle_root else None,
            "miner_info": json.loads(header.miner_info.to_json()) if header.miner_info else None,
            "pow_difficulty": header.pow_difficulty,
            "pow_energy": header.pow_energy,
            "diversity": header.diversity,
            "num_solutions": header.num_solutions
        }
