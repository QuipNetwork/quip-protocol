import asyncio
import json
import socket
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Callable
from datetime import datetime
import aiohttp
from aiohttp import web
import logging

import copy

from shared.block import MinerInfo
from shared.node import Node

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_public_ip() -> Optional[str]:
    """
    Get the public IP address by querying external services.

    Returns:
        Public IP address as string, or None if unable to determine
    """
    # List of reliable IP detection services
    services = [
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://ipecho.net/plain",
        "https://checkip.amazonaws.com",
        "https://ident.me"
    ]

    for service in services:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(service, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        ip = (await response.text()).strip()
                        # Basic validation - check if it looks like an IP
                        if ip and '.' in ip and len(ip.split('.')) == 4:
                            logger.info(f"Detected public IP: {ip}")
                            return ip
        except Exception as e:
            logger.debug(f"Failed to get IP from {service}: {e}")
            continue

    logger.warning("Unable to determine public IP address")
    return None


def get_local_ip() -> str:
    """
    Get the local IP address (best guess).

    Returns:
        Local IP address as string
    """
    try:
        # Connect to a remote address to determine which local interface would be used
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Use Google's DNS server - we don't actually send data
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        # Fallback to localhost
        return "127.0.0.1"

@dataclass
class Message:
    """Base message structure for P2P communication."""
    type: str
    sender: str
    timestamp: float
    data: dict


class NetworkNode(Node):
    """Peer-to-peer node for quantum blockchain network."""
    
    def __init__(self, config: dict):
        self.bind_address = config.get("listen", "127.0.0.1")
        self.port = config.get("port", 20049)

        self.node_name = config.get("node_name", socket.getfqdn())
        self.public_host = config.get("public_host", f"{get_public_ip()}:{self.port}")
        
        self.secret = config.get("secret", "quip network node")
        self.auto_mine = config.get("auto_mine", False)

        self.heartbeat_interval = config.get("heartbeat_interval", 15)
        self.heartbeat_timeout = config.get("heartbeat_timeout", 300)
        self.node_timeout = config.get("node_timeout", 3)

        self.initial_peers = config.get("peer", ["nodes.quip.network:20049"])

        self.net_lock = asyncio.Lock()
        self.running = False
        self.heartbeats = {}
        
        # Callbacks
        self.on_new_node: Optional[Callable] = None
        self.on_node_lost: Optional[Callable] = None
        self.on_block_received: Optional[Callable] = None
        
        # Web server
        self.app = web.Application()
        self.setup_routes()
        self.runner = None
        
        # Background tasks
        self.heartbeat_task = None
        self.cleanup_task = None
        self.running = False

        self.miners_config = {
            "cpu": config.get("cpu", None),
            "gpu": config.get("gpu", None),
            "qpu": config.get("qpu", None)
        }
        super().__init__(self.node_name, self.miners_config, secret=self.secret)

    def setup_routes(self):
        """Setup HTTP routes for P2P communication."""
        self.app.router.add_post('/join', self.handle_join)
        self.app.router.add_post('/heartbeat', self.handle_heartbeat)
        self.app.router.add_post('/peers', self.handle_get_peers)
        self.app.router.add_post('/broadcast', self.handle_broadcast)
        self.app.router.add_post('/block', self.handle_new_block)
        self.app.router.add_get('/status', self.handle_status)
    
    async def start(self):
        """Start the P2P node."""
        self.running = True
        
        # Start web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.bind_address, self.port)
        await site.start()
        
        logger.info(f"Network node {self.node_name} ({self.crypto.ecdsa_public_key_hex[:8]}) started at {self.bind_address}:{self.port} with public address {self.public_host}")
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self.cleanup_loop())
    
    async def stop(self):
        """Stop the P2P node."""
        self.running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Stop web server
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Network node stopped")
    
    async def connect_to_peer(self, peer_address: str) -> bool:
        """Connect to a peer and join the network."""
        try:
            # Send join request
            async with aiohttp.ClientSession() as session:
                data = {
                    "address": self.public_host,
                    "version": "0.0.0",
                    "capabilities": ["mining", "relay"],
                    "info": self.info().to_network()
                }
                
                async with session.post(f"http://{peer_address}/join", json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        
                        # Add all nodes from the peer's node list
                        peers_found = 0
                        for peers_data in result.get("peers", {}):
                            for peer_host, peer_info in peers_data.items():
                                # except ourselves
                                if peer_host == self.public_host:
                                    continue
                                await self.add_node(peer_host, peer_info)
                                peers_found += 1
                        
                        logger.info(f"Successfully joined network via {peer_address}")
                        logger.info(f"Discovered {peers_found} peers") # type: ignore
                        return True
                    else:
                        logger.error(f"Failed to join via {peer_address}: {resp.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error connecting to peer {peer_address}: {e}")
            return False
    
    async def add_node(self, host: str, info: MinerInfo) -> bool:
        """Add a node to our registry."""
        if host == self.public_host:
            logger.warning(f"Skipping adding ourselves as a peer: {host}")
            return False 
        
        async with self.net_lock:
            is_new = self.add_or_update_peer(host, info)
            
            if is_new:
                logger.info(f"New node discovered: {host}: {info.miner_id} ({info.ecdsa_public_key.hex()[:8]})")
                if self.on_new_node:
                    asyncio.create_task(self.on_new_node(host, info))
                    
                # Broadcast new node to all other nodes
                asyncio.create_task(self.broadcast_new_node(host, info))
            
            return is_new
    
    async def remove_node(self, host: str):
        """Remove a node from our registry."""
        async with self.net_lock:
            if host in self.peers:
                del self.peers[host]
                logger.info(f"Node removed: {host}")
                if self.on_node_lost:
                    asyncio.create_task(self.on_node_lost(host))
    
    async def broadcast_new_node(self, new_node_address: str, new_node_info: MinerInfo):
        """Broadcast a new node to all known nodes."""
        message = Message(
            type="new_node",
            sender=self.public_host,
            timestamp=time.time(),
            data={
                "host": new_node_address,
                "info": new_node_info.to_json()
            }
        )
        
        await self.broadcast_message(message)
    
    async def broadcast_message(self, message: Message):
        """Broadcast a message to all known nodes."""
        tasks = []
        async with self.net_lock:
            for node_host in self.peers:
                if node_host != message.data.get("host"):  # Don't send to the node itself
                    task = asyncio.create_task(self.send_message(node_host, message))
                    tasks.append(task)
        
        # Wait for all broadcasts to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_message(self, node_host: str, message: Message) -> bool:
        """Send a message to a specific node."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{node_host}/broadcast",
                    json=asdict(message),
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.debug(f"Failed to send message to {node_host}: {e}")
            return False
    
    async def heartbeat_loop(self):
        """Send heartbeats to all known nodes."""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat to all nodes
                tasks = []
                async with self.net_lock:
                    for node_host in list(self.peers.keys()):
                        task = asyncio.create_task(self.send_heartbeat(node_host))
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def send_heartbeat(self, node_host: str) -> bool:
        """Send heartbeat to a specific node."""
        try:
            async with aiohttp.ClientSession() as session:
                data = {"sender": self.public_host}
                async with session.post(
                    f"http://{node_host}/heartbeat",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def get_peer_status(self, host: str) -> Optional[dict]:
        """Get status information from a peer node."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}/status",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.debug(f"Failed to get status from {host}: HTTP {resp.status}")
                        return None
        except Exception as e:
            logger.debug(f"Error getting status from {host}: {e}")
            return None

    async def refresh_peer_info(self, host: str) -> bool:
        """Refresh peer information by calling their status endpoint."""
        peer_status = await self.get_peer_status(host)
        if peer_status and 'info' in peer_status:
            try:
                info = MinerInfo.from_json(peer_status['info'])
                async with self.net_lock:
                    self.peers[host] = info

                logger.debug(f"Refreshed info for peer {host}")
                return True
            except Exception as e:
                logger.debug(f"Error parsing peer info from {host}: {e}")
                return False
        return False

    async def refresh_all_peer_info(self):
        """Refresh information for all known peers by calling their status endpoints."""
        async with self.net_lock:
            peer_hosts = list(self.peers.keys())

        for host in peer_hosts:
            await self.refresh_peer_info(host)
    
    async def cleanup_loop(self):
        """Remove dead nodes from registry."""
        while self.running:
            try:
                await asyncio.sleep(self.node_timeout / 2)
                
                # Find dead nodes
                current_time = time.time()
                dead_nodes = []
                
                async with self.net_lock:
                    for host, node_info in list(self.peers.items()):
                        if host not in self.heartbeats or current_time - self.heartbeats[host] > self.node_timeout:
                            dead_nodes.append(host)
                
                # Remove dead nodes
                for host in dead_nodes:
                    await self.remove_node(host)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    # HTTP Handlers
    
    async def handle_join(self, request: web.Request) -> web.Response:
        """Handle join request from a new node."""
        try:
            data = await request.json()
            new_node_address = data.get("host")
            new_node_info = MinerInfo.from_network(data.get("info"))
            
            if not new_node_address:
                return web.json_response({"error": "Missing address"}, status=400)
            
            # Add the new node
            await self.add_node(new_node_address, new_node_info)
            
            # Return our node list
            async with self.net_lock:
                peers_data = copy.deepcopy(self.peers)

            # Add ourself
            peers_data[self.public_host] = self.info()
            
            return web.json_response({
                "status": "ok",
                "peers": peers_data
            })
            
        except Exception as e:
            logger.error(f"Error handling join: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle heartbeat from another node."""
        try:
            data = await request.json()
            sender = data.get("sender")
            
            if sender:
                async with self.net_lock:
                    if sender in self.peers:
                        self.heartbeats[sender] = time.time()
                    else:
                        # New node discovered via heartbeat - get their info in background
                        logger.info(f"New node discovered via heartbeat: {sender}")
                        asyncio.create_task(self.refresh_peer_info(sender))
            
            return web.json_response({"status": "ok"})
            
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_get_peers(self, request: web.Request) -> web.Response:
        """Return list of known nodes."""
        async with self.net_lock:
            peers_data = copy.deepcopy(self.peers)
        
        return web.json_response({"peers": peers_data})
    
    async def handle_broadcast(self, request: web.Request) -> web.Response:
        """Handle broadcast message from another node."""
        try:
            data = await request.json()
            message = Message(**data)
            
            # Handle different message types
            if message.type == "new_node":
                new_host = message.data.get("host")
                new_info = MinerInfo.from_json(message.data.get("info", "{}"))
                if new_host:
                    await self.add_node(new_host, new_info)
            
            elif message.type == "block":
                if self.on_block_received:
                    asyncio.create_task(self.on_block_received(message.data))
            
            return web.json_response({"status": "ok"})
            
        except Exception as e:
            logger.error(f"Error handling broadcast: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_new_block(self, request: web.Request) -> web.Response:
        """Handle new block announcement."""
        try:
            block_data = await request.json()
            
            if self.on_block_received:
                asyncio.create_task(self.on_block_received(block_data))
            
            return web.json_response({"status": "ok"})
            
        except Exception as e:
            logger.error(f"Error handling new block: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_status(self, request: web.Request) -> web.Response:
        """Return node status."""
        
        return web.json_response({
            "host": self.public_host,
            "info": self.info().to_json(),
            "running": self.running,
            "total_peers": len(self.peers),
            "uptime": time.time() if self.running else 0
        })
    
    async def broadcast_block(self, block_data: dict):
        """Broadcast a new block to the network."""
        message = Message(
            type="block",
            sender=self.public_host,
            timestamp=time.time(),
            data=block_data
        )
        
        await self.broadcast_message(message)