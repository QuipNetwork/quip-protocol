import asyncio
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Callable
from datetime import datetime
import aiohttp
from aiohttp import web
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a network node."""
    address: str  # host:port
    last_seen: float
    version: str = "1.0.0"
    capabilities: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["mining", "relay"]
    
    def is_alive(self, timeout: float = 60.0) -> bool:
        """Check if node is still alive based on last heartbeat."""
        return (time.time() - self.last_seen) < timeout


@dataclass
class Message:
    """Base message structure for P2P communication."""
    type: str
    sender: str
    timestamp: float
    data: dict


class P2PNode:
    """Peer-to-peer node for quantum blockchain network."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, 
                 node_timeout: float = 60.0, heartbeat_interval: float = 15.0):
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.node_timeout = node_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Node registry
        self.nodes: Dict[str, NodeInfo] = {}
        self.nodes_lock = asyncio.Lock()
        
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
        
    def setup_routes(self):
        """Setup HTTP routes for P2P communication."""
        self.app.router.add_post('/join', self.handle_join)
        self.app.router.add_post('/heartbeat', self.handle_heartbeat)
        self.app.router.add_post('/nodes', self.handle_get_nodes)
        self.app.router.add_post('/broadcast', self.handle_broadcast)
        self.app.router.add_post('/block', self.handle_new_block)
        self.app.router.add_get('/status', self.handle_status)
    
    async def start(self):
        """Start the P2P node."""
        self.running = True
        
        # Start web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        
        logger.info(f"P2P node started at {self.address}")
        
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
        
        logger.info("P2P node stopped")
    
    async def connect_to_peer(self, peer_address: str) -> bool:
        """Connect to a peer and join the network."""
        try:
            # Send join request
            async with aiohttp.ClientSession() as session:
                data = {
                    "address": self.address,
                    "version": "1.0.0",
                    "capabilities": ["mining", "relay"]
                }
                
                async with session.post(f"http://{peer_address}/join", json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        
                        # Add the peer we connected to
                        await self.add_node(peer_address)
                        
                        # Add all nodes from the peer's node list
                        for node_data in result.get("nodes", []):
                            node_info = NodeInfo(**node_data)
                            if node_info.address != self.address:  # Don't add ourselves
                                await self.add_node(node_info.address)
                        
                        logger.info(f"Successfully joined network via {peer_address}")
                        logger.info(f"Discovered {len(self.nodes)} peers")
                        return True
                    else:
                        logger.error(f"Failed to join via {peer_address}: {resp.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error connecting to peer {peer_address}: {e}")
            return False
    
    async def add_node(self, address: str) -> bool:
        """Add a node to our registry."""
        if address == self.address:
            return False  # Don't add ourselves
        
        async with self.nodes_lock:
            is_new = address not in self.nodes
            self.nodes[address] = NodeInfo(
                address=address,
                last_seen=time.time()
            )
            
            if is_new:
                logger.info(f"New node discovered: {address}")
                if self.on_new_node:
                    asyncio.create_task(self.on_new_node(address))
                    
                # Broadcast new node to all other nodes
                asyncio.create_task(self.broadcast_new_node(address))
            
            return is_new
    
    async def remove_node(self, address: str):
        """Remove a node from our registry."""
        async with self.nodes_lock:
            if address in self.nodes:
                del self.nodes[address]
                logger.info(f"Node removed: {address}")
                if self.on_node_lost:
                    asyncio.create_task(self.on_node_lost(address))
    
    async def broadcast_new_node(self, new_node_address: str):
        """Broadcast a new node to all known nodes."""
        message = Message(
            type="new_node",
            sender=self.address,
            timestamp=time.time(),
            data={"address": new_node_address}
        )
        
        await self.broadcast_message(message)
    
    async def broadcast_message(self, message: Message):
        """Broadcast a message to all known nodes."""
        tasks = []
        async with self.nodes_lock:
            for node_address in self.nodes:
                if node_address != message.data.get("address"):  # Don't send to the node itself
                    task = asyncio.create_task(self.send_message(node_address, message))
                    tasks.append(task)
        
        # Wait for all broadcasts to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_message(self, node_address: str, message: Message) -> bool:
        """Send a message to a specific node."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{node_address}/broadcast",
                    json=asdict(message),
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.debug(f"Failed to send message to {node_address}: {e}")
            return False
    
    async def heartbeat_loop(self):
        """Send heartbeats to all known nodes."""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat to all nodes
                tasks = []
                async with self.nodes_lock:
                    for node_address in list(self.nodes.keys()):
                        task = asyncio.create_task(self.send_heartbeat(node_address))
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def send_heartbeat(self, node_address: str) -> bool:
        """Send heartbeat to a specific node."""
        try:
            async with aiohttp.ClientSession() as session:
                data = {"sender": self.address}
                async with session.post(
                    f"http://{node_address}/heartbeat",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def cleanup_loop(self):
        """Remove dead nodes from registry."""
        while self.running:
            try:
                await asyncio.sleep(self.node_timeout / 2)
                
                # Find dead nodes
                current_time = time.time()
                dead_nodes = []
                
                async with self.nodes_lock:
                    for address, node_info in list(self.nodes.items()):
                        if not node_info.is_alive(self.node_timeout):
                            dead_nodes.append(address)
                
                # Remove dead nodes
                for address in dead_nodes:
                    await self.remove_node(address)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    # HTTP Handlers
    
    async def handle_join(self, request: web.Request) -> web.Response:
        """Handle join request from a new node."""
        try:
            data = await request.json()
            new_node_address = data.get("address")
            
            if not new_node_address:
                return web.json_response({"error": "Missing address"}, status=400)
            
            # Add the new node
            await self.add_node(new_node_address)
            
            # Return our node list
            async with self.nodes_lock:
                nodes_data = [asdict(node) for node in self.nodes.values()]
            
            return web.json_response({
                "status": "ok",
                "nodes": nodes_data
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
                async with self.nodes_lock:
                    if sender in self.nodes:
                        self.nodes[sender].last_seen = time.time()
                    else:
                        # New node discovered via heartbeat
                        await self.add_node(sender)
            
            return web.json_response({"status": "ok"})
            
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_get_nodes(self, request: web.Request) -> web.Response:
        """Return list of known nodes."""
        async with self.nodes_lock:
            nodes_data = [asdict(node) for node in self.nodes.values()]
        
        return web.json_response({"nodes": nodes_data})
    
    async def handle_broadcast(self, request: web.Request) -> web.Response:
        """Handle broadcast message from another node."""
        try:
            data = await request.json()
            message = Message(**data)
            
            # Handle different message types
            if message.type == "new_node":
                new_address = message.data.get("address")
                if new_address:
                    await self.add_node(new_address)
            
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
        async with self.nodes_lock:
            active_nodes = sum(1 for node in self.nodes.values() if node.is_alive(self.node_timeout))
        
        return web.json_response({
            "address": self.address,
            "running": self.running,
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "uptime": time.time() if self.running else 0
        })
    
    async def broadcast_block(self, block_data: dict):
        """Broadcast a new block to the network."""
        message = Message(
            type="block",
            sender=self.address,
            timestamp=time.time(),
            data=block_data
        )
        
        await self.broadcast_message(message)


async def main():
    """Example usage of P2P node."""
    
    # Get port from command line or use default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    # Create node
    node = P2PNode(port=port)
    
    # Define callbacks
    async def on_new_node(address):
        logger.info(f"New node joined: {address}")

    async def on_node_lost(address):
        logger.info(f"Node lost: {address}")

    async def on_block_received(block_data):
        logger.info(f"New block received: {block_data.get('index', 'unknown')}")
    
    node.on_new_node = on_new_node
    node.on_node_lost = on_node_lost
    node.on_block_received = on_block_received
    
    # Start node
    await node.start()
    
    # If peer address provided, connect to it
    if len(sys.argv) > 2:
        peer_address = sys.argv[2]
        logger.info(f"Connecting to peer: {peer_address}")
        success = await node.connect_to_peer(peer_address)
        if not success:
            logger.warning("Failed to connect to peer!")

    logger.info(f"Node running at {node.address}")
    logger.info("Press Ctrl+C to stop")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())