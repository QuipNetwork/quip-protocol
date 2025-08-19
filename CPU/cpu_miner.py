"""CPU-based Simulated Annealing miner for quantum blockchain."""

import asyncio
import logging
import sys
sys.path.append('..')

from shared.blockchain_shared import SharedMiningNode

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for CPU miner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CPU Mining Node')
    parser.add_argument('--id', type=int, default=1, help='Node ID')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--peer', help='Peer address to connect to (host:port)')
    parser.add_argument('--num-sweeps', type=int, default=4096, 
                       help='Number of sweeps for simulated annealing')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create CPU mining node
    node = SharedMiningNode(
        miner_type="CPU",
        miner_id=args.id,
        host=args.host,
        port=args.port,
        num_sweeps=args.num_sweeps
    )
    
    try:
        # Start node
        await node.start()
        
        if args.peer:
            # Connect to network
            logger.info(f"Connecting to peer: {args.peer}")
            success = await node.connect_to_network(args.peer)
            if success:
                logger.info("Successfully joined network!")
            else:
                logger.error("Failed to join network!")
                # Start mining anyway with genesis block
                await node.start_mining()
        else:
            # No peer specified, start mining from genesis
            logger.info("Starting as bootstrap node")
            await node.start_mining()
        
        logger.info(f"CPU Mining Node {args.id} running at {args.host}:{args.port}")
        logger.info(f"Simulated Annealing with {args.num_sweeps} sweeps")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down CPU miner...")
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())