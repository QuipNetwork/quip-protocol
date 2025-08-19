"""GPU-accelerated miner for quantum blockchain using Modal."""

import asyncio
import logging
import sys
sys.path.append('..')

from shared.blockchain_shared import SharedMiningNode

logger = logging.getLogger(__name__)

# Check if Modal is available
try:
    import modal
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("Modal not installed. GPU mining disabled.")


async def main():
    """Main entry point for GPU miner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Mining Node')
    parser.add_argument('--id', type=int, default=1, help='Node ID')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--peer', help='Peer address to connect to (host:port)')
    parser.add_argument('--gpu-type', default='t4', choices=['t4', 'a10g', 'a100'],
                       help='GPU type to use')
    parser.add_argument('--num-sweeps', type=int, default=512,
                       help='Number of sweeps for GPU annealing')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check GPU availability
    if not GPU_AVAILABLE:
        logger.warning("Modal not available, using CPU miner instead")
        miner_type = "CPU"
        gpu_type = None
        # Increase sweeps for CPU fallback
        num_sweeps = args.num_sweeps * 8
    else:
        miner_type = "GPU"
        gpu_type = args.gpu_type
        num_sweeps = args.num_sweeps
    
    # Create mining node
    node = SharedMiningNode(
        miner_type=miner_type,
        miner_id=args.id,
        host=args.host,
        port=args.port,
        num_sweeps=num_sweeps,
        gpu_type=gpu_type
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
        
        if miner_type == "GPU":
            logger.info(f"GPU Mining Node {args.id} running at {args.host}:{args.port}")
            logger.info(f"GPU Type: {args.gpu_type.upper()}, Sweeps: {args.num_sweeps}")
        else:
            logger.info(f"CPU Mining Node {args.id} (GPU fallback) running at {args.host}:{args.port}")
            logger.info(f"Simulated Annealing with {num_sweeps} sweeps")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down GPU miner...")
        await node.stop()


def run_with_modal():
    """Run the miner with Modal context if available."""
    if GPU_AVAILABLE:
        from quantum_blockchain import gpu_app
        with gpu_app.run():
            asyncio.run(main())
    else:
        asyncio.run(main())


if __name__ == "__main__":
    run_with_modal()