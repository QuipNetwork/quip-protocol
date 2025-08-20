"""QPU (Quantum Processing Unit) miner for quantum blockchain using D-Wave."""

import asyncio
import logging
import sys
import os
sys.path.append('..')

from shared.blockchain_shared import SharedMiningNode

logger = logging.getLogger(__name__)

# Check if D-Wave is available
try:
    from dwave.system import DWaveSampler
    QPU_AVAILABLE = True
except ImportError:
    QPU_AVAILABLE = False
    logger.warning("D-Wave Ocean SDK not installed. QPU mining disabled.")


async def main():
    """Main entry point for QPU miner."""
    import argparse

    parser = argparse.ArgumentParser(description='QPU Mining Node')
    parser.add_argument('--id', type=int, default=1, help='Node ID')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--peer', help='Peer address to connect to (host:port)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check QPU availability
    if not QPU_AVAILABLE or not os.getenv('DWAVE_API_TOKEN'):
        if not QPU_AVAILABLE:
            logger.warning("D-Wave Ocean SDK not installed, using CPU miner instead")
        else:
            logger.warning("DWAVE_API_TOKEN not set, using CPU miner instead")
        miner_type = "CPU"
        # Use high sweep count for CPU fallback
        num_sweeps = 8192
    else:
        # Test QPU connection
        try:
            test_sampler = DWaveSampler()
            logger.info(f"Connected to QPU: {test_sampler.properties.get('chip_id', 'Unknown')}")
            miner_type = "QPU"
            num_sweeps = None  # Not used for QPU
        except Exception as e:
            logger.error(f"Failed to connect to QPU: {e}")
            logger.warning("Using CPU miner instead")
            miner_type = "CPU"
            num_sweeps = 8192

    # Create mining node
    node = SharedMiningNode(
        miner_type=miner_type,
        miner_id=args.id,
        host=args.host,
        port=args.port,
        num_sweeps=num_sweeps
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

        if miner_type == "QPU":
            logger.info(f"QPU Mining Node {args.id} running at {args.host}:{args.port}")
            logger.info("Using D-Wave quantum annealer")
        else:
            logger.info(f"CPU Mining Node {args.id} (QPU fallback) running at {args.host}:{args.port}")
            logger.info(f"Simulated Annealing with {num_sweeps} sweeps")

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down QPU miner...")
        await node.stop()



def cli():
    """Console-script entry point."""
    import asyncio as _asyncio
    _asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())