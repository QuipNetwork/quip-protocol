"""CPU worker process for CPU-based simulated annealing mining."""

import random
import sys
import time
import numpy as np

from CPU.sa_miner import SimulatedAnnealingMiner
from shared.base_miner import MiningResult


def cpu_mine_block_process(miner_data, block_header: str, result_queue, stop_event):
    """CPU-specific mining process function.
    
    Args:
        miner_data: Serialized miner data (type, id, config)
        block_header: Block header to mine
        result_queue: Queue to put results
        stop_event: Event to signal stop
    """
    miner_id = miner_data['id']
    miner_config = miner_data.get('config', {})
    
    # Create CPU miner
    miner = SimulatedAnnealingMiner(
        miner_id=miner_id
    )
    
    # Call the mine_block method
    miner.mine_block(block_header, result_queue, stop_event)