"""CPU worker process for CPU-based simulated annealing mining."""

import hashlib
import random
import sys
import time
import numpy as np

from .sa_sampler import SimulatedAnnealingStructuredSampler
from shared.miner import Miner, MiningResult


def cpu_mine_block_process(miner_data, block_header: str, result_queue, stop_event):
    """CPU-specific mining process function.
    
    Args:
        miner_data: Serialized miner data (type, id, config)
        block_header: Block header to mine
        result_queue: Queue to put results
        stop_event: Event to signal stop
    """
    miner_type = miner_data['type']
    miner_id = miner_data['id']
    miner_config = miner_data.get('config', {})
    
    # Create CPU sampler
    sampler = SimulatedAnnealingStructuredSampler()
    
    miner = Miner(
        miner_id, 
        miner_id, 
        sampler, 
        difficulty_energy=miner_config['difficulty_energy'],
        min_diversity=miner_config['min_diversity'],
        min_solutions=miner_config['min_solutions']
    )
    
    # Call the original Miner.mine_block method
    miner.mine_block(block_header, result_queue, stop_event)