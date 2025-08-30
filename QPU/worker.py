"""QPU worker process for D-Wave quantum annealing mining."""

from .dwave_sampler import DWaveSamplerWrapper
from shared.miner import Miner, MiningResult


def qpu_mine_block_process(miner_data, block, requirements, result_queue, stop_event):
    """QPU-specific mining process function.
    
    Args:
        miner_data: Serialized miner data (type, id, config)
        block: Block object to mine
        requirements: NextBlockRequirements object with difficulty settings
        result_queue: Queue to put results
        stop_event: Event to signal stop
    """
    miner_id = miner_data['id']
    miner_config = miner_data.get('config', {})
    
    # Create QPU miner
    from QPU.dwave_miner import DWaveMiner
    miner = DWaveMiner(miner_id=miner_id)
    
    # Call the mine_block method
    miner.mine_block(block, requirements, result_queue, stop_event)